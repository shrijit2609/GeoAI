"""Regression tests for the M2/M3 training dataset loaders.

Background: the Colab M2 run failed inside a DataLoader worker with
``TypeError: argument should be a str or an os.PathLike object ... not
'method'``. ``discover_pairs`` builds a DataFrame with a column literally named
``mask``, and ``self.df.iloc[index]`` returns a ``pandas.Series``, which defines
a built-in ``.mask()`` method. Python resolves that real class attribute before
pandas' column fallback, so ``row.mask`` yielded the bound method rather than
the path.

These tests pin the fix (bracket access) and guard against regressions.
"""
from __future__ import annotations

import ast
import contextlib
import importlib
import inspect
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
M2_PATH = REPO_ROOT / "training/model2_building_extractor/train.py"
M3_PATH = REPO_ROOT / "training/model3_change_detection/train.py"


def _purge() -> None:
    for name in list(sys.modules):
        if name == "train" or name.startswith("ssa_test_"):
            del sys.modules[name]


@contextlib.contextmanager
def load_trainer(directory: Path):
    """Import ``training/<x>/train.py`` under its real module name ``train``.

    Using the real name (and putting the directory on ``sys.path``) keeps the
    module importable by name inside spawned DataLoader workers on Windows,
    which must re-import it to unpickle the Dataset. Colab uses fork and is
    unaffected either way.
    """
    _purge()
    sys.path.insert(0, str(directory))
    try:
        yield importlib.import_module("train")
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(directory))
        _purge()


@pytest.fixture
def m2_train():
    with load_trainer(M2_PATH.parent) as module:
        yield module


@pytest.fixture
def m3_train():
    with load_trainer(M3_PATH.parent) as module:
        yield module


# ---------------------------------------------------------------- core bug
def test_series_mask_attribute_is_the_pandas_method_not_the_column():
    """Documents the trap: attribute access silently returns a bound method."""
    row = pd.DataFrame([{"id": "a", "image": "/i/a.png", "mask": "/m/a.png"}]).iloc[0]
    assert callable(row.mask)
    assert isinstance(row["mask"], str)


def test_trainers_never_access_dataframe_rows_by_attribute():
    """Static guard: no `row.<name>` access anywhere in either trainer."""
    for path in (M2_PATH, M3_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders = [
            f"row.{node.attr}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "row"
        ]
        assert not offenders, f"{path.name} uses attribute access on rows: {offenders}"


def test_building_dataset_getitem_uses_bracket_access(m2_train):
    source = inspect.getsource(m2_train.BuildingDataset.__getitem__)
    assert 'row["mask"]' in source
    assert 'row["image"]' in source
    # Assert on the parsed tree, not on raw text: the explanatory comment
    # legitimately mentions `row.mask`.
    tree = ast.parse(textwrap.dedent(source))
    offenders = [
        f"row.{node.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "row"
    ]
    assert not offenders, f"__getitem__ still uses attribute access: {offenders}"


# ------------------------------------------------------------------- M2
def _build_whu_fixture(root: Path, per_split: int = 40) -> Path:
    from PIL import Image

    dataset = root / "whu"
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        for kind in ("Image", "Mask"):
            (dataset / split / kind).mkdir(parents=True, exist_ok=True)
        for i in range(per_split):
            img = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
            msk = (rng.integers(0, 2, (256, 256)) * 255).astype("uint8")
            Image.fromarray(img).save(dataset / split / "Image" / f"{split}_{i:04d}.png")
            Image.fromarray(msk).save(dataset / split / "Mask" / f"{split}_{i:04d}.png")
    return dataset


@pytest.fixture
def whu_frames(m2_train, tmp_path):
    pytest.importorskip("cv2", reason="opencv is only required for the raster readers")
    return m2_train.discover_pairs(_build_whu_fixture(tmp_path))


def test_discover_pairs_columns_are_plain_strings(whu_frames, m2_train):
    train_df = whu_frames[0]
    assert len(train_df) > 0
    row = train_df.iloc[0]
    assert isinstance(row["image"], str) and Path(row["image"]).is_file()
    assert isinstance(row["mask"], str) and Path(row["mask"]).is_file()


def test_building_dataset_returns_rgb_image_and_single_channel_mask(whu_frames, m2_train):
    train_df = whu_frames[0]
    image, mask = m2_train.BuildingDataset(train_df, training=True)[0]
    assert tuple(image.shape) == (3, m2_train.IMAGE_SIZE, m2_train.IMAGE_SIZE)
    assert tuple(mask.shape) == (1, m2_train.IMAGE_SIZE, m2_train.IMAGE_SIZE)
    assert image.dtype is torch.float32 and mask.dtype is torch.float32
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_building_dataloader_workers_produce_a_batch(whu_frames, m2_train):
    """This is the exact configuration that raised the Colab TypeError."""
    from torch.utils.data import DataLoader

    train_df = whu_frames[0]
    loader = DataLoader(
        m2_train.BuildingDataset(train_df, training=True),
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    images, masks = next(iter(loader))
    assert images.shape[0] == 8
    assert tuple(images.shape[1:]) == (3, m2_train.IMAGE_SIZE, m2_train.IMAGE_SIZE)
    assert tuple(masks.shape[1:]) == (1, m2_train.IMAGE_SIZE, m2_train.IMAGE_SIZE)


def test_building_batch_transfers_to_device(whu_frames, m2_train):
    from torch.utils.data import DataLoader

    train_df = whu_frames[0]
    loader = DataLoader(m2_train.BuildingDataset(train_df), batch_size=2, num_workers=0)
    images, _masks = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert images.to(device).device.type == device.type


# ------------------------------------------------------------------- M3
def test_change_dataset_rows_are_positional_and_bracket_accessed(m3_train):
    assert "row[" in inspect.getsource(m3_train.extract_row)
    assert "self.rows[index]" in inspect.getsource(m3_train.ChangeDataset.__getitem__)


def test_change_dataset_shapes_and_worker_batch(m3_train):
    from torch.utils.data import DataLoader

    rng = np.random.default_rng(1)
    rows = [
        (
            rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
            rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
            # load_splits()/extract_row() emit binary masks in {0, 1}; the
            # dataset must consume exactly that, it does not rescale.
            rng.integers(0, 2, (256, 256)).astype(np.uint8),
        )
        for _ in range(8)
    ]
    before, after, target = m3_train.ChangeDataset(rows, augment=True)[0]
    size = m3_train.IMAGE_SIZE
    assert tuple(before.shape) == (3, size, size)
    assert tuple(after.shape) == (3, size, size)
    assert tuple(target.shape) == (1, size, size)
    assert set(torch.unique(target).tolist()) <= {0.0, 1.0}

    loader = DataLoader(m3_train.ChangeDataset(rows, augment=True), batch_size=4, num_workers=2)
    b, a, t = next(iter(loader))
    assert b.shape[0] == 4 and tuple(b.shape[2:]) == (size, size)
    assert tuple(t.shape[1:]) == (1, size, size)


# ------------------------------------------------------------- contracts
def test_training_contracts_are_unchanged(m2_train, m3_train):
    assert m2_train.EPOCHS == 12
    assert m3_train.EPOCHS == 5
    assert m2_train.IMAGE_SIZE == 256
    assert m3_train.IMAGE_SIZE == 256
    assert "clip_grad_norm_(model.parameters(), 1.0)" in M3_PATH.read_text(encoding="utf-8")

