"""Notebook-cell-82 WHU Building Dataset training pipeline for Google Colab/T4."""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
IMAGE_SIZE = 256
EPOCHS = 12
TRAIN_BATCH = 8
EVAL_BATCH = 8


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_stem(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in ("_mask", "_masks", "_label", "_labels", "_gt", "_groundtruth", "_ground_truth"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def discover_pairs(dataset_dir: Path) -> pd.DataFrame:
    all_rasters = [path for path in dataset_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    image_files: list[Path] = []
    mask_files: list[Path] = []
    for path in all_rasters:
        lower = str(path).lower()
        if "/mask/" in lower or "\\mask\\" in lower or "mask" in path.name.lower() or "label" in path.name.lower():
            mask_files.append(path)
        elif "/image/" in lower or "\\image\\" in lower or "image" in path.name.lower():
            image_files.append(path)
    if not image_files or not mask_files:
        raise RuntimeError("Could not identify WHU image and mask paths; inspect the downloaded dataset layout rather than training on unpaired files.")
    image_map = {clean_stem(path): path for path in image_files}
    mask_map = {clean_stem(path): path for path in mask_files}
    common_ids = sorted(image_map.keys() & mask_map.keys())
    if len(common_ids) < 100:
        raise RuntimeError(f"Only {len(common_ids)} image/mask pairs found; notebook cell 82 requires at least 100.")
    pairs = pd.DataFrame([{"id": sample_id, "image": str(image_map[sample_id]), "mask": str(mask_map[sample_id])} for sample_id in common_ids])

    def infer_split(path: str) -> str | None:
        lower = path.lower()
        if "/train/" in lower or "\\train\\" in lower:
            return "train"
        if any(token in lower for token in ("/val/", "\\val\\", "/validation/", "\\validation\\")):
            return "val"
        if "/test/" in lower or "\\test\\" in lower:
            return "test"
        return None

    pairs["split"] = pairs["image"].map(infer_split)
    if pairs["split"].notna().all():
        train_df = pairs[pairs.split == "train"].copy()
        val_df = pairs[pairs.split == "val"].copy()
        test_df = pairs[pairs.split == "test"].copy()
    else:
        train_df, temporary = train_test_split(pairs, test_size=0.30, random_state=42)
        val_df, test_df = train_test_split(temporary, test_size=0.50, random_state=42)
    splits = [frame.drop_duplicates("id").reset_index(drop=True) for frame in (train_df, val_df, test_df)]
    ids = [set(frame.id) for frame in splits]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise RuntimeError("WHU source-level split leakage detected")
    return splits


class BuildingDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, training: bool = False):
        self.df = dataframe.reset_index(drop=True)
        self.training = training

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def read_image(path: str) -> np.ndarray:
        import cv2

        if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg"}:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot read image: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            with rasterio.open(path) as source:
                image = np.transpose(source.read(), (1, 2, 0))
            if image.shape[2] < 3:
                raise ValueError(f"Image has fewer than 3 channels: {path}")
            image = image[:, :, :3]
        image = image.astype(np.float32)
        if image.max() > 1:
            image /= 255.0
        return np.clip(image, 0, 1)

    @staticmethod
    def read_mask(path: str) -> np.ndarray:
        import cv2

        if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg"}:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Cannot read mask: {path}")
        else:
            with rasterio.open(path) as source:
                mask = source.read(1)
        return (mask > 0).astype(np.float32)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        import cv2

        row = self.df.iloc[index]
        image = cv2.resize(self.read_image(row.image), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(self.read_mask(row.mask), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        if self.training:
            if random.random() < 0.5:
                image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
            if random.random() < 0.5:
                image, mask = np.flipud(image).copy(), np.flipud(mask).copy()
            if random.random() < 0.5:
                k = random.randint(1, 3)
                image, mask = np.rot90(image, k).copy(), np.rot90(mask, k).copy()
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask[None, :, :]).float()
        return image_tensor, mask_tensor


def build_model(pretrained: bool = True) -> nn.Module:
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet50(weights=weights)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    return model


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probabilities = torch.sigmoid(logits).reshape(logits.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    intersection = (probabilities * targets).sum(dim=1)
    dice = (2 * intersection + smooth) / (probabilities.sum(dim=1) + targets.sum(dim=1) + smooth)
    return 1 - dice.mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return 0.5 * F.binary_cross_entropy_with_logits(logits, targets) + 0.5 * dice_loss(logits, targets)


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            loss = combined_loss(model(images)["out"], masks)
            total += loss.item() * images.size(0)
            count += images.size(0)
    return total / max(count, 1)


def evaluate_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            predictions = (torch.sigmoid(model(images)["out"]) >= 0.5).bool()
            actual = masks.bool()
            tp += int((predictions & actual).sum())
            fp += int((predictions & ~actual).sum())
            fn += int((~predictions & actual).sum())
            tn += int((~predictions & ~actual).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + 1e-9),
        "iou": tp / (tp + fp + fn + 1e-9),
        "pixel_accuracy": (tp + tn) / (tp + tn + fp + fn + 1e-9),
        "threshold": 0.5,
    }


def train(dataset_dir: Path, project_root: Path, resume: bool = False) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("This training job is intended for Google Colab with a CUDA/T4 runtime. No CPU fallback training is started.")
    seed_everything(42)
    device = torch.device("cuda")
    train_df, val_df, test_df = discover_pairs(dataset_dir)
    train_loader = DataLoader(BuildingDataset(train_df, True), batch_size=TRAIN_BATCH, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(BuildingDataset(val_df), batch_size=EVAL_BATCH, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(BuildingDataset(test_df), batch_size=EVAL_BATCH, shuffle=False, num_workers=2, pin_memory=True)

    model_dir = project_root / "models/model2_building_extractor"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "building_deeplabv3_resnet50_best.pt"
    metrics_path = model_dir / "metrics.json"
    history_path = model_dir / "training_history.json"
    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    start_epoch = 0
    if resume and best_path.is_file():
        saved = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        if "optimizer_state_dict" in saved:
            optimizer.load_state_dict(saved["optimizer_state_dict"])
        best_validation_loss = float(saved.get("val_loss", best_validation_loss))
        start_epoch = int(saved.get("epoch", 0))
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print(f"Resuming from epoch {start_epoch + 1} (best_val_loss={best_validation_loss:.5f})")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        started = time.time()
        total = 0.0
        seen = 0
        for images, masks in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = combined_loss(model(images)["out"], masks)
            loss.backward()
            optimizer.step()
            total += loss.item() * images.size(0)
            seen += images.size(0)
        train_loss = total / max(seen, 1)
        validation_loss = evaluate_loss(model, val_loader, device)
        scheduler.step(validation_loss)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": validation_loss, "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.time() - started})
        print(f"Epoch {epoch + 1:02d}/{EPOCHS} | train={train_loss:.5f} val={validation_loss:.5f}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch + 1, "val_loss": validation_loss, "architecture": "DeepLabV3-ResNet50", "task": "binary_building_segmentation", "input_size": IMAGE_SIZE}, best_path)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    torch.save({"model_state_dict": model.state_dict(), "architecture": "DeepLabV3-ResNet50", "task": "binary_building_segmentation", "input_channels": 3, "input_size": IMAGE_SIZE, "threshold": 0.5, "best_epoch": checkpoint["epoch"]}, model_dir / "building_deeplabv3_resnet50_final.pt")
    metrics = evaluate_metrics(model, test_loader, device)
    metrics.update({"dataset": "WHU Building Dataset", "training_samples": len(train_df), "validation_samples": len(val_df), "test_samples": len(test_df), "best_epoch": checkpoint["epoch"], "best_validation_loss": best_validation_loss})
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    preprocessing = {"input_size": [IMAGE_SIZE, IMAGE_SIZE], "scale": 1.0 / 255.0, "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0], "channel_order": "RGB", "resize": "OpenCV INTER_LINEAR", "mask_resize": "OpenCV INTER_NEAREST", "threshold": 0.5}
    (model_dir / "building_preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    card = f"""# Model 2: Building footprint extraction\n\nArchitecture: torchvision DeepLabV3-ResNet50, pretrained DEFAULT weights, one-channel binary classifier head.\nDataset: WHU Building Dataset aerial mirror (`giswqs/WHU-Building-Dataset`).\nInput: RGB 256x256, scaled to [0,1], no channel mean/std normalization. Training augmentation: horizontal flip p=.5, vertical flip p=.5, random 90-degree rotation p=.5.\nLoss: 0.5 BCEWithLogits + 0.5 Dice. Optimizer: AdamW lr=1e-4, weight_decay=1e-4. Scheduler: ReduceLROnPlateau(mode=min, factor=.5, patience=2). Epochs: 12. Best checkpoint: lowest validation loss. Test threshold: .5.\n\nMetrics are recorded in `metrics.json`; they describe only the held-out WHU split, not Indian deployment accuracy.\n"""
    (model_dir / "model_card.md").write_text(card, encoding="utf-8")
    return metrics


def download_dataset(dataset_dir: Path, attempts: int = 3) -> Path:
    """Download the WHU mirror, retrying transient Hugging Face failures.

    The previous Colab run died here: a single ``snapshot_download`` call over a
    multi-gigabyte repository aborts the whole job on any transient network or
    CDN error. ``snapshot_download`` is resumable, so retrying is safe and a
    partially downloaded mirror is completed rather than restarted.
    """

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise SystemExit(
            "huggingface_hub is required for --download. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    dataset_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"Downloading WHU mirror (attempt {attempt}/{attempts}) -> {dataset_dir}")
            snapshot_download(
                repo_id="giswqs/WHU-Building-Dataset",
                repo_type="dataset",
                local_dir=str(dataset_dir),
                max_workers=4,
            )
            break
        except Exception as exc:  # noqa: BLE001 - retried below
            last_error = exc
            print(f"  download attempt {attempt} failed: {exc}")
            if attempt == attempts:
                raise SystemExit(
                    f"WHU download failed after {attempts} attempts: {exc}"
                ) from exc
    verify_dataset(dataset_dir)
    return dataset_dir


def verify_dataset(dataset_dir: Path) -> None:
    """Fail loudly and early when the mirror is incomplete or unpaired."""

    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")
    rasters = [
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not rasters:
        raise SystemExit(
            f"No image/mask rasters were found under {dataset_dir}. The download did "
            "not complete; re-run with --download to resume it."
        )
    train_df, val_df, test_df = discover_pairs(dataset_dir)
    print(
        "Dataset verified: "
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"(total {len(rasters)} rasters)"
    )


def preflight(dataset_dir: Path, project_root: Path) -> dict[str, Any]:
    """GPU sanity check: dataset, one batch, forward, loss, backward, optimizer."""

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Select a Google Colab Tesla T4 runtime first."
        )
    device = torch.device("cuda")
    seed_everything(42)
    verify_dataset(dataset_dir)
    train_df, val_df, test_df = discover_pairs(dataset_dir)
    loader = DataLoader(
        BuildingDataset(train_df, True), batch_size=TRAIN_BATCH, shuffle=True, num_workers=2
    )
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)
    print(f"batch ok: images={tuple(images.shape)} masks={tuple(masks.shape)}")

    model = build_model(pretrained=True).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    output = model(images)
    logits = output["out"] if isinstance(output, dict) else output
    if logits.shape[1] != 1:
        raise SystemExit(f"expected a one-channel building head, got {tuple(logits.shape)}")
    loss = combined_loss(logits, masks)
    loss.backward()
    optimizer.step()
    if not torch.isfinite(loss):
        raise SystemExit(f"preflight loss is not finite: {loss.item()}")
    print(f"forward/backward/optimizer ok: loss={loss.item():.6f} on {torch.cuda.get_device_name(0)}")
    del model, optimizer
    torch.cuda.empty_cache()
    return {
        "device": torch.cuda.get_device_name(0),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "batch_images": list(images.shape),
        "batch_masks": list(masks.shape),
        "loss": float(loss.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, help="Local extracted WHU aerial image/mask mirror")
    parser.add_argument("--project-root", type=Path, default=Path("/content/drive/MyDrive/SpatialShiftAI"))
    parser.add_argument("--download", action="store_true", help="Download giswqs/WHU-Building-Dataset into the project raw-data area")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run the GPU sanity check (dataset, batch, forward, loss, backward, optimizer) and exit",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from an existing best checkpoint and training history",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Select a Google Colab Tesla T4 runtime before downloading data or training.")
    dataset_dir = args.dataset_dir
    if args.download:
        dataset_dir = download_dataset(args.project_root / "data/raw/whu_building_hf")
    if dataset_dir is None:
        raise SystemExit("Pass --dataset-dir or --download. Dataset files are not included in this repository.")
    if args.preflight:
        print(json.dumps(preflight(dataset_dir, args.project_root), indent=2))
        return
    metrics = train(dataset_dir, args.project_root, resume=args.resume)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
