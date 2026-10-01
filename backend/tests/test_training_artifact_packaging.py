from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_packager():
    script = Path(__file__).resolve().parents[2] / "training" / "package_trained_models.py"
    spec = importlib.util.spec_from_file_location("package_trained_models", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_packager_is_idempotent_for_canonical_training_outputs(tmp_path: Path):
    project = tmp_path / "SpatialShiftAI"
    model2 = project / "models/model2_building_extractor"
    model3 = project / "models/model3_change_detection"
    model2.mkdir(parents=True)
    model3.mkdir(parents=True)
    for path in (
        model2 / "building_deeplabv3_resnet50_best.pt",
        model2 / "building_deeplabv3_resnet50_final.pt",
        model3 / "siamese_resnet18_change_best.pt",
        model3 / "siamese_resnet18_change_final.pt",
    ):
        path.write_bytes(b"checkpoint")

    installed = _load_packager().install_artifacts(project)

    assert len(installed) == 6
    assert all(path.is_file() for path in installed)
    assert (model2 / "building_preprocessing.json").is_file()
    assert (model3 / "change_preprocessing.json").is_file()
    assert _load_packager().install_artifacts(project) == installed


def test_packager_refuses_to_claim_missing_checkpoints(tmp_path: Path):
    project = tmp_path / "empty-project"
    try:
        _load_packager().install_artifacts(project)
    except FileNotFoundError as exc:
        assert "Expected trained artifact" in str(exc)
    else:
        raise AssertionError("packager must not fabricate trained checkpoints")
