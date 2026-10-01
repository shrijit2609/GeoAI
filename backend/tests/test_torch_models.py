"""Adapter tests for the PyTorch models (1, 2, 3).

Checkpoints written here use the real architectures with untrained weights so
that the loading, preprocessing and output contract can be exercised. They
carry no information about the quality of the trained SIH artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.core.errors import (
    InvalidInputError,
    ModelArtifactMissingError,
    ModelCompatibilityError,
)
from app.ml.common import ModelStatus
from app.ml.imaging import LoadedImage
from app.ml.model1_parcel_matcher import sample_parcel_boundary
from app.ml.model_registry import ModelRegistry
from shapely.geometry import MultiPolygon, Polygon

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

PREPROCESSING = {
    "input_size": [64, 64],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "scale": 1 / 255.0,
}


class TinySiamese(torch.nn.Module):
    """Stand-in for the trained encoder, saved as a whole module."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )

    def forward(self, a, b):
        return self.encoder(a), self.encoder(b)


def rgb_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


def test_model1_boundary_sampling_matches_training_notebook():
    square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    small = Polygon([(10, 10), (11, 10), (11, 11), (10, 11), (10, 10)])
    points = sample_parcel_boundary(MultiPolygon([square, small]), n=4)

    expected = np.asarray(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ],
        dtype=np.float32,
    ) / np.sqrt(2.0)
    np.testing.assert_allclose(points, expected, rtol=1e-6, atol=1e-6)


@pytest.fixture
def building_checkpoint(empty_model_root: Path) -> Path:
    from torchvision.models.segmentation import deeplabv3_resnet50

    model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=1)
    path = empty_model_root / "model2_building_extractor" / "building_deeplabv3_resnet50_best.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "preprocessing": PREPROCESSING, "epoch": 3},
        path,
    )
    return path


@pytest.fixture
def change_checkpoint(empty_model_root: Path) -> Path:
    from app.ml.model3_change_detector import build_reference_network

    model = build_reference_network(num_classes=1)
    path = empty_model_root / "model3_change_detection" / "siamese_resnet18_change_best.pt"
    torch.save({"state_dict": model.state_dict(), "preprocessing": PREPROCESSING}, path)
    return path


# ----------------------------------------------------------------------
def test_missing_checkpoints_are_not_faked(registry: ModelRegistry):
    for key in ("parcel_matcher", "building_extractor", "change_detector"):
        health = registry.health(key)
        expected = (
            ModelStatus.TRAINING_REQUIRED
            if key in ("building_extractor", "change_detector")
            else ModelStatus.MISSING_ARTIFACT
        )
        assert health.status is expected
        assert health.loaded is False

    with pytest.raises(ModelArtifactMissingError):
        registry.get("building_extractor").predict(rgb_image())


def test_building_extractor_real_inference(
    registry: ModelRegistry, building_checkpoint: Path
):
    assert registry.health("building_extractor").status is ModelStatus.READY

    result = registry.get("building_extractor").predict(rgb_image())
    evidence = result.evidence
    assert result.status == "success"
    assert result.model_version.endswith("epoch3")
    assert 0.0 <= evidence["building_pixel_percentage"] <= 100.0
    assert evidence["mask_shape"] == [64, 64]
    assert 0.0 <= evidence["probability_mask_summary"]["min"] <= 1.0
    assert result.inference_ms is not None


def test_building_polygons_are_marked_as_pixel_coordinates(
    registry: ModelRegistry, building_checkpoint: Path
):
    result = registry.get("building_extractor").predict(rgb_image())
    polygons = result.evidence["polygons"]
    assert polygons["coordinate_space"] == "pixel"
    assert polygons["is_pixel_coordinates"] is True
    assert polygons["crs"] is None
    assert any("pixel coordinates" in warning for warning in result.warnings)


def test_building_polygons_use_the_crs_when_georeferenced(
    registry: ModelRegistry, building_checkpoint: Path
):
    georeferenced = LoadedImage(
        array=rgb_image(),
        crs="EPSG:32643",
        transform=(0.5, 0.0, 700000.0, 0.0, -0.5, 3100000.0),
        source="test",
    )
    result = registry.get("building_extractor").predict(georeferenced)
    polygons = result.evidence["polygons"]
    assert polygons["coordinate_space"] == "crs"
    assert polygons["crs"] == "EPSG:32643"
    assert polygons["is_pixel_coordinates"] is False


def test_checkpoint_without_preprocessing_is_rejected(
    registry: ModelRegistry, empty_model_root: Path
):
    from torchvision.models.segmentation import deeplabv3_resnet50

    model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=1)
    torch.save(
        model.state_dict(),
        empty_model_root / "model2_building_extractor" / "building_deeplabv3_resnet50_best.pt",
    )
    with pytest.raises(ModelCompatibilityError) as excinfo:
        registry.get("building_extractor").load()
    assert "preprocessing" in str(excinfo.value)


def test_preprocessing_sidecar_is_used_when_present(
    registry: ModelRegistry, empty_model_root: Path
):
    from torchvision.models.segmentation import deeplabv3_resnet50

    directory = empty_model_root / "model2_building_extractor"
    model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=1)
    torch.save(model.state_dict(), directory / "building_deeplabv3_resnet50_best.pt")
    (directory / "building_preprocessing.json").write_text(json.dumps(PREPROCESSING))

    registry.get("building_extractor").load()
    assert registry.health("building_extractor").status is ModelStatus.READY


def test_architecture_mismatch_is_reported(registry: ModelRegistry, empty_model_root: Path):
    from torchvision.models.segmentation import deeplabv3_resnet50

    model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=1)
    state = model.state_dict()
    state.pop("backbone.conv1.weight")
    torch.save(
        {"model_state_dict": state, "preprocessing": PREPROCESSING},
        empty_model_root / "model2_building_extractor" / "building_deeplabv3_resnet50_best.pt",
    )
    with pytest.raises(ModelCompatibilityError):
        registry.get("building_extractor").load()


# ----------------------------------------------------------------------
def test_change_detector_real_inference(registry: ModelRegistry, change_checkpoint: Path):
    result = registry.get("change_detector").predict(rgb_image(1), rgb_image(2))
    evidence = result.evidence
    assert result.decision in ("change_detected", "no_change_detected")
    assert 0.0 <= evidence["changed_pixel_percentage"] <= 100.0
    assert evidence["total_pixels"] == 64 * 64
    assert evidence["change_polygons"]["is_pixel_coordinates"] is True


def test_change_detector_rejects_mismatched_epochs(
    registry: ModelRegistry, change_checkpoint: Path
):
    small = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(InvalidInputError):
        registry.get("change_detector").predict(rgb_image(1), small)


# ----------------------------------------------------------------------
def test_parcel_matcher_refuses_to_guess_the_architecture(
    registry: ModelRegistry, empty_model_root: Path
):
    """A weights-only Siamese checkpoint cannot be rebuilt safely."""

    from torchvision.models import resnet18

    torch.save(
        {"model_state_dict": resnet18(weights=None).state_dict(), "preprocessing": PREPROCESSING},
        empty_model_root / "model1_parcel_matcher" / "parcel_siamese_v2_best.pt",
    )
    with pytest.raises(ModelCompatibilityError) as excinfo:
        registry.get("parcel_matcher").load()
    assert "architecture" in str(excinfo.value)

    health = registry.health("parcel_matcher")
    assert health.status is ModelStatus.ARTIFACT_PRESENT_BUT_INFERENCE_BLOCKED
    assert health.error


def test_parcel_matcher_runs_when_the_module_is_saved(
    registry: ModelRegistry, empty_model_root: Path
):
    torch.save(
        {"__object__": TinySiamese(), "preprocessing": PREPROCESSING, "version": "test-1"},
        empty_model_root / "model1_parcel_matcher" / "parcel_siamese_v2_best.pt",
    )
    result = registry.get("parcel_matcher").predict(rgb_image(3), rgb_image(4))
    assert result.model == "parcel_matcher"
    assert 0.0 <= result.confidence <= 1.0
    assert result.decision in ("match", "no_match")
    assert result.evidence["embedding_distance"] is not None
    assert any("threshold" in warning for warning in result.warnings)


def test_recovered_parcel_checkpoint_loads_and_runs_notebook_preprocessing():
    from shapely.geometry import Polygon

    model_root = Path(__file__).resolve().parents[2] / "models"
    registry = ModelRegistry(model_root=model_root, device="cpu")
    health = registry.health("parcel_matcher", probe=True)
    result = registry.get("parcel_matcher").predict(
        Polygon([(0, 0), (10, 0), (10, 8), (0, 8)]),
        Polygon([(0, 0), (10, 0), (10, 8), (0, 8)]),
    )

    assert health.status is ModelStatus.READY
    assert health.loaded is True
    assert result.status == "success"
    assert result.evidence["preprocessing"]["boundary_points"] == 64
    assert result.evidence["embedding_distance"] is not None
