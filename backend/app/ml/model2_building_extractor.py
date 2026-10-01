"""Model 2 adapter - building footprint extraction (DeepLabV3-ResNet50)."""

from __future__ import annotations

import base64
import io
import time
from typing import Any

import numpy as np

from app.core.errors import InferenceError, ModelCompatibilityError
from app.ml.base import BaseModelAdapter
from app.ml.common import ModelResult
from app.ml.imaging import ImageInput, LoadedImage, prepare_array, read_image
from app.ml.polygonize import polygonize_mask
from app.ml.torch_utils import (
    checkpoint_version,
    extract_state_dict,
    load_checkpoint,
    load_strict,
    read_preprocessing,
    require_torch,
)

DEFAULT_BINARY_THRESHOLD = 0.5


class BuildingExtractor(BaseModelAdapter):
    """Segments building footprints from a single image."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._module = None
        self._preprocessing: dict[str, Any] = {}
        self._num_classes = 1

    def _load(self) -> None:
        torch = require_torch(self.spec.key)
        checkpoint = load_checkpoint(
            self.spec.key,
            self.artifact_path("building_deeplabv3_resnet50_best.pt"),
            self.device,
        )
        self._model_version = checkpoint_version(
            checkpoint, "building_deeplabv3_resnet50_best"
        )
        self._preprocessing = read_preprocessing(
            self.spec.key,
            checkpoint,
            self.optional_artifact("building_preprocessing.json"),
        )

        whole = checkpoint.get("__object__")
        if isinstance(whole, torch.nn.Module):
            self._module = whole
        else:
            state_dict = extract_state_dict(self.spec.key, checkpoint)
            self._num_classes = _infer_num_classes(self.spec.key, state_dict)
            self._module = _build_deeplabv3(
                self.spec.key,
                self._num_classes,
                auxiliary="aux_classifier.4.weight" in state_dict,
            )
            load_strict(self.spec.key, self._module, state_dict)

        self._module.to(self.device)
        self._module.eval()

    # ------------------------------------------------------------------
    def predict(
        self, image: ImageInput, threshold: float = DEFAULT_BINARY_THRESHOLD
    ) -> ModelResult:
        self.load()
        torch = require_torch(self.spec.key)
        if self._module is None:  # pragma: no cover
            raise InferenceError("building extractor is not loaded")

        loaded = read_image(image)
        prepared = prepare_array(self.spec.key, loaded.array, self._preprocessing)
        tensor = torch.from_numpy(prepared).unsqueeze(0).to(self.device)

        started = time.perf_counter()
        with torch.inference_mode():
            output = self._module(tensor)
        elapsed = (time.perf_counter() - started) * 1000

        logits = output["out"] if isinstance(output, dict) else output
        probability = _to_probability(torch, logits)
        binary = (probability >= threshold).astype(np.uint8)
        building_pixel_percentage = float(binary.mean() * 100.0)

        warnings: list[str] = []
        if not loaded.georeferenced:
            warnings.append(
                "input raster has no CRS/geotransform; polygons are in pixel coordinates"
            )

        polygons = polygonize_mask(binary, loaded.transform, loaded.crs)
        mask_buffer = io.BytesIO()
        from PIL import Image

        Image.fromarray((binary * 255).astype(np.uint8)).save(mask_buffer, format="PNG")

        return self.build_result(
            confidence=None,
            decision=None,
            inference_ms=elapsed,
            warnings=warnings,
            evidence={
                "binary_threshold": threshold,
                "building_pixel_percentage": building_pixel_percentage,
                "mask_shape": list(binary.shape),
                "mask_png_base64": base64.b64encode(mask_buffer.getvalue()).decode("ascii"),
                "probability_mask_summary": _mask_summary(probability),
                "mean_building_probability": float(probability.mean()),
                "polygon_space": polygons["coordinate_space"],
                "crs": loaded.crs,
                "polygon_count": len(polygons["features"]),
                "polygons": polygons,
                "input": loaded.describe(),
                "preprocessing": self._preprocessing,
            },
        )

    def predict_masks(
        self, image: ImageInput, threshold: float = DEFAULT_BINARY_THRESHOLD
    ) -> tuple[np.ndarray, np.ndarray, LoadedImage]:
        """Return the raw probability mask and binary mask for downstream use."""

        self.load()
        torch = require_torch(self.spec.key)
        loaded = read_image(image)
        prepared = prepare_array(self.spec.key, loaded.array, self._preprocessing)
        tensor = torch.from_numpy(prepared).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self._module(tensor)
        logits = output["out"] if isinstance(output, dict) else output
        probability = _to_probability(torch, logits)
        return probability, (probability >= threshold).astype(np.uint8), loaded


def _build_deeplabv3(
    model_key: str, num_classes: int, auxiliary: bool = False
):
    try:
        from torchvision.models.segmentation import deeplabv3_resnet50
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ModelCompatibilityError(
            model_key, "torchvision is required for DeepLabV3-ResNet50"
        ) from exc
    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=21 if auxiliary else num_classes,
        aux_loss=auxiliary,
    )
    if auxiliary:
        import torch

        model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def _infer_num_classes(model_key: str, state_dict: dict[str, Any]) -> int:
    for key in ("classifier.4.weight", "classifier.4.bias"):
        tensor = state_dict.get(key)
        if tensor is not None and hasattr(tensor, "shape"):
            return int(tensor.shape[0])
    raise ModelCompatibilityError(
        model_key,
        "checkpoint does not expose the DeepLabV3 classifier head, so the number "
        "of output classes cannot be determined",
        required=["classifier.4.weight in the state dict"],
    )


def _to_probability(torch, logits) -> np.ndarray:
    tensor = logits[0] if logits.ndim == 4 else logits
    if tensor.shape[0] == 1:
        probability = torch.sigmoid(tensor[0])
    else:
        probability = torch.softmax(tensor, dim=0)[1]
    return probability.detach().cpu().numpy().astype(np.float32)


def _mask_summary(mask: np.ndarray) -> dict[str, float]:
    return {
        "min": float(mask.min()),
        "max": float(mask.max()),
        "mean": float(mask.mean()),
        "std": float(mask.std()),
    }
