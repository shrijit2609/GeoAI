"""Model 3 adapter - temporal change detection (Siamese ResNet18 + decoder)."""

from __future__ import annotations

import base64
import io
import time
from typing import Any

import numpy as np

from app.core.errors import InferenceError, InvalidInputError, ModelCompatibilityError
from app.ml.base import BaseModelAdapter
from app.ml.common import ModelResult
from app.ml.imaging import ImageInput, prepare_array, read_image
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


def build_reference_network(num_classes: int = 1):
    """Reference implementation of the trained change-detection network.

    A Siamese ResNet18 encoder, an absolute feature difference and a small
    upsampling decoder. Weights are always loaded strictly: if the trained
    checkpoint used a different decoder the load fails loudly rather than
    silently producing meaningless masks.
    """

    import torch
    from torch import nn
    from torchvision.models import resnet18

    class SiameseChangeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            backbone = resnet18(weights=None)
            self.encoder = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(256, 128, 2, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(128, 64, 2, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(64, 32, 2, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(32, 16, 2, stride=2),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(16, 8, 2, stride=2),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),

                nn.Conv2d(8, num_classes, 1),
            )

        def forward(self, before, after):
            feat_before = self.encoder(before)
            feat_after = self.encoder(after)
            diff = torch.abs(feat_after - feat_before)
            logits = self.decoder(diff)
            return nn.functional.interpolate(
                logits, size=before.shape[-2:], mode="bilinear", align_corners=False
            )

    return SiameseChangeNet()


class ChangeDetector(BaseModelAdapter):
    """Detects change between two co-registered image epochs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._module = None
        self._preprocessing: dict[str, Any] = {}

    def _load(self) -> None:
        torch = require_torch(self.spec.key)
        checkpoint = load_checkpoint(
            self.spec.key,
            self.artifact_path("siamese_resnet18_change_best.pt"),
            self.device,
        )
        self._model_version = checkpoint_version(
            checkpoint, "siamese_resnet18_change_best"
        )
        self._preprocessing = read_preprocessing(
            self.spec.key,
            checkpoint,
            self.optional_artifact("change_preprocessing.json"),
        )

        whole = checkpoint.get("__object__")
        if isinstance(whole, torch.nn.Module):
            self._module = whole
        else:
            state_dict = extract_state_dict(self.spec.key, checkpoint)
            num_classes = _infer_num_classes(self.spec.key, state_dict)
            self._module = build_reference_network(num_classes)
            load_strict(self.spec.key, self._module, state_dict)

        self._module.to(self.device)
        self._module.eval()

    # ------------------------------------------------------------------
    def predict(
        self,
        before_image: ImageInput,
        after_image: ImageInput,
        threshold: float = DEFAULT_BINARY_THRESHOLD,
    ) -> ModelResult:
        self.load()
        torch = require_torch(self.spec.key)
        if self._module is None:  # pragma: no cover
            raise InferenceError("change detector is not loaded")

        before = read_image(before_image)
        after = read_image(after_image)
        if before.array.shape[:2] != after.array.shape[:2]:
            raise InvalidInputError(
                "before/after rasters differ in size "
                f"({before.array.shape[:2]} vs {after.array.shape[:2]}); "
                "co-register the epochs before running change detection"
            )

        warnings: list[str] = []
        if before.georeferenced != after.georeferenced or before.crs != after.crs:
            warnings.append(
                "before/after rasters do not share the same georeferencing; "
                "results are reported in the geometry of the before image"
            )
        if not before.georeferenced:
            warnings.append(
                "no CRS/geotransform available; change polygons are in pixel coordinates"
            )

        tensor_before = torch.from_numpy(
            prepare_array(self.spec.key, before.array, self._preprocessing)
        ).unsqueeze(0).to(self.device)
        tensor_after = torch.from_numpy(
            prepare_array(self.spec.key, after.array, self._preprocessing)
        ).unsqueeze(0).to(self.device)

        started = time.perf_counter()
        with torch.inference_mode():
            output = self._module(tensor_before, tensor_after)
        elapsed = (time.perf_counter() - started) * 1000

        logits = output["out"] if isinstance(output, dict) else output
        probability = _to_probability(torch, logits)
        binary = (probability >= threshold).astype(np.uint8)
        changed_percentage = float(binary.mean() * 100.0)

        changed = binary.sum()
        mean_confidence = (
            float(probability[binary == 1].mean()) if changed else None
        )
        mask_buffer = io.BytesIO()
        from PIL import Image

        Image.fromarray((binary * 255).astype(np.uint8)).save(mask_buffer, format="PNG")

        return self.build_result(
            confidence=mean_confidence,
            decision="change_detected" if changed else "no_change_detected",
            inference_ms=elapsed,
            warnings=warnings,
            evidence={
                "binary_threshold": threshold,
                "changed_pixel_percentage": changed_percentage,
                "changed_pixel_count": int(changed),
                "total_pixels": int(binary.size),
                "mask_shape": list(binary.shape),
                "mask_png_base64": base64.b64encode(mask_buffer.getvalue()).decode("ascii"),
                "probability_mask_summary": {
                    "min": float(probability.min()),
                    "max": float(probability.max()),
                    "mean": float(probability.mean()),
                    "std": float(probability.std()),
                },
                "change_polygons": polygonize_mask(
                    binary, before.transform, before.crs
                ),
                "before": before.describe(),
                "after": after.describe(),
                "preprocessing": self._preprocessing,
            },
        )


def _infer_num_classes(model_key: str, state_dict: dict[str, Any]) -> int:
    for key, tensor in reversed(list(state_dict.items())):
        if key.endswith("weight") and hasattr(tensor, "shape") and tensor.ndim == 4:
            return int(tensor.shape[0])
    raise ModelCompatibilityError(
        model_key,
        "checkpoint does not expose a convolutional output head, so the number "
        "of change classes cannot be determined",
    )


def _to_probability(torch, logits) -> np.ndarray:
    tensor = logits[0] if logits.ndim == 4 else logits
    if tensor.shape[0] == 1:
        probability = torch.sigmoid(tensor[0])
    else:
        probability = torch.softmax(tensor, dim=0)[1]
    return probability.detach().cpu().numpy().astype(np.float32)
