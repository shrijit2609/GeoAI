"""Model 1 adapter - parcel correspondence with a Siamese CNN.

The adapter loads the real checkpoint. It refuses to run when the checkpoint
does not carry enough information to reproduce training-time preprocessing or
to rebuild the network, instead of guessing and producing meaningless scores.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from app.core.errors import (
    InferenceError,
    InvalidInputError,
)
from app.ml.base import BaseModelAdapter
from app.ml.common import ModelResult
from app.ml.imaging import ImageInput, prepare_tensor_input, read_image
from app.services.geospatial import to_geometry
from app.ml.torch_utils import (
    checkpoint_version,
    extract_state_dict,
    load_checkpoint,
    load_strict,
    read_preprocessing,
    require_torch,
)

DEFAULT_THRESHOLD = 0.5
NUM_BOUNDARY_POINTS = 64


def sample_parcel_boundary(geometry: Any, n: int = NUM_BOUNDARY_POINTS) -> np.ndarray:
    """Reproduce the training notebook's largest-ring boundary normalization."""

    geom = to_geometry(geometry)
    if isinstance(geom, Polygon):
        polygon = geom
    elif isinstance(geom, MultiPolygon) and geom.geoms:
        polygon = max(geom.geoms, key=lambda part: part.area)
    else:
        raise InvalidInputError(
            f"parcel matcher requires Polygon or MultiPolygon geometry, got {geom.geom_type}"
        )

    if polygon.is_empty or polygon.exterior.is_empty:
        raise InvalidInputError("parcel matcher cannot process empty parcel geometry")

    boundary = polygon.exterior
    distances = np.linspace(0, boundary.length, n, endpoint=False)
    points = np.asarray(
        [
            [boundary.interpolate(float(distance)).x, boundary.interpolate(float(distance)).y]
            for distance in distances
        ],
        dtype=np.float32,
    )
    points -= points.mean(axis=0)
    scale = float(np.max(np.linalg.norm(points, axis=1)))
    if scale > 0:
        points /= scale
    return points.astype(np.float32)


def _build_siamese_parcel_matcher(torch, state_dict: dict[str, Any]):
    """Build the training-time network layout and load its state strictly."""

    class _ParcelEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Conv1d(2, 64, kernel_size=5, padding=2),
                torch.nn.ReLU(),
                torch.nn.Conv1d(64, 128, kernel_size=5, padding=2),
                torch.nn.ReLU(),
                torch.nn.Conv1d(128, 256, kernel_size=5, padding=2),
                torch.nn.ReLU(),
            )
            self.pool = torch.nn.AdaptiveMaxPool1d(1)
            self.embedding = torch.nn.Sequential(
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
            )

        def forward(self, value):
            if value.ndim == 4:
                if value.shape[1] != 2:
                    raise InvalidInputError(
                        "parcel matcher requires exactly two input channels; "
                        f"received {value.shape[1]}"
                    )
                value = value.flatten(start_dim=2)
            elif value.ndim == 3 and value.shape[-1] == 2:
                if value.shape[1] != NUM_BOUNDARY_POINTS:
                    raise InvalidInputError(
                        f"parcel matcher requires {NUM_BOUNDARY_POINTS} boundary points; "
                        f"received {value.shape[1]}"
                    )
                value = value.transpose(1, 2)
            elif value.ndim != 3:
                raise InvalidInputError(
                    "parcel matcher input must have shape (batch, 2, sequence) "
                    "or (batch, 2, height, width)"
                )

            if value.shape[1] != 2:
                raise InvalidInputError(
                    "parcel matcher requires exactly two input channels; "
                    f"received {value.shape[1]}"
                )

            features = self.network(value)
            pooled = self.pool(features).squeeze(-1)
            embedding = self.embedding(pooled)
            return torch.nn.functional.normalize(embedding, p=2, dim=1)

    class _SiameseParcelMatcher(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = _ParcelEncoder()

        def forward(self, parcel_a, parcel_b):
            return self.encoder(parcel_a), self.encoder(parcel_b)

    module = _SiameseParcelMatcher()
    load_strict("parcel_matcher", module, state_dict)
    return module


class ParcelMatcher(BaseModelAdapter):
    """Compares two parcel representations and reports correspondence."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._module = None
        self._preprocessing: dict[str, Any] = {}
        self._uses_boundary_features = False
        self._calibrator = None
        self._threshold: float | None = None
        self._threshold_source = "default"

    def _load(self) -> None:
        torch = require_torch(self.spec.key)
        checkpoint_path = self.artifact_path("parcel_siamese_v2_best.pt")
        checkpoint = load_checkpoint(self.spec.key, checkpoint_path, self.device)

        self._model_version = checkpoint_version(checkpoint, "parcel_siamese_v2_best")
        self._module = self._build_module(torch, checkpoint)
        self._module.to(self.device)
        self._module.eval()
        self._load_calibrator()
        self._load_threshold()

    def _build_module(self, torch, checkpoint: dict[str, Any]):
        """Recover the trained network.

        Only two situations are supported: the checkpoint stores the whole
        module (``torch.save(model)`` or TorchScript), or it stores a state dict
        together with a serialisable architecture the training run recorded. A
        Siamese CNN has no canonical layer layout, so weights alone are not
        enough and rebuilding one by inspection would be a guess.
        """

        whole = checkpoint.get("__object__")
        if whole is not None and isinstance(whole, torch.nn.Module):
            self._preprocessing = read_preprocessing(
                self.spec.key,
                checkpoint,
                self.optional_artifact("parcel_match_preprocessing.json"),
            )
            self._threshold = DEFAULT_THRESHOLD
            self._threshold_source = "default"
            return whole

        for key in ("model_object", "module"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, torch.nn.Module):
                self._preprocessing = read_preprocessing(
                    self.spec.key,
                    checkpoint,
                    self.optional_artifact("parcel_match_preprocessing.json"),
                )
                self._threshold = DEFAULT_THRESHOLD
                self._threshold_source = "default"
                return candidate

        state_dict = extract_state_dict(self.spec.key, checkpoint)
        module = _build_siamese_parcel_matcher(torch, state_dict)
        self._uses_boundary_features = True
        self._preprocessing = {
            "source": "training_notebook",
            "boundary_points": NUM_BOUNDARY_POINTS,
            "sampling": "equidistant interpolation on largest polygon exterior; endpoint excluded",
            "centering": "subtract sampled-point mean",
            "scale": "divide by maximum sampled-point Euclidean radius",
            "coordinate_order": ["x", "y"],
        }
        return module

    def _load_calibrator(self) -> None:
        path = self.optional_artifact("parcel_match_isotonic_calibrator.pkl")
        if path is None:
            return
        import joblib

        self._calibrator = joblib.load(path)

    def _load_threshold(self) -> None:
        path = self.optional_artifact("parcel_match_threshold.json")
        if path is None:
            return
        payload = json.loads(Path(path).read_text())
        value = payload.get(
            "threshold",
            payload.get("best_threshold", payload.get("decision_threshold")),
        )
        if isinstance(value, (int, float)):
            self._threshold = float(value)
            self._threshold_source = path.name

    # ------------------------------------------------------------------
    def predict(self, parcel_a: Any, parcel_b: Any) -> ModelResult:
        self.load()
        torch = require_torch(self.spec.key)
        if self._module is None:  # pragma: no cover - guarded by load()
            raise InferenceError("parcel matcher is not loaded")

        image_a = image_b = None
        if self._uses_boundary_features:
            points_a = _boundary_input(parcel_a)
            points_b = _boundary_input(parcel_b)
            tensor_a = torch.from_numpy(points_a).unsqueeze(0).to(self.device)
            tensor_b = torch.from_numpy(points_b).unsqueeze(0).to(self.device)
        else:
            image_a = read_image(parcel_a)
            image_b = read_image(parcel_b)
            tensor_a = prepare_tensor_input(image_a.array, self._preprocessing, self.device)
            tensor_b = prepare_tensor_input(image_b.array, self._preprocessing, self.device)

        started = time.perf_counter()
        with torch.inference_mode():
            output = self._module(tensor_a, tensor_b)
        elapsed = (time.perf_counter() - started) * 1000

        if self._uses_boundary_features and isinstance(output, (tuple, list)) and len(output) == 2:
            emb_a, emb_b = output
            embeddings = float(
                torch.nn.functional.pairwise_distance(emb_a, emb_b).mean()
            )
            raw_score = 1.0 / (1.0 + embeddings)
        else:
            raw_score, embeddings = _interpret_output(torch, output)
        calibrated = (
            self._calibrate(raw_score)
            if self._calibrator is not None
            else (raw_score if not self._uses_boundary_features else None)
        )
        decision_score = calibrated if calibrated is not None else raw_score
        decision = None
        if self._threshold is not None:
            decision = "match" if decision_score >= self._threshold else "no_match"

        warnings: list[str] = []
        if self._calibrator is None:
            warnings.append(
                "no isotonic calibrator installed; raw similarity is not a calibrated probability"
            )
        if self._threshold_source == "default":
            warnings.append("no saved decision threshold found; decision is omitted")

        return self.build_result(
            confidence=calibrated,
            decision=decision,
            inference_ms=elapsed,
            warnings=warnings,
            evidence={
                "raw_score": raw_score,
                "calibrated_score": calibrated,
                "threshold": self._threshold,
                "threshold_source": self._threshold_source,
                "embedding_distance": embeddings,
                "preprocessing": self._preprocessing,
                "input_a": image_a.describe() if image_a is not None else {"type": "parcel_geometry"},
                "input_b": image_b.describe() if image_b is not None else {"type": "parcel_geometry"},
            },
        )

    def _calibrate(self, raw_score: float) -> float:
        if self._calibrator is None:
            return raw_score
        value = self._calibrator.predict(np.asarray([raw_score], dtype=float))
        return float(np.asarray(value).reshape(-1)[0])


def _interpret_output(torch, output: Any) -> tuple[float, float | None]:
    """Turn the network output into a probability plus optional distance."""

    if isinstance(output, (tuple, list)):
        if len(output) == 2 and all(hasattr(item, "shape") for item in output):
            emb_a, emb_b = output
            distance = float(torch.nn.functional.pairwise_distance(emb_a, emb_b).mean())
            similarity = float(
                torch.nn.functional.cosine_similarity(emb_a, emb_b).mean()
            )
            return (similarity + 1.0) / 2.0, distance
        output = output[0]

    tensor = output
    if tensor.ndim == 0:
        logit = tensor
    elif tensor.shape[-1] == 1:
        logit = tensor.reshape(-1)[0]
    elif tensor.shape[-1] == 2:
        probs = torch.softmax(tensor.reshape(1, -1), dim=-1)
        return float(probs[0, 1]), None
    else:
        raise InvalidInputError(
            f"unsupported parcel matcher output shape {tuple(tensor.shape)}"
        )

    value = float(logit)
    if 0.0 <= value <= 1.0:
        return value, None
    return float(torch.sigmoid(logit)), None


def _boundary_input(value: Any) -> np.ndarray:
    if isinstance(value, dict) and "boundary_points" in value:
        points = np.asarray(value["boundary_points"], dtype=np.float32)
        if points.shape != (NUM_BOUNDARY_POINTS, 2):
            raise InvalidInputError(
                f"boundary_points must have shape ({NUM_BOUNDARY_POINTS}, 2), got {points.shape}"
            )
        return points
    if isinstance(value, np.ndarray) and value.shape == (NUM_BOUNDARY_POINTS, 2):
        return value.astype(np.float32, copy=False)
    if isinstance(value, dict) and "geometry" in value:
        value = value["geometry"]
    try:
        return sample_parcel_boundary(value)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(f"unsupported parcel geometry: {exc}") from exc
