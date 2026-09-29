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

from app.core.errors import (
    InferenceError,
    InvalidInputError,
    ModelCompatibilityError,
)
from app.ml.base import BaseModelAdapter
from app.ml.common import ModelResult
from app.ml.imaging import ImageInput, prepare_tensor_input, read_image
from app.ml.torch_utils import (
    checkpoint_version,
    extract_state_dict,
    load_checkpoint,
    read_preprocessing,
    require_torch,
)

DEFAULT_THRESHOLD = 0.5


class ParcelMatcher(BaseModelAdapter):
    """Compares two parcel representations and reports correspondence."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._module = None
        self._preprocessing: dict[str, Any] = {}
        self._calibrator = None
        self._threshold: float = DEFAULT_THRESHOLD
        self._threshold_source = "default"

    def _load(self) -> None:
        torch = require_torch(self.spec.key)
        checkpoint_path = self.artifact_path("parcel_siamese_v2_best.pt")
        checkpoint = load_checkpoint(self.spec.key, checkpoint_path, self.device)

        self._model_version = checkpoint_version(checkpoint, "parcel_siamese_v2_best")
        self._preprocessing = read_preprocessing(
            self.spec.key,
            checkpoint,
            self.optional_artifact("parcel_match_preprocessing.json"),
        )
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
            return whole

        for key in ("model_object", "module"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, torch.nn.Module):
                return candidate

        state_dict = extract_state_dict(self.spec.key, checkpoint)
        raise ModelCompatibilityError(
            self.spec.key,
            "the checkpoint only contains weights and no architecture definition, "
            "so the Siamese encoder cannot be rebuilt without guessing",
            required=[
                "the training-time nn.Module (torch.save(model) or TorchScript), or",
                "an architecture descriptor plus the matching builder added to this adapter",
                f"observed {len(state_dict)} weight tensors, first keys: "
                + ", ".join(list(state_dict)[:5]),
            ],
        )

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
        value = payload.get("threshold", payload.get("best_threshold"))
        if isinstance(value, (int, float)):
            self._threshold = float(value)
            self._threshold_source = path.name

    # ------------------------------------------------------------------
    def predict(self, parcel_a: ImageInput, parcel_b: ImageInput) -> ModelResult:
        self.load()
        torch = require_torch(self.spec.key)
        if self._module is None:  # pragma: no cover - guarded by load()
            raise InferenceError("parcel matcher is not loaded")

        image_a = read_image(parcel_a)
        image_b = read_image(parcel_b)
        tensor_a = prepare_tensor_input(image_a.array, self._preprocessing, self.device)
        tensor_b = prepare_tensor_input(image_b.array, self._preprocessing, self.device)

        started = time.perf_counter()
        with torch.inference_mode():
            output = self._module(tensor_a, tensor_b)
        elapsed = (time.perf_counter() - started) * 1000

        raw_score, embeddings = _interpret_output(torch, output)
        calibrated = self._calibrate(raw_score)
        decision = "match" if calibrated >= self._threshold else "no_match"

        warnings: list[str] = []
        if self._calibrator is None:
            warnings.append(
                "no isotonic calibrator installed; the score is the raw model output"
            )
        if self._threshold_source == "default":
            warnings.append(
                f"no saved threshold found; using the default of {DEFAULT_THRESHOLD}"
            )

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
                "input_a": image_a.describe(),
                "input_b": image_b.describe(),
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
