"""Helpers for loading PyTorch checkpoints produced by the training notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.errors import ModelCompatibilityError, ModelLoadError

STATE_DICT_KEYS = (
    "model_state_dict",
    "state_dict",
    "model",
    "weights",
)

PREPROCESSING_KEYS = ("preprocessing", "preprocess", "transform_config", "config")


def require_torch(model_key: str):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ModelLoadError(
            model_key, "PyTorch is not installed in this environment"
        ) from exc
    return torch


def load_checkpoint(model_key: str, path: Path, device: str) -> dict[str, Any]:
    """Load a checkpoint file and normalise it to a dict."""

    torch = require_torch(model_key)
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(model_key, f"could not read {path.name}: {exc}") from exc

    if isinstance(payload, dict):
        return payload
    return {"__object__": payload}


def extract_state_dict(model_key: str, checkpoint: dict[str, Any]) -> dict[str, Any]:
    for key in STATE_DICT_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            return _strip_module_prefix(value)

    tensor_like = {
        key: value
        for key, value in checkpoint.items()
        if hasattr(value, "shape") and hasattr(value, "dtype")
    }
    if tensor_like:
        return _strip_module_prefix(tensor_like)

    raise ModelCompatibilityError(
        model_key,
        "checkpoint does not contain a recognisable state_dict",
        required=[f"one of {', '.join(STATE_DICT_KEYS)} or a flat tensor mapping"],
    )


def _strip_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    if all(key.startswith("module.") for key in state):
        return {key[len("module.") :]: value for key, value in state.items()}
    return state


def read_preprocessing(
    model_key: str,
    checkpoint: dict[str, Any],
    sidecar: Path | None,
) -> dict[str, Any]:
    """Return the preprocessing descriptor recorded with the model.

    Preprocessing is never guessed: if the checkpoint has no descriptor and no
    sidecar JSON was installed, a compatibility error is raised describing what
    must be supplied.
    """

    for key in PREPROCESSING_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            return dict(value)

    if sidecar is not None and sidecar.is_file():
        try:
            return json.loads(sidecar.read_text())
        except json.JSONDecodeError as exc:
            raise ModelCompatibilityError(
                model_key, f"{sidecar.name} is not valid JSON: {exc}"
            ) from exc

    raise ModelCompatibilityError(
        model_key,
        "the checkpoint records no preprocessing descriptor, so inference "
        "preprocessing cannot be reconstructed safely",
        required=[
            "input_size (height, width)",
            "normalisation mean/std",
            "channel order and scaling used during training",
            "supplied either inside the checkpoint or as a sidecar JSON file",
        ],
    )


def checkpoint_version(checkpoint: dict[str, Any], fallback: str) -> str:
    for key in ("version", "model_version", "run_id", "experiment"):
        value = checkpoint.get(key)
        if isinstance(value, (str, int, float)):
            return str(value)
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, int):
        return f"{fallback}+epoch{epoch}"
    return fallback


def load_strict(model_key: str, module, state_dict: dict[str, Any]) -> None:
    """Load weights strictly and translate mismatches into a clear error."""

    try:
        module.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ModelCompatibilityError(
            model_key,
            "checkpoint weights do not match the expected architecture",
            required=[str(exc)],
        ) from exc
