"""Model inventory and health endpoints.

``/api/models/health`` really inspects the configured MODEL_ROOT and attempts a
load; it never returns a hardcoded status.
"""

from __future__ import annotations

import base64
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from app.core.errors import InvalidInputError, ModelArtifactMissingError
from app.ml.model_registry import ModelRegistry, UnknownModelError, get_registry
from app.ml.orchestrator import Orchestrator

router = APIRouter(prefix="/models", tags=["models"])


def registry_dependency() -> ModelRegistry:
    return get_registry()


@router.get("")
def list_models(registry: ModelRegistry = Depends(registry_dependency)) -> dict[str, Any]:
    """Declared models and their expected artifacts (no loading performed)."""

    models = []
    for spec in registry.specs:
        adapter = registry.get(spec.key)
        health = adapter.health(probe=False)
        models.append(
            {
                "key": spec.key,
                "model_key": spec.key,
                "title": spec.title,
                "name": spec.title,
                "purpose": spec.purpose or spec.task,
                "dataset": spec.dataset,
                "architecture": spec.architecture,
                "architecture_name": spec.architecture,
                "task": spec.task,
                "runtime": spec.runtime.value,
                "artifact": next((artifact.name for artifact in spec.artifacts if artifact.role in {"checkpoint", "classifier"}), None),
                "artifact_path": str(adapter.artifact_path(next((artifact.name for artifact in spec.artifacts if artifact.role in {"checkpoint", "classifier"}), ""))),
                "version": adapter.model_version,
                "benchmark_metrics": dict(spec.benchmark_metrics),
                "status": health.status.value,
                "status_group": health.status_group,
                "loaded": health.loaded,
                "live_inference_available": health.live_inference_available,
                "readiness_reason": health.readiness_reason,
                "model_dir": str(adapter.model_dir),
                "notes": list(spec.notes),
                "artifacts": [
                    report.model_dump()
                    for report in adapter.artifact_reports(include_hash=False)
                ],
            }
        )
    return {
        "model_root": str(registry.model_root),
        "model_root_exists": registry.model_root.is_dir(),
        "count": len(models),
        "models": models,
    }


@router.get("/health")
@router.get("/readiness")
def models_health(
    probe: bool = Query(True, description="Attempt a real load of each model"),
    include_hash: bool = Query(False, description="Include sha256 of present artifacts"),
    registry: ModelRegistry = Depends(registry_dependency),
) -> dict[str, Any]:
    summary = registry.summary(probe=probe)
    summary["models"] = {
        key: health.model_dump(mode="json") for key, health in summary["models"].items()
    }
    return summary


@router.get("/pipeline")
def pipeline(
    probe: bool = Query(True),
    registry: ModelRegistry = Depends(registry_dependency),
) -> dict[str, Any]:
    orchestrator = Orchestrator(registry=registry)
    return {"stages": orchestrator.pipeline(probe=probe)}


@router.get("/{model_key}/health")
def model_health(
    model_key: str,
    probe: bool = Query(True),
    include_hash: bool = Query(False),
    registry: ModelRegistry = Depends(registry_dependency),
) -> dict[str, Any]:
    try:
        health = registry.health(model_key, probe=probe, include_hash=include_hash)
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return health.model_dump(mode="json")


@router.post("/{model_key}/infer")
@router.post("/{model_key}/predict")
@router.post("/{model_key}/inference")
def model_inference(
    model_key: str,
    payload: dict[str, Any] | None = Body(default=None),
    registry: ModelRegistry = Depends(registry_dependency),
) -> dict[str, Any]:
    return infer_with_registry(model_key, payload, registry)


def infer_with_registry(
    model_key: str,
    payload: dict[str, Any] | None,
    registry: ModelRegistry,
) -> dict[str, Any]:
    try:
        adapter = registry.get(model_key)
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    missing = adapter.missing_required_artifacts()
    if missing:
        raise ModelArtifactMissingError(adapter.spec.key, missing)

    request_payload = dict(payload or {})
    if model_key == "building_extractor":
        for key in ("image", "input", "data"):
            if key in request_payload:
                request_payload[key] = _decode_image_value(request_payload[key])
    elif model_key == "change_detector":
        for key in ("before_image", "before", "after_image", "after"):
            if key in request_payload:
                request_payload[key] = _decode_image_value(request_payload[key])
    try:
        result = adapter.infer(request_payload)
    except InvalidInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return result.model_dump(mode="json")


def _decode_image_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    encoded = value.get("base64") or value.get("data")
    if not isinstance(encoded, str):
        return value
    if encoded.startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise HTTPException(status_code=422, detail="image base64 payload is invalid") from exc
