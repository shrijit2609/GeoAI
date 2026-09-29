"""Model inventory and health endpoints.

``/api/models/health`` really inspects the configured MODEL_ROOT and attempts a
load; it never returns a hardcoded status.
"""

from __future__ import annotations

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
        models.append(
            {
                "key": spec.key,
                "title": spec.title,
                "architecture": spec.architecture,
                "task": spec.task,
                "runtime": spec.runtime.value,
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
    try:
        adapter = registry.get(model_key)
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    missing = adapter.missing_required_artifacts()
    if missing:
        raise ModelArtifactMissingError(adapter.spec.key, missing)

    try:
        result = adapter.infer(payload or {})
    except InvalidInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return result.model_dump(mode="json")
