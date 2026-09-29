"""Coordination layer over the model registry.

Phase 1 scope: expose a single place that knows which pipeline stages exist,
which of them are currently backed by a usable artifact, and which are skipped.
Stages are never simulated - a stage without a loadable artifact is reported as
skipped with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.core.errors import (
    ModelArtifactMissingError,
    ModelCompatibilityError,
    ModelLoadError,
    SpatialShiftError,
)
from app.ml.common import ModelResult, ModelStatus
from app.ml.model_registry import ModelRegistry, get_registry


@dataclass(frozen=True)
class PipelineStage:
    name: str
    model_key: str
    description: str


PIPELINE: tuple[PipelineStage, ...] = (
    PipelineStage(
        "parcel_correspondence",
        "parcel_matcher",
        "Match parcel representations across cadastral sources.",
    ),
    PipelineStage(
        "building_extraction",
        "building_extractor",
        "Extract building footprints from orthorectified imagery.",
    ),
    PipelineStage(
        "change_detection",
        "change_detector",
        "Detect change between two imagery epochs.",
    ),
    PipelineStage(
        "entity_resolution",
        "entity_resolver",
        "Resolve corresponding land-record entities.",
    ),
    PipelineStage(
        "record_anomaly_review",
        "anomaly_detector",
        "Flag land records that require human review.",
    ),
)


class StageOutcome(dict):
    """Serialisable outcome of a single pipeline stage."""


class Orchestrator:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or get_registry()

    def pipeline(self, probe: bool = True) -> list[dict[str, Any]]:
        health = self.registry.health_report(probe=probe)
        stages: list[dict[str, Any]] = []
        for stage in PIPELINE:
            model_health = health[stage.model_key]
            stages.append(
                {
                    "stage": stage.name,
                    "model": stage.model_key,
                    "description": stage.description,
                    "status": model_health.status.value,
                    "available": model_health.status == ModelStatus.READY,
                    "blocker": model_health.error
                    or (
                        "missing artifacts: " + ", ".join(model_health.missing_artifacts)
                        if model_health.missing_artifacts
                        else None
                    ),
                }
            )
        return stages

    def run_stage(
        self, stage_name: str, invoke: Callable[[Any], ModelResult]
    ) -> StageOutcome:
        """Run one stage, converting artifact problems into an explicit skip."""

        stage = next((item for item in PIPELINE if item.name == stage_name), None)
        if stage is None:
            raise SpatialShiftError(f"unknown pipeline stage '{stage_name}'")

        adapter = self.registry.get(stage.model_key)
        try:
            result = invoke(adapter)
        except ModelArtifactMissingError as exc:
            return StageOutcome(
                stage=stage.name,
                model=stage.model_key,
                status="skipped",
                reason="missing_artifact",
                detail=str(exc),
            )
        except (ModelLoadError, ModelCompatibilityError) as exc:
            return StageOutcome(
                stage=stage.name,
                model=stage.model_key,
                status="skipped",
                reason="unusable_artifact",
                detail=str(exc),
            )
        return StageOutcome(
            stage=stage.name,
            model=stage.model_key,
            status="completed",
            result=result.model_dump(mode="json"),
        )
