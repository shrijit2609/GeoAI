"""Lazy, cached registry of the trained model adapters."""

from __future__ import annotations

import threading
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.errors import SpatialShiftError
from app.core.logging import get_logger
from app.ml.base import BaseModelAdapter
from app.ml.common import ModelHealth, ModelSpec, ModelStatus, resolve_device
from app.ml.manifest import MODEL_SPECS, MODEL_SPECS_BY_KEY
from app.ml.model1_parcel_matcher import ParcelMatcher
from app.ml.model2_building_extractor import BuildingExtractor
from app.ml.model3_change_detector import ChangeDetector
from app.ml.model4_entity_resolver import EntityResolver
from app.ml.model5_anomaly_detector import AnomalyDetector

logger = get_logger(__name__)

ADAPTERS: dict[str, type[BaseModelAdapter]] = {
    "parcel_matcher": ParcelMatcher,
    "building_extractor": BuildingExtractor,
    "change_detector": ChangeDetector,
    "entity_resolver": EntityResolver,
    "anomaly_detector": AnomalyDetector,
}


class UnknownModelError(SpatialShiftError):
    def __init__(self, key: str):
        super().__init__(
            f"unknown model '{key}'; known models: {', '.join(MODEL_SPECS_BY_KEY)}"
        )


class ModelRegistry:
    """Owns adapter instances, their lifetime and their health reporting.

    Adapters are constructed eagerly (cheap) but checkpoints are only read on
    first use, so large models never land on the GPU at import time.
    """

    def __init__(self, model_root: Path, device: str = "auto") -> None:
        self.model_root = Path(model_root)
        self.requested_device = device
        self.device = resolve_device(device)
        self._lock = threading.Lock()
        self._adapters: dict[str, BaseModelAdapter] = {}

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ModelRegistry":
        settings = settings or get_settings()
        return cls(model_root=settings.resolved_model_root, device=settings.device)

    # ------------------------------------------------------------------
    @property
    def specs(self) -> tuple[ModelSpec, ...]:
        return MODEL_SPECS

    def keys(self) -> list[str]:
        return [spec.key for spec in MODEL_SPECS]

    def get(self, key: str) -> BaseModelAdapter:
        spec = MODEL_SPECS_BY_KEY.get(key)
        if spec is None:
            raise UnknownModelError(key)
        with self._lock:
            adapter = self._adapters.get(key)
            if adapter is None:
                adapter = ADAPTERS[key](
                    spec=spec, model_root=self.model_root, device=self.requested_device
                )
                self._adapters[key] = adapter
            return adapter

    def load(self, key: str) -> BaseModelAdapter:
        adapter = self.get(key)
        adapter.load()
        return adapter

    def unload_all(self) -> None:
        with self._lock:
            for adapter in self._adapters.values():
                adapter.unload()
            self._adapters.clear()

    # ------------------------------------------------------------------
    def health(self, key: str, probe: bool = True, include_hash: bool = False) -> ModelHealth:
        return self.get(key).health(probe=probe, include_hash=include_hash)

    def health_report(
        self, probe: bool = True, include_hash: bool = False
    ) -> dict[str, ModelHealth]:
        return {
            key: self.health(key, probe=probe, include_hash=include_hash)
            for key in self.keys()
        }

    def summary(self, probe: bool = True) -> dict[str, object]:
        report = self.health_report(probe=probe)
        counts: dict[str, int] = {}
        for health in report.values():
            counts[health.status.value] = counts.get(health.status.value, 0) + 1
        return {
            "model_root": str(self.model_root),
            "model_root_exists": self.model_root.is_dir(),
            "device": self.device,
            "requested_device": self.requested_device,
            "total": len(report),
            "ready": counts.get(ModelStatus.READY.value, 0),
            "status_counts": counts,
            "models": report,
        }


_registry: ModelRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = ModelRegistry.from_settings()
        return _registry


def reset_registry() -> None:
    global _registry
    with _registry_lock:
        if _registry is not None:
            _registry.unload_all()
        _registry = None
