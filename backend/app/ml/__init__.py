"""Machine learning adapters and registry for SpatialShiftAI."""

from app.ml.common import ModelHealth, ModelResult, ModelStatus, Provenance
from app.ml.manifest import MODEL_SPECS, MODEL_SPECS_BY_KEY
from app.ml.model_registry import ModelRegistry, get_registry, reset_registry

__all__ = [
    "MODEL_SPECS",
    "MODEL_SPECS_BY_KEY",
    "ModelHealth",
    "ModelRegistry",
    "ModelResult",
    "ModelStatus",
    "Provenance",
    "get_registry",
    "reset_registry",
]
