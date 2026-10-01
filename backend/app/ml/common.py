"""Shared ML primitives: device selection, artifact manifest and result contract."""

from __future__ import annotations

import datetime as _dt
import hashlib
import platform
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger(__name__)


class ModelStatus(str, Enum):
    """Lifecycle status reported for every registered model."""

    READY = "ready"
    READY_FOR_TEST = "ready_for_test"
    ARTIFACT_PRESENT_BUT_PREPROCESSING_BLOCKED = (
        "artifact_present_but_preprocessing_blocked"
    )
    ARTIFACT_PRESENT_BUT_INFERENCE_BLOCKED = "artifact_present_but_inference_blocked"
    TRAINING_REQUIRED = "training_required"
    MISSING_ARTIFACT = "missing_artifact"
    ERROR = "error"
    LOAD_ERROR = "error"
    UNSUPPORTED_RUNTIME = "unsupported_runtime"


class ModelRuntime(str, Enum):
    TORCH = "torch"
    SKLEARN = "sklearn"


@dataclass(frozen=True)
class ArtifactSpec:
    """A single file that belongs to a model."""

    name: str
    role: str
    required: bool = True

    def path(self, model_dir: Path) -> Path:
        return model_dir / self.name

    def exists(self, model_dir: Path) -> bool:
        return self.path(model_dir).is_file()


@dataclass(frozen=True)
class ModelSpec:
    """Declarative description of a trained model and its artifacts."""

    key: str
    title: str
    directory: str
    runtime: ModelRuntime
    architecture: str
    task: str
    artifacts: tuple[ArtifactSpec, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)
    readiness_requirements: tuple[str, ...] = field(default_factory=tuple)
    missing_status: ModelStatus = ModelStatus.MISSING_ARTIFACT
    missing_status_reason: str | None = None

    def model_dir(self, model_root: Path) -> Path:
        return model_root / self.directory

    def required_artifacts(self) -> tuple[ArtifactSpec, ...]:
        return tuple(a for a in self.artifacts if a.required)


class ArtifactReport(BaseModel):
    """Per-artifact availability information."""

    name: str
    role: str
    required: bool
    present: bool
    path: str
    size_bytes: int | None = None
    sha256: str | None = None


class ModelHealth(BaseModel):
    """Runtime health of a single model."""

    model_config = {"protected_namespaces": ()}

    key: str
    title: str
    architecture: str
    task: str
    runtime: ModelRuntime
    status: ModelStatus
    model_dir: str
    loaded: bool = False
    load_time_ms: float | None = None
    device: str | None = None
    error: str | None = None
    readiness_reason: str | None = None
    missing_artifacts: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactReport] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    """Traceability metadata attached to every model result."""

    model_config = {"protected_namespaces": ()}

    model_key: str
    model_version: str
    artifacts: list[str] = Field(default_factory=list)
    device: str | None = None
    runtime: str | None = None
    library_versions: dict[str, str] = Field(default_factory=dict)
    host: str = Field(default_factory=platform.node)
    generated_at: _dt.datetime = Field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc)
    )


class ModelResult(BaseModel):
    """Common structured result contract returned by every model adapter."""

    model_config = {"protected_namespaces": ()}

    model: str
    model_version: str
    status: str
    confidence: float | None = None
    decision: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    provenance: Provenance | None = None
    inference_ms: float | None = None


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str) -> str:
    """Resolve the configured device name to an actual torch device string.

    ``auto`` selects CUDA only when a CUDA build of torch reports an available
    device; otherwise CPU is used. When torch is not installed the CPU device
    name is returned so that sklearn based models keep working.
    """

    requested = (requested or "auto").lower()
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            logger.warning("CUDA requested but torch is not installed; using cpu")
        return "cpu"

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to cpu")
            return "cpu"
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def library_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for module_name in ("torch", "sklearn", "numpy", "shapely", "pyproj", "rasterio"):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        versions[module_name] = getattr(module, "__version__", "unknown")
    return versions
