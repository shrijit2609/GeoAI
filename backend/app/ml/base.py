"""Base class shared by every model adapter."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.core.errors import (
    InferenceError,
    InvalidInputError,
    ModelArtifactMissingError,
    ModelCompatibilityError,
    ModelLoadError,
)
from app.core.logging import get_logger
from app.ml.common import (
    ArtifactReport,
    ModelHealth,
    ModelResult,
    ModelSpec,
    ModelStatus,
    Provenance,
    library_versions,
    resolve_device,
)

logger = get_logger(__name__)


class BaseModelAdapter:
    """Lazily loads a trained artifact and exposes a structured contract.

    Subclasses implement :meth:`_load` and their own inference entry point. No
    adapter ever returns a synthetic prediction: when an artifact is absent the
    adapter raises, and the registry surfaces the status instead.
    """

    def __init__(self, spec: ModelSpec, model_root: Path, device: str = "auto") -> None:
        self.spec = spec
        self.model_root = Path(model_root)
        self.requested_device = device
        self.device = resolve_device(device)
        self._loaded = False
        self._load_time_ms: float | None = None
        self._error: str | None = None
        self._status: ModelStatus = ModelStatus.MISSING_ARTIFACT
        self._model_version: str = "unknown"

    # ------------------------------------------------------------------
    # artifact handling
    # ------------------------------------------------------------------
    @property
    def model_dir(self) -> Path:
        return self.spec.model_dir(self.model_root)

    def artifact_path(self, name: str) -> Path:
        return self.model_dir / name

    def optional_artifact(self, name: str) -> Path | None:
        path = self.artifact_path(name)
        return path if path.is_file() else None

    def missing_required_artifacts(self) -> list[str]:
        return [
            artifact.name
            for artifact in self.spec.required_artifacts()
            if not artifact.exists(self.model_dir)
        ]

    def artifact_reports(self, include_hash: bool = False) -> list[ArtifactReport]:
        from app.ml.common import sha256_of

        reports: list[ArtifactReport] = []
        for artifact in self.spec.artifacts:
            path = artifact.path(self.model_dir)
            present = path.is_file()
            reports.append(
                ArtifactReport(
                    name=artifact.name,
                    role=artifact.role,
                    required=artifact.required,
                    present=present,
                    path=str(path),
                    size_bytes=path.stat().st_size if present else None,
                    sha256=sha256_of(path) if present and include_hash else None,
                )
            )
        return reports

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def load(self, force: bool = False) -> None:
        if self._loaded and not force:
            return

        missing = self.missing_required_artifacts()
        if missing:
            self._status = ModelStatus.MISSING_ARTIFACT
            self._error = None
            raise ModelArtifactMissingError(self.spec.key, missing)

        started = time.perf_counter()
        try:
            self._load()
        except ModelCompatibilityError as exc:
            self._loaded = False
            self._status = ModelStatus.LOAD_ERROR
            self._error = f"{type(exc).__name__}: {exc}"
            logger.warning("Model %s is not usable: %s", self.spec.key, exc)
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced through health reporting
            self._loaded = False
            self._status = ModelStatus.LOAD_ERROR
            self._error = f"{type(exc).__name__}: {exc}"
            logger.exception("Failed to load model %s", self.spec.key)
            raise ModelLoadError(self.spec.key, str(exc)) from exc

        self._load_time_ms = (time.perf_counter() - started) * 1000
        self._loaded = True
        self._error = None
        self._status = ModelStatus.READY

    def _load(self) -> None:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def unload(self) -> None:
        self._loaded = False
        self._load_time_ms = None
        self._status = ModelStatus.MISSING_ARTIFACT

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # health
    # ------------------------------------------------------------------
    def health(self, probe: bool = True, include_hash: bool = False) -> ModelHealth:
        """Inspect artifact availability and, optionally, attempt a real load."""

        missing = self.missing_required_artifacts()
        status = self._status
        error = self._error

        if missing:
            status = ModelStatus.MISSING_ARTIFACT
            error = None
        elif self._loaded:
            status = ModelStatus.READY
        elif probe:
            try:
                self.load()
                status = ModelStatus.READY
                error = None
            except (ModelLoadError, ModelCompatibilityError) as exc:
                status = ModelStatus.LOAD_ERROR
                error = exc.reason
            except ModelArtifactMissingError as exc:
                status = ModelStatus.MISSING_ARTIFACT
                missing = exc.missing
        else:
            status = ModelStatus.READY_FOR_TEST

        return ModelHealth(
            key=self.spec.key,
            title=self.spec.title,
            architecture=self.spec.architecture,
            task=self.spec.task,
            runtime=self.spec.runtime,
            status=status,
            model_dir=str(self.model_dir),
            loaded=self._loaded,
            load_time_ms=self._load_time_ms,
            device=self.device,
            error=error,
            missing_artifacts=missing,
            artifacts=self.artifact_reports(include_hash=include_hash),
            notes=list(self.spec.notes),
        )

    # ------------------------------------------------------------------
    # results
    # ------------------------------------------------------------------
    @property
    def model_version(self) -> str:
        return self._model_version

    def provenance(self) -> Provenance:
        return Provenance(
            model_key=self.spec.key,
            model_version=self.model_version,
            artifacts=[
                str(artifact.path(self.model_dir))
                for artifact in self.spec.artifacts
                if artifact.exists(self.model_dir)
            ],
            device=self.device,
            runtime=self.spec.runtime.value,
            library_versions=library_versions(),
        )

    def infer(self, *args: Any, **kwargs: Any) -> ModelResult:
        """Dispatch a model-specific request to the adapter's inference method."""

        payload: dict[str, Any] = {}
        if len(args) == 1 and isinstance(args[0], dict):
            payload = dict(args[0])
        elif args:
            payload = {"_args": list(args)}
        payload.update(kwargs)

        key = self.spec.key
        if key == "parcel_matcher":
            parcel_a = payload.get("parcel_a", payload.get("parcelA", payload.get("a")))
            parcel_b = payload.get("parcel_b", payload.get("parcelB", payload.get("b")))
            if parcel_a is None or parcel_b is None:
                raise InvalidInputError(
                    "parcel_matcher inference requires parcel_a and parcel_b"
                )
            return self.predict(parcel_a, parcel_b)

        if key == "building_extractor":
            image = payload.get("image", payload.get("input", payload.get("data")))
            if image is None:
                raise InvalidInputError("building_extractor inference requires an image")
            return self.predict(image)

        if key == "change_detector":
            before = payload.get("before_image", payload.get("before"))
            after = payload.get("after_image", payload.get("after"))
            if before is None or after is None:
                raise InvalidInputError(
                    "change_detector inference requires before_image and after_image"
                )
            return self.predict(before, after)

        if key == "entity_resolver":
            record_a = payload.get("record_a", payload.get("recordA", payload.get("a")))
            record_b = payload.get("record_b", payload.get("recordB", payload.get("b")))
            if record_a is None or record_b is None:
                raise InvalidInputError(
                    "entity_resolver inference requires record_a and record_b"
                )
            return self.resolve(record_a, record_b)

        if key == "anomaly_detector":
            record = payload.get("record", payload.get("data", payload.get("payload")))
            if record is None:
                raise InvalidInputError("anomaly_detector inference requires a record")
            return self.predict(record)

        raise InferenceError(f"model '{key}' does not define a generic inference contract")

    def build_result(
        self,
        *,
        status: str = "success",
        confidence: float | None = None,
        decision: str | None = None,
        evidence: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        inference_ms: float | None = None,
    ) -> ModelResult:
        return ModelResult(
            model=self.spec.key,
            model_version=self.model_version,
            status=status,
            confidence=confidence,
            decision=decision,
            evidence=evidence or {},
            warnings=warnings or [],
            provenance=self.provenance(),
            inference_ms=inference_ms,
        )
