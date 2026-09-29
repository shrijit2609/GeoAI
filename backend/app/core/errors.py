"""Domain specific exceptions."""

from __future__ import annotations


class SpatialShiftError(Exception):
    """Base class for all application errors."""


class ConfigurationError(SpatialShiftError):
    """Raised when the runtime configuration is invalid."""


class ModelArtifactMissingError(SpatialShiftError):
    """Raised when a required model artifact is not present on disk."""

    def __init__(self, model_key: str, missing: list[str]):
        self.model_key = model_key
        self.missing = missing
        super().__init__(
            f"Model '{model_key}' is missing required artifact(s): {', '.join(missing)}"
        )


class ModelLoadError(SpatialShiftError):
    """Raised when an artifact exists but cannot be loaded."""

    def __init__(self, model_key: str, reason: str):
        self.model_key = model_key
        self.reason = reason
        super().__init__(f"Model '{model_key}' failed to load: {reason}")


class ModelCompatibilityError(SpatialShiftError):
    """Raised when an artifact cannot be used safely.

    This is used instead of guessing preprocessing or architecture details that
    are not recorded in the checkpoint.
    """

    def __init__(self, model_key: str, reason: str, required: list[str] | None = None):
        self.model_key = model_key
        self.reason = reason
        self.required = required or []
        super().__init__(f"Model '{model_key}' is not usable: {reason}")


class InferenceError(SpatialShiftError):
    """Raised when inference fails at runtime."""


class InvalidInputError(SpatialShiftError):
    """Raised for malformed caller supplied input."""


class GeometryError(SpatialShiftError):
    """Raised for geometry parsing/validation failures."""


class CRSError(SpatialShiftError):
    """Raised for CRS detection/transformation failures."""
