"""Runtime configuration for the SpatialShiftAI backend.

All configuration is environment driven. No personal or machine specific paths
are baked into the source tree.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DeviceName = Literal["auto", "cpu", "cuda"]

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings.

    Environment variables are prefixed with ``SPATIALSHIFT_`` except for the
    documented ``MODEL_ROOT`` variable which is read as-is.
    """

    model_config = SettingsConfigDict(
        env_prefix="SPATIALSHIFT_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = "SpatialShiftAI"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api"

    device: DeviceName = "auto"
    model_root: Path = Field(default_factory=lambda: REPO_ROOT / "models")

    # Preloading large checkpoints at startup is deliberately disabled.
    eager_model_load: bool = False

    @field_validator("device", mode="before")
    @classmethod
    def _normalise_device(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("model_root", mode="before")
    @classmethod
    def _expand_model_root(cls, value: object) -> object:
        if isinstance(value, str):
            return Path(os.path.expanduser(value))
        return value

    @property
    def resolved_model_root(self) -> Path:
        return self.model_root.expanduser()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    overrides: dict[str, object] = {}
    model_root_env = os.environ.get("MODEL_ROOT")
    if model_root_env:
        overrides["model_root"] = model_root_env
    return Settings(**overrides)


def reset_settings_cache() -> None:
    """Clear the cached settings (used by tests that patch the environment)."""

    get_settings.cache_clear()
