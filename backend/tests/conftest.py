from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import Settings, reset_settings_cache
from app.ml.manifest import MODEL_SPECS
from app.ml.model_registry import ModelRegistry, reset_registry


@pytest.fixture
def empty_model_root(tmp_path: Path) -> Path:
    """A MODEL_ROOT with the expected directories but no artifacts."""

    root = tmp_path / "models"
    for spec in MODEL_SPECS:
        (root / spec.directory).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def registry(empty_model_root: Path) -> ModelRegistry:
    return ModelRegistry(model_root=empty_model_root, device="cpu")


@pytest.fixture
def settings(empty_model_root: Path) -> Settings:
    return Settings(model_root=empty_model_root, device="cpu")


@pytest.fixture(autouse=True)
def _clean_globals():
    reset_settings_cache()
    reset_registry()
    yield
    reset_settings_cache()
    reset_registry()


@pytest.fixture
def api_client(empty_model_root: Path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_ROOT", str(empty_model_root))
    monkeypatch.setenv("SPATIALSHIFT_DEVICE", "cpu")
    reset_settings_cache()
    reset_registry()

    from app.main import create_app

    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def entity_resolver_schema_payload() -> dict:
    return {
        "version": "test-1",
        "features": [
            {"name": "village_sim", "field": "village", "comparator": "jaro_winkler"},
            {"name": "khasra_exact", "field": "khasra", "comparator": "exact"},
            {"name": "area_ratio", "field": "area", "comparator": "ratio"},
        ],
    }


@pytest.fixture
def write_json():
    def _write(path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return path

    return _write
