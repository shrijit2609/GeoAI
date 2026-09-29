from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings, get_settings, reset_settings_cache


def test_defaults_point_at_repo_models_dir():
    settings = Settings()
    assert settings.model_root.name == "models"
    assert settings.device == "auto"
    assert settings.eager_model_load is False


def test_model_root_env_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MODEL_ROOT", str(tmp_path))
    reset_settings_cache()
    assert get_settings().resolved_model_root == tmp_path


def test_device_env_override(monkeypatch):
    monkeypatch.setenv("SPATIALSHIFT_DEVICE", "CPU")
    reset_settings_cache()
    assert get_settings().device == "cpu"


def test_invalid_device_rejected(monkeypatch):
    monkeypatch.setenv("SPATIALSHIFT_DEVICE", "tpu")
    reset_settings_cache()
    with pytest.raises(Exception):
        get_settings()
    reset_settings_cache()


def test_settings_cache_is_reset_between_calls(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MODEL_ROOT", str(tmp_path / "a"))
    reset_settings_cache()
    first = get_settings().resolved_model_root
    monkeypatch.setenv("MODEL_ROOT", str(tmp_path / "b"))
    assert get_settings().resolved_model_root == first  # cached
    reset_settings_cache()
    assert get_settings().resolved_model_root == tmp_path / "b"
