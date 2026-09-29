"""Adapter tests for Model 4 and Model 5.

The artifacts used here are built inside the test from scikit-learn on
synthetic data. They exercise the adapter contract (schema driven features,
scaling, calibration, thresholds) - they are not the trained SIH models and
say nothing about model quality.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.core.errors import (
    InvalidInputError,
    ModelArtifactMissingError,
    ModelCompatibilityError,
)
from app.ml.common import ModelStatus
from app.ml.model_registry import ModelRegistry

joblib = pytest.importorskip("joblib")
sklearn = pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def _fit_logistic(n_features: int) -> tuple[LogisticRegression, StandardScaler]:
    rng = np.random.default_rng(0)
    features = rng.random((200, n_features))
    labels = (features.mean(axis=1) > 0.5).astype(int)
    scaler = StandardScaler().fit(features)
    model = LogisticRegression().fit(scaler.transform(features), labels)
    return model, scaler


@pytest.fixture
def entity_resolver_artifacts(empty_model_root: Path, entity_resolver_schema_payload):
    directory = empty_model_root / "model4_entity_resolution"
    model, scaler = _fit_logistic(3)
    joblib.dump(model, directory / "model4_entity_resolver_logistic.joblib")
    joblib.dump(scaler, directory / "model4_feature_scaler.joblib")
    (directory / "model4_feature_schema.json").write_text(
        json.dumps(entity_resolver_schema_payload)
    )
    (directory / "model4_threshold.json").write_text(json.dumps({"threshold": 0.42}))
    return directory


@pytest.fixture
def anomaly_artifacts(empty_model_root: Path):
    directory = empty_model_root / "model5_anomaly_detection"
    model, scaler = _fit_logistic(3)
    joblib.dump(model, directory / "model5_anomaly_classifier.joblib")
    joblib.dump(scaler, directory / "model5_feature_scaler.joblib")
    (directory / "model5_feature_schema.json").write_text(
        json.dumps(
            {
                "version": "test-1",
                "features": [
                    {"name": "khasra_missing", "field": "khasra", "comparator": "missing"},
                    {"name": "village_len", "field": "village", "comparator": "length"},
                    {"name": "area_value", "field": "area", "comparator": "value"},
                ],
            }
        )
    )
    return directory


# ----------------------------------------------------------------------
def test_entity_resolver_without_artifacts_raises(registry: ModelRegistry):
    with pytest.raises(ModelArtifactMissingError):
        registry.get("entity_resolver").resolve({}, {})


def test_entity_resolver_real_inference(
    registry: ModelRegistry, entity_resolver_artifacts: Path
):
    adapter = registry.get("entity_resolver")
    assert registry.health("entity_resolver").status is ModelStatus.READY

    result = adapter.resolve(
        {"village": "Sultanpur", "khasra": "121/2", "area": 1000},
        {"village": "Sultanpur", "khasra": "121/2", "area": 1010},
    )
    assert result.model == "entity_resolver"
    assert 0.0 <= result.confidence <= 1.0
    assert result.decision in ("match", "no_match")
    assert result.evidence["threshold"] == 0.42
    assert result.evidence["threshold_source"] == "model4_threshold.json"
    assert "khasra_exact" in result.evidence["matched_fields"]
    assert result.provenance.model_key == "entity_resolver"
    assert result.provenance.library_versions["python"]
    assert result.inference_ms is not None


def test_entity_resolver_reports_conflicting_fields(
    registry: ModelRegistry, entity_resolver_artifacts: Path
):
    result = registry.get("entity_resolver").resolve(
        {"village": "Sultanpur", "khasra": "121/2", "area": 1000},
        {"village": "Chhapraula", "khasra": "88", "area": 4000},
    )
    assert "khasra_exact" in result.evidence["conflicting_fields"]


def test_entity_resolver_ignores_owner_name_with_a_warning(
    registry: ModelRegistry, entity_resolver_artifacts: Path
):
    result = registry.get("entity_resolver").resolve(
        {"village": "Sultanpur", "khasra": "1", "area": 1, "owner_name": "A"},
        {"village": "Sultanpur", "khasra": "1", "area": 1, "owner_name": "B"},
    )
    assert any("owner names" in warning for warning in result.warnings)
    assert "owner_name" not in result.evidence["features"]


def test_entity_resolver_rejects_malformed_input(
    registry: ModelRegistry, entity_resolver_artifacts: Path
):
    with pytest.raises(InvalidInputError):
        registry.get("entity_resolver").resolve("not-a-record", {})


def test_unknown_comparator_is_a_compatibility_error(
    registry: ModelRegistry, empty_model_root: Path
):
    directory = empty_model_root / "model4_entity_resolution"
    model, scaler = _fit_logistic(1)
    joblib.dump(model, directory / "model4_entity_resolver_logistic.joblib")
    joblib.dump(scaler, directory / "model4_feature_scaler.joblib")
    (directory / "model4_feature_schema.json").write_text(
        json.dumps({"features": [{"name": "x", "field": "village", "comparator": "magic"}]})
    )
    with pytest.raises(ModelCompatibilityError):
        registry.get("entity_resolver").resolve({"village": "a"}, {"village": "b"})


def test_uninterpretable_feature_name_is_a_compatibility_error(
    registry: ModelRegistry, empty_model_root: Path
):
    directory = empty_model_root / "model4_entity_resolution"
    model, scaler = _fit_logistic(1)
    joblib.dump(model, directory / "model4_entity_resolver_logistic.joblib")
    joblib.dump(scaler, directory / "model4_feature_scaler.joblib")
    (directory / "model4_feature_schema.json").write_text(
        json.dumps({"features": ["mystery_feature_one"]})
    )
    with pytest.raises(ModelCompatibilityError):
        registry.load("entity_resolver").resolve({}, {})


# ----------------------------------------------------------------------
def test_anomaly_detector_without_artifacts_raises(registry: ModelRegistry):
    with pytest.raises(ModelArtifactMissingError):
        registry.get("anomaly_detector").predict({})


def test_anomaly_detector_real_inference(registry: ModelRegistry, anomaly_artifacts: Path):
    result = registry.get("anomaly_detector").predict(
        {"village": "Sultanpur", "khasra": None, "area": 1200}
    )
    assert result.decision in ("review_required", "no_review_required")
    assert 0.0 <= result.confidence <= 1.0
    assert result.evidence["missing_fields"] == ["khasra"]
    assert "missing critical attribute" in result.evidence["explanation"]
    assert any("geometry or topology" in warning for warning in result.warnings)


def test_anomaly_detector_language_never_alleges_fraud(
    registry: ModelRegistry, anomaly_artifacts: Path
):
    result = registry.get("anomaly_detector").predict(
        {"village": "Sultanpur", "khasra": "1", "area": 10}
    )
    text = json.dumps(result.model_dump(mode="json")).lower()
    for term in ("fraud", "illegal", "encroachment"):
        assert term not in text


def test_anomaly_detector_rejects_malformed_input(
    registry: ModelRegistry, anomaly_artifacts: Path
):
    with pytest.raises(InvalidInputError):
        registry.get("anomaly_detector").predict(["not", "a", "record"])


def test_anomaly_detector_rejects_non_numeric_value_feature(
    registry: ModelRegistry, anomaly_artifacts: Path
):
    with pytest.raises(ModelCompatibilityError):
        registry.get("anomaly_detector").predict(
            {"village": "Sultanpur", "khasra": "1", "area": "not-a-number"}
        )
