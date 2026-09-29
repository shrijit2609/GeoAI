from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import ModelArtifactMissingError
from app.ml.common import ModelStatus
from app.ml.manifest import MODEL_SPECS
from app.ml.model_registry import ModelRegistry, UnknownModelError


def test_registry_exposes_all_five_models(registry: ModelRegistry):
    assert registry.keys() == [
        "parcel_matcher",
        "building_extractor",
        "change_detector",
        "entity_resolver",
        "anomaly_detector",
    ]


def test_adapters_are_cached(registry: ModelRegistry):
    assert registry.get("parcel_matcher") is registry.get("parcel_matcher")


def test_unknown_model_raises(registry: ModelRegistry):
    with pytest.raises(UnknownModelError):
        registry.get("nope")


def test_missing_artifacts_are_reported_not_faked(registry: ModelRegistry):
    report = registry.health_report()
    assert set(report) == {spec.key for spec in MODEL_SPECS}
    for health in report.values():
        assert health.status is ModelStatus.MISSING_ARTIFACT
        assert health.loaded is False
        assert health.missing_artifacts
        assert health.error is None


def test_loading_a_missing_model_raises(registry: ModelRegistry):
    with pytest.raises(ModelArtifactMissingError) as excinfo:
        registry.load("entity_resolver")
    assert "model4_entity_resolver_logistic.joblib" in excinfo.value.missing


def test_summary_counts_statuses(registry: ModelRegistry):
    summary = registry.summary()
    assert summary["total"] == 5
    assert summary["ready"] == 0
    assert summary["status_counts"]["missing_artifact"] == 5
    assert summary["device"] == "cpu"


def test_artifact_detection_reflects_the_filesystem(
    registry: ModelRegistry, empty_model_root: Path
):
    adapter = registry.get("anomaly_detector")
    assert "model5_anomaly_classifier.joblib" in adapter.missing_required_artifacts()

    (empty_model_root / "model5_anomaly_detection" / "model5_anomaly_classifier.joblib").write_bytes(
        b"not-a-real-model"
    )
    reports = {report.name: report for report in adapter.artifact_reports()}
    assert reports["model5_anomaly_classifier.joblib"].present is True
    assert reports["model5_anomaly_classifier.joblib"].size_bytes == 16
    assert reports["model5_threshold.json"].present is False


def test_corrupt_artifact_reports_load_error(
    registry: ModelRegistry, empty_model_root: Path, write_json
):
    directory = empty_model_root / "model5_anomaly_detection"
    (directory / "model5_anomaly_classifier.joblib").write_bytes(b"corrupt")
    write_json(directory / "model5_feature_schema.json", {"features": ["area_value"]})

    health = registry.health("anomaly_detector")
    assert health.status is ModelStatus.LOAD_ERROR
    assert health.error
    assert health.loaded is False


def test_probe_false_reports_ready_for_test(
    registry: ModelRegistry, empty_model_root: Path, write_json
):
    directory = empty_model_root / "model5_anomaly_detection"
    (directory / "model5_anomaly_classifier.joblib").write_bytes(b"corrupt")
    write_json(directory / "model5_feature_schema.json", {"features": ["area_value"]})

    health = registry.health("anomaly_detector", probe=False)
    assert health.status is ModelStatus.READY_FOR_TEST
