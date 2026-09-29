"""Model 5 adapter - land-record anomaly / conflict detection.

Outputs are review signals about record attributes. They never assert fraud,
illegality or encroachment, and they say nothing about geometry or topology.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.core.errors import InferenceError, InvalidInputError
from app.ml.base import BaseModelAdapter
from app.ml.common import ModelResult
from app.ml.record_features import FeatureSchema, compute_record_features

DEFAULT_THRESHOLD = 0.5

REVIEW_TERMS = {
    "missing_attribute": "missing critical attribute",
    "attribute_conflict": "attribute conflict",
    "identity_conflict": "identity conflict",
    "statistical": "anomalous record",
}


class AnomalyDetector(BaseModelAdapter):
    """Flags land records that require human review."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._classifier = None
        self._scaler = None
        self._calibrator = None
        self._schema: FeatureSchema | None = None
        self._threshold = DEFAULT_THRESHOLD
        self._threshold_source = "default"

    def _load(self) -> None:
        import joblib

        self._classifier = joblib.load(
            self.artifact_path("model5_anomaly_classifier.joblib")
        )
        self._schema = FeatureSchema.load(
            self.spec.key, self.artifact_path("model5_feature_schema.json")
        )
        self._model_version = f"model5-{self._schema.version}"

        scaler_path = self.optional_artifact("model5_feature_scaler.joblib")
        if scaler_path is not None:
            self._scaler = joblib.load(scaler_path)

        calibrator_path = self.optional_artifact("model5_isotonic_calibrator.joblib")
        if calibrator_path is not None:
            self._calibrator = joblib.load(calibrator_path)

        threshold_path = self.optional_artifact("model5_threshold.json")
        if threshold_path is not None:
            payload = json.loads(Path(threshold_path).read_text())
            value = payload.get("threshold", payload.get("best_threshold"))
            if isinstance(value, (int, float)):
                self._threshold = float(value)
                self._threshold_source = threshold_path.name

    # ------------------------------------------------------------------
    def predict(self, record: dict[str, Any]) -> ModelResult:
        self.load()
        if self._schema is None or self._classifier is None:  # pragma: no cover
            raise InferenceError("anomaly detector is not loaded")
        if not isinstance(record, dict):
            raise InvalidInputError("record must be a mapping")

        features, detail = compute_record_features(self._schema, record)

        started = time.perf_counter()
        scaled = self._scaler.transform(features) if self._scaler is not None else features
        probability = float(self._classifier.predict_proba(scaled)[0, 1])
        calibrated = probability
        if self._calibrator is not None:
            calibrated = float(
                np.asarray(
                    self._calibrator.predict(np.asarray([probability], dtype=float))
                ).reshape(-1)[0]
            )
        elapsed = (time.perf_counter() - started) * 1000

        warnings = [
            "this model addresses record attribute anomalies only; it does not "
            "establish geometry or topology anomalies",
        ]
        if self._calibrator is None:
            warnings.append("no isotonic calibrator installed; probability is uncalibrated")
        if self._threshold_source == "default":
            warnings.append(
                f"no saved threshold found; using the default of {DEFAULT_THRESHOLD}"
            )

        anomaly_type = self._anomaly_type(detail)
        flagged = calibrated >= self._threshold

        return self.build_result(
            confidence=calibrated,
            decision="review_required" if flagged else "no_review_required",
            inference_ms=elapsed,
            warnings=warnings,
            evidence={
                "raw_probability": probability,
                "calibrated_probability": calibrated,
                "threshold": self._threshold,
                "threshold_source": self._threshold_source,
                "anomaly_type": anomaly_type,
                "explanation": self._explain(detail, anomaly_type, flagged),
                "feature_schema_version": self._schema.version,
                "features": detail,
                "missing_fields": [
                    item["field"]
                    for item in detail.values()
                    if item["comparator"] in ("missing", "is_missing", "any_missing")
                    and item["value"] == 1.0
                ],
            },
        )

    def _anomaly_type(self, detail: dict[str, Any]) -> str | None:
        classes = getattr(self._classifier, "classes_", None)
        if classes is not None and len(classes) > 2:
            return "multiclass_label_available"
        missing = [
            item["field"]
            for item in detail.values()
            if item["comparator"] in ("missing", "is_missing", "any_missing")
            and item["value"] == 1.0
        ]
        if missing:
            return REVIEW_TERMS["missing_attribute"]
        return None

    def _explain(
        self, detail: dict[str, Any], anomaly_type: str | None, flagged: bool
    ) -> str:
        missing = [
            item["field"]
            for item in detail.values()
            if item["comparator"] in ("missing", "is_missing", "any_missing")
            and item["value"] == 1.0
        ]
        parts: list[str] = []
        if missing:
            parts.append("missing critical attribute(s): " + ", ".join(sorted(set(missing))))
        if flagged:
            parts.append(
                "the trained classifier scored this record above the review threshold"
            )
        else:
            parts.append(
                "the trained classifier scored this record below the review threshold"
            )
        if anomaly_type:
            parts.append(f"anomaly type: {anomaly_type}")
        return "; ".join(parts)
