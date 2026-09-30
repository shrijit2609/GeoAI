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
import pandas as pd

from app.core.errors import InferenceError, InvalidInputError, ModelCompatibilityError
from app.ml.base import BaseModelAdapter
from app.ml.common import ModelResult
from app.ml.record_features import (
    FeatureSchema,
    compute_model5_notebook_features,
    compute_record_features,
)

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
        self._schema_payload: dict[str, Any] | None = None
        self._feature_names: list[str] = []
        self._notebook_inference_config: dict[str, Any] | None = None
        self._isolation = None
        self._threshold = DEFAULT_THRESHOLD
        self._threshold_source = "default"

    def _load(self) -> None:
        import joblib

        schema_path = self.artifact_path("model5_feature_schema.json")
        try:
            schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelCompatibilityError(
                self.spec.key, f"{schema_path.name} could not be read: {exc}"
            ) from exc
        if not isinstance(schema_payload, dict):
            raise ModelCompatibilityError(
                self.spec.key, "Model 5 feature schema must be a JSON object"
            )
        raw_features = schema_payload.get("features")
        if not isinstance(raw_features, list):
            raise ModelCompatibilityError(
                self.spec.key, "Model 5 feature schema must list its ordered features"
            )
        self._schema_payload = schema_payload
        is_notebook_schema = all(isinstance(name, str) for name in raw_features) and any(
            "__" in name for name in raw_features
        )
        if is_notebook_schema:
            self._feature_names = list(raw_features)
        else:
            self._schema = FeatureSchema.load(self.spec.key, schema_path)
            self._feature_names = self._schema.names
        self._model_version = f"model5-{schema_payload.get('version', '1')}"

        self._classifier = joblib.load(
            self.artifact_path("model5_anomaly_classifier.joblib")
        )
        expected_features = len(self._feature_names)

        scaler_path = self.optional_artifact("model5_feature_scaler.joblib")
        if scaler_path is not None:
            self._scaler = joblib.load(scaler_path)

        if is_notebook_schema:
            isolation_path = self.optional_artifact("model5_isolation_forest.joblib")
            config_path = self.optional_artifact("model5_inference_config.json")
            if isolation_path is None or config_path is None or self._scaler is None:
                raise ModelCompatibilityError(
                    self.spec.key,
                    "notebook-trained Model 5 requires its scaler, isolation forest, and inference config",
                )
            self._isolation = joblib.load(isolation_path)
            self._notebook_inference_config = json.loads(
                Path(config_path).read_text(encoding="utf-8")
            )
            for name, estimator in (
                ("classifier", self._classifier),
                ("scaler", self._scaler),
                ("isolation forest", self._isolation),
            ):
                dimensions = getattr(estimator, "n_features_in_", None)
                if dimensions != expected_features:
                    raise ModelCompatibilityError(
                        self.spec.key,
                        f"{name} expects {dimensions} features; schema lists {expected_features}",
                    )
        if self._scaler is not None:
            scaler_names = getattr(self._scaler, "feature_names_in_", None)
            if scaler_names is not None and list(scaler_names) != self._feature_names:
                raise ModelCompatibilityError(
                    self.spec.key,
                    "saved scaler feature names/order do not match model5_feature_schema.json",
                )

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
        if self._schema_payload is None or self._classifier is None:  # pragma: no cover
            raise InferenceError("anomaly detector is not loaded")
        if not isinstance(record, dict):
            raise InvalidInputError("record must be a mapping")

        if self._schema_payload is None:  # pragma: no cover
            raise InferenceError("anomaly detector feature schema is not loaded")
        if self._notebook_inference_config is not None:
            features, detail = compute_model5_notebook_features(
                self._schema_payload,
                self._notebook_inference_config,
                record,
            )
        elif self._schema is not None:
            features, detail = compute_record_features(self._schema, record)
        else:  # pragma: no cover
            raise InferenceError("anomaly detector feature builder is not loaded")

        expected_shape = (1, len(self._feature_names))
        if features.shape != expected_shape:
            raise ModelCompatibilityError(
                self.spec.key,
                f"feature builder returned {features.shape}; expected {expected_shape}",
            )

        started = time.perf_counter()
        if self._scaler is not None:
            scaled = self._scaler.transform(
                pd.DataFrame(features, columns=self._feature_names)
            )
        else:
            scaled = features
        supervised_probability = float(self._classifier.predict_proba(scaled)[0, 1])
        isolation_score = None
        if self._isolation is not None and self._notebook_inference_config is not None:
            raw_isolation = float(-self._isolation.decision_function(scaled)[0])
            config = self._notebook_inference_config
            low = float(config["isolation_score_min"])
            high = float(config["isolation_score_max"])
            isolation_score = float(np.clip((raw_isolation - low) / (high - low + 1e-9), 0.0, 1.0))
            hybrid_weights = config.get("hybrid_weights", {"supervised": 0.75, "isolation_forest": 0.25})
            probability = float(
                hybrid_weights["supervised"] * supervised_probability
                + hybrid_weights["isolation_forest"] * isolation_score
            )
        else:
            probability = supervised_probability
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
                "supervised_probability": supervised_probability,
                "normalized_isolation_score": isolation_score,
                "calibrated_probability": calibrated,
                "threshold": self._threshold,
                "threshold_source": self._threshold_source,
                "anomaly_type": anomaly_type,
                "explanation": self._explain(detail, anomaly_type, flagged),
                "feature_schema_version": (
                    self._schema.version
                    if self._schema is not None
                    else str(self._schema_payload.get("version", "1"))
                ),
                "features": detail,
                "missing_fields": self._missing_fields(detail),
            },
        )

    def _missing_fields(self, detail: dict[str, Any]) -> list[str]:
        if self._notebook_inference_config is not None:
            return [
                field
                for field in ("district", "khasra", "pargana", "village", "tehsil", "block")
                if detail.get(f"{field}__present", {}).get("value") == 0.0
            ]
        return [
            item["field"]
            for item in detail.values()
            if item.get("comparator") in ("missing", "is_missing", "any_missing")
            and item.get("value") == 1.0
        ]

    def _anomaly_type(self, detail: dict[str, Any]) -> str | None:
        if self._notebook_inference_config is not None:
            return (
                REVIEW_TERMS["missing_attribute"]
                if self._missing_fields(detail)
                else None
            )
        classes = getattr(self._classifier, "classes_", None)
        if classes is not None and len(classes) > 2:
            return "multiclass_label_available"
        missing = self._missing_fields(detail)
        if missing:
            return REVIEW_TERMS["missing_attribute"]
        return None

    def _explain(
        self, detail: dict[str, Any], anomaly_type: str | None, flagged: bool
    ) -> str:
        missing = self._missing_fields(detail)
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
