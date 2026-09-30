"""Model 4 adapter - land-record entity resolution."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.errors import (
    InferenceError,
    InvalidInputError,
    ModelCompatibilityError,
)
from app.ml.base import BaseModelAdapter
from app.ml.common import ModelResult
from app.ml.record_features import (
    FeatureSchema,
    compute_model4_features,
    compute_pair_features,
)

DEFAULT_THRESHOLD = 0.5
MATCH_EVIDENCE_CUTOFF = 0.9
CONFLICT_EVIDENCE_CUTOFF = 0.2

# Owner names are not part of the training data for this model.
UNSUPPORTED_FIELDS = ("owner", "owner_name", "owner_names", "khatedar")


class EntityResolver(BaseModelAdapter):
    """Decides whether two land records describe the same entity."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._classifier = None
        self._scaler = None
        self._calibrator = None
        self._schema: FeatureSchema | None = None
        self._schema_payload: dict[str, Any] | None = None
        self._feature_names: list[str] = []
        self._threshold = DEFAULT_THRESHOLD
        self._threshold_source = "default"

    def _load(self) -> None:
        import joblib

        schema_path = self.artifact_path("model4_feature_schema.json")
        try:
            schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelCompatibilityError(
                self.spec.key, f"{schema_path.name} could not be read: {exc}"
            ) from exc
        if not isinstance(schema_payload, dict):
            raise ModelCompatibilityError(
                self.spec.key, "Model 4 feature schema must be a JSON object"
            )

        raw_features = (
            schema_payload.get("features")
            or schema_payload.get("feature_names")
            or schema_payload.get("columns")
        )
        if not isinstance(raw_features, list):
            raise ModelCompatibilityError(
                self.spec.key,
                "Model 4 feature schema must contain an ordered feature list",
            )
        self._schema_payload = schema_payload

        # Older unit fixtures use declarative comparator schemas. The recovered
        # notebook artifact uses its exact flat, double-underscore feature names.
        is_notebook_schema = (
            all(isinstance(name, str) for name in raw_features)
            and isinstance(schema_payload.get("field_mapping"), dict)
            and any("__" in name for name in raw_features)
        )
        if is_notebook_schema:
            self._feature_names = list(raw_features)
        else:
            self._schema = FeatureSchema(self.spec.key, schema_payload)
            self._feature_names = self._schema.names
        self._model_version = f"model4-{schema_payload.get('version', '1')}"

        self._classifier = joblib.load(
            self.artifact_path("model4_entity_resolver_logistic.joblib")
        )
        self._scaler = joblib.load(self.artifact_path("model4_feature_scaler.joblib"))

        expected_features = len(self._feature_names)
        declared_features = schema_payload.get("feature_count")
        if declared_features is not None and declared_features != expected_features:
            raise ModelCompatibilityError(
                self.spec.key,
                f"feature schema declares {declared_features} entries but lists {expected_features}",
            )
        for label, estimator in (("classifier", self._classifier), ("scaler", self._scaler)):
            actual_features = getattr(estimator, "n_features_in_", None)
            if actual_features != expected_features:
                raise ModelCompatibilityError(
                    self.spec.key,
                    f"{label} expects {actual_features} features; schema lists {expected_features}",
                )
        scaler_names = getattr(self._scaler, "feature_names_in_", None)
        if scaler_names is not None and list(scaler_names) != self._feature_names:
            raise ModelCompatibilityError(
                self.spec.key,
                "saved scaler feature names/order do not match model4_feature_schema.json",
            )

        calibrator_path = self.optional_artifact("model4_isotonic_calibrator.joblib")
        if calibrator_path is not None:
            self._calibrator = joblib.load(calibrator_path)

        threshold_path = self.optional_artifact("model4_threshold.json")
        if threshold_path is not None:
            payload = json.loads(Path(threshold_path).read_text())
            value = payload.get(
                "decision_threshold",
                payload.get("threshold", payload.get("best_threshold")),
            )
            if isinstance(value, (int, float)):
                self._threshold = float(value)
                self._threshold_source = threshold_path.name

    # ------------------------------------------------------------------
    def resolve(self, record_a: dict[str, Any], record_b: dict[str, Any]) -> ModelResult:
        self.load()
        if self._classifier is None or self._scaler is None:  # pragma: no cover
            raise InferenceError("entity resolver is not loaded")
        if not isinstance(record_a, dict) or not isinstance(record_b, dict):
            raise InvalidInputError("record_a and record_b must be mappings")

        warnings: list[str] = []
        for field in UNSUPPORTED_FIELDS:
            if field in record_a or field in record_b:
                warnings.append(
                    f"field '{field}' was supplied but is not used: the training "
                    "data contains no owner names"
                )
                break

        if self._schema_payload is None:  # pragma: no cover - guarded by load()
            raise InferenceError("entity resolver feature schema is not loaded")
        if self._schema is not None:
            features, detail = compute_pair_features(
                self._schema, record_a, record_b
            )
        else:
            features, detail = compute_model4_features(
                self._schema_payload, record_a, record_b
            )

        expected_shape = (1, len(self._feature_names))
        if features.shape != expected_shape:
            raise ModelCompatibilityError(
                self.spec.key,
                f"feature builder returned shape {features.shape}; expected {expected_shape}",
            )

        started = time.perf_counter()
        ordered_features = pd.DataFrame(
            features,
            columns=self._feature_names,
        )
        scaled = self._scaler.transform(ordered_features)
        if scaled.shape != expected_shape:
            raise ModelCompatibilityError(
                self.spec.key,
                f"scaler returned shape {scaled.shape}; expected {expected_shape}",
            )
        probability = float(self._classifier.predict_proba(scaled)[0, 1])
        calibrated = probability
        if self._calibrator is not None:
            calibrated = float(
                np.asarray(
                    self._calibrator.predict(np.asarray([probability], dtype=float))
                ).reshape(-1)[0]
            )
        elapsed = (time.perf_counter() - started) * 1000

        if self._calibrator is None:
            warnings.append("no isotonic calibrator installed; probability is uncalibrated")
        if self._threshold_source == "default":
            warnings.append(
                f"no saved threshold found; using the default of {DEFAULT_THRESHOLD}"
            )

        matched_fields = [
            name
            for name, item in detail.items()
            if item["value"] >= MATCH_EVIDENCE_CUTOFF and not item["a_missing"]
            and not item["b_missing"]
        ]
        conflicting_fields = [
            name
            for name, item in detail.items()
            if item["value"] <= CONFLICT_EVIDENCE_CUTOFF and not item["a_missing"]
            and not item["b_missing"]
        ]
        missing_fields = [
            name for name, item in detail.items() if item["a_missing"] or item["b_missing"]
        ]

        return self.build_result(
            confidence=calibrated,
            decision="match" if calibrated >= self._threshold else "no_match",
            inference_ms=elapsed,
            warnings=warnings,
            evidence={
                "raw_probability": probability,
                "calibrated_probability": calibrated,
                "threshold": self._threshold,
                "threshold_source": self._threshold_source,
                "feature_schema_version": (
                    self._schema.version
                    if self._schema is not None
                    else str(self._schema_payload.get("version", "1"))
                ),
                "features": detail,
                "matched_fields": matched_fields,
                "conflicting_fields": conflicting_fields,
                "fields_missing_in_one_record": missing_fields,
            },
        )
