"""Model 4 adapter - land-record entity resolution."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.core.errors import InferenceError, InvalidInputError
from app.ml.base import BaseModelAdapter
from app.ml.common import ModelResult
from app.ml.record_features import FeatureSchema, compute_pair_features

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
        self._threshold = DEFAULT_THRESHOLD
        self._threshold_source = "default"

    def _load(self) -> None:
        import joblib

        self._classifier = joblib.load(
            self.artifact_path("model4_entity_resolver_logistic.joblib")
        )
        self._scaler = joblib.load(self.artifact_path("model4_feature_scaler.joblib"))
        self._schema = FeatureSchema.load(
            self.spec.key, self.artifact_path("model4_feature_schema.json")
        )
        self._model_version = f"model4-{self._schema.version}"

        calibrator_path = self.optional_artifact("model4_isotonic_calibrator.joblib")
        if calibrator_path is not None:
            self._calibrator = joblib.load(calibrator_path)

        threshold_path = self.optional_artifact("model4_threshold.json")
        if threshold_path is not None:
            payload = json.loads(Path(threshold_path).read_text())
            value = payload.get("threshold", payload.get("best_threshold"))
            if isinstance(value, (int, float)):
                self._threshold = float(value)
                self._threshold_source = threshold_path.name

    # ------------------------------------------------------------------
    def resolve(self, record_a: dict[str, Any], record_b: dict[str, Any]) -> ModelResult:
        self.load()
        if self._schema is None or self._classifier is None:  # pragma: no cover
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

        features, detail = compute_pair_features(self._schema, record_a, record_b)

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
                "feature_schema_version": self._schema.version,
                "features": detail,
                "matched_fields": matched_fields,
                "conflicting_fields": conflicting_fields,
                "fields_missing_in_one_record": missing_fields,
            },
        )
