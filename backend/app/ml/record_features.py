"""Feature computation for the tabular land-record models (Model 4 and 5).

Features are derived strictly from the feature schema saved alongside the
trained artifacts. Nothing is invented: an unknown comparator or an unknown
feature name is a compatibility error, not a zero.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.core.errors import ModelCompatibilityError

WHITESPACE = re.compile(r"\s+")
NON_ALNUM = re.compile(r"[^0-9a-z ]+")


def normalise_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = NON_ALNUM.sub(" ", text.lower())
    return WHITESPACE.sub(" ", text).strip()


def to_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def _jaro_winkler(a: str, b: str) -> float:
    from rapidfuzz.distance import JaroWinkler

    return float(JaroWinkler.similarity(a, b))


def _levenshtein_ratio(a: str, b: str) -> float:
    from rapidfuzz.distance import Levenshtein

    return float(Levenshtein.normalized_similarity(a, b))


def _token_sort(a: str, b: str) -> float:
    from rapidfuzz import fuzz

    return float(fuzz.token_sort_ratio(a, b) / 100.0)


def _exact(a: str, b: str) -> float:
    return 1.0 if a and b and a == b else 0.0


def _prefix(a: str, b: str, length: int = 3) -> float:
    if not a or not b:
        return 0.0
    return 1.0 if a[:length] == b[:length] else 0.0


STRING_COMPARATORS: dict[str, Callable[[str, str], float]] = {
    "exact": _exact,
    "equal": _exact,
    "match": _exact,
    "jaro": _jaro_winkler,
    "jaro_winkler": _jaro_winkler,
    "jw": _jaro_winkler,
    "levenshtein": _levenshtein_ratio,
    "lev": _levenshtein_ratio,
    "edit": _levenshtein_ratio,
    "ratio": _levenshtein_ratio,
    "sim": _levenshtein_ratio,
    "similarity": _levenshtein_ratio,
    "token_sort": _token_sort,
    "tokensort": _token_sort,
    "prefix": _prefix,
}


def _numeric_ratio(a: float, b: float) -> float:
    denominator = max(abs(a), abs(b))
    if denominator == 0:
        return 1.0
    return float(min(abs(a), abs(b)) / denominator)


def _abs_diff(a: float, b: float) -> float:
    return float(abs(a - b))


def _rel_diff(a: float, b: float) -> float:
    denominator = max(abs(a), abs(b))
    if denominator == 0:
        return 0.0
    return float(abs(a - b) / denominator)


def _log_ratio(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return float(abs(math.log(a / b)))


NUMERIC_COMPARATORS: dict[str, Callable[[float, float], float]] = {
    "ratio": _numeric_ratio,
    "area_ratio": _numeric_ratio,
    "diff": _abs_diff,
    "abs_diff": _abs_diff,
    "absdiff": _abs_diff,
    "rel_diff": _rel_diff,
    "relative_diff": _rel_diff,
    "pct_diff": _rel_diff,
    "log_ratio": _log_ratio,
}

PRESENCE_COMPARATORS = {
    "both_present": lambda a, b: 1.0 if a and b else 0.0,
    "any_missing": lambda a, b: 0.0 if a and b else 1.0,
    "missing": lambda a, b: 0.0 if a and b else 1.0,
}


class FeatureSchema:
    """Parsed representation of a saved feature schema."""

    def __init__(self, model_key: str, payload: dict[str, Any]):
        self.model_key = model_key
        self.raw = payload
        self.version = str(payload.get("version", payload.get("schema_version", "1")))
        self.features = self._parse_features(payload)
        self.field_map: dict[str, str] = payload.get("field_map", {}) or {}

    @classmethod
    def load(cls, model_key: str, path: Path) -> "FeatureSchema":
        try:
            payload = json.loads(Path(path).read_text())
        except json.JSONDecodeError as exc:
            raise ModelCompatibilityError(
                model_key, f"{Path(path).name} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            payload = {"features": payload}
        return cls(model_key, payload)

    def _parse_features(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        raw_features = (
            payload.get("features")
            or payload.get("feature_names")
            or payload.get("columns")
        )
        if not raw_features:
            raise ModelCompatibilityError(
                self.model_key,
                "feature schema contains no feature list",
                required=["features / feature_names / columns"],
            )

        parsed: list[dict[str, Any]] = []
        for entry in raw_features:
            if isinstance(entry, dict):
                name = entry.get("name")
                if not name:
                    raise ModelCompatibilityError(
                        self.model_key, f"feature entry without a name: {entry!r}"
                    )
                parsed.append(
                    {
                        "name": name,
                        "field": entry.get("field") or entry.get("source_field"),
                        "comparator": (
                            entry.get("comparator")
                            or entry.get("type")
                            or entry.get("op")
                        ),
                        "kind": entry.get("kind"),
                    }
                )
            elif isinstance(entry, str):
                parsed.append(self._parse_flat_name(entry))
            else:
                raise ModelCompatibilityError(
                    self.model_key, f"unsupported feature entry: {entry!r}"
                )
        return parsed

    def _parse_flat_name(self, name: str) -> dict[str, Any]:
        """Interpret ``<field>_<comparator>`` style feature names."""

        lowered = name.lower()
        for comparator in sorted(
            set(STRING_COMPARATORS) | set(NUMERIC_COMPARATORS) | set(PRESENCE_COMPARATORS),
            key=len,
            reverse=True,
        ):
            suffix = f"_{comparator}"
            if lowered.endswith(suffix):
                return {
                    "name": name,
                    "field": name[: -len(suffix)],
                    "comparator": comparator,
                    "kind": None,
                }
        raise ModelCompatibilityError(
            self.model_key,
            f"feature '{name}' does not declare a comparator and its name cannot "
            "be interpreted unambiguously",
            required=[
                "a feature schema entry of the form "
                '{"name": ..., "field": ..., "comparator": ...}'
            ],
        )

    @property
    def names(self) -> list[str]:
        return [feature["name"] for feature in self.features]

    def source_field(self, feature: dict[str, Any]) -> str:
        field = feature["field"]
        return self.field_map.get(field, field)


def compute_pair_features(
    schema: FeatureSchema,
    record_a: dict[str, Any],
    record_b: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the schema's feature vector for a pair of records."""

    values: list[float] = []
    detail: dict[str, Any] = {}

    for feature in schema.features:
        field = schema.source_field(feature)
        comparator = (feature["comparator"] or "").lower()
        raw_a = record_a.get(field)
        raw_b = record_b.get(field)

        if comparator in PRESENCE_COMPARATORS:
            value = PRESENCE_COMPARATORS[comparator](
                normalise_text(raw_a), normalise_text(raw_b)
            )
        elif comparator in NUMERIC_COMPARATORS and (
            feature.get("kind") != "string"
        ) and (to_number(raw_a) is not None or to_number(raw_b) is not None):
            number_a = to_number(raw_a)
            number_b = to_number(raw_b)
            if number_a is None or number_b is None:
                value = 0.0
            else:
                value = NUMERIC_COMPARATORS[comparator](number_a, number_b)
        elif comparator in STRING_COMPARATORS:
            value = STRING_COMPARATORS[comparator](
                normalise_text(raw_a), normalise_text(raw_b)
            )
        else:
            raise ModelCompatibilityError(
                schema.model_key,
                f"unsupported comparator '{comparator}' for feature "
                f"'{feature['name']}'",
                required=[
                    "supported comparators: "
                    + ", ".join(
                        sorted(
                            set(STRING_COMPARATORS)
                            | set(NUMERIC_COMPARATORS)
                            | set(PRESENCE_COMPARATORS)
                        )
                    )
                ],
            )

        values.append(float(value))
        detail[feature["name"]] = {
            "field": field,
            "comparator": comparator,
            "value": float(value),
            "a": raw_a,
            "b": raw_b,
            "a_missing": raw_a in (None, ""),
            "b_missing": raw_b in (None, ""),
        }

    return np.asarray([values], dtype=float), detail


def compute_record_features(
    schema: FeatureSchema, record: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute single-record features (Model 5) from the saved schema."""

    values: list[float] = []
    detail: dict[str, Any] = {}

    for feature in schema.features:
        field = schema.source_field(feature)
        comparator = (feature["comparator"] or "value").lower()
        raw = record.get(field)

        if comparator in ("missing", "is_missing", "any_missing"):
            value = 1.0 if raw in (None, "") else 0.0
        elif comparator in ("present", "both_present", "is_present"):
            value = 0.0 if raw in (None, "") else 1.0
        elif comparator in ("value", "numeric", "number"):
            number = to_number(raw)
            if number is None:
                raise ModelCompatibilityError(
                    schema.model_key,
                    f"feature '{feature['name']}' requires a numeric value for "
                    f"field '{field}' but received {raw!r}",
                )
            value = number
        elif comparator in ("length", "len"):
            value = float(len(normalise_text(raw)))
        elif comparator in ("log", "log1p"):
            number = to_number(raw)
            value = float(math.log1p(number)) if number and number > 0 else 0.0
        else:
            raise ModelCompatibilityError(
                schema.model_key,
                f"unsupported single-record comparator '{comparator}' for feature "
                f"'{feature['name']}'",
                required=["missing, present, value, length or log"],
            )

        values.append(float(value))
        detail[feature["name"]] = {
            "field": field,
            "comparator": comparator,
            "value": float(value),
            "raw": raw,
        }

    return np.asarray([values], dtype=float), detail
