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
import pandas as pd
from rapidfuzz import fuzz

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


MODEL4_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "village": ("village", "vill_name", "vname_e", "vname_h"),
    "district": ("district",),
    "khasra": ("khasra", "sno"),
    "tehsil": ("tehsil", "tahsil"),
    "block": ("block", "block_name", "block_name_1"),
    "pargana": ("pargana",),
    "village_code": ("village_code", "vill_code", "village_co", "ccode11"),
    "district_code": ("district_code", "district_c", "dtcode11"),
    "tehsil_code": ("tehsil_code", "tehsil_cod"),
    "area": ("area", "area_srs", "area_decim"),
    "parcel_type": ("parcel_type", "par_type"),
    "remarks": ("remarks",),
}


def _model4_safe_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _model4_normalize_text(value: Any) -> str:
    text = _model4_safe_string(value)
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[_\-/\\]+", " ", text)
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _model4_normalize_numeric(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return re.sub(r"[^0-9.\-]", "", text)


def _model4_script_type(value: Any) -> str:
    text = str(value)
    if not text.strip():
        return "empty"
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    if has_devanagari and has_latin:
        return "mixed"
    if has_devanagari:
        return "devanagari"
    if has_latin:
        return "latin"
    return "other"


def _model4_record_value(record: dict[str, Any], field: str) -> Any:
    for key in MODEL4_FIELD_ALIASES.get(field, (field,)):
        if key in record:
            return record[key]
    return ""


def compute_model4_features(
    schema: dict[str, Any],
    record_a: dict[str, Any],
    record_b: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reproduce Model 4's notebook feature engineering in saved schema order."""

    feature_names = schema.get("features")
    field_mapping = schema.get("field_mapping")
    if not isinstance(feature_names, list) or not isinstance(field_mapping, dict):
        raise ModelCompatibilityError(
            "entity_resolver",
            "Model 4 schema must contain ordered 'features' and 'field_mapping'",
        )

    generated: dict[str, float] = {}
    detail: dict[str, Any] = {}
    field_values: dict[str, tuple[Any, Any, str, str]] = {}
    numeric_fields: set[str] = set()

    for field in field_mapping:
        raw_a = _model4_record_value(record_a, field)
        raw_b = _model4_record_value(record_b, field)
        if f"{field}__relative_diff" in feature_names:
            numeric_fields.add(field)
        field_values[field] = (
            raw_a,
            raw_b,
            _model4_normalize_text(raw_a),
            _model4_normalize_text(raw_b),
        )

    for field, (raw_a, raw_b, norm_a, norm_b) in field_values.items():
        if field in numeric_fields:
            value_a = _model4_normalize_numeric(raw_a)
            value_b = _model4_normalize_numeric(raw_b)
            generated[f"{field}__exact"] = float(
                bool(value_a) and bool(value_b) and value_a == value_b
            )
            generated[f"{field}__both_empty"] = float(
                not value_a and not value_b
            )
            generated[f"{field}__one_empty"] = float(
                bool(value_a) != bool(value_b)
            )
            try:
                numeric_a = float(value_a)
                numeric_b = float(value_b)
                difference = abs(numeric_a - numeric_b)
                denominator = max(abs(numeric_a), abs(numeric_b), 1e-9)
                relative_difference = difference / denominator
                generated[f"{field}__relative_diff"] = float(
                    min(relative_difference, 10.0)
                )
                generated[f"{field}__close_1pct"] = float(
                    relative_difference <= 0.01
                )
                generated[f"{field}__close_5pct"] = float(
                    relative_difference <= 0.05
                )
                generated[f"{field}__close_10pct"] = float(
                    relative_difference <= 0.10
                )
            except (TypeError, ValueError, ZeroDivisionError):
                generated[f"{field}__relative_diff"] = 1.0
                generated[f"{field}__close_1pct"] = 0.0
                generated[f"{field}__close_5pct"] = 0.0
                generated[f"{field}__close_10pct"] = 0.0
            continue

        generated[f"{field}__both_empty"] = float(not norm_a and not norm_b)
        generated[f"{field}__one_empty"] = float(bool(norm_a) != bool(norm_b))
        generated[f"{field}__exact"] = float(bool(norm_a) and norm_a == norm_b)
        generated[f"{field}__len_a"] = float(min(len(norm_a), 100))
        generated[f"{field}__len_b"] = float(min(len(norm_b), 100))

        if norm_a and norm_b:
            generated[f"{field}__len_ratio"] = (
                min(len(norm_a), len(norm_b)) / max(len(norm_a), len(norm_b))
            )
            generated[f"{field}__ratio"] = fuzz.ratio(norm_a, norm_b) / 100.0
            generated[f"{field}__wratio"] = fuzz.WRatio(norm_a, norm_b) / 100.0
            generated[f"{field}__token_sort"] = (
                fuzz.token_sort_ratio(norm_a, norm_b) / 100.0
            )
            generated[f"{field}__token_set"] = (
                fuzz.token_set_ratio(norm_a, norm_b) / 100.0
            )
            generated[f"{field}__partial"] = (
                fuzz.partial_ratio(norm_a, norm_b) / 100.0
            )

            def ngrams(text: str, size: int = 3) -> set[str]:
                if len(text) < size:
                    return {text}
                return {text[index : index + size] for index in range(len(text) - size + 1)}

            grams_a = ngrams(norm_a)
            grams_b = ngrams(norm_b)
            union_size = len(grams_a | grams_b)
            generated[f"{field}__char3_jaccard"] = (
                len(grams_a & grams_b) / union_size if union_size else 0.0
            )
            tokens_a = set(norm_a.split())
            tokens_b = set(norm_b.split())
            generated[f"{field}__token_overlap"] = (
                len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))
            )
            script_a = _model4_script_type(raw_a)
            script_b = _model4_script_type(raw_b)
            generated[f"{field}__same_script"] = float(script_a == script_b)
            generated[f"{field}__cross_script"] = float(
                {script_a, script_b} == {"latin", "devanagari"}
            )
        else:
            for suffix in (
                "len_ratio",
                "ratio",
                "wratio",
                "token_sort",
                "token_set",
                "partial",
                "char3_jaccard",
                "token_overlap",
                "same_script",
                "cross_script",
            ):
                generated[f"{field}__{suffix}"] = 0.0

    def normalized_field(field: str, side: int) -> str:
        values = field_values.get(field)
        return values[2 + side] if values is not None else ""

    district_match = int(
        bool(normalized_field("district", 0))
        and normalized_field("district", 0) == normalized_field("district", 1)
    )
    khasra_match = int(
        bool(normalized_field("khasra", 0))
        and normalized_field("khasra", 0) == normalized_field("khasra", 1)
    )
    pargana_match = int(
        bool(normalized_field("pargana", 0))
        and normalized_field("pargana", 0) == normalized_field("pargana", 1)
    )
    village_match = int(
        bool(normalized_field("village", 0))
        and normalized_field("village", 0) == normalized_field("village", 1)
    )

    generated["cross__district_khasra_exact"] = float(district_match * khasra_match)
    generated["cross__district_pargana_exact"] = float(district_match * pargana_match)
    generated["cross__district_village_exact"] = float(district_match * village_match)
    generated["cross__strong_cadastral_identity"] = float(
        bool(district_match and (khasra_match or pargana_match or village_match))
    )

    exact_count = 0
    available_count = 0
    for raw_a, raw_b, _, _ in field_values.values():
        normalized_a = _model4_normalize_text(raw_a)
        normalized_b = _model4_normalize_text(raw_b)
        if normalized_a or normalized_b:
            available_count += 1
        if normalized_a and normalized_b and normalized_a == normalized_b:
            exact_count += 1
    generated["cross__exact_field_count"] = float(exact_count)
    generated["cross__available_field_count"] = float(available_count)
    generated["cross__exact_field_ratio"] = float(
        exact_count / max(1, available_count)
    )

    missing = [name for name in feature_names if name not in generated]
    extra = [name for name in generated if name not in feature_names]
    if missing or extra:
        raise ModelCompatibilityError(
            "entity_resolver",
            "notebook feature construction does not exactly match the saved schema",
            required=[
                f"missing features: {missing[:10]}",
                f"unexpected features: {extra[:10]}",
            ],
        )

    values = [float(generated[name]) for name in feature_names]
    for name, value in zip(feature_names, values):
        field = name.split("__", 1)[0]
        raw_a, raw_b, _, _ = field_values.get(field, (None, None, "", ""))
        detail[name] = {
            "field": field,
            "feature": name,
            "value": value,
            "a": raw_a,
            "b": raw_b,
            "a_missing": raw_a in (None, ""),
            "b_missing": raw_b in (None, ""),
        }
    matrix = np.asarray([values], dtype=np.float64)
    if matrix.shape[1] != len(feature_names):
        raise ModelCompatibilityError(
            "entity_resolver",
            f"Model 4 generated {matrix.shape[1]} features; schema declares {len(feature_names)}",
        )
    return matrix, detail


def compute_model5_notebook_features(
    schema: dict[str, Any],
    inference_config: dict[str, Any],
    record: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reproduce Model 5's notebook features for one canonical record."""

    feature_names = schema.get("features")
    if not isinstance(feature_names, list) or not all(
        isinstance(name, str) for name in feature_names
    ):
        raise ModelCompatibilityError(
            "anomaly_detector", "Model 5 feature schema must list ordered string names"
        )

    generated: dict[str, float] = {}
    detail: dict[str, Any] = {}
    fields = tuple(
        field
        for field in (
            "village", "district", "khasra", "pargana", "tehsil", "block",
            "village_code", "district_code", "tehsil_code", "parcel_type",
            "development",
        )
        if f"{field}__present" in feature_names
    )
    values = {field: _model4_normalize_text(_model4_record_value(record, field)) for field in fields}

    for field in fields:
        value = values[field]
        generated[f"{field}__present"] = float(bool(value))
        generated[f"{field}__length"] = float(min(len(value), 100))

    present_count = sum(bool(value) for value in values.values())
    if "identity__field_count" in feature_names:
        generated["identity__field_count"] = float(present_count)
    if "identity__missing_count" in feature_names:
        generated["identity__missing_count"] = float(len(fields) - present_count)

    for field in ("district", "village", "khasra", "pargana"):
        if field not in values:
            continue
        value = values[field]
        generated[f"{field}__value_length"] = float(len(value))
        generated[f"{field}__digit_count"] = float(sum(character.isdigit() for character in value))
        generated[f"{field}__token_count"] = float(len(value.split()))

    khasra = values.get("khasra", "")
    raw_khasra = _model4_safe_string(_model4_record_value(record, "khasra"))
    if "khasra__numeric" in feature_names:
        generated["khasra__numeric"] = float(
            bool(khasra) and khasra.replace(".", "", 1).isdigit()
        )
    if "khasra__has_slash" in feature_names:
        generated["khasra__has_slash"] = float("/" in raw_khasra)
    if "khasra__has_dash" in feature_names:
        generated["khasra__has_dash"] = float("-" in raw_khasra)
    if "khasra__has_alpha" in feature_names:
        generated["khasra__has_alpha"] = float(any(character.isalpha() for character in khasra))

    for field in ("district", "village", "pargana"):
        key = f"cross__{field}_present"
        if key in feature_names:
            generated[key] = float(bool(values.get(field, "")))
    if "cross__strong_identity" in feature_names:
        generated["cross__strong_identity"] = float(
            all(bool(values.get(field, "")) for field in ("district", "village", "khasra"))
        )
    if "cross__admin_identity_count" in feature_names:
        generated["cross__admin_identity_count"] = float(
            sum(bool(values.get(field, "")) for field in ("district", "village", "pargana"))
        )

    if "source__row_group" in feature_names:
        try:
            generated["source__row_group"] = float(record.get("source_row_group", 0.0) or 0.0)
        except (TypeError, ValueError):
            generated["source__row_group"] = 0.0

    frequency_fields = inference_config.get("frequency_fields", {})
    for field in ("district", "village", "khasra", "pargana"):
        frequency_key = f"{field}__frequency"
        rarity_key = f"{field}__rarity"
        if frequency_key not in feature_names and rarity_key not in feature_names:
            continue
        frequencies = frequency_fields.get(field)
        if not isinstance(frequencies, dict):
            raise ModelCompatibilityError(
                "anomaly_detector",
                f"Model 5 inference config is missing training frequencies for '{field}'",
            )
        frequency = float(frequencies.get(values.get(field, ""), 0.0))
        generated[frequency_key] = frequency
        generated[rarity_key] = 1.0 / (1.0 + frequency)

    missing = [name for name in feature_names if name not in generated]
    extra = [name for name in generated if name not in feature_names]
    if missing or extra:
        raise ModelCompatibilityError(
            "anomaly_detector",
            "Model 5 notebook feature construction does not match its saved schema",
            required=[f"missing features: {missing[:10]}", f"unexpected features: {extra[:10]}"],
        )

    ordered_values = [float(generated[name]) for name in feature_names]
    matrix = np.asarray([ordered_values], dtype=np.float64)
    for name, value in zip(feature_names, ordered_values):
        detail[name] = {"field": name.split("__", 1)[0], "feature": name, "value": value}
    if matrix.shape != (1, len(feature_names)):
        raise ModelCompatibilityError(
            "anomaly_detector",
            f"Model 5 feature builder returned shape {matrix.shape}",
        )
    return matrix, detail


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
