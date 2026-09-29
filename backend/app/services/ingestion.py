"""Source adapters and canonical ingestion pipeline for vector geospatial data."""

from __future__ import annotations

import datetime as dt
import json
import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import geopandas as gpd

from app.core.errors import GeometryError, InvalidInputError
from app.services.geospatial import to_geojson, to_geometry, transform_geometry, validate_geometry


_CANONICAL_FIELD_ALIASES = {
    "internal_id": "internal_id",
    "source_id": "source_id",
    "source_name": "source_name",
    "village": "village",
    "village_code": "village_code",
    "district": "district",
    "district_code": "district_code",
    "tehsil": "tehsil",
    "tehsil_code": "tehsil_code",
    "block": "block",
    "pargana": "pargana",
    "khasra": "khasra",
    "parcel_type": "parcel_type",
    "area": "area",
    "status": "development_status",
    "development_status": "development_status",
    "remarks": "remarks",
    "source_timestamp": "source_timestamp",
    "timestamp": "source_timestamp",
    "observed_at": "source_timestamp",
    "confidence": "confidence",
    "quality_flags": "quality_flags",
    "source": "source_name",
    "parcel_id": "source_id",
}


def _normalise_key(key: str) -> str:
    cleaned = str(key).strip().lower().replace(" ", "_")
    return cleaned.replace("-", "_")


def _normalise_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalise_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_value(v) for v in value]
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _normalise_properties(properties: dict[str, Any] | None) -> dict[str, Any]:
    if not properties:
        return {}
    out: dict[str, Any] = {}
    for key, value in properties.items():
        out[_normalise_key(key)] = _normalise_value(value)
    return out


def _coerce_datetime(value: Any) -> dt.datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time.min)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _geometry_coords(geometry: Any) -> list[tuple[float, float]]:
    try:
        geom = to_geometry(geometry)
    except GeometryError:
        return []
    if geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [(float(geom.x), float(geom.y))]
    if geom.geom_type in {"LineString", "LinearRing"}:
        return [tuple(float(v) for v in point) for point in geom.coords]
    if geom.geom_type == "Polygon":
        return [tuple(float(v) for v in point) for point in geom.exterior.coords]
    if geom.geom_type == "MultiPoint":
        points: list[tuple[float, float]] = []
        for part in geom.geoms:
            points.extend(_geometry_coords(part))
        return points
    if geom.geom_type in {"MultiPolygon", "GeometryCollection"}:
        pieces: list[tuple[float, float]] = []
        for part in getattr(geom, "geoms", []):
            pieces.extend(_geometry_coords(part))
        return pieces
    return []


def _infer_crs_from_geometry(geometry: Any) -> str | None:
    coords = _geometry_coords(geometry)
    if not coords:
        return None
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    if all(-180.0 <= x <= 180.0 and -90.0 <= y <= 90.0 for x, y in zip(xs, ys)):
        return "EPSG:4326"
    return None


def _explicit_crs_from_geojson(document: dict[str, Any] | None) -> str | None:
    if not document:
        return None
    crs = document.get("crs")
    if not crs:
        return None
    if isinstance(crs, str):
        return crs
    if isinstance(crs, dict):
        props = crs.get("properties") or {}
        name = props.get("name") or crs.get("name")
        if name:
            return str(name)
    return None


def _canonical_value(raw: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw:
            return raw.get(key)
    for key in keys:
        alias = _CANONICAL_FIELD_ALIASES.get(key)
        if alias and alias in raw:
            return raw.get(alias)
    return None


def _to_canonical_record(
    *,
    source_name: str,
    source_type: str,
    source_id: str,
    geometry: Any,
    detected_crs: str | None,
    project_crs: str | None,
    properties: dict[str, Any],
    warnings: list[str],
    provenance: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    raw = _normalise_properties(properties)
    geom_obj = to_geometry(geometry)
    final_crs = detected_crs if project_crs is None else project_crs
    if detected_crs and project_crs and detected_crs != project_crs:
        try:
            geom_obj = transform_geometry(geom_obj, detected_crs, project_crs)
            final_crs = project_crs
            provenance["transformation"] = [
                *(provenance.get("transformation", [])),
                {"type": "crs_transform", "from": detected_crs, "to": project_crs},
            ]
        except Exception as exc:  # pragma: no cover - defensive path
            warnings.append(f"unable to transform CRS from {detected_crs} to {project_crs}: {exc}")

    if detected_crs is None and project_crs:
        warnings.append("CRS is missing in source metadata; project CRS was requested but source CRS could not be determined.")

    source_timestamp = _coerce_datetime(_canonical_value(raw, "source_timestamp", "timestamp", "observed_at"))

    record = {
        "internal_id": str(uuid.uuid4()),
        "source_id": source_id,
        "source_name": source_name,
        "source_type": source_type,
        "geometry": to_geojson(geom_obj),
        "geometry_type": geom_obj.geom_type,
        "crs": final_crs,
        "village": _canonical_value(raw, "village") or None,
        "village_code": _canonical_value(raw, "village_code") or None,
        "district": _canonical_value(raw, "district") or None,
        "district_code": _canonical_value(raw, "district_code") or None,
        "tehsil": _canonical_value(raw, "tehsil") or None,
        "tehsil_code": _canonical_value(raw, "tehsil_code") or None,
        "block": _canonical_value(raw, "block") or None,
        "pargana": _canonical_value(raw, "pargana") or None,
        "khasra": _canonical_value(raw, "khasra") or None,
        "parcel_type": _canonical_value(raw, "parcel_type") or None,
        "area": _canonical_value(raw, "area") or None,
        "development_status": _canonical_value(raw, "status", "development_status") or None,
        "remarks": _canonical_value(raw, "remarks") or None,
        "source_timestamp": source_timestamp.isoformat() if source_timestamp else None,
        "ingestion_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "confidence": _canonical_value(raw, "confidence"),
        "quality_flags": [],
        "provenance": provenance,
        "raw_attributes": raw,
        "source_fields": sorted(raw.keys()),
    }

    missing_fields = [
        field for field in ["village", "district", "tehsil", "khasra", "parcel_type"]
        if record.get(field) is None
    ]
    if missing_fields:
        record["quality_flags"].append("missing_optional_fields")
        warnings.extend(f"missing optional field: {field}" for field in missing_fields)

    record["quality_flags"] = list(dict.fromkeys(record["quality_flags"]))
    return record, warnings


class SourceAdapter(ABC):
    """Base contract for source-specific ingest adapters."""

    source_type = "unknown"

    def __init__(self, *, source_name: str, source_type: str | None = None, project_crs: str | None = None) -> None:
        self.source_name = source_name
        self.source_type = source_type or self.source_type
        self.project_crs = project_crs

    @abstractmethod
    def ingest(self, source: Any) -> dict[str, Any]:
        """Return canonicalized records plus ingestion statistics."""


class GeoJSONAdapter(SourceAdapter):
    source_type = "geojson"

    def ingest(self, source: Any) -> dict[str, Any]:
        payload = source
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise InvalidInputError(f"source file not found: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise InvalidInputError("GeoJSON payload must be a mapping")
        return _ingest_geojson_payload(payload, source_name=self.source_name, source_type=self.source_type, project_crs=self.project_crs)


class GeoPackageAdapter(SourceAdapter):
    source_type = "geopackage"

    def ingest(self, source: Any) -> dict[str, Any]:
        path = Path(source)
        if not path.exists():
            raise InvalidInputError(f"source file not found: {path}")
        frame = gpd.read_file(path)
        return _ingest_dataframe(frame, source_name=self.source_name, source_type=self.source_type, project_crs=self.project_crs, source_id=path.stem)


class ShapefileAdapter(SourceAdapter):
    source_type = "shapefile"

    def ingest(self, source: Any) -> dict[str, Any]:
        path = Path(source)
        if not path.exists():
            raise InvalidInputError(f"source file not found: {path}")
        frame = gpd.read_file(path)
        return _ingest_dataframe(frame, source_name=self.source_name, source_type=self.source_type, project_crs=self.project_crs, source_id=path.stem)


class ParquetAdapter(SourceAdapter):
    source_type = "parquet"

    def ingest(self, source: Any) -> dict[str, Any]:
        path = Path(source)
        if not path.exists():
            raise InvalidInputError(f"source file not found: {path}")
        frame = gpd.read_parquet(path)
        return _ingest_dataframe(frame, source_name=self.source_name, source_type=self.source_type, project_crs=self.project_crs, source_id=path.stem)


def _ingest_geojson_payload(
    payload: dict[str, Any],
    *,
    source_name: str,
    source_type: str,
    project_crs: str | None,
) -> dict[str, Any]:
    document_crs = _explicit_crs_from_geojson(payload)
    features = payload.get("features")
    if not isinstance(features, list):
        raise InvalidInputError("GeoJSON document must contain a 'features' list")

    total_records = 0
    valid_records = 0
    invalid_records = 0
    geometry_types: set[str] = set()
    detected_fields: set[str] = set()
    warnings: list[str] = []
    errors: list[str] = []
    records: list[dict[str, Any]] = []

    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            invalid_records += 1
            errors.append(f"feature {index} is not an object")
            continue
        total_records += 1
        source_id = str(feature.get("id") or feature.get("properties", {}).get("source_id") or feature.get("properties", {}).get("parcel_id") or f"{source_name}-{index}")
        properties = feature.get("properties") or {}
        detected_fields.update(_normalise_properties(properties).keys())
        geometry = feature.get("geometry")
        if geometry is None:
            invalid_records += 1
            errors.append(f"record '{source_id}' has no geometry")
            continue
        try:
            geom_obj = to_geometry(geometry)
        except GeometryError as exc:
            invalid_records += 1
            errors.append(f"record '{source_id}' failed geometry parsing: {exc}")
            continue
        validation = validate_geometry(geom_obj)
        if geom_obj.is_empty:
            invalid_records += 1
            errors.append(f"record '{source_id}' has an empty geometry")
            continue
        if not validation["is_valid"]:
            invalid_records += 1
            errors.append(f"record '{source_id}' has invalid geometry: {validation['reason']}")
            continue

        detected_crs = document_crs or _infer_crs_from_geometry(geom_obj)
        geometry_types.add(geom_obj.geom_type)
        provenance = {
            "source": source_name,
            "source_record_id": source_id,
            "ingestion_operation": "geojson_ingest",
            "transformation": ["attribute_normalization", "geometry_validation"],
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_type": source_type,
            "detected_crs": detected_crs,
            "project_crs": project_crs,
        }
        record, record_warnings = _to_canonical_record(
            source_name=source_name,
            source_type=source_type,
            source_id=source_id,
            geometry=geom_obj,
            detected_crs=detected_crs,
            project_crs=project_crs,
            properties=properties,
            warnings=warnings,
            provenance=provenance,
        )
        if record_warnings:
            warnings.extend(record_warnings)
        records.append(record)
        valid_records += 1

    stats = {
        "source_id": str(uuid.uuid4()),
        "source_name": source_name,
        "source_type": source_type,
        "total_records": total_records,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "detected_crs": document_crs or (features[0].get("geometry") and _infer_crs_from_geometry(features[0].get("geometry")) if features else None),
        "geometry_types": sorted(geometry_types),
        "detected_fields": sorted(detected_fields),
        "missing_expected_fields": [
            field for field in ["village", "district", "tehsil", "khasra"] if field not in detected_fields
        ],
        "warnings": list(dict.fromkeys(warnings)),
        "errors": errors,
        "provenance": {
            "source_name": source_name,
            "ingestion_operation": "geojson_ingest",
            "source_type": source_type,
            "ingested_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "project_crs": project_crs,
        },
        "records": records,
    }
    return stats


def _ingest_dataframe(
    frame: gpd.GeoDataFrame,
    *,
    source_name: str,
    source_type: str,
    project_crs: str | None,
    source_id: str,
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    geometry_types: set[str] = set()
    detected_fields: set[str] = set()
    detected_crs = frame.crs.to_string() if frame.crs is not None else None

    for index, row in frame.iterrows():
        row_dict = row.to_dict()
        properties = {k: v for k, v in row_dict.items() if k != "geometry"}
        key = str(properties.get("source_id") or properties.get("parcel_id") or f"{source_name}-{index}")
        detected_fields.update(_normalise_properties(properties).keys())
        if row_dict.get("geometry") is None:
            errors.append(f"record '{key}' has no geometry")
            continue
        try:
            geom_obj = to_geometry(row_dict["geometry"])
        except GeometryError as exc:
            errors.append(f"record '{key}' failed geometry parsing: {exc}")
            continue
        validation = validate_geometry(geom_obj)
        if geom_obj.is_empty:
            errors.append(f"record '{key}' has an empty geometry")
            continue
        if not validation["is_valid"]:
            errors.append(f"record '{key}' has invalid geometry: {validation['reason']}")
            continue

        geometry_types.add(geom_obj.geom_type)
        provenance = {
            "source": source_name,
            "source_record_id": key,
            "ingestion_operation": "vector_ingest",
            "transformation": ["attribute_normalization", "geometry_validation"],
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_type": source_type,
            "detected_crs": detected_crs,
            "project_crs": project_crs,
        }
        record, record_warnings = _to_canonical_record(
            source_name=source_name,
            source_type=source_type,
            source_id=key,
            geometry=geom_obj,
            detected_crs=detected_crs,
            project_crs=project_crs,
            properties=properties,
            warnings=warnings,
            provenance=provenance,
        )
        if record_warnings:
            warnings.extend(record_warnings)
        records.append(record)

    return {
        "source_id": source_id,
        "source_name": source_name,
        "source_type": source_type,
        "total_records": len(frame),
        "valid_records": len(records),
        "invalid_records": len(frame) - len(records),
        "detected_crs": detected_crs,
        "geometry_types": sorted(geometry_types),
        "detected_fields": sorted(detected_fields),
        "missing_expected_fields": [
            field for field in ["village", "district", "tehsil", "khasra"] if field not in detected_fields
        ],
        "warnings": list(dict.fromkeys(warnings)),
        "errors": errors,
        "provenance": {
            "source_name": source_name,
            "ingestion_operation": "vector_ingest",
            "source_type": source_type,
            "ingested_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "project_crs": project_crs,
        },
        "records": records,
    }


def get_adapter_for_source(
    *,
    source_name: str,
    source_type: str | None = None,
    project_crs: str | None = None,
    source_path: str | Path | None = None,
    payload: dict[str, Any] | None = None,
) -> SourceAdapter:
    if payload is not None:
        return GeoJSONAdapter(source_name=source_name, source_type=source_type, project_crs=project_crs)
    if source_path is None:
        raise InvalidInputError("source_path or payload is required")
    path = Path(source_path)
    suffix = path.suffix.lower()
    if suffix in {".geojson", ".json"}:
        return GeoJSONAdapter(source_name=source_name, source_type=source_type, project_crs=project_crs)
    if suffix == ".gpkg":
        return GeoPackageAdapter(source_name=source_name, source_type=source_type, project_crs=project_crs)
    if suffix == ".shp":
        return ShapefileAdapter(source_name=source_name, source_type=source_type, project_crs=project_crs)
    if suffix in {".parquet", ".pq"}:
        return ParquetAdapter(source_name=source_name, source_type=source_type, project_crs=project_crs)
    raise InvalidInputError(f"unsupported source format: {suffix or path.name}")


def ingest_payload(payload: dict[str, Any], *, source_name: str, source_type: str | None = None, project_crs: str | None = None) -> dict[str, Any]:
    adapter = GeoJSONAdapter(source_name=source_name, source_type=source_type, project_crs=project_crs)
    return adapter.ingest(payload)


def ingest_file(source: str | Path, *, source_name: str | None = None, source_type: str | None = None, project_crs: str | None = None) -> dict[str, Any]:
    path = Path(source)
    if not path.exists():
        raise InvalidInputError(f"source file not found: {path}")
    adapter = get_adapter_for_source(
        source_name=source_name or path.stem,
        source_type=source_type,
        project_crs=project_crs,
        source_path=path,
    )
    return adapter.ingest(path)


def _write_upload_tempfile(filename: str, content: bytes) -> Path:
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".geojson") as handle:
        handle.write(content)
        return Path(handle.name)
