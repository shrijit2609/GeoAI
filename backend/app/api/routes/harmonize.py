"""Deterministic harmonization jobs over registered sources."""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.geospatial import to_geometry, transform_geometry
from app.services.source_catalog import source_catalog

router = APIRouter(prefix="/harmonize", tags=["harmonization"])
IDENTITY_FIELDS = ("ulpin", "district", "village", "tehsil", "khasra")
HARMONIZED_FIELDS = (
    "ulpin", "district", "district_code", "village", "village_code", "tehsil",
    "tehsil_code", "block", "pargana", "khasra", "parcel_type", "area",
    "development_status", "remarks",
)


class HarmonizeRequest(BaseModel):
    source_ids: list[str] = Field(min_length=1, max_length=20)


def _normalise(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold() if value is not None else ""


def _identity(record: dict[str, Any]) -> tuple[str, ...] | None:
    attributes = record.get("raw_attributes") or {}
    values = []
    for field in IDENTITY_FIELDS:
        value = record.get(field)
        if value is None:
            value = attributes.get(field)
        values.append(_normalise(value))
    if values[0]:
        return ("ulpin", values[0])
    composite = values[1:]
    if all(composite):
        return ("cadastral", *composite)
    return None


@router.post("")
def create_harmonization(request: HarmonizeRequest) -> dict[str, Any]:
    sources = []
    for source_id in request.source_ids:
        source = source_catalog.get_source(source_id)
        if source is None:
            raise HTTPException(status_code=404, detail=f"source '{source_id}' is not registered in this process")
        sources.append(source)

    grouped: dict[tuple[str, ...], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for source in sources:
        for record in source.get("records", []):
            identity = _identity(record)
            identity = identity or ("source-record", source["source_id"], str(record.get("source_id")))
            grouped.setdefault(identity, []).append((source, record))

    features = []
    conflicts = []
    for identity, rows in grouped.items():
        first_source, first_record = rows[0]
        attributes = []
        properties = {}
        for field in HARMONIZED_FIELDS:
            observed = []
            for source, record in rows:
                raw = record.get("raw_attributes") or {}
                value = record.get(field)
                if value is None:
                    value = raw.get(field)
                if value is not None and not any(_normalise(value) == _normalise(item["value"]) for item in observed):
                    observed.append({"source_id": source["source_id"], "source_name": source.get("source_name"), "value": value})
            if len(observed) > 1:
                properties[field] = None
                conflicts.append({
                    "identity": list(identity),
                    "field": field,
                    "values": observed,
                    "status": "unresolved",
                    "selected_value": None,
                    "confidence": None,
                    "rule": "no automatic source precedence",
                })
            else:
                properties[field] = observed[0]["value"] if observed else None
                if observed:
                    attributes.append({"field": field, **observed[0]})

        properties.update({
            "harmonized_id": "|".join(identity),
            "matching_method": "exact_normalized_identifier",
            "source_count": len({source["source_id"] for source, _ in rows}),
            "source_records": [
                {"source_id": source["source_id"], "source_name": source.get("source_name"), "record_id": record.get("source_id")}
                for source, record in rows
            ],
            "provenance": [record.get("provenance") for _, record in rows],
        })
        geometry = first_record.get("geometry")
        geometry_crs = first_record.get("crs")
        if geometry is not None:
            if not geometry_crs:
                geometry = None
            elif geometry_crs.upper() not in {"EPSG:4326", "OGC:CRS84"}:
                geometry = transform_geometry(
                    to_geometry(geometry), geometry_crs, "EPSG:4326"
                ).__geo_interface__
        features.append({
            "type": "Feature",
            "id": properties["harmonized_id"],
            "geometry": geometry,
            "properties": properties,
        })

    output_crs = "EPSG:4326" if sources and all(
        not record.get("geometry") or record.get("crs")
        for source in sources for record in source.get("records", [])
    ) else None
    job = source_catalog.save_job({
        "status": "completed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_ids": request.source_ids,
        "matching_method": "exact_normalized_identifier",
        "model_inference_used": False,
        "feature_count": len(features),
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "feature_collection": {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {"crs": output_crs, "confidence": None},
        },
        "limitations": [
            "Records are linked only by exact normalized ULPIN or complete district/village/tehsil/khasra identifiers.",
            "Conflicting fields are left unresolved; no authoritative source precedence is assumed.",
            "No unavailable ML model prediction is substituted for identifier matching.",
        ],
    })
    return {key: value for key, value in job.items() if key != "feature_collection"} | {
        "feature_collection": job["feature_collection"]
    }


@router.get("/{job_id}")
def get_harmonization(job_id: str) -> dict[str, Any]:
    job = source_catalog.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="harmonization job not found in this backend process")
    return job