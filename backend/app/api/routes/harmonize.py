"""Deterministic harmonization jobs over registered sources."""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.geospatial import to_geometry, transform_geometry
from app.services.source_catalog import source_catalog
from app.ml.common import ModelStatus
from app.ml.model_registry import get_registry

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
    flat_records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for source in sources:
        for record in source.get("records", []):
            flat_records.append((source, record))
            identity = _identity(record)
            identity = identity or ("source-record", source["source_id"], str(record.get("source_id")))
            grouped.setdefault(identity, []).append((source, record))

    registry = get_registry()
    health = {key: registry.health(key, probe=True) for key in registry.keys()}
    stage_names = (
        "validation", "crs", "geometry", "candidate_matching", "parcel_matching",
        "entity_resolution", "anomaly_detection", "building_extraction",
        "change_detection", "provenance", "export",
    )
    stages = {name: {"status": "COMPLETED", "reason": None} for name in stage_names}
    model_evidence: dict[tuple[str, str], list[dict[str, Any]]] = {}
    candidates: list[dict[str, Any]] = []
    candidates_examined = 0
    candidate_limit = 250
    m1_ready = health["parcel_matcher"].status is ModelStatus.READY
    m4_ready = health["entity_resolver"].status is ModelStatus.READY
    stages["parcel_matching"] = {
        "status": "COMPLETED" if m1_ready else "SKIPPED",
        "reason": None if m1_ready else health["parcel_matcher"].readiness_reason,
    }
    stages["entity_resolution"] = {
        "status": "COMPLETED" if m4_ready else "SKIPPED",
        "reason": None if m4_ready else health["entity_resolver"].readiness_reason,
    }
    for stage_name, model_key in (("building_extraction", "building_extractor"), ("change_detection", "change_detector")):
        stages[stage_name] = {
            "status": "SKIPPED",
            "reason": f"No image inputs were attached to the selected source records. {health[model_key].readiness_reason}",
        }
    stages["anomaly_detection"] = {
        "status": "COMPLETED" if health["anomaly_detector"].status is ModelStatus.READY else "SKIPPED",
        "reason": None if health["anomaly_detector"].status is ModelStatus.READY else health["anomaly_detector"].readiness_reason,
    }

    def record_for_model(record: dict[str, Any]) -> dict[str, Any]:
        payload = dict(record.get("raw_attributes") or {})
        payload.update({key: value for key, value in record.items() if key not in {"geometry", "raw_attributes", "provenance"}})
        return payload

    geometry_errors = 0

    def candidate_geometry(row: tuple[dict[str, Any], dict[str, Any]]):
        nonlocal geometry_errors
        source, record = row
        geometry = record.get("geometry")
        crs = record.get("crs") or source.get("detected_crs")
        if geometry is None or crs is None:
            return None
        try:
            parsed = to_geometry(geometry)
            if str(crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
                parsed = transform_geometry(parsed, crs, "EPSG:4326")
            return parsed
        except Exception:  # Invalid source geometry is withheld from candidates, not fatal to the job.
            geometry_errors += 1
            return None

    union_parent = list(range(len(flat_records)))

    def find(index: int) -> int:
        while union_parent[index] != index:
            union_parent[index] = union_parent[union_parent[index]]
            index = union_parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            union_parent[root_right] = root_left

    record_indices: dict[tuple[str, str], int] = {}
    for index, (source, record) in enumerate(flat_records):
        record_indices[(source["source_id"], str(record.get("source_id")))] = index

    exact_group_first: dict[tuple[str, ...], int] = {}
    for identity, rows in grouped.items():
        for source, record in rows:
            index = record_indices[(source["source_id"], str(record.get("source_id")))]
            if identity in exact_group_first:
                union(exact_group_first[identity], index)
            else:
                exact_group_first[identity] = index

    if len(flat_records) > 1 and (m1_ready or m4_ready):
        geometries = [candidate_geometry(row) for row in flat_records]
        by_index: dict[int, tuple[str, str]] = {}
        for identity, rows in grouped.items():
            for source, record in rows:
                index = record_indices.get((source["source_id"], str(record.get("source_id"))))
                if index is not None:
                    by_index[index] = identity
        pair_checks = 0
        pair_check_limit = 100_000
        for left_index, left_row in enumerate(flat_records):
            left_geometry = geometries[left_index]
            if left_geometry is None:
                continue
            for right_index in range(left_index + 1, len(flat_records)):
                if candidates_examined >= candidate_limit or pair_checks >= pair_check_limit:
                    break
                pair_checks += 1
                right_row = flat_records[right_index]
                if left_row[0]["source_id"] == right_row[0]["source_id"]:
                    continue
                right_geometry = geometries[right_index]
                if right_geometry is None or not left_geometry.intersects(right_geometry):
                    continue
                candidates_examined += 1
                item = {"source_a": left_row[0]["source_id"], "record_a": left_row[1].get("source_id"), "source_b": right_row[0]["source_id"], "record_b": right_row[1].get("source_id"), "model_evidence": {}}
                try:
                    if m1_ready:
                        result = registry.get("parcel_matcher").predict(left_geometry, right_geometry)
                        item["model_evidence"]["parcel_matcher"] = result.model_dump(mode="json")
                    if m4_ready:
                        result = registry.get("entity_resolver").resolve(record_for_model(left_row[1]), record_for_model(right_row[1]))
                        item["model_evidence"]["entity_resolver"] = result.model_dump(mode="json")
                    outcomes = [evidence.get("decision") for evidence in item["model_evidence"].values() if evidence.get("decision") is not None]
                    if outcomes and all(outcome == "match" for outcome in outcomes):
                        union(left_index, right_index)
                        item["status"] = "linked"
                    else:
                        item["status"] = "review"
                except Exception as exc:  # model evidence must never abort ingestion/harmonization
                    item["status"] = "skipped"
                    item["reason"] = f"Candidate inference failed: {exc}"
                candidates.append(item)
        if geometry_errors:
            stages["geometry"]["reason"] = f"{geometry_errors} record geometry/CRS value(s) could not be normalized for candidate generation; those records were preserved and not compared."
        if candidates_examined >= candidate_limit or pair_checks >= pair_check_limit:
            stages["candidate_matching"]["reason"] = f"Candidate evaluation was bounded at {candidate_limit} intersections / {pair_check_limit} pair checks."

        merged: dict[tuple[str, ...], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        for index, row in enumerate(flat_records):
            identity = by_index.get(index, ("source-record", row[0]["source_id"], str(row[1].get("source_id"))))
            merged.setdefault(("component", str(find(index))), []).append(row)
        grouped = merged
    else:
        stages["candidate_matching"] = {
            "status": "SKIPPED",
            "reason": "No READY parcel/entity model or no cross-source records with comparable geometry were available.",
        }

    features = []
    conflicts = []
    anomalies = []
    processing_time = dt.datetime.now(dt.timezone.utc).isoformat()
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
                identity_conflict = field == "ulpin"
                conflict = {
                    "record": "|".join(identity),
                    "identity": list(identity),
                    "field": field,
                    "source_values": observed,
                    "values": observed,
                    "conflict_type": "identity_conflict" if identity_conflict else "attribute_conflict",
                    "severity": "critical" if identity_conflict else "high" if field in {"district", "village", "khasra"} else "medium",
                    "status": "Requires review",
                    "resolution_status": "unresolved",
                    "selected_value": None,
                    "confidence": None,
                    "rule": "no automatic source precedence",
                    "explanation": "Sources provide different non-empty values; no authoritative precedence was configured.",
                }
                conflicts.append(conflict)
            elif not observed and field in {"district", "village", "khasra"}:
                properties[field] = None
                for source, record in rows:
                    conflicts.append({
                        "record": "|".join(identity),
                        "identity": list(identity),
                        "field": field,
                        "source_values": [{"source_id": source["source_id"], "source_name": source.get("source_name"), "value": None}],
                        "values": [],
                        "conflict_type": "missing_attribute",
                        "severity": "low",
                        "status": "Requires review",
                        "resolution_status": "unresolved",
                        "selected_value": None,
                        "confidence": None,
                        "explanation": f"A critical cadastral field ({field}) is missing from this source record.",
                    })
            else:
                properties[field] = observed[0]["value"] if observed else None
                if observed:
                    attributes.append({"field": field, **observed[0]})

        source_records = [
            {"source_id": source["source_id"], "source_name": source.get("source_name"), "source_type": source.get("source_type"), "record_id": record.get("source_id"), "source_timestamp": source.get("created_at") or source.get("uploaded_at")}
            for source, record in rows
        ]
        row_indices = [record_indices[(source["source_id"], str(record.get("source_id")))] for source, record in rows]
        evidence_for_group = [candidate for candidate in candidates if any(record_indices.get((candidate.get("source_a"), str(candidate.get("record_a")))) == index for index in row_indices) or any(record_indices.get((candidate.get("source_b"), str(candidate.get("record_b")))) == index for index in row_indices)]
        linked_evidence = [candidate for candidate in evidence_for_group if candidate.get("status") == "linked"]
        serialized_evidence = [candidate.get("model_evidence", {}) for candidate in evidence_for_group]
        model4_confidences = [
            result.get("confidence")
            for evidence in serialized_evidence
            for result in [evidence.get("entity_resolver", {})]
            if isinstance(result.get("confidence"), (int, float)) and result.get("decision") == "match"
        ]
        resolution_confidence = min(model4_confidences) if model4_confidences else None
        resolution_reason = (
            "Model 1 parcel correspondence and Model 4 entity resolution linked spatially intersecting source records."
            if linked_evidence and len(rows) > 1
            else "Exact normalized identifier match."
            if len(rows) > 1 and _identity(first_record)
            else "Single source record; no cross-source link was established."
        )
        properties.update({
            "harmonized_id": "|".join(identity),
            "matching_method": "model_and_spatial_candidate" if linked_evidence and len(rows) > 1 else "exact_normalized_identifier",
            "source_count": len({source["source_id"] for source, _ in rows}),
            "source_records": source_records,
            "provenance": [record.get("provenance") for _, record in rows],
            "source_id": [source["source_id"] for source, _ in rows],
            "source_name": [source.get("source_name") for source, _ in rows],
            "source_type": [source.get("source_type") for source, _ in rows],
            "source_timestamp": [source.get("created_at") or source.get("uploaded_at") for source, _ in rows],
            "source_fields": [sorted((record.get("raw_attributes") or {}).keys()) for _, record in rows],
            "original_attributes": [record.get("raw_attributes") or {} for _, record in rows],
            "model_evidence": serialized_evidence,
            "confidence": resolution_confidence,
            "conflicts": [item for item in conflicts if item.get("record") == "|".join(identity)],
            "resolution_reason": resolution_reason,
            "processing_timestamp": processing_time,
        })
        if health["anomaly_detector"].status is ModelStatus.READY:
            try:
                canonical_record = dict(properties)
                anomaly_result = registry.get("anomaly_detector").predict(canonical_record)
                properties["model_evidence"].append({"anomaly_detector": anomaly_result.model_dump(mode="json")})
                if anomaly_result.decision == "review_required":
                    anomaly = {
                        "record": "|".join(identity),
                        "type": "potential_anomaly",
                        "severity": "medium",
                        "confidence": anomaly_result.confidence,
                        "source": source_records,
                        "explanation": anomaly_result.evidence.get("explanation", "Model 5 flagged record attributes for human review."),
                        "status": "Requires review",
                        "evidence": anomaly_result.evidence,
                    }
                    anomalies.append(anomaly)
                    conflicts.append({**anomaly, "field": None, "conflict_type": "potential_anomaly", "source_values": [], "resolution_status": "unresolved"})
                    properties["conflicts"].append(anomaly)
            except Exception as exc:
                stages["anomaly_detection"] = {"status": "SKIPPED", "reason": f"Model 5 inference failed for one or more records: {exc}"}
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
        "anomaly_count": len(anomalies),
        "conflicts": conflicts,
        "anomalies": anomalies,
        "stages": stages,
        "candidate_count": candidates_examined,
        "candidate_results": candidates,
        "model_statuses": {key: value.status.value for key, value in health.items()},
        "model_inference_used": bool(candidates_examined or anomalies),
        "feature_collection": {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {"crs": output_crs, "confidence": None},
        },
        "limitations": [
            "Spatial candidate links are accepted only when every available READY M1/M4 model returns match; uncalibrated model scores are retained as evidence, not labeled calibrated confidence.",
            "Conflicting attributes remain unresolved; no authoritative source precedence is assumed.",
            "Unavailable image models are marked SKIPPED with their exact readiness reason.",
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