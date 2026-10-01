"""Compatibility API routes for the production GIS dashboard and existing clients."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from app.api.routes.data import (
    export_source,
    get_source,
    list_layers,
    list_sources,
    upload_data,
)
from app.api.routes.harmonize import HarmonizeRequest, create_harmonization, get_harmonization
from app.api.routes.models import infer_with_registry, model_inference, models_health
from app.api.routes.ulpin import lookup_ulpin
from app.ml.model_registry import get_registry
from app.services.source_catalog import source_catalog

router = APIRouter()


@router.get("/dashboard/summary")
def dashboard_summary() -> dict[str, Any]:
    sources = source_catalog.list_sources()
    jobs = [job for job in source_catalog._jobs.values()]
    registry_summary = get_registry().summary(probe=True)

    feature_count = sum(len(source.get("records", [])) for source in sources)
    conflict_count = sum(len(job.get("conflicts", [])) for job in jobs)
    anomaly_count = sum(len(job.get("anomalies", [])) for job in jobs)
    harmonized_parcels = sum(int(job.get("feature_count", 0)) for job in jobs)
    confidence_values: list[float] = []
    for source in sources:
        for record in source.get("records", []):
            value = record.get("confidence")
            if isinstance(value, (int, float)):
                confidence_values.append(float(value))
    for job in jobs:
        for feature in (job.get("feature_collection") or {}).get("features", []):
            value = (feature.get("properties") or {}).get("confidence")
            if isinstance(value, (int, float)):
                confidence_values.append(float(value))
    average_confidence = (
        round(sum(confidence_values) / len(confidence_values), 4)
        if confidence_values
        else 0.0
    )

    latest_job = sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)[0] if jobs else {}
    return {
        "registered_sources": len(sources),
        "total_features": feature_count,
        "harmonized_parcels": harmonized_parcels,
        "conflicts": conflict_count,
        "anomalies": anomaly_count,
        "average_confidence": average_confidence,
        "models_ready": int(registry_summary.get("ready", 0)),
        "processing_jobs": len(jobs),
        "source_health": {
            "healthy": len([source for source in sources if not source.get("warnings")]),
            "warning_count": sum(len(source.get("warnings", [])) for source in sources),
        },
        "model_health": {
            "ready": int(registry_summary.get("ready", 0)),
            "total": int(registry_summary.get("total", 0)),
            "status_counts": registry_summary.get("status_counts", {}),
        },
        "recent_activity": latest_job.get("source_ids", []),
        "processing_status": latest_job.get("status", "PENDING"),
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }


@router.get("/sources")
def sources_compat() -> dict[str, Any]:
    payload = list_sources()
    return {"sources": payload["sources"], "count": payload["count"]}


@router.post("/sources/upload")
async def source_upload_compat(
    file: UploadFile | None = File(default=None),
    payload: str | None = Form(default=None),
    source_name: str | None = Form(default=None),
    source_type: str | None = Form(default=None),
    source_crs: str | None = Form(default=None),
    project_crs: str | None = Form(default=None),
    register_source: bool = Form(default=False, alias="register"),
):
    return await upload_data(file=file, payload=payload, source_name=source_name, source_type=source_type, source_crs=source_crs, project_crs=project_crs, register_source=register_source)


@router.post("/sources/register")
def source_register_compat(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="source payload must be a JSON object")
    source = payload.get("source") if isinstance(payload.get("source"), dict) else payload
    if not source.get("source_name"):
        raise HTTPException(status_code=422, detail="source_name is required for registration")
    if not isinstance(source.get("records"), list):
        raise HTTPException(status_code=422, detail="source must contain validated records")
    return source_catalog.register(source)


@router.get("/sources/{source_id}")
def source_detail_compat(source_id: str) -> dict[str, Any]:
    payload = get_source(source_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="source not found")
    return payload


@router.delete("/sources/{source_id}")
def delete_source_compat(source_id: str) -> dict[str, Any]:
    if not source_catalog.delete_source(source_id):
        raise HTTPException(status_code=404, detail="source not found")
    return {"source_id": source_id, "deleted": True}


@router.get("/map/layers")
def map_layers_compat() -> dict[str, Any]:
    return {"layers": list_layers()["layers"]}


@router.get("/map/layers/{layer_id}")
def map_layer_detail_compat(layer_id: str) -> dict[str, Any]:
    return source_catalog.get_source(layer_id) or {}


@router.post("/harmonization/run")
def harmonization_run_compat(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    request = HarmonizeRequest(**payload)
    return create_harmonization(request)


@router.get("/harmonization/{job_id}")
def harmonization_status_compat(job_id: str) -> dict[str, Any]:
    job = get_harmonization(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.get("/harmonization/{job_id}/results")
def harmonization_results_compat(job_id: str) -> dict[str, Any]:
    job = get_harmonization(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "result_count": job.get("feature_count", 0),
        "results": job.get("feature_collection", {"type": "FeatureCollection", "features": []}),
        "conflicts": job.get("conflicts", []),
    }


@router.get("/models/readiness")
def readiness_compat() -> dict[str, Any]:
    summary = get_registry().summary(probe=True)
    summary["models"] = {
        key: health.model_dump(mode="json") for key, health in summary["models"].items()
    }
    return summary


@router.post("/entity-resolution/infer")
def entity_resolution_compat(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    return infer_with_registry("entity_resolver", payload, get_registry())


@router.post("/anomaly/infer")
def anomaly_infer_compat(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    return infer_with_registry("anomaly_detector", payload, get_registry())


@router.post("/ulpin/lookup")
def lookup_compat(payload: dict[str, Any] | None = Body(default=None), ulpin: str | None = None) -> dict[str, Any]:
    query = ulpin or (payload or {}).get("ulpin") or (payload or {}).get("query")
    if not query:
        raise HTTPException(status_code=422, detail="ULPIN is required")
    return lookup_ulpin(str(query))


@router.get("/exports/{job_id}")
def export_compat(job_id: str, format: str = "geojson") -> Response:
    job = get_harmonization(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    collection = job.get("feature_collection", {"type": "FeatureCollection", "features": []})
    if format.lower() == "geojson":
        return JSONResponse(collection, headers={"Content-Disposition": f'attachment; filename="{job_id}.geojson"'})
    if format.lower() == "csv":
        output = []
        for feature in collection.get("features", []):
            props = feature.get("properties", {})
            output.append({**props, "geometry": json.dumps(feature.get("geometry"))})
        if not output:
            return Response("", media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{job_id}.csv"'})
        header = sorted({key for row in output for key in row.keys()})
        lines = [",".join(header)]
        for row in output:
            values = [str(row.get(column, "")).replace('"', '""') for column in header]
            lines.append(",".join(f'"{value}"' if "," in value or '"' in value else value for value in values))
        body = "\n".join(lines)
        return Response(body, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{job_id}.csv"'})
    if format.lower() == "conflicts":
        payload = json.dumps(job.get("conflicts", []), ensure_ascii=False, indent=2)
        return Response(payload, media_type="application/json", headers={"Content-Disposition": f'attachment; filename="{job_id}-conflicts.json"'})
    if format.lower() == "provenance":
        report = [
            {"record_id": feature.get("id"), "provenance": feature.get("properties", {}).get("provenance"), "model_evidence": feature.get("properties", {}).get("model_evidence"), "source_records": feature.get("properties", {}).get("source_records"), "resolution_reason": feature.get("properties", {}).get("resolution_reason"), "processing_timestamp": feature.get("properties", {}).get("processing_timestamp")}
            for feature in collection.get("features", [])
        ]
        payload = json.dumps(report, ensure_ascii=False, indent=2)
        return Response(payload, media_type="application/json", headers={"Content-Disposition": f'attachment; filename="{job_id}-provenance.json"'})
    raise HTTPException(status_code=400, detail="unsupported export format")
