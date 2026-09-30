"""Bounded upload, in-memory source catalog, layers and exports."""

from __future__ import annotations

import csv
import io
import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from app.core.config import get_settings
from app.core.errors import InvalidInputError
from app.services.geospatial import to_geometry, transform_geometry
from app.services.ingestion import ingest_file, ingest_payload
from app.services.source_catalog import source_catalog

router = APIRouter(prefix="/data", tags=["data"])
layers_router = APIRouter(prefix="/layers", tags=["layers"])
SUPPORTED_EXTENSIONS = {".geojson", ".json", ".gpkg", ".shp", ".zip", ".parquet", ".pq", ".csv", ".kml"}


def _safe_upload_name(filename: str | None) -> str:
    name = (filename or "").replace("\\", "/")
    clean = PurePosixPath(name).name
    if not clean or clean in {".", ".."}:
        raise InvalidInputError("uploaded file must have a valid filename")
    suffix = Path(clean).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise InvalidInputError(f"unsupported upload extension: {suffix or '(none)'}")
    return clean


def _ingest_shapefile_zip(path: Path, *, source_name: str, source_type: str | None, source_crs: str | None, project_crs: str | None) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory(prefix="spatialshift-shp-") as temp_dir:
        members = archive.infolist()
        zip_limit = get_settings().upload_max_zip_expanded_bytes
        if sum(member.file_size for member in members) > zip_limit:
            raise InvalidInputError(f"Shapefile ZIP exceeds the {zip_limit}-byte expanded size limit")
        root = Path(temp_dir).resolve()
        shapefiles = []
        for member in members:
            if member.is_dir():
                continue
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise InvalidInputError("ZIP contains an unsafe path")
            destination = (root / Path(*relative.parts)).resolve()
            if root not in destination.parents:
                raise InvalidInputError("ZIP contains an unsafe path")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                target.write(source.read(zip_limit + 1))
            if destination.suffix.lower() == ".shp":
                shapefiles.append(destination)
        if len(shapefiles) != 1:
            raise InvalidInputError("Shapefile ZIP must contain exactly one .shp file")
        return ingest_file(
            shapefiles[0],
            source_name=source_name,
            source_type=source_type or "shapefile",
            source_crs=source_crs,
            project_crs=project_crs,
        )


async def _read_bounded(file: UploadFile) -> bytes:
    upload_limit = get_settings().upload_max_bytes
    content = await file.read(upload_limit + 1)
    if len(content) > upload_limit:
        raise InvalidInputError(f"upload exceeds the configured {upload_limit}-byte request limit")
    return content


@router.post("/upload")
async def upload_data(
    file: UploadFile | None = File(default=None),
    payload: str | None = Form(default=None),
    source_name: str | None = Form(default=None),
    source_type: str | None = Form(default=None),
    source_crs: str | None = Form(default=None),
    project_crs: str | None = Form(default=None),
):
    try:
        if payload is not None:
            result = ingest_payload(
                json.loads(payload),
                source_name=source_name or "uploaded-geojson",
                source_type=source_type,
                source_crs=source_crs,
                project_crs=project_crs,
            )
        elif file is not None:
            name = _safe_upload_name(file.filename)
            suffix = Path(name).suffix.lower()
            content = await _read_bounded(file)
            with tempfile.TemporaryDirectory(prefix="spatialshift-upload-") as temp_dir:
                path = Path(temp_dir) / name
                path.write_bytes(content)
                ingest_kwargs = {
                    "source_name": source_name or Path(name).stem,
                    "source_type": source_type,
                    "project_crs": project_crs,
                    "source_crs": source_crs,
                }
                if suffix == ".zip":
                    result = _ingest_shapefile_zip(
                        path,
                        source_name=ingest_kwargs["source_name"],
                        source_type=source_type,
                        source_crs=source_crs,
                        project_crs=project_crs,
                    )
                else:
                    result = ingest_file(path, **ingest_kwargs)
        else:
            raise InvalidInputError("provide a vector file or a GeoJSON payload")
        summary = source_catalog.register(result)
        return {**result, "source_id": summary["source_id"]}
    except (InvalidInputError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/sources")
def list_sources() -> dict[str, Any]:
    return {"sources": source_catalog.list_sources(), "count": len(source_catalog.list_sources())}


@router.get("/layers")
def list_layers() -> dict[str, Any]:
    return {"layers": source_catalog.list_layers()}


def _as_feature_collection(source: dict[str, Any]) -> dict[str, Any]:
    features = []
    for record in source.get("records", []):
        geometry = record.get("geometry")
        record_crs = record.get("crs")
        if geometry is not None:
            if not record_crs:
                geometry = None
            elif record_crs.upper() not in {"EPSG:4326", "OGC:CRS84"}:
                geometry = transform_geometry(to_geometry(geometry), record_crs, "EPSG:4326").__geo_interface__
        properties = dict(record.get("raw_attributes") or {})
        properties.update({key: value for key, value in record.items() if key not in {"geometry", "raw_attributes"}})
        features.append({"type": "Feature", "id": record.get("source_id"), "geometry": geometry, "properties": properties})
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "source_id": source["source_id"],
            "source_name": source.get("source_name"),
            "source_crs": source.get("detected_crs"),
            "display_crs": "EPSG:4326" if source.get("detected_crs") else None,
            "warnings": list(source.get("warnings", [])) + (["CRS is unknown for one or more records; those geometries are withheld from geographic map display."] if any(record.get("geometry") is not None and not record.get("crs") for record in source.get("records", [])) else []),
        },
    }


@router.get("/{source_id}/export")
def export_source(source_id: str, format: str = "geojson"):
    source = source_catalog.get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="source not found in this backend process")
    collection = _as_feature_collection(source)
    if format.lower() == "geojson":
        return JSONResponse(collection, headers={"Content-Disposition": f'attachment; filename="{source_id}.geojson"'})
    if format.lower() == "csv":
        output = io.StringIO()
        fields = sorted({key for feature in collection["features"] for key in (feature.get("properties") or {})})
        writer = csv.DictWriter(output, fieldnames=[*fields, "geometry_geojson"], extrasaction="ignore")
        writer.writeheader()
        for feature in collection["features"]:
            writer.writerow({**(feature.get("properties") or {}), "geometry_geojson": json.dumps(feature.get("geometry"))})
        return Response(output.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{source_id}.csv"'})
    raise HTTPException(status_code=400, detail="format must be geojson or csv")


@router.get("/{source_id}")
def get_source(source_id: str) -> dict[str, Any]:
    source = source_catalog.get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="source not found in this backend process")
    return source


@router.get("/layers/{layer_id}")
def get_layer(layer_id: str) -> dict[str, Any]:
    source = source_catalog.get_source(layer_id)
    if source is None:
        raise HTTPException(status_code=404, detail="layer not found in this backend process")
    return _as_feature_collection(source)


@layers_router.get("")
def list_layers_alias() -> dict[str, Any]:
    return list_layers()


@layers_router.get("/{layer_id}")
def get_layer_alias(layer_id: str) -> dict[str, Any]:
    return get_layer(layer_id)