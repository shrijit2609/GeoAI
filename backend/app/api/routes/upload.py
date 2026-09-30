"""Upload and ingest vector source files into the canonical parcel model."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.errors import InvalidInputError
from app.services.ingestion import ingest_file, ingest_payload

router = APIRouter(prefix="/upload", tags=["ingestion"])


@router.post("")
async def upload_source(
    file: UploadFile | None = File(default=None),
    source_name: str = Form(default="uploaded-source"),
    source_type: str = Form(default="other"),
    project_crs: str | None = Form(default=None),
    payload: str | None = Form(default=None),
):
    """Upload a GeoJSON, GeoPackage, Shapefile or Parquet source."""
    try:
        if payload is not None:
            if len(payload.encode("utf-8")) > get_settings().upload_max_bytes:
                raise InvalidInputError("JSON payload exceeds the configured upload limit")
            json_payload = json.loads(payload)
            return ingest_payload(
                json_payload,
                source_name=source_name,
                source_type=source_type,
                project_crs=project_crs,
            )

        if file is None:
            raise InvalidInputError("a source file or JSON payload is required")

        suffix = Path(file.filename or "source.json").suffix.lower() or ".geojson"
        if suffix not in {".geojson", ".json", ".gpkg", ".shp", ".parquet", ".pq"}:
            raise InvalidInputError(f"unsupported source format: {suffix}")
        limit = get_settings().upload_max_bytes
        content = await file.read(limit + 1)
        if len(content) > limit:
            raise InvalidInputError("upload exceeds the configured request limit")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(content)
            tmp_path = Path(handle.name)

        try:
            result = ingest_file(
                tmp_path,
                source_name=source_name,
                source_type=source_type,
                project_crs=project_crs,
            )
            return result
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except TypeError:
                if tmp_path.exists():
                    tmp_path.unlink()
    except (InvalidInputError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
