"""Local ULPIN lookup with an optional explicitly configured external service."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException

from app.services.source_catalog import source_catalog

router = APIRouter(prefix="/ulpin", tags=["land-record identity"])


@router.get("/{ulpin}")
def lookup_ulpin(ulpin: str) -> dict:
    if not ulpin.strip():
        raise HTTPException(status_code=422, detail="ULPIN must not be empty")
    local_records = source_catalog.ulpin_matches(ulpin)
    endpoint = os.environ.get("SPATIALSHIFT_ULPIN_ENDPOINT", "").strip()
    if local_records:
        return {
            "status": "local_match",
            "ulpin": ulpin,
            "records": local_records,
            "external_service": "configured" if endpoint else "not_configured",
        }
    if not endpoint:
        return {
            "status": "not_found_locally",
            "ulpin": ulpin,
            "records": [],
            "external_service": "not_configured",
            "detail": "External ULPIN service is not configured; no government record was queried.",
        }

    url = endpoint.replace("{ulpin}", quote(ulpin, safe=""))
    if "{ulpin}" not in endpoint:
        url = f"{endpoint.rstrip('/')}/{quote(ulpin, safe='')}"
    try:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read(2_000_001))
            return {"status": "external_response", "ulpin": ulpin, "records": payload, "external_service": "configured"}
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Configured ULPIN service could not be queried: {type(exc).__name__}") from exc