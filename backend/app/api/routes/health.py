"""Health and version endpoints."""

from __future__ import annotations

import datetime as _dt
import platform
from typing import Any

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.ml.common import library_versions, resolve_device

router = APIRouter(tags=["health"])


@router.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Liveness plus the runtime facts that actually apply to this process."""

    model_root = settings.resolved_model_root
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "time": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "device": resolve_device(settings.device),
        "requested_device": settings.device,
        "model_root": str(model_root),
        "model_root_exists": model_root.is_dir(),
    }


@router.get("/version")
def version(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": library_versions(),
    }
