"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import api_router
from app.core.config import get_settings
from app.core.errors import (
    InvalidInputError,
    ModelArtifactMissingError,
    ModelCompatibilityError,
    ModelLoadError,
    SpatialShiftError,
)
from app.core.logging import configure_logging, get_logger
from app.ml.model_registry import get_registry

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    registry = get_registry()
    logger.info(
        "SpatialShiftAI starting: model_root=%s device=%s",
        registry.model_root,
        registry.device,
    )
    if settings.eager_model_load:
        for key in registry.keys():
            try:
                registry.load(key)
            except SpatialShiftError as exc:
                logger.warning("eager load skipped for %s: %s", key, exc)
    yield
    registry.unload_all()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Automated integration and intelligent harmonization of multi-source "
            "geospatial data for urban land record management (SIH PS 26013)."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix=settings.api_prefix)

    @app.exception_handler(ModelArtifactMissingError)
    async def _missing_artifact(request: Request, exc: ModelArtifactMissingError):
        return JSONResponse(
            status_code=503,
            content={
                "error": "missing_artifact",
                "model": exc.model_key,
                "missing": exc.missing,
                "detail": str(exc),
            },
        )

    @app.exception_handler(ModelCompatibilityError)
    async def _incompatible(request: Request, exc: ModelCompatibilityError):
        return JSONResponse(
            status_code=503,
            content={
                "error": "incompatible_artifact",
                "model": exc.model_key,
                "required": exc.required,
                "detail": str(exc),
            },
        )

    @app.exception_handler(ModelLoadError)
    async def _load_error(request: Request, exc: ModelLoadError):
        return JSONResponse(
            status_code=503,
            content={
                "error": "load_error",
                "model": exc.model_key,
                "detail": str(exc),
            },
        )

    @app.exception_handler(InvalidInputError)
    async def _invalid_input(request: Request, exc: InvalidInputError):
        return JSONResponse(
            status_code=422, content={"error": "invalid_input", "detail": str(exc)}
        )

    @app.exception_handler(SpatialShiftError)
    async def _domain_error(request: Request, exc: SpatialShiftError):
        return JSONResponse(
            status_code=400, content={"error": "domain_error", "detail": str(exc)}
        )

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "docs": "/docs",
            "health": f"{settings.api_prefix}/health",
        }

    return app


app = create_app()
