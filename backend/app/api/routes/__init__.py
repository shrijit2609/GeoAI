"""API route modules."""

from fastapi import APIRouter

from app.api.routes import data, harmonize, health, models, ulpin, upload

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(models.router)
api_router.include_router(upload.router)
api_router.include_router(data.router)
api_router.include_router(data.layers_router)
api_router.include_router(harmonize.router)
api_router.include_router(ulpin.router)

__all__ = ["api_router"]
