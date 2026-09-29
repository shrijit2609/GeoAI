"""API route modules."""

from fastapi import APIRouter

from app.api.routes import health, models, upload

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(models.router)
api_router.include_router(upload.router)

__all__ = ["api_router"]
