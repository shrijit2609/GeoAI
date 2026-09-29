"""Deterministic (non-ML) domain services."""

from app.services.confidence import ConfidenceAggregator, ConfidenceComponents
from app.services.conflicts import ConflictEngine
from app.services.ingestion import (
    GeoJSONAdapter,
    GeoPackageAdapter,
    ParquetAdapter,
    ShapefileAdapter,
    SourceAdapter,
    ingest_file,
    ingest_payload,
)
from app.services.topology import TopologyService

__all__ = [
    "ConfidenceAggregator",
    "ConfidenceComponents",
    "ConflictEngine",
    "GeoJSONAdapter",
    "GeoPackageAdapter",
    "ParquetAdapter",
    "ShapefileAdapter",
    "SourceAdapter",
    "TopologyService",
    "ingest_file",
    "ingest_payload",
]
