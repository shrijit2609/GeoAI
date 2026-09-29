"""Spatial and attribute conflict schemas."""

from __future__ import annotations

import datetime as _dt
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConflictType(str, Enum):
    GEOMETRY_CONFLICT = "GEOMETRY_CONFLICT"
    ATTRIBUTE_CONFLICT = "ATTRIBUTE_CONFLICT"
    IDENTITY_CONFLICT = "IDENTITY_CONFLICT"
    TEMPORAL_CHANGE = "TEMPORAL_CHANGE"
    BUILDING_BOUNDARY_CONFLICT = "BUILDING_BOUNDARY_CONFLICT"
    CRS_CONFLICT = "CRS_CONFLICT"
    TOPOLOGY_ERROR = "TOPOLOGY_ERROR"
    MISSING_ATTRIBUTE = "MISSING_ATTRIBUTE"
    SOURCE_DISAGREEMENT = "SOURCE_DISAGREEMENT"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResolutionStatus(str, Enum):
    UNRESOLVED = "unresolved"
    UNDER_REVIEW = "under_review"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class Conflict(BaseModel):
    """A detected disagreement between two sources or within one record.

    ``confidence`` is only populated when a deterministic measurement or a
    model actually produced a value; it is never filled in with a guess.
    """

    conflict_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: ConflictType
    severity: Severity
    source_a: str
    source_b: str | None = None
    parcel_id: str | None = None
    ulpin: str | None = None
    field: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    detector: str | None = Field(
        default=None, description="Rule id or model key that produced the conflict"
    )
    resolution_status: ResolutionStatus = ResolutionStatus.UNRESOLVED
    resolution_note: str | None = None
    detected_at: _dt.datetime = Field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc)
    )


class ConflictSet(BaseModel):
    parcel_id: str | None = None
    conflicts: list[Conflict] = Field(default_factory=list)

    def add(self, conflict: Conflict) -> "ConflictSet":
        self.conflicts.append(conflict)
        return self

    def by_type(self, conflict_type: ConflictType) -> list[Conflict]:
        return [item for item in self.conflicts if item.type == conflict_type]

    @property
    def unresolved(self) -> list[Conflict]:
        return [
            item
            for item in self.conflicts
            if item.resolution_status == ResolutionStatus.UNRESOLVED
        ]
