"""Topology validation and repair schemas."""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TopologyIssueType(str, Enum):
    INVALID_GEOMETRY = "invalid_geometry"
    SELF_INTERSECTION = "self_intersection"
    EMPTY_GEOMETRY = "empty_geometry"
    DUPLICATE_GEOMETRY = "duplicate_geometry"
    OVERLAP = "overlap"
    GAP = "gap"
    UNPARSEABLE_GEOMETRY = "unparseable_geometry"


class TopologyIssue(BaseModel):
    type: TopologyIssueType
    severity: str = "medium"
    feature_ids: list[str] = Field(default_factory=list)
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class RepairProposal(BaseModel):
    """A proposed geometry change that a human must approve.

    Authoritative cadastral geometry is never modified in place; the original
    and the proposal are both retained.
    """

    feature_id: str
    operation: str
    reason: str
    original_geometry: dict[str, Any] | None
    proposed_geometry: dict[str, Any] | None
    confidence: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: _dt.datetime = Field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc)
    )
    applied: bool = False


class TopologyReport(BaseModel):
    feature_count: int
    checked: list[str] = Field(default_factory=list)
    issues: list[TopologyIssue] = Field(default_factory=list)
    repairs: list[RepairProposal] = Field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.issues
