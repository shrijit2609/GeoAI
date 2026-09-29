"""Provenance structures for harmonized outputs."""

from __future__ import annotations

import datetime as _dt
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class ProvenanceActorType(str, Enum):
    MODEL = "model"
    RULE = "rule"
    TRANSFORMATION = "transformation"
    HUMAN_REVIEWER = "human_reviewer"
    IMPORT = "import"


class ReviewAction(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    MODIFIED = "modified"
    DEFERRED = "deferred"


class SourceReference(BaseModel):
    source_system: str
    source_record_id: str
    source_dataset: str | None = None
    source_type: str | None = None


class ReviewerAction(BaseModel):
    reviewer_id: str
    action: ReviewAction
    comment: str | None = None
    acted_at: _dt.datetime = Field(default_factory=_now)


class ProvenanceEntry(BaseModel):
    """A single traceable step that contributed to a harmonized output."""

    model_config = {"protected_namespaces": ()}

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    actor_type: ProvenanceActorType
    actor: str = Field(description="Model key, rule id, or transformation name")
    actor_version: str | None = None
    description: str | None = None
    sources: list[SourceReference] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    reviewer_action: ReviewerAction | None = None
    created_at: _dt.datetime = Field(default_factory=_now)


class ProvenanceChain(BaseModel):
    """Ordered provenance for one harmonized target object."""

    target_type: str
    target_id: str
    entries: list[ProvenanceEntry] = Field(default_factory=list)

    def add(self, entry: ProvenanceEntry) -> "ProvenanceChain":
        self.entries.append(entry)
        return self

    def record_model(
        self,
        model_key: str,
        model_version: str,
        *,
        confidence: float | None = None,
        description: str | None = None,
        sources: list[SourceReference] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> ProvenanceEntry:
        entry = ProvenanceEntry(
            actor_type=ProvenanceActorType.MODEL,
            actor=model_key,
            actor_version=model_version,
            description=description,
            confidence=confidence,
            sources=sources or [],
            parameters=parameters or {},
        )
        self.add(entry)
        return entry

    def record_rule(
        self,
        rule_id: str,
        *,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        confidence: float | None = None,
    ) -> ProvenanceEntry:
        entry = ProvenanceEntry(
            actor_type=ProvenanceActorType.RULE,
            actor=rule_id,
            description=description,
            parameters=parameters or {},
            confidence=confidence,
        )
        self.add(entry)
        return entry

    def record_transformation(
        self,
        name: str,
        *,
        parameters: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> ProvenanceEntry:
        entry = ProvenanceEntry(
            actor_type=ProvenanceActorType.TRANSFORMATION,
            actor=name,
            description=description,
            parameters=parameters or {},
        )
        self.add(entry)
        return entry

    def record_review(self, action: ReviewerAction, target: str) -> ProvenanceEntry:
        entry = ProvenanceEntry(
            actor_type=ProvenanceActorType.HUMAN_REVIEWER,
            actor=action.reviewer_id,
            description=f"{action.action.value} {target}",
            reviewer_action=action,
        )
        self.add(entry)
        return entry
