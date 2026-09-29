"""Harmonization confidence aggregation.

The aggregator keeps the individual evidence channels separate and combines
only the ones that were actually measured.

Formula
-------
Given the set ``A`` of components that are present (each with a value
``v_i`` in ``[0, 1]`` and a weight ``w_i``), the base confidence is the
weighted geometric mean::

    base = exp( sum_i(w_i * ln(max(v_i, eps))) / sum_i(w_i) )

A geometric mean is used instead of an arithmetic average so that a single
weak channel (for example a geometric agreement of 0.05) cannot be hidden by
strong channels.

Unresolved conflicts then apply a multiplicative penalty::

    final = base * prod_over_unresolved_conflicts(1 - penalty(severity))

with ``penalty(critical)=0.5``, ``high=0.3``, ``medium=0.15``, ``low=0.05``,
``info=0``. Missing channels are never replaced by a default value; they are
reported in ``missing_components`` and simply do not take part in the mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from pydantic import BaseModel, Field

from app.core.errors import InvalidInputError
from app.schemas.conflict import Conflict, ResolutionStatus, Severity

EPSILON = 1e-6

DEFAULT_WEIGHTS: dict[str, float] = {
    "model_confidence": 0.35,
    "geometric_agreement": 0.25,
    "attribute_agreement": 0.2,
    "source_agreement": 0.1,
    "temporal_evidence": 0.1,
}

SEVERITY_PENALTY: dict[Severity, float] = {
    Severity.CRITICAL: 0.5,
    Severity.HIGH: 0.3,
    Severity.MEDIUM: 0.15,
    Severity.LOW: 0.05,
    Severity.INFO: 0.0,
}


@dataclass
class ConfidenceComponents:
    """Individually measured evidence channels. ``None`` means 'not measured'."""

    model_confidence: float | None = None
    geometric_agreement: float | None = None
    attribute_agreement: float | None = None
    source_agreement: float | None = None
    temporal_evidence: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "model_confidence": self.model_confidence,
            "geometric_agreement": self.geometric_agreement,
            "attribute_agreement": self.attribute_agreement,
            "source_agreement": self.source_agreement,
            "temporal_evidence": self.temporal_evidence,
        }


class ConfidenceBreakdown(BaseModel):
    components: dict[str, float | None]
    weights: dict[str, float]
    missing_components: list[str] = Field(default_factory=list)
    base_confidence: float | None = None
    conflict_penalty: float = 1.0
    final_confidence: float | None = None
    formula: str = (
        "final = exp(sum(w_i*ln(v_i))/sum(w_i)) * prod(1 - penalty(severity))"
    )
    notes: list[str] = Field(default_factory=list)


class ConfidenceAggregator:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            unknown = set(weights) - set(DEFAULT_WEIGHTS)
            if unknown:
                raise InvalidInputError(
                    f"unknown confidence components: {', '.join(sorted(unknown))}"
                )
            self.weights.update(weights)

    def aggregate(
        self,
        components: ConfidenceComponents,
        conflicts: Sequence[Conflict] | None = None,
    ) -> ConfidenceBreakdown:
        values = components.as_dict()
        for name, value in values.items():
            if value is not None and not 0.0 <= value <= 1.0:
                raise InvalidInputError(
                    f"confidence component '{name}' must be in [0, 1], got {value}"
                )

        present = {name: value for name, value in values.items() if value is not None}
        missing = [name for name, value in values.items() if value is None]
        notes: list[str] = []

        if not present:
            return ConfidenceBreakdown(
                components=values,
                weights=self.weights,
                missing_components=missing,
                base_confidence=None,
                conflict_penalty=1.0,
                final_confidence=None,
                notes=["no evidence channel was measured; confidence is undefined"],
            )

        weight_sum = sum(self.weights[name] for name in present)
        log_sum = sum(
            self.weights[name] * math.log(max(value, EPSILON))
            for name, value in present.items()
        )
        base = math.exp(log_sum / weight_sum)

        penalty = self._conflict_penalty(conflicts or [])
        if missing:
            notes.append(
                "not measured: " + ", ".join(missing) + " (excluded from the mean)"
            )

        return ConfidenceBreakdown(
            components=values,
            weights=self.weights,
            missing_components=missing,
            base_confidence=base,
            conflict_penalty=penalty,
            final_confidence=max(0.0, min(1.0, base * penalty)),
            notes=notes,
        )

    @staticmethod
    def _conflict_penalty(conflicts: Iterable[Conflict]) -> float:
        penalty = 1.0
        for conflict in conflicts:
            if conflict.resolution_status != ResolutionStatus.UNRESOLVED:
                continue
            penalty *= 1.0 - SEVERITY_PENALTY.get(conflict.severity, 0.0)
        return penalty


def attribute_agreement_from_fields(
    matched: Sequence[str], conflicting: Sequence[str]
) -> float | None:
    """Fraction of compared attributes that agree; ``None`` if none compared."""

    total = len(matched) + len(conflicting)
    if total == 0:
        return None
    return len(matched) / total
