from __future__ import annotations

import math

import pytest

from app.core.errors import InvalidInputError
from app.schemas.conflict import Conflict, ConflictType, ResolutionStatus, Severity
from app.schemas.parcel import CanonicalParcel, SourceRecord
from app.schemas.provenance import (
    ProvenanceActorType,
    ProvenanceChain,
    ReviewAction,
    ReviewerAction,
    SourceReference,
)
from app.services.confidence import (
    ConfidenceAggregator,
    ConfidenceComponents,
    attribute_agreement_from_fields,
)


def test_missing_channels_are_excluded_not_defaulted():
    breakdown = ConfidenceAggregator().aggregate(
        ConfidenceComponents(model_confidence=0.9, geometric_agreement=0.8)
    )
    assert set(breakdown.missing_components) == {
        "attribute_agreement",
        "source_agreement",
        "temporal_evidence",
    }
    expected = math.exp(
        (0.35 * math.log(0.9) + 0.25 * math.log(0.8)) / (0.35 + 0.25)
    )
    assert breakdown.base_confidence == pytest.approx(expected)
    assert breakdown.final_confidence == pytest.approx(expected)


def test_no_measurement_yields_undefined_confidence():
    breakdown = ConfidenceAggregator().aggregate(ConfidenceComponents())
    assert breakdown.final_confidence is None
    assert breakdown.base_confidence is None


def test_weak_channel_is_not_hidden_by_the_average():
    breakdown = ConfidenceAggregator().aggregate(
        ConfidenceComponents(model_confidence=0.99, geometric_agreement=0.05)
    )
    arithmetic = (0.35 * 0.99 + 0.25 * 0.05) / 0.6
    assert breakdown.final_confidence < arithmetic


def test_unresolved_conflicts_apply_a_penalty():
    components = ConfidenceComponents(model_confidence=0.9, geometric_agreement=0.9)
    aggregator = ConfidenceAggregator()
    clean = aggregator.aggregate(components)

    conflict = Conflict(
        type=ConflictType.IDENTITY_CONFLICT,
        severity=Severity.CRITICAL,
        source_a="a",
        source_b="b",
    )
    penalised = aggregator.aggregate(components, [conflict])
    assert penalised.conflict_penalty == pytest.approx(0.5)
    assert penalised.final_confidence == pytest.approx(clean.final_confidence * 0.5)

    conflict.resolution_status = ResolutionStatus.RESOLVED
    resolved = aggregator.aggregate(components, [conflict])
    assert resolved.final_confidence == pytest.approx(clean.final_confidence)


def test_out_of_range_component_rejected():
    with pytest.raises(InvalidInputError):
        ConfidenceAggregator().aggregate(ConfidenceComponents(model_confidence=1.4))


def test_unknown_weight_rejected():
    with pytest.raises(InvalidInputError):
        ConfidenceAggregator(weights={"vibes": 1.0})


def test_attribute_agreement_helper():
    assert attribute_agreement_from_fields(["a", "b"], ["c"]) == pytest.approx(2 / 3)
    assert attribute_agreement_from_fields([], []) is None


# ----------------------------------------------------------------------
def test_provenance_chain_records_every_actor():
    chain = ProvenanceChain(target_type="canonical_parcel", target_id="P-1")
    chain.record_model(
        "entity_resolver",
        "model4-1",
        confidence=0.87,
        sources=[SourceReference(source_system="revenue", source_record_id="R-1")],
    )
    chain.record_rule("topology.repair", parameters={"operation": "make_valid"})
    chain.record_transformation("crs_transform", parameters={"from": "EPSG:4326", "to": "EPSG:32643"})
    chain.record_review(
        ReviewerAction(reviewer_id="officer-7", action=ReviewAction.ACCEPTED), "P-1"
    )

    assert [entry.actor_type for entry in chain.entries] == [
        ProvenanceActorType.MODEL,
        ProvenanceActorType.RULE,
        ProvenanceActorType.TRANSFORMATION,
        ProvenanceActorType.HUMAN_REVIEWER,
    ]
    model_entry = chain.entries[0]
    assert model_entry.actor_version == "model4-1"
    assert model_entry.confidence == 0.87
    assert model_entry.sources[0].source_record_id == "R-1"
    assert chain.entries[-1].reviewer_action.action is ReviewAction.ACCEPTED
    assert all(entry.created_at for entry in chain.entries)


def test_canonical_parcel_links_records_and_adopts_ulpin():
    parcel = CanonicalParcel(parcel_id="P-1")
    assert parcel.ulpin is None

    parcel.link(SourceRecord(source_system="revenue", source_record_id="R-1"))
    assert parcel.ulpin is None  # never generated

    parcel.link(
        SourceRecord(
            source_system="municipal", source_record_id="M-1", ulpin="UP-XX-0001"
        )
    )
    assert parcel.ulpin == "UP-XX-0001"
    assert parcel.ulpin_source == "municipal:M-1"
    assert parcel.source_keys == ["revenue:R-1", "municipal:M-1"]


def test_ulpin_disagreement_is_surfaced():
    parcel = CanonicalParcel(parcel_id="P-1")
    parcel.link(SourceRecord(source_system="a", source_record_id="1", ulpin="U-1"))
    parcel.link(SourceRecord(source_system="b", source_record_id="2", ulpin="U-2"))
    assert parcel.ulpin_disagreements() == ["U-1", "U-2"]


def test_blank_ulpin_is_normalised_to_none():
    record = SourceRecord(source_system="a", source_record_id="1", ulpin="   ")
    assert record.ulpin is None
