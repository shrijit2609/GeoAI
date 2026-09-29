from __future__ import annotations

from app.schemas.conflict import Conflict, ConflictType, ResolutionStatus, Severity
from app.schemas.parcel import LandRecordAttributes, SourceRecord, SourceType
from app.services.conflicts import ConflictEngine

SQUARE = {
    "type": "Polygon",
    "coordinates": [[[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]],
}
SHIFTED = {
    "type": "Polygon",
    "coordinates": [[[5, 0], [5, 10], [15, 10], [15, 0], [5, 0]]],
}


def make_record(**overrides) -> SourceRecord:
    payload = {
        "source_system": "revenue",
        "source_record_id": "R-1",
        "source_type": SourceType.REVENUE_RECORD,
        "crs": "EPSG:32643",
        "geometry": SQUARE,
        "attributes": LandRecordAttributes(
            village="Sultanpur", district="Gautam Buddh Nagar", khasra="121/2", area=1000.0
        ),
    }
    payload.update(overrides)
    return SourceRecord(**payload)


def types_of(conflict_set) -> set[ConflictType]:
    return {conflict.type for conflict in conflict_set.conflicts}


def test_identical_records_produce_no_conflicts():
    result = ConflictEngine().compare_records(
        make_record(), make_record(source_system="municipal", source_record_id="M-1")
    )
    assert result.conflicts == []


def test_geometry_conflict_reported_with_measured_evidence():
    result = ConflictEngine().compare_records(
        make_record(),
        make_record(source_system="municipal", source_record_id="M-1", geometry=SHIFTED),
    )
    geometry = result.by_type(ConflictType.GEOMETRY_CONFLICT)
    assert geometry
    assert geometry[0].evidence["iou"] < 0.9
    assert geometry[0].confidence is None
    assert geometry[0].resolution_status is ResolutionStatus.UNRESOLVED


def test_crs_conflict_blocks_geometry_comparison():
    result = ConflictEngine().compare_records(
        make_record(),
        make_record(source_system="municipal", source_record_id="M-1", crs="EPSG:4326"),
    )
    assert ConflictType.CRS_CONFLICT in types_of(result)
    assert ConflictType.GEOMETRY_CONFLICT not in types_of(result)


def test_identity_conflict_on_differing_ulpin():
    result = ConflictEngine().compare_records(
        make_record(ulpin="UP-XX-0001"),
        make_record(
            source_system="municipal", source_record_id="M-1", ulpin="UP-XX-0002"
        ),
    )
    identity = result.by_type(ConflictType.IDENTITY_CONFLICT)
    assert identity
    assert identity[0].severity is Severity.CRITICAL
    assert identity[0].field == "ulpin"


def test_attribute_and_area_conflicts():
    other = make_record(
        source_system="municipal",
        source_record_id="M-1",
        attributes=LandRecordAttributes(
            village="Sultanpur",
            district="Gautam Buddh Nagar",
            khasra="121/3",
            area=1500.0,
        ),
    )
    result = ConflictEngine().compare_records(make_record(), other)
    fields = {c.field for c in result.by_type(ConflictType.ATTRIBUTE_CONFLICT)}
    assert {"khasra", "area"} <= fields


def test_area_within_tolerance_is_not_a_conflict():
    other = make_record(
        source_system="municipal",
        source_record_id="M-1",
        attributes=LandRecordAttributes(
            village="Sultanpur",
            district="Gautam Buddh Nagar",
            khasra="121/2",
            area=1020.0,
        ),
    )
    result = ConflictEngine().compare_records(make_record(), other)
    assert not result.by_type(ConflictType.ATTRIBUTE_CONFLICT)


def test_missing_critical_attribute_reported():
    incomplete = make_record(
        source_system="municipal",
        source_record_id="M-1",
        attributes=LandRecordAttributes(village="Sultanpur", district=None, khasra=None),
    )
    result = ConflictEngine().compare_records(make_record(), incomplete)
    missing = {c.field for c in result.by_type(ConflictType.MISSING_ATTRIBUTE)}
    assert {"district", "khasra"} <= missing


def test_conflict_object_defaults():
    conflict = Conflict(
        type=ConflictType.SOURCE_DISAGREEMENT,
        severity=Severity.LOW,
        source_a="revenue:R-1",
        source_b="municipal:M-1",
    )
    assert conflict.conflict_id
    assert conflict.resolution_status is ResolutionStatus.UNRESOLVED
    assert conflict.confidence is None
    assert conflict.detected_at is not None


def test_conflicts_from_topology_report():
    from app.services.topology import TopologyService

    report = TopologyService().validate(
        [("a", "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))"),
         ("b", "POLYGON ((5 0, 5 10, 15 10, 15 0, 5 0))")]
    )
    conflicts = ConflictEngine().from_topology_report(report, source="cadastral_layer")
    assert conflicts.conflicts
    assert conflicts.conflicts[0].detector == "topology_service"
