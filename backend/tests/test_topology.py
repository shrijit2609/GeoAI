from __future__ import annotations

from app.schemas.topology import TopologyIssueType
from app.services.topology import TopologyService

SQUARE = "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))"
OVERLAPPING = "POLYGON ((5 0, 5 10, 15 10, 15 0, 5 0))"
BOWTIE = "POLYGON ((0 0, 10 10, 10 0, 0 10, 0 0))"
EMPTY = "POLYGON EMPTY"


def issue_types(report) -> set[TopologyIssueType]:
    return {issue.type for issue in report.issues}


def test_clean_features_produce_no_issues():
    report = TopologyService().validate(
        [("a", SQUARE), ("b", "POLYGON ((20 0, 20 10, 30 10, 30 0, 20 0))")],
        crs="EPSG:32643",
    )
    assert report.is_clean
    assert report.feature_count == 2
    assert not report.repairs


def test_self_intersection_detected_and_repair_proposed():
    report = TopologyService().validate([("bowtie", BOWTIE)], crs="EPSG:32643")
    assert TopologyIssueType.SELF_INTERSECTION in issue_types(report)

    assert len(report.repairs) == 1
    proposal = report.repairs[0]
    assert proposal.feature_id == "bowtie"
    assert proposal.operation in ("make_valid", "buffer_zero")
    assert proposal.original_geometry is not None
    assert proposal.proposed_geometry is not None
    assert proposal.applied is False
    assert proposal.provenance["actor"] == "topology_service"
    assert "area_before" in proposal.metrics


def test_empty_geometry_detected():
    report = TopologyService().validate([("empty", EMPTY)])
    assert TopologyIssueType.EMPTY_GEOMETRY in issue_types(report)


def test_unparseable_geometry_detected():
    report = TopologyService().validate([("bad", "POLYGON (())")])
    assert TopologyIssueType.UNPARSEABLE_GEOMETRY in issue_types(report)


def test_duplicate_geometry_detected():
    report = TopologyService().validate([("a", SQUARE), ("b", SQUARE)])
    duplicates = [
        issue for issue in report.issues if issue.type == TopologyIssueType.DUPLICATE_GEOMETRY
    ]
    assert duplicates
    assert set(duplicates[0].feature_ids) == {"a", "b"}


def test_overlap_detected_with_metrics():
    report = TopologyService().validate([("a", SQUARE), ("b", OVERLAPPING)])
    overlaps = [issue for issue in report.issues if issue.type == TopologyIssueType.OVERLAP]
    assert overlaps
    assert overlaps[0].evidence["intersection_area"] == 50.0


def test_gap_detected_between_surrounding_parcels():
    ring = [
        ("n", "POLYGON ((0 20, 30 20, 30 30, 0 30, 0 20))"),
        ("s", "POLYGON ((0 0, 30 0, 30 10, 0 10, 0 0))"),
        ("w", "POLYGON ((0 10, 10 10, 10 20, 0 20, 0 10))"),
        ("e", "POLYGON ((20 10, 30 10, 30 20, 20 20, 20 10))"),
    ]
    report = TopologyService().validate(ring)
    gaps = [issue for issue in report.issues if issue.type == TopologyIssueType.GAP]
    assert gaps
    assert gaps[0].evidence["area"] == 100.0


def test_gap_detection_can_be_disabled():
    ring = [
        ("n", "POLYGON ((0 20, 30 20, 30 30, 0 30, 0 20))"),
        ("s", "POLYGON ((0 0, 30 0, 30 10, 0 10, 0 0))"),
        ("w", "POLYGON ((0 10, 10 10, 10 20, 0 20, 0 10))"),
        ("e", "POLYGON ((20 10, 30 10, 30 20, 20 20, 20 10))"),
    ]
    report = TopologyService(detect_gaps=False).validate(ring)
    assert TopologyIssueType.GAP not in issue_types(report)
