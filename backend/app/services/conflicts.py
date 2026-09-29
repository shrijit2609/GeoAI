"""Deterministic spatial/attribute conflict detection.

Every conflict produced here is backed by a measurement or an explicit
disagreement between two source records. Nothing is inferred probabilistically.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.core.errors import GeometryError
from app.schemas.conflict import Conflict, ConflictSet, ConflictType, Severity
from app.schemas.parcel import SourceRecord
from app.schemas.topology import TopologyIssueType, TopologyReport
from app.services.geospatial import crs_matches, overlap_metrics, to_geometry

DETECTOR = "deterministic_conflict_engine"

DEFAULT_CRITICAL_ATTRIBUTES = ("village", "district", "khasra")
DEFAULT_GEOMETRY_IOU_THRESHOLD = 0.9
DEFAULT_AREA_TOLERANCE = 0.05


class ConflictEngine:
    def __init__(
        self,
        critical_attributes: Sequence[str] = DEFAULT_CRITICAL_ATTRIBUTES,
        geometry_iou_threshold: float = DEFAULT_GEOMETRY_IOU_THRESHOLD,
        area_tolerance: float = DEFAULT_AREA_TOLERANCE,
    ) -> None:
        self.critical_attributes = tuple(critical_attributes)
        self.geometry_iou_threshold = geometry_iou_threshold
        self.area_tolerance = area_tolerance

    # ------------------------------------------------------------------
    def compare_records(
        self,
        record_a: SourceRecord,
        record_b: SourceRecord,
        parcel_id: str | None = None,
    ) -> ConflictSet:
        """Compare two source records that are believed to describe one parcel."""

        conflicts = ConflictSet(parcel_id=parcel_id)
        key_a = f"{record_a.source_system}:{record_a.source_record_id}"
        key_b = f"{record_b.source_system}:{record_b.source_record_id}"

        self._crs_conflict(conflicts, record_a, record_b, key_a, key_b, parcel_id)
        self._geometry_conflict(conflicts, record_a, record_b, key_a, key_b, parcel_id)
        self._identity_conflict(conflicts, record_a, record_b, key_a, key_b, parcel_id)
        self._attribute_conflicts(conflicts, record_a, record_b, key_a, key_b, parcel_id)
        self._missing_attributes(conflicts, record_a, key_a, parcel_id)
        self._missing_attributes(conflicts, record_b, key_b, parcel_id)
        return conflicts

    # ------------------------------------------------------------------
    def _crs_conflict(
        self,
        conflicts: ConflictSet,
        a: SourceRecord,
        b: SourceRecord,
        key_a: str,
        key_b: str,
        parcel_id: str | None,
    ) -> None:
        if a.geometry is None or b.geometry is None:
            return
        if a.crs is None or b.crs is None:
            conflicts.add(
                Conflict(
                    type=ConflictType.CRS_CONFLICT,
                    severity=Severity.HIGH,
                    source_a=key_a,
                    source_b=key_b,
                    parcel_id=parcel_id,
                    detector=DETECTOR,
                    evidence={"crs_a": a.crs, "crs_b": b.crs, "reason": "crs not declared"},
                )
            )
            return
        if not crs_matches(a.crs, b.crs):
            conflicts.add(
                Conflict(
                    type=ConflictType.CRS_CONFLICT,
                    severity=Severity.MEDIUM,
                    source_a=key_a,
                    source_b=key_b,
                    parcel_id=parcel_id,
                    detector=DETECTOR,
                    evidence={
                        "crs_a": a.crs,
                        "crs_b": b.crs,
                        "reason": "geometries must be reprojected before comparison",
                    },
                )
            )

    def _geometry_conflict(
        self,
        conflicts: ConflictSet,
        a: SourceRecord,
        b: SourceRecord,
        key_a: str,
        key_b: str,
        parcel_id: str | None,
    ) -> None:
        if a.geometry is None or b.geometry is None:
            return
        if not crs_matches(a.crs, b.crs):
            return  # already reported as a CRS conflict; comparison would be invalid
        try:
            geom_a = to_geometry(a.geometry)
            geom_b = to_geometry(b.geometry)
        except GeometryError as exc:
            conflicts.add(
                Conflict(
                    type=ConflictType.TOPOLOGY_ERROR,
                    severity=Severity.HIGH,
                    source_a=key_a,
                    source_b=key_b,
                    parcel_id=parcel_id,
                    detector=DETECTOR,
                    evidence={"error": str(exc)},
                )
            )
            return

        metrics = overlap_metrics(geom_a.buffer(0), geom_b.buffer(0))
        if metrics["iou"] < self.geometry_iou_threshold:
            conflicts.add(
                Conflict(
                    type=ConflictType.GEOMETRY_CONFLICT,
                    severity=Severity.HIGH if metrics["iou"] < 0.5 else Severity.MEDIUM,
                    source_a=key_a,
                    source_b=key_b,
                    parcel_id=parcel_id,
                    detector=DETECTOR,
                    confidence=None,
                    evidence={
                        **metrics,
                        "crs": a.crs,
                        "iou_threshold": self.geometry_iou_threshold,
                    },
                )
            )

    def _identity_conflict(
        self,
        conflicts: ConflictSet,
        a: SourceRecord,
        b: SourceRecord,
        key_a: str,
        key_b: str,
        parcel_id: str | None,
    ) -> None:
        if a.ulpin and b.ulpin and a.ulpin != b.ulpin:
            conflicts.add(
                Conflict(
                    type=ConflictType.IDENTITY_CONFLICT,
                    severity=Severity.CRITICAL,
                    source_a=key_a,
                    source_b=key_b,
                    parcel_id=parcel_id,
                    field="ulpin",
                    detector=DETECTOR,
                    evidence={"ulpin_a": a.ulpin, "ulpin_b": b.ulpin},
                )
            )
        if (
            a.source_parcel_id
            and b.source_parcel_id
            and a.source_system == b.source_system
            and a.source_parcel_id != b.source_parcel_id
        ):
            conflicts.add(
                Conflict(
                    type=ConflictType.IDENTITY_CONFLICT,
                    severity=Severity.HIGH,
                    source_a=key_a,
                    source_b=key_b,
                    parcel_id=parcel_id,
                    field="source_parcel_id",
                    detector=DETECTOR,
                    evidence={
                        "source_parcel_id_a": a.source_parcel_id,
                        "source_parcel_id_b": b.source_parcel_id,
                    },
                )
            )

    def _attribute_conflicts(
        self,
        conflicts: ConflictSet,
        a: SourceRecord,
        b: SourceRecord,
        key_a: str,
        key_b: str,
        parcel_id: str | None,
    ) -> None:
        attrs_a = a.attributes.as_record()
        attrs_b = b.attributes.as_record()
        for field in sorted(set(attrs_a) & set(attrs_b)):
            value_a = attrs_a[field]
            value_b = attrs_b[field]
            if value_a in (None, "") or value_b in (None, ""):
                continue
            if field == "area":
                self._area_conflict(
                    conflicts, value_a, value_b, key_a, key_b, parcel_id, attrs_a, attrs_b
                )
                continue
            if _normalise(value_a) != _normalise(value_b):
                conflicts.add(
                    Conflict(
                        type=ConflictType.ATTRIBUTE_CONFLICT,
                        severity=(
                            Severity.HIGH
                            if field in self.critical_attributes
                            else Severity.LOW
                        ),
                        source_a=key_a,
                        source_b=key_b,
                        parcel_id=parcel_id,
                        field=field,
                        detector=DETECTOR,
                        evidence={"value_a": value_a, "value_b": value_b},
                    )
                )

    def _area_conflict(
        self,
        conflicts: ConflictSet,
        value_a: Any,
        value_b: Any,
        key_a: str,
        key_b: str,
        parcel_id: str | None,
        attrs_a: dict[str, Any],
        attrs_b: dict[str, Any],
    ) -> None:
        try:
            area_a = float(value_a)
            area_b = float(value_b)
        except (TypeError, ValueError):
            return
        unit_a = attrs_a.get("area_unit")
        unit_b = attrs_b.get("area_unit")
        if unit_a and unit_b and unit_a != unit_b:
            conflicts.add(
                Conflict(
                    type=ConflictType.ATTRIBUTE_CONFLICT,
                    severity=Severity.MEDIUM,
                    source_a=key_a,
                    source_b=key_b,
                    parcel_id=parcel_id,
                    field="area_unit",
                    detector=DETECTOR,
                    evidence={"unit_a": unit_a, "unit_b": unit_b},
                )
            )
            return
        denominator = max(abs(area_a), abs(area_b))
        if denominator == 0:
            return
        relative = abs(area_a - area_b) / denominator
        if relative > self.area_tolerance:
            conflicts.add(
                Conflict(
                    type=ConflictType.ATTRIBUTE_CONFLICT,
                    severity=Severity.MEDIUM,
                    source_a=key_a,
                    source_b=key_b,
                    parcel_id=parcel_id,
                    field="area",
                    detector=DETECTOR,
                    evidence={
                        "area_a": area_a,
                        "area_b": area_b,
                        "relative_difference": relative,
                        "tolerance": self.area_tolerance,
                        "unit": unit_a or unit_b,
                    },
                )
            )

    def _missing_attributes(
        self,
        conflicts: ConflictSet,
        record: SourceRecord,
        key: str,
        parcel_id: str | None,
    ) -> None:
        attributes = record.attributes.as_record()
        for field in self.critical_attributes:
            if not attributes.get(field):
                conflicts.add(
                    Conflict(
                        type=ConflictType.MISSING_ATTRIBUTE,
                        severity=Severity.MEDIUM,
                        source_a=key,
                        parcel_id=parcel_id,
                        field=field,
                        detector=DETECTOR,
                        evidence={"reason": "critical attribute not supplied"},
                    )
                )

    # ------------------------------------------------------------------
    def from_topology_report(
        self, report: TopologyReport, source: str, parcel_id: str | None = None
    ) -> ConflictSet:
        conflicts = ConflictSet(parcel_id=parcel_id)
        for issue in report.issues:
            conflict_type = (
                ConflictType.GEOMETRY_CONFLICT
                if issue.type == TopologyIssueType.OVERLAP
                else ConflictType.TOPOLOGY_ERROR
            )
            conflicts.add(
                Conflict(
                    type=conflict_type,
                    severity=Severity(issue.severity)
                    if issue.severity in Severity._value2member_map_
                    else Severity.MEDIUM,
                    source_a=source,
                    source_b=None,
                    parcel_id=parcel_id,
                    detector="topology_service",
                    evidence={
                        "issue_type": issue.type.value,
                        "message": issue.message,
                        "feature_ids": issue.feature_ids,
                        **issue.evidence,
                    },
                )
            )
        return conflicts


def _normalise(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())
