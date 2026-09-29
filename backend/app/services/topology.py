"""Deterministic topology validation and repair proposals."""

from __future__ import annotations

import datetime as _dt
from typing import Any, Iterable, Sequence

from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from app.core.errors import GeometryError
from app.schemas.topology import (
    RepairProposal,
    TopologyIssue,
    TopologyIssueType,
    TopologyReport,
)
from app.services.geospatial import (
    overlap_metrics,
    repair_geometry,
    to_geojson,
    to_geometry,
    validate_geometry,
)

DEFAULT_OVERLAP_TOLERANCE = 1e-9


class TopologyService:
    """Checks a feature collection for topology problems.

    Repairs are returned as proposals: the service never mutates the input
    geometry, so a reviewer can compare original and proposed geometry.
    """

    def __init__(
        self,
        overlap_tolerance: float = DEFAULT_OVERLAP_TOLERANCE,
        detect_gaps: bool = True,
    ) -> None:
        self.overlap_tolerance = overlap_tolerance
        self.detect_gaps = detect_gaps

    # ------------------------------------------------------------------
    def validate(
        self, features: Sequence[tuple[str, Any]], crs: str | None = None
    ) -> TopologyReport:
        """Validate ``(feature_id, geometry)`` pairs sharing a single CRS."""

        parsed: list[tuple[str, BaseGeometry]] = []
        issues: list[TopologyIssue] = []
        repairs: list[RepairProposal] = []
        checked = [
            "invalid_geometry",
            "self_intersection",
            "empty_geometry",
            "duplicate_geometry",
            "overlap",
        ]
        if self.detect_gaps:
            checked.append("gap")

        for feature_id, raw in features:
            try:
                geometry = to_geometry(raw)
            except GeometryError as exc:
                issues.append(
                    TopologyIssue(
                        type=TopologyIssueType.UNPARSEABLE_GEOMETRY,
                        severity="high",
                        feature_ids=[feature_id],
                        message=str(exc),
                    )
                )
                continue

            parsed.append((feature_id, geometry))
            issues.extend(self._single_feature_issues(feature_id, geometry))
            proposal = self._repair_proposal(feature_id, geometry, crs)
            if proposal is not None:
                repairs.append(proposal)

        issues.extend(self._duplicate_issues(parsed))
        issues.extend(self._overlap_issues(parsed))
        if self.detect_gaps:
            issues.extend(self._gap_issues(parsed))

        return TopologyReport(
            feature_count=len(features),
            checked=checked,
            issues=issues,
            repairs=repairs,
        )

    # ------------------------------------------------------------------
    def _single_feature_issues(
        self, feature_id: str, geometry: BaseGeometry
    ) -> list[TopologyIssue]:
        issues: list[TopologyIssue] = []
        report = validate_geometry(geometry)

        if report["is_empty"]:
            issues.append(
                TopologyIssue(
                    type=TopologyIssueType.EMPTY_GEOMETRY,
                    severity="high",
                    feature_ids=[feature_id],
                    message="geometry is empty",
                    evidence=report,
                )
            )
            return issues

        if not report["is_valid"]:
            reason = report["reason"] or "invalid geometry"
            issue_type = (
                TopologyIssueType.SELF_INTERSECTION
                if "self-intersection" in reason.lower()
                else TopologyIssueType.INVALID_GEOMETRY
            )
            issues.append(
                TopologyIssue(
                    type=issue_type,
                    severity="high",
                    feature_ids=[feature_id],
                    message=reason,
                    evidence=report,
                )
            )
        return issues

    def _repair_proposal(
        self, feature_id: str, geometry: BaseGeometry, crs: str | None
    ) -> RepairProposal | None:
        if geometry.is_valid or geometry.is_empty:
            return None
        try:
            repaired, operation = repair_geometry(geometry)
        except GeometryError as exc:
            return RepairProposal(
                feature_id=feature_id,
                operation="none",
                reason=f"no safe repair available: {exc}",
                original_geometry=to_geojson(geometry),
                proposed_geometry=None,
                provenance=self._provenance(crs),
            )

        area_before = float(geometry.buffer(0).area)
        area_after = float(repaired.area)
        area_delta = area_after - area_before
        return RepairProposal(
            feature_id=feature_id,
            operation=operation,
            reason=validate_geometry(geometry)["reason"] or "invalid geometry",
            original_geometry=to_geojson(geometry),
            proposed_geometry=to_geojson(repaired),
            confidence=None,
            metrics={
                "area_before": area_before,
                "area_after": area_after,
                "area_delta": area_delta,
                "relative_area_change": (
                    abs(area_delta) / area_before if area_before else None
                ),
                "geometry_type_changed": geometry.geom_type != repaired.geom_type,
            },
            provenance=self._provenance(crs),
        )

    @staticmethod
    def _provenance(crs: str | None) -> dict[str, Any]:
        return {
            "actor_type": "rule",
            "actor": "topology_service",
            "library": "shapely",
            "crs": crs,
            "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }

    def _duplicate_issues(
        self, parsed: Sequence[tuple[str, BaseGeometry]]
    ) -> list[TopologyIssue]:
        issues: list[TopologyIssue] = []
        seen: dict[str, str] = {}
        for feature_id, geometry in parsed:
            if geometry.is_empty:
                continue
            key = geometry.normalize().wkb_hex
            previous = seen.get(key)
            if previous is not None:
                issues.append(
                    TopologyIssue(
                        type=TopologyIssueType.DUPLICATE_GEOMETRY,
                        severity="medium",
                        feature_ids=[previous, feature_id],
                        message="identical geometry present more than once",
                    )
                )
            else:
                seen[key] = feature_id
        return issues

    def _overlap_issues(
        self, parsed: Sequence[tuple[str, BaseGeometry]]
    ) -> list[TopologyIssue]:
        issues: list[TopologyIssue] = []
        for index, (id_a, geom_a) in enumerate(parsed):
            if geom_a.is_empty or geom_a.area == 0:
                continue
            for id_b, geom_b in parsed[index + 1 :]:
                if geom_b.is_empty or geom_b.area == 0:
                    continue
                if not geom_a.intersects(geom_b):
                    continue
                metrics = overlap_metrics(geom_a.buffer(0), geom_b.buffer(0))
                if metrics["intersection_area"] <= self.overlap_tolerance:
                    continue
                issues.append(
                    TopologyIssue(
                        type=TopologyIssueType.OVERLAP,
                        severity="high" if metrics["iou"] > 0.1 else "medium",
                        feature_ids=[id_a, id_b],
                        message="parcel geometries overlap",
                        evidence=metrics,
                    )
                )
        return issues

    def _gap_issues(
        self, parsed: Sequence[tuple[str, BaseGeometry]]
    ) -> list[TopologyIssue]:
        """Detect interior holes in the union of an otherwise contiguous block."""

        polygons = [
            geom.buffer(0)
            for _, geom in parsed
            if not geom.is_empty and geom.geom_type in ("Polygon", "MultiPolygon")
        ]
        if len(polygons) < 2:
            return []

        merged = unary_union(polygons)
        issues: list[TopologyIssue] = []
        for polygon in _iter_polygons(merged):
            for interior in polygon.interiors:
                from shapely.geometry import Polygon

                hole = Polygon(interior)
                issues.append(
                    TopologyIssue(
                        type=TopologyIssueType.GAP,
                        severity="medium",
                        feature_ids=[fid for fid, geom in parsed if geom.intersects(hole)],
                        message="gap enclosed by adjacent parcels",
                        evidence={
                            "area": float(hole.area),
                            "bounds": list(hole.bounds),
                        },
                    )
                )
        return issues


def _iter_polygons(geometry: BaseGeometry) -> Iterable[Any]:
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in geometry.geoms:
            if part.geom_type == "Polygon":
                yield part
