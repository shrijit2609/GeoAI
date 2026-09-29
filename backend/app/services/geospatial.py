"""Deterministic geospatial utilities.

These are rule based, not model based, and are used wherever an exact
geometric answer is required (CRS handling, validity, overlap, distance).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

from shapely import wkt
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.validation import explain_validity, make_valid

from app.core.errors import CRSError, GeometryError

GeometryInput = Any


def to_geometry(value: GeometryInput) -> BaseGeometry:
    """Accept WKT, GeoJSON mappings, __geo_interface__ objects or geometries."""

    if isinstance(value, BaseGeometry):
        return value
    if isinstance(value, str):
        try:
            return wkt.loads(value)
        except Exception as exc:  # noqa: BLE001
            raise GeometryError(f"could not parse WKT geometry: {exc}") from exc
    if isinstance(value, dict):
        payload = value.get("geometry", value)
        try:
            return shape(payload)
        except Exception as exc:  # noqa: BLE001
            raise GeometryError(f"could not parse GeoJSON geometry: {exc}") from exc
    if hasattr(value, "__geo_interface__"):
        return shape(value.__geo_interface__)
    raise GeometryError(f"unsupported geometry input: {type(value).__name__}")


def to_geojson(geometry: BaseGeometry) -> dict[str, Any]:
    return mapping(geometry)


# ----------------------------------------------------------------------
# CRS
# ----------------------------------------------------------------------
@lru_cache(maxsize=64)
def _crs(identifier: str):
    from pyproj import CRS

    try:
        return CRS.from_user_input(identifier)
    except Exception as exc:  # noqa: BLE001
        raise CRSError(f"unknown CRS '{identifier}': {exc}") from exc


def describe_crs(identifier: str) -> dict[str, Any]:
    crs = _crs(identifier)
    return {
        "input": identifier,
        "srs": crs.to_string(),
        "epsg": crs.to_epsg(),
        "name": crs.name,
        "is_geographic": bool(crs.is_geographic),
        "is_projected": bool(crs.is_projected),
        "unit": crs.axis_info[0].unit_name if crs.axis_info else None,
    }


@lru_cache(maxsize=64)
def _transformer(source: str, target: str):
    from pyproj import Transformer

    return Transformer.from_crs(_crs(source), _crs(target), always_xy=True)


def transform_geometry(
    geometry: GeometryInput, source_crs: str, target_crs: str
) -> BaseGeometry:
    """Reproject a geometry between two CRS, always in x/y (lon/lat) order."""

    geom = to_geometry(geometry)
    if source_crs == target_crs:
        return geom
    transformer = _transformer(source_crs, target_crs)
    try:
        return shapely_transform(
            lambda x, y, z=None: transformer.transform(x, y), geom
        )
    except Exception as exc:  # noqa: BLE001
        raise CRSError(
            f"failed to transform geometry from {source_crs} to {target_crs}: {exc}"
        ) from exc


def crs_matches(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return _crs(a) == _crs(b)
    except CRSError:
        return a.strip().lower() == b.strip().lower()


# ----------------------------------------------------------------------
# validity and repair
# ----------------------------------------------------------------------
def validate_geometry(geometry: GeometryInput) -> dict[str, Any]:
    geom = to_geometry(geometry)
    return {
        "geometry_type": geom.geom_type,
        "is_valid": bool(geom.is_valid),
        "is_empty": bool(geom.is_empty),
        "is_simple": bool(geom.is_simple),
        "reason": None if geom.is_valid else explain_validity(geom),
        "area": float(geom.area),
        "length": float(geom.length),
        "bounds": list(geom.bounds) if not geom.is_empty else None,
    }


def repair_geometry(geometry: GeometryInput) -> tuple[BaseGeometry, str]:
    """Return a repaired geometry and the operation used.

    Only safe Shapely operations are applied; the caller decides whether the
    proposal is accepted.
    """

    geom = to_geometry(geometry)
    if geom.is_valid:
        return geom, "none"

    repaired = make_valid(geom)
    if repaired.is_valid and not repaired.is_empty:
        return repaired, "make_valid"

    buffered = geom.buffer(0)
    if buffered.is_valid and not buffered.is_empty:
        return buffered, "buffer_zero"

    raise GeometryError("geometry could not be repaired with safe operations")


# ----------------------------------------------------------------------
# relations and measurements
# ----------------------------------------------------------------------
def intersects(a: GeometryInput, b: GeometryInput) -> bool:
    return to_geometry(a).intersects(to_geometry(b))


def contains(a: GeometryInput, b: GeometryInput) -> bool:
    return to_geometry(a).contains(to_geometry(b))


def intersection(a: GeometryInput, b: GeometryInput) -> BaseGeometry:
    return to_geometry(a).intersection(to_geometry(b))


def overlap_metrics(a: GeometryInput, b: GeometryInput) -> dict[str, float]:
    """Area overlap between two geometries in their shared CRS units."""

    geom_a = to_geometry(a)
    geom_b = to_geometry(b)
    inter = geom_a.intersection(geom_b)
    union = geom_a.union(geom_b)
    inter_area = float(inter.area)
    return {
        "intersection_area": inter_area,
        "union_area": float(union.area),
        "area_a": float(geom_a.area),
        "area_b": float(geom_b.area),
        "iou": float(inter_area / union.area) if union.area else 0.0,
        "overlap_fraction_a": float(inter_area / geom_a.area) if geom_a.area else 0.0,
        "overlap_fraction_b": float(inter_area / geom_b.area) if geom_b.area else 0.0,
    }


def distance(a: GeometryInput, b: GeometryInput) -> float:
    return float(to_geometry(a).distance(to_geometry(b)))


def bounding_box(geometry: GeometryInput) -> dict[str, float]:
    minx, miny, maxx, maxy = to_geometry(geometry).bounds
    return {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}


def total_bounds(geometries: Iterable[GeometryInput]) -> dict[str, float] | None:
    boxes = [to_geometry(geom).bounds for geom in geometries]
    if not boxes:
        return None
    return {
        "minx": min(box[0] for box in boxes),
        "miny": min(box[1] for box in boxes),
        "maxx": max(box[2] for box in boxes),
        "maxy": max(box[3] for box in boxes),
    }
