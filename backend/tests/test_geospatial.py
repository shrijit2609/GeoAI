from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from app.core.errors import CRSError, GeometryError
from app.services import geospatial

SQUARE = "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))"
BOWTIE = "POLYGON ((0 0, 10 10, 10 0, 0 10, 0 0))"


def test_parse_wkt_and_geojson():
    from_wkt = geospatial.to_geometry(SQUARE)
    from_geojson = geospatial.to_geometry(geospatial.to_geojson(from_wkt))
    assert from_wkt.equals(from_geojson)


def test_invalid_geometry_input_raises():
    with pytest.raises(GeometryError):
        geospatial.to_geometry("not a geometry")
    with pytest.raises(GeometryError):
        geospatial.to_geometry(42)


def test_describe_crs():
    described = geospatial.describe_crs("EPSG:4326")
    assert described["epsg"] == 4326
    assert described["is_geographic"] is True


def test_unknown_crs_raises():
    with pytest.raises(CRSError):
        geospatial.describe_crs("EPSG:999999")


def test_transform_wgs84_to_utm_and_back():
    point = "POINT (77.2090 28.6139)"  # Delhi
    utm = geospatial.transform_geometry(point, "EPSG:4326", "EPSG:32643")
    assert 700000 < utm.x < 730000
    assert 3_100_000 < utm.y < 3_200_000

    back = geospatial.transform_geometry(utm, "EPSG:32643", "EPSG:4326")
    assert back.x == pytest.approx(77.2090, abs=1e-6)
    assert back.y == pytest.approx(28.6139, abs=1e-6)


def test_transform_is_a_noop_for_identical_crs():
    geom = geospatial.transform_geometry(SQUARE, "EPSG:4326", "EPSG:4326")
    assert geom.equals(geospatial.to_geometry(SQUARE))


def test_crs_matches_handles_aliases():
    assert geospatial.crs_matches("EPSG:4326", "epsg:4326")
    assert not geospatial.crs_matches("EPSG:4326", "EPSG:32643")
    assert not geospatial.crs_matches(None, "EPSG:4326")


def test_validate_and_repair_self_intersection():
    report = geospatial.validate_geometry(BOWTIE)
    assert report["is_valid"] is False
    assert "self-intersection" in report["reason"].lower()

    repaired, operation = geospatial.repair_geometry(BOWTIE)
    assert repaired.is_valid
    assert operation in ("make_valid", "buffer_zero")


def test_repair_is_noop_for_valid_geometry():
    repaired, operation = geospatial.repair_geometry(SQUARE)
    assert operation == "none"
    assert repaired.equals(geospatial.to_geometry(SQUARE))


def test_overlap_metrics_and_distance():
    a = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    b = Polygon([(5, 0), (5, 10), (15, 10), (15, 0)])
    metrics = geospatial.overlap_metrics(a, b)
    assert metrics["intersection_area"] == pytest.approx(50.0)
    assert metrics["iou"] == pytest.approx(50 / 150)
    assert metrics["overlap_fraction_a"] == pytest.approx(0.5)

    far = Polygon([(100, 100), (100, 110), (110, 110), (110, 100)])
    assert geospatial.distance(a, far) > 0
    assert geospatial.intersects(a, b)
    assert not geospatial.intersects(a, far)
    assert geospatial.contains(a, Polygon([(1, 1), (1, 2), (2, 2), (2, 1)]))


def test_bounding_boxes():
    assert geospatial.bounding_box(SQUARE) == {
        "minx": 0.0,
        "miny": 0.0,
        "maxx": 10.0,
        "maxy": 10.0,
    }
    combined = geospatial.total_bounds([SQUARE, "POLYGON ((20 20, 20 30, 30 30, 20 20))"])
    assert combined["maxx"] == 30.0
    assert geospatial.total_bounds([]) is None
