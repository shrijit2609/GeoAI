from __future__ import annotations

import geopandas as gpd
import pytest
from pathlib import Path
from shapely.geometry import Polygon

from app.core.errors import InvalidInputError
from app.services.ingestion import ingest_file, ingest_payload


@pytest.fixture
def geojson_payload() -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "parcel-001",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[77.0, 28.0], [77.1, 28.0], [77.1, 28.1], [77.0, 28.1], [77.0, 28.0]]
                    ],
                },
                "properties": {
                    "village": "Rampur",
                    "district": "Agra",
                    "tehsil": "Sikandra",
                    "block": "Block A",
                    "khasra": "12/3",
                    "parcel_type": "agricultural",
                    "area": 1250.0,
                    "status": "active",
                    "remarks": "inspection pending",
                },
            },
            {
                "type": "Feature",
                "id": "parcel-002",
                "geometry": {
                    "type": "Point",
                    "coordinates": [77.03, 28.04],
                },
                "properties": {"village": "Rampur"},
            },
        ],
    }


def test_geojson_ingestion_normalizes_and_tracks_provenance(geojson_payload: dict):
    result = ingest_payload(
        geojson_payload,
        source_name="revenue_source",
        source_type="revenue_record",
        project_crs="EPSG:32643",
    )

    assert result["source_name"] == "revenue_source"
    assert result["total_records"] == 2
    assert result["valid_records"] == 2
    assert result["invalid_records"] == 0
    assert result["detected_crs"] == "EPSG:4326"
    assert result["geometry_types"]
    assert result["provenance"]["source_name"] == "revenue_source"

    first = result["records"][0]
    assert first["source_id"] == "parcel-001"
    assert first["village"] == "Rampur"
    assert first["district"] == "Agra"
    assert first["tehsil"] == "Sikandra"
    assert first["khasra"] == "12/3"
    assert first["source_timestamp"] is None
    assert first["ingestion_timestamp"]
    assert first["provenance"]["source_record_id"] == "parcel-001"


def test_missing_optional_fields_remain_null():
    payload = {"type": "FeatureCollection", "features": [
        {
            "type": "Feature",
            "id": "parcel-003",
            "geometry": {"type": "Polygon", "coordinates": [[[0,0],[1,0],[1,1],[0,1],[0,0]]]},
            "properties": {"village": "Alpha"},
        }
    ]}

    result = ingest_payload(payload, source_name="municipal", project_crs="EPSG:3857")
    record = result["records"][0]
    assert record["district"] is None
    assert record["tehsil"] is None
    assert record["khasra"] is None
    assert record["area"] is None


def test_invalid_and_empty_geometry_are_reported():
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "bad-1",
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 0], [0, 0], [0, 0]]]},
                "properties": {"village": "Bad"},
            },
            {
                "type": "Feature",
                "id": "empty-1",
                "geometry": {"type": "MultiPoint", "coordinates": []},
                "properties": {"village": "Empty"},
            },
        ],
    }

    result = ingest_payload(payload, source_name="invalid_demo")
    assert result["total_records"] == 2
    assert result["valid_records"] == 0
    assert result["invalid_records"] == 2
    assert result["errors"]


def test_crs_detection_and_reprojection():
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "parcel-4326",
                "geometry": {"type": "Point", "coordinates": [77.0, 28.0]},
                "properties": {"village": "Sample"},
            }
        ],
    }
    result = ingest_payload(payload, source_name="gnss", project_crs="EPSG:3857")
    record = result["records"][0]
    assert result["detected_crs"] == "EPSG:4326"
    assert record["crs"] == "EPSG:3857"
    assert record["geometry"]


def test_unsupported_format_raises():
    path = Path("does_not_exist.txt")
    with pytest.raises(InvalidInputError):
        ingest_file(path, source_name="bad")


def test_shapefile_ingestion_works(tmp_path: Path):
    shapefile_path = tmp_path / "demo_shp.shp"
    gdf = gpd.GeoDataFrame(
        {
            "village": ["North"],
            "district": ["Lucknow"],
            "khasra": ["88/11"],
        },
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])],
        crs="EPSG:4326",
    )
    gdf.to_file(shapefile_path, driver="ESRI Shapefile")

    result = ingest_file(shapefile_path, source_name="shp_demo")
    assert result["total_records"] == 1
    assert result["valid_records"] == 1
    assert result["invalid_records"] == 0
    assert result["detected_crs"] == "EPSG:4326"
    assert result["records"][0]["village"] == "North"
