from __future__ import annotations


def _payload(ulpin: str, district: str, parcel_id: str) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": parcel_id,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[77, 28], [77.01, 28], [77.01, 28.01], [77, 28.01], [77, 28]]],
                },
                "properties": {
                    "ulpin": ulpin,
                    "district": district,
                    "village": "Rampur",
                    "tehsil": "Sikandra",
                    "khasra": "12/3",
                },
            }
        ],
    }


def test_upload_registers_map_layer_export_and_local_ulpin(api_client):
    response = api_client.post(
        "/api/data/upload",
        data={"payload": __import__("json").dumps(_payload("UP-123", "Agra", "parcel-a")), "source_name": "revenue"},
    )
    assert response.status_code == 200
    source_id = response.json()["source_id"]
    assert api_client.get("/api/data/sources").json()["count"] == 1

    layer = api_client.get(f"/api/data/layers/{source_id}")
    assert layer.status_code == 200
    assert layer.json()["features"][0]["properties"]["ulpin"] == "UP-123"

    exported = api_client.get(f"/api/data/{source_id}/export?format=geojson")
    assert exported.status_code == 200
    assert exported.json()["features"][0]["id"] == "parcel-a"

    lookup = api_client.get("/api/ulpin/UP-123").json()
    assert lookup["status"] == "local_match"
    assert lookup["records"][0]["source_id"] == source_id


def test_harmonization_links_only_exact_ids_and_preserves_conflicts(api_client):
    source_ids = []
    for name, district, record_id in (("municipal", "Agra", "a"), ("revenue", "Mathura", "b")):
        response = api_client.post(
            "/api/data/upload",
            data={"payload": __import__("json").dumps(_payload("UP-456", district, record_id)), "source_name": name},
        )
        assert response.status_code == 200
        source_ids.append(response.json()["source_id"])

    result = api_client.post("/api/harmonize", json={"source_ids": source_ids})
    assert result.status_code == 200
    job = result.json()
    assert job["matching_method"] == "exact_normalized_identifier"
    assert job["model_inference_used"] is False
    assert job["conflict_count"] == 1
    assert job["conflicts"][0]["field"] == "district"
    assert job["conflicts"][0]["selected_value"] is None
    assert job["feature_collection"]["features"][0]["properties"]["district"] is None
    persisted = api_client.get(f"/api/harmonize/{job['job_id']}")
    assert persisted.status_code == 200


def test_ulpin_unavailable_is_not_fabricated(api_client):
    result = api_client.get("/api/ulpin/NOT-IN-LOCAL-DATA").json()
    assert result["status"] == "not_found_locally"
    assert result["records"] == []
    assert result["external_service"] == "not_configured"


def test_model_readiness_alias_reports_actual_registry(api_client):
    response = api_client.get("/api/models/readiness")
    assert response.status_code == 200
    assert response.json()["total"] == 5


def test_top_level_layers_alias_returns_registered_layers(api_client):
    uploaded = api_client.post(
        "/api/data/upload",
        data={"payload": __import__("json").dumps(_payload("UP-789", "Agra", "layer-row"))},
    )
    source_id = uploaded.json()["source_id"]
    response = api_client.get("/api/layers")
    assert response.status_code == 200
    assert any(item["layer_id"] == source_id for item in response.json()["layers"])
    assert api_client.get(f"/api/layers/{source_id}").status_code == 200


def test_csv_upload_requires_or_records_explicit_source_crs(api_client, tmp_path):
    csv_path = tmp_path / "points.csv"
    csv_path.write_text("latitude,longitude,parcel_id\n28.0,77.0,P-1\n")
    with csv_path.open("rb") as handle:
        response = api_client.post(
            "/api/data/upload",
            files={"file": ("points.csv", handle, "text/csv")},
            data={"source_crs": "EPSG:4326"},
        )
    assert response.status_code == 200
    assert response.json()["records"][0]["crs"] == "EPSG:4326"


def test_unknown_crs_geometry_is_withheld_from_map(api_client):
    payload = _payload("UP-999", "Agra", "unknown-crs")
    payload["features"][0]["geometry"]["coordinates"] = [[[500000, 3000000], [500010, 3000000], [500010, 3000010], [500000, 3000010], [500000, 3000000]]]
    uploaded = api_client.post("/api/data/upload", data={"payload": __import__("json").dumps(payload)})
    source_id = uploaded.json()["source_id"]
    layer = api_client.get(f"/api/layers/{source_id}").json()
    assert layer["features"][0]["geometry"] is None
    assert any("CRS is unknown" in warning for warning in layer["metadata"]["warnings"])


def test_explicit_geojson_source_crs_enables_safe_map_reprojection(api_client):
    payload = _payload("UP-1000", "Agra", "projected-row")
    payload["features"][0]["geometry"]["coordinates"] = [[[500000, 3000000], [500010, 3000000], [500010, 3000010], [500000, 3000010], [500000, 3000000]]]
    response = api_client.post(
        "/api/data/upload",
        data={"payload": __import__("json").dumps(payload), "source_crs": "EPSG:32643"},
    )
    assert response.status_code == 200
    source_id = response.json()["source_id"]
    layer = api_client.get(f"/api/layers/{source_id}").json()
    assert layer["features"][0]["geometry"] is not None
    assert layer["metadata"]["display_crs"] == "EPSG:4326"