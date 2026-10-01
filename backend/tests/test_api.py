from __future__ import annotations


def test_health_reports_real_runtime_state(api_client):
    payload = api_client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["device"] == "cpu"
    assert payload["model_root_exists"] is True


def test_root_serves_the_frontend_dashboard(api_client):
    response = api_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Land Record Harmonization" in response.text
    assert api_client.get("/ui/app.js").status_code == 200


def test_version_lists_installed_libraries(api_client):
    payload = api_client.get("/api/version").json()
    assert payload["version"]
    assert "python" in payload["libraries"]


def test_models_endpoint_lists_expected_artifacts(api_client):
    payload = api_client.get("/api/models").json()
    assert payload["count"] == 5
    keys = [model["key"] for model in payload["models"]]
    assert "parcel_matcher" in keys
    parcel = next(m for m in payload["models"] if m["key"] == "parcel_matcher")
    names = [artifact["name"] for artifact in parcel["artifacts"]]
    assert "parcel_siamese_v2_best.pt" in names
    for field in ("model_key", "name", "purpose", "architecture_name", "artifact", "status", "live_inference_available", "readiness_reason", "version", "benchmark_metrics"):
        assert field in parcel
    assert all(artifact["present"] is False for artifact in parcel["artifacts"])


def test_models_health_reports_missing_artifacts(api_client):
    payload = api_client.get("/api/models/health").json()
    assert payload["ready"] == 0
    for key, health in payload["models"].items():
        expected = "training_required" if key in {
            "building_extractor", "change_detector", "anomaly_detector"
        } else "missing_artifact"
        assert health["status"] == expected, key
        assert health["missing_artifacts"]


def test_single_model_health(api_client):
    payload = api_client.get("/api/models/entity_resolver/health").json()
    assert payload["key"] == "entity_resolver"
    assert payload["status"] == "missing_artifact"


def test_unknown_model_returns_404(api_client):
    response = api_client.get("/api/models/not_a_model/health")
    assert response.status_code == 404


def test_pipeline_marks_stages_unavailable(api_client):
    stages = api_client.get("/api/models/pipeline").json()["stages"]
    assert len(stages) == 5
    assert all(stage["available"] is False for stage in stages)
    assert all(stage["blocker"] for stage in stages)


def test_inference_endpoint_reports_missing_artifacts_safely(api_client):
    response = api_client.post(
        "/api/models/entity_resolver/infer",
        json={"record_a": {"village": "A"}, "record_b": {"village": "B"}},
    )

    assert response.status_code == 503
    payload = response.json()
    assert payload["error"] == "missing_artifact"
    assert payload["model"] == "entity_resolver"


def test_predict_alias_uses_the_same_missing_artifact_contract(api_client):
    response = api_client.post(
        "/api/models/parcel_matcher/predict",
        json={"parcel_a": {"image": "a"}, "parcel_b": {"image": "b"}},
    )

    assert response.status_code == 503
    payload = response.json()
    assert payload["error"] == "missing_artifact"
    assert payload["model"] == "parcel_matcher"


def test_dashboard_summary_returns_zeroed_metrics_for_empty_catalog(api_client):
    payload = api_client.get("/api/dashboard/summary").json()
    assert payload["registered_sources"] == 0
    assert payload["total_features"] == 0
    assert payload["harmonized_parcels"] == 0
    assert payload["conflicts"] == 0
    assert payload["anomalies"] == 0
    assert payload["processing_jobs"] == 0
    assert payload["models_ready"] == 0


def test_compatibility_routes_expose_source_and_model_endpoints(api_client):
    sources = api_client.get("/api/sources").json()
    assert sources["count"] == 0
    layers = api_client.get("/api/map/layers").json()
    assert layers["layers"] == []
    readiness = api_client.get("/api/models/readiness").json()
    assert readiness["total"] == 5
    assert readiness["ready"] == 0
    assert readiness["models"]["building_extractor"]["status"] == "training_required"
    assert readiness["models"]["building_extractor"]["readiness_reason"]


def test_model_inference_compatibility_routes_use_the_registry(api_client):
    for path, payload in (
        ("/api/entity-resolution/infer", {"record_a": {}, "record_b": {}}),
        ("/api/anomaly/infer", {"record": {}}),
    ):
        response = api_client.post(path, json=payload)
        assert response.status_code == 503
        assert response.json()["error"] == "missing_artifact"


def test_sources_preview_register_detail_and_delete(api_client):
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"district": "Agra", "village": "Rampur", "khasra": "12"},
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            }
        ],
    }
    preview = api_client.post(
        "/api/sources/upload",
        files={"file": ("revenue.geojson", __import__("json").dumps(geojson), "application/geo+json")},
        data={"source_name": "Revenue demo", "source_type": "revenue_record", "source_crs": "EPSG:4326", "register": "false"},
    )
    assert preview.status_code == 200
    preview_payload = preview.json()
    assert preview_payload["registered"] is False
    assert preview_payload["valid_records"] == 1

    registered = api_client.post("/api/sources/register", json=preview_payload)
    assert registered.status_code == 200
    source_id = registered.json()["source_id"]
    assert api_client.get("/api/sources").json()["count"] == 1
    assert api_client.get(f"/api/sources/{source_id}").status_code == 200
    assert api_client.get("/api/map/layers").json()["layers"][0]["feature_count"] == 1
    deleted = api_client.delete(f"/api/sources/{source_id}")
    assert deleted.json()["deleted"] is True
    assert api_client.get(f"/api/sources/{source_id}").status_code == 404


def test_harmonization_returns_skipped_model_stages_and_provenance(api_client):
    source_ids = []
    for suffix, village in (("a", "Rampur"), ("b", "Ramnagar")):
        response = api_client.post(
            "/api/sources/register",
            json={
                "source_name": f"source-{suffix}",
                "source_type": "revenue_record",
                "detected_crs": "EPSG:4326",
                "records": [
                    {
                        "source_id": f"record-{suffix}",
                        "crs": "EPSG:4326",
                        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
                        "raw_attributes": {"ulpin": "UP-1", "district": "Agra", "village": village, "khasra": "12"},
                    }
                ],
            },
        )
        source_ids.append(response.json()["source_id"])

    job = api_client.post("/api/harmonization/run", json={"source_ids": source_ids}).json()
    assert job["status"] == "completed"
    assert job["stages"]["building_extraction"]["status"] == "SKIPPED"
    assert job["stages"]["change_detection"]["status"] == "SKIPPED"
    assert job["stages"]["building_extraction"]["reason"]
    properties = job["feature_collection"]["features"][0]["properties"]
    for field in (
        "source_id", "source_name", "source_type", "source_timestamp", "source_fields",
        "original_attributes", "model_evidence", "confidence", "conflicts",
        "resolution_reason", "processing_timestamp",
    ):
        assert field in properties
    assert job["conflicts"][0]["conflict_type"] == "attribute_conflict"
    assert job["conflicts"][0]["status"] == "Requires review"
    assert api_client.get(f"/api/harmonization/{job['job_id']}").status_code == 200
    assert api_client.get(f"/api/harmonization/{job['job_id']}/results").status_code == 200
    assert api_client.get(f"/api/exports/{job['job_id']}?format=conflicts").status_code == 200
    assert api_client.get(f"/api/exports/{job['job_id']}?format=provenance").status_code == 200
    for source_id in source_ids:
        assert api_client.delete(f"/api/sources/{source_id}").status_code == 200
