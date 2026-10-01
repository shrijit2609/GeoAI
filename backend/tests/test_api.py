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
    assert all(artifact["present"] is False for artifact in parcel["artifacts"])


def test_models_health_reports_missing_artifacts(api_client):
    payload = api_client.get("/api/models/health").json()
    assert payload["ready"] == 0
    for key, health in payload["models"].items():
        assert health["status"] == "missing_artifact", key
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
