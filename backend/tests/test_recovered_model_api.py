from __future__ import annotations

from pathlib import Path


def test_recovered_models_are_ready_and_infer_through_http(monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import reset_settings_cache
    from app.ml.model_registry import reset_registry
    from app.main import create_app

    repository_root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("MODEL_ROOT", str(repository_root / "models"))
    monkeypatch.setenv("SPATIALSHIFT_DEVICE", "cpu")
    reset_settings_cache()
    reset_registry()

    with TestClient(create_app()) as client:
        readiness = client.get("/api/models/readiness").json()
        assert readiness["models"]["parcel_matcher"]["status_group"] == "READY"
        assert readiness["models"]["entity_resolver"]["status_group"] == "READY"
        assert readiness["models"]["anomaly_detector"]["status_group"] == "READY"
        assert readiness["models"]["building_extractor"]["status_group"] == "TRAINING_REQUIRED"
        assert readiness["models"]["change_detector"]["status_group"] == "TRAINING_REQUIRED"

        parcel = client.post(
            "/api/models/parcel_matcher/infer",
            json={
                "parcel_a": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 8], [0, 8], [0, 0]]]},
                "parcel_b": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 8], [0, 8], [0, 0]]]},
            },
        )
        assert parcel.status_code == 200
        assert parcel.json()["status"] == "success"
        assert parcel.json()["evidence"]["preprocessing"]["boundary_points"] == 64

        pair = {
            "record_a": {"district": "Agra", "village": "Sultanpur", "khasra": "121/2", "tehsil": "Sikandra", "block": "A", "pargana": "P", "area": 1000},
            "record_b": {"district": "Agra", "village": "Sultanpur", "khasra": "121/2", "tehsil": "Sikandra", "block": "A", "pargana": "P", "area": 1010},
        }
        resolver = client.post("/api/models/entity_resolver/infer", json=pair)
        assert resolver.status_code == 200
        assert resolver.json()["status"] == "success"
        assert len(resolver.json()["evidence"]["features"]) == 104
        assert resolver.json()["evidence"]["threshold"] == 0.05

        anomaly = client.post(
            "/api/anomaly/infer",
            json={"record": {"district": "Agra", "village": "Sultanpur", "khasra": "121/2", "pargana": "P", "tehsil": "Sikandra", "block": "A", "area": 1000}},
        )
        assert anomaly.status_code == 200
        assert anomaly.json()["status"] == "success"
        assert len(anomaly.json()["evidence"]["features"]) == 54

    reset_registry()
    reset_settings_cache()
