# SpatialShiftAI

Automated integration and intelligent harmonization of multi-source geospatial
data for urban land record management (SIH 2026, problem statement 26013).

This repository contains the FastAPI backend, canonical vector ingestion,
deterministic harmonization services, and a browser dashboard served by the
backend. Model endpoints never fabricate predictions: unavailable or
incompatible artifacts are reported with their actual readiness state.

## Layout

```
backend/
  app/
    api/routes/     FastAPI routers (health, models, data, upload, harmonize, ULPIN)
    core/           settings, logging, error types
    ml/             model manifest, adapters for models 1-5, registry, orchestrator
    schemas/        parcel, provenance, conflict, topology contracts
    services/       geospatial, ingestion, topology, conflict, confidence engines
  tests/            pytest suite (small generated fixtures; no large weights required)
models/             trained artifacts, loaded at runtime, never committed
data/               sample inputs and JSON schemas
database/           migrations (Phase 2)
docs/               architecture, model and API notes
frontend/           responsive Leaflet dashboard (served at `/`)
```

## Local Setup (Windows)

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r backend/requirements.txt
copy .env.example .env
.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --reload --port 8000
```

Open `http://localhost:8000/` for the dashboard or `http://localhost:8000/docs`
for OpenAPI. The dashboard uses the current origin as its API URL by default.

## Docker / Render

```bash
docker compose up --build
```

- `GET /api/health` – process liveness, environment, resolved device and model root
- `GET /api/models` – the artifact manifest and per-model artifact presence
- `GET /api/models/health` – real artifact inspection, and by default a real load attempt
- `GET /api/models/{model_key}/health` – the same for a single model
- `POST /api/data/upload` – bounded vector upload, validation and source registration
- `GET /api/data/sources`, `/api/layers` – registered sources and map layers
- `POST /api/harmonize` – deterministic identifier linkage, unresolved conflicts and provenance
- `GET /api/ulpin/{id}` – local uploaded-source lookup or explicit unconfigured-service status
- `POST /api/upload` – legacy vector ingestion route
- `GET /api/models/pipeline` – the declared pipeline stages and whether each can run
- `GET /api/version` – build/runtime metadata
- `GET /docs` – OpenAPI UI

## Tests

```bash
cd backend
../.venv/bin/pytest
```

The suite builds its own scikit-learn and torchvision checkpoints in temporary
directories so the adapters are exercised end to end without the trained SIH
artifacts. Those synthetic checkpoints say nothing about model quality; they
only verify the loading, preprocessing and output contracts.

## Model artifacts

Artifacts are **not** stored in Git. Set `MODEL_ROOT` to the directory that
holds them; it defaults to `./models`. Expected layout:

```
$MODEL_ROOT/
  model1_parcel_matcher/     parcel_siamese_v2_best.pt [+ calibrator, threshold, preprocessing]
  model2_building_extractor/ building_deeplabv3_resnet50_best.pt [+ preprocessing]
  model3_change_detection/   siamese_resnet18_change_best.pt [+ preprocessing]
  model4_entity_resolution/  model4_entity_resolver_logistic.joblib, model4_feature_scaler.joblib,
                             model4_feature_schema.json [+ calibrator, threshold]
  model5_anomaly_detection/  model5_anomaly_classifier.joblib, model5_feature_schema.json
                             [+ scaler, calibrator, threshold]
```

`GET /api/models/health` reports one of `ready`, `missing_artifact`,
`load_error` or `unsupported_runtime` per model. A model is only `ready` after
its artifact has actually been loaded in this process.

See [docs/models.md](docs/models.md) for per-model contracts and limitations.
The recovered Model 1 weights strictly load using their 5-wide Conv1d weight
shapes, but inference remains blocked because preprocessing and feature
construction are unavailable. The recovered Model 4 bundle is incompatible
with the current feature parser. Models 2, 3 and 5 have no recovered inference
artifacts. Do not interpret controlled benchmark metrics as deployment
accuracy.

## Canonical geospatial model and ingestion

The canonical parcel schema supports source metadata, CRS, geometry, parcel
identity fields, source timestamps, quality flags, and provenance. It keeps
missing values explicit and never fabricates source values.

Supported vector sources include GeoJSON, GeoPackage, Shapefile ZIP, Parquet,
CSV with latitude/longitude columns and KML where the installed GDAL/Fiona
driver supports it. Uploads to `/api/data/upload` are limited to 64 MiB (ZIP
expanded content to 128 MiB). The ingestion layer validates geometry, detects
CRS when possible, normalizes attributes, keeps raw properties, records
invalid rows without silently discarding them, and returns statistics and
provenance. Unknown-CRS geometries are withheld from geographic map display.

Registered sources and harmonization jobs are held in process memory and are
lost on backend restart. Harmonization links exact normalized ULPINs or complete
district/village/tehsil/khasra keys; disagreements remain unresolved and no
source is silently preferred. It is deterministic and does not substitute for
unavailable ML inference.

The dashboard displays uploaded geometries on a Leaflet map with OpenStreetMap
tiles, layer visibility, fit-to-bounds, local ULPIN lookup, and source/export
controls. It uses only records uploaded to the backend; no sample features are
hardcoded.

## Production model runtime

The backend exposes the five model adapters through the registry. Each resolves
artifacts under `MODEL_ROOT`; model weights remain external to Git. The current
recovered bundle does not make any model inference-ready, and readiness checks
report the specific missing or incompatible requirements.

Model keys and expected directories:

- `parcel_matcher` → `MODEL_ROOT/model1_parcel_matcher`
- `building_extractor` → `MODEL_ROOT/model2_building_extractor`
- `change_detector` → `MODEL_ROOT/model3_change_detection`
- `entity_resolver` → `MODEL_ROOT/model4_entity_resolution`
- `anomaly_detector` → `MODEL_ROOT/model5_anomaly_detection`

The runtime contract is explicit and safe:

- missing artifacts remain `missing_artifact`
- a file path alone is not treated as readiness
- a model can fail to load without crashing the rest of the API
- inference never returns fabricated predictions when a model is unavailable

Inference endpoints are exposed under the model registry:

- `POST /api/models/{model_key}/infer`
- `POST /api/models/{model_key}/predict`

The same route family is used for the parcel matcher, building extractor,
change detector, entity resolver and anomaly detector.

Benchmark figures and their evaluation-set scope are summarized in
[docs/models.md](docs/models.md). They are training benchmarks, not deployment
performance.

## Documentation

- [docs/architecture.md](docs/architecture.md)
- [docs/models.md](docs/models.md)
- [docs/api.md](docs/api.md)
