# SpatialShiftAI

Automated integration and intelligent harmonization of multi-source geospatial
data for urban land record management (SIH 2026, problem statement 26013).

This repository currently contains **Phase 1: the backend and ML foundation**.
There is no frontend yet, and there are no simulated model outputs anywhere in
the codebase: every model endpoint either runs a real trained artifact or
reports that the artifact is missing.

## Layout

```
backend/
  app/
    api/routes/     FastAPI routers (health, models)
    core/           settings, logging, error types
    ml/             model manifest, adapters for models 1-5, registry, orchestrator
    schemas/        parcel, provenance, conflict, topology contracts
    services/       geospatial, topology, conflict, confidence engines
  tests/            pytest suite (no network, no trained artifacts required)
models/             trained artifacts, loaded at runtime, never committed
data/               sample inputs and JSON schemas
database/           migrations (Phase 2)
docs/               architecture, model and API notes
frontend/           Phase 2
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
cp .env.example .env          # then point MODEL_ROOT at your artifact directory
```

## Running

```bash
cd backend
../.venv/bin/uvicorn app.main:app --reload --port 8000
```

- `GET /api/health` – process liveness, environment, resolved device and model root
- `GET /api/models` – the artifact manifest and per-model artifact presence
- `GET /api/models/health` – real artifact inspection, and by default a real load attempt
- `GET /api/models/{model_key}/health` – the same for a single model
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

See <docs/models.md> for the per-model contract and the known limits (for
example: the parcel matcher needs the trained `nn.Module`, not a bare state
dict; Model 4 does not use owner names; Model 5 output is a review signal, not
a fraud determination).

## Documentation

- [docs/architecture.md](docs/architecture.md)
- [docs/models.md](docs/models.md)
- [docs/api.md](docs/api.md)
