# API

Base path: `/api`. OpenAPI UI at `/docs`.

The backend exposes model introspection and the Phase 2 canonical ingestion API.
The ingestion endpoints validate and normalize geospatial sources without
fabricating missing values or model outputs.

## Production model runtime

The backend keeps all trained model artifacts external to Git by resolving them
under `MODEL_ROOT`. The runtime contract is explicit and safe: missing artifacts
return a structured missing-artifact response, a file path alone does not mark a
model as ready, and the rest of the API stays operational when a model is
unavailable.

### Inference endpoints

- `POST /api/models/{model_key}/infer`
- `POST /api/models/{model_key}/predict`

The supported model keys are:

- `parcel_matcher`
- `building_extractor`
- `change_detector`
- `entity_resolver`
- `anomaly_detector`

Each route validates the supplied request payload, executes the model adapter,
and returns the normalized model result with `model`, `model_version`,
`status`, `confidence`, `decision`, `evidence`, `warnings`, `provenance`, and
`inference_ms` when the model is available.

## `GET /api/health`

Process liveness plus environment, timestamp, resolved device and the
configured model root (and whether it exists). Does not load any model; use
`/api/models/health` for readiness.

## `GET /api/version`

Application version, Python version and the versions of the libraries that
matter for reproducing an inference run (torch, torchvision, scikit-learn,
numpy, shapely, rasterio, pyproj).

## `POST /api/upload`

Upload and ingest a vector source. The endpoint accepts either a JSON payload or
an uploaded file. Supported formats are GeoJSON, GeoPackage, Shapefile and
Parquet.

Response fields include:

- `source_name`, `source_type`, `source_id`
- `total_records`, `valid_records`, `invalid_records`
- `detected_crs`, `geometry_types`, `detected_fields`
- `warnings`, `errors`, `provenance`
- `records` — canonicalized parcel/entity records

## `GET /api/models`

The manifest: for each model its key, title, architecture, task, runtime,
resolved directory under `MODEL_ROOT`, notes, and every expected artifact with
`required`, `present`, `path` and `size_bytes`. No loading is attempted.

## `GET /api/models/health`

Query parameters:

- `probe` (default `true`) — attempt a real load of each model
- `include_hash` (default `false`) — include the sha256 of present artifacts

Returns per model: `status`, `loaded`, `device`, `load_time_ms`,
`model_version`, `error` (the real exception text when loading failed),
`missing_artifacts` and the artifact reports. With `probe=false` a model with
all files present reports `ready_for_test` rather than `ready`.

## `GET /api/models/{model_key}/health`

The same payload for one model; `404` for an unknown key.

## `GET /api/models/pipeline`

The declared stages (parcel correspondence → building extraction → change
detection → entity resolution → anomaly review) with, for each, whether it can
run and — when it cannot — the concrete reason (missing artifact, load error,
unsupported runtime). Stages that cannot run are reported as skipped, never
simulated.

## Errors

| exception | status | `error` |
| --- | --- | --- |
| `InvalidInputError` | 422 | `invalid_input` |
| `ModelArtifactMissingError` | 503 | `missing_artifact` |
| `ModelCompatibilityError` | 503 | `incompatible_artifact` |
| `ModelLoadError` | 503 | `load_error` |
| other `SpatialShiftError` (geometry, CRS, inference) | 400 | `domain_error` |
| unknown model key | 404 | FastAPI `detail` |

Each error response carries `error`, `detail` and, for model errors, the model
key.
