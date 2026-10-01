# API

Base path: `/api`. OpenAPI UI at `/docs`.

The backend exposes model introspection and the Phase 2 canonical ingestion API.
The ingestion endpoints validate and normalize geospatial sources without
fabricating missing values or model outputs.

## Production model runtime

The backend resolves trained model artifacts under `MODEL_ROOT`. The runtime
contract distinguishes ready models, reproducible training-required models,
missing files, present-but-blocked artifacts, and load errors. A file path alone
does not mark a model as ready, and the rest of the API stays operational when a
model is unavailable.

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

## Data, Layers and Exports

- `POST /api/data/upload` — bounded upload for GeoJSON, GeoPackage, Shapefile
	ZIP, Parquet, CSV coordinates and KML when the installed GDAL/Fiona driver
	supports it. The request limit is 64 MiB; expanded ZIP content is limited to
	128 MiB.
- `GET /api/data/sources` — source summaries registered in this backend process.
- `GET /api/data/{source_id}` — source records for the current process.
- `GET /api/data/{source_id}/export?format=geojson|csv` — export one source.
- `GET /api/layers` and `GET /api/layers/{layer_id}` — layer summaries and
	GeoJSON transformed to EPSG:4326 only when the source CRS is known.

The source catalog is process-local and is cleared when the backend restarts.
CRS-unknown geometries are withheld from geographic map output rather than
silently interpreted as longitude/latitude.

## Harmonization

- `POST /api/harmonize` with `{ "source_ids": ["..."] }` links records only
	when normalized ULPINs or complete district/village/tehsil/khasra keys match
	exactly. Conflicting attributes remain unresolved and confidence is not
	invented.
- `GET /api/harmonize/{job_id}` retrieves a result held in process memory.

This deterministic workflow does not call model inference or apply an
authoritative source-precedence rule.

## ULPIN

- `GET /api/ulpin/{ulpin}` searches uploaded sources. If
	`SPATIALSHIFT_ULPIN_ENDPOINT` is configured, a local miss can be forwarded to
	that endpoint (replace `{ulpin}` in the URL or append the identifier). With
	no configured endpoint, the response states that no external service was
	queried.

## `GET /api/models`

The manifest: for each model its key, title, architecture, task, runtime,
resolved directory under `MODEL_ROOT`, notes, and every expected artifact with
`required`, `present`, `path` and `size_bytes`. No loading is attempted.

## `GET /api/models/health`

Query parameters:

- `probe` (default `true`) — attempt a real load of each model
- `include_hash` (default `false`) — include the sha256 of present artifacts

Returns per model: `status`, `loaded`, `device`, `load_time_ms`,
`model_version`, `readiness_reason`, `error` (the real exception text when
loading failed), `missing_artifacts` and the artifact reports. Statuses
distinguish `ready`, `training_required`, `missing_artifact`,
`artifact_present_but_preprocessing_blocked`,
`artifact_present_but_inference_blocked`, and `error`. With `probe=false` a
model with all files present reports `ready_for_test` rather than `ready`.

## `GET /api/models/{model_key}/health`

The same payload for one model; `404` for an unknown key.

## `GET /api/models/pipeline`

The declared stages (parcel correspondence → building extraction → change
detection → entity resolution → anomaly review) with, for each, whether it can
run and — when it cannot — the concrete readiness reason. Stages that cannot
run are reported as skipped, never
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
