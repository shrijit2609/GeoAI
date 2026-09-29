# Architecture (Phase 1)

```
              FastAPI (app/main.py)
                 |            |
         api/routes/health   api/routes/models
                                  |
                          ml/model_registry  --- lazy, cached adapters
                                  |                 |
                          ml/orchestrator      ml/model{1..5}_*.py
                                                    |
                                            MODEL_ROOT/<model dir>/artifacts

  services/geospatial   services/topology   services/conflicts   services/confidence
        (deterministic, no ML — always available)
```

## Layers

**core** – `config.py` (pydantic-settings; `MODEL_ROOT`, `SPATIALSHIFT_DEVICE`,
log level, CORS, eager loading), `logging.py`, `errors.py`. Every failure mode
has a typed exception, and the API maps them to HTTP status codes.

**ml** – `manifest.py` is the single source of truth for which files each model
needs and which are optional. `base.py` implements lazy loading, load timing,
artifact reporting (path, size, presence, optional sha256), device resolution
and the `ModelResult` builder. `model_registry.py` caches one adapter per model
and exposes health without forcing a load. `orchestrator.py` declares the
pipeline stages and marks a stage as skipped — with the reason — when its model
cannot run. No stage is ever simulated.

**schemas** – `SourceRecord` / `CanonicalParcel` (a ULPIN is only ever copied
from an authoritative source record, never generated), `ProvenanceChain`
(model, rule, transformation and reviewer entries), `Conflict` / `ConflictSet`,
`TopologyIssue` / `RepairProposal`.

**services** – deterministic geospatial work that does not need ML:

- `geospatial.py`: CRS description and transformation (pyproj), geometry
  validation and repair, intersection, containment, overlap metrics, distance,
  bounding boxes.
- `topology.py`: invalid geometry, self-intersection, empty geometry,
  duplicates, overlaps and gaps. Repairs are returned as **proposals**; the
  service never mutates authoritative cadastral geometry in place.
- `conflicts.py`: rule-based comparison of two source records producing CRS,
  geometry, identity, attribute and missing-attribute conflicts, each with the
  evidence that produced it. Topology reports can be folded into the same
  conflict set.
- `confidence.py`: weighted geometric mean over the channels that were actually
  measured, with a multiplicative penalty per unresolved conflict. Channels
  that were not measured are listed in `missing_components` rather than being
  filled with a default — if nothing was measured, the confidence is `None`.

## Design rules held throughout

1. A model is `ready` only after a real load succeeded in this process.
2. Preprocessing and architecture are read from the artifact, never guessed.
3. Pixel-space outputs are labelled as such and never presented as map
   coordinates.
4. Repairs and resolutions are proposals; authoritative data is not rewritten.
5. Every model result carries provenance: model key, version, device, artifact
   paths and library versions.
