# Models

All five adapters share one contract (`app/ml/common.py`):

```python
ModelResult(model, model_version, status, confidence, decision,
            evidence, warnings, provenance, inference_ms)
```

`confidence` is the model's own (optionally calibrated) score. It is never
synthesised: if a model cannot run, the caller gets an error or a
`missing_artifact` health entry rather than a number.

Adapters are lazy: the artifact is loaded on first use, cached in the registry,
and placed on the device chosen by `SPATIALSHIFT_DEVICE` (`auto` picks CUDA when
available). PyTorch inference runs under `torch.inference_mode()`.

## Statuses

| status | meaning |
| --- | --- |
| `ready` | the artifact was loaded successfully in this process |
| `ready_for_test` | all required files are present, load not yet attempted |
| `missing_artifact` | at least one required file is absent under `MODEL_ROOT` |
| `load_error` | the file exists but failed to load or is not usable as-is |
| `unsupported_runtime` | the runtime (e.g. torch) needed for the artifact is unavailable |

## Model 1 — parcel correspondence (Siamese CNN)

`parcel_siamese_v2_best.pt`, optional isotonic calibrator, threshold and
preprocessing sidecars.

A Siamese encoder has no canonical layer layout, so a checkpoint containing
only weights cannot be rebuilt without guessing the architecture. In that case
the adapter raises `ModelCompatibilityError` and health reports `load_error`
with the observed tensor names, instead of inventing an encoder that would
produce meaningless similarity scores. Supported checkpoints carry the trained
`nn.Module` (or TorchScript). Without a saved threshold the decision boundary
falls back to 0.5 and the result carries a warning.

## Model 2 — building footprint extraction (DeepLabV3-ResNet50)

`building_deeplabv3_resnet50_best.pt` plus recorded preprocessing (input size,
mean, std, scale) either inside the checkpoint or in
`building_preprocessing.json`. Preprocessing is never guessed.

Output: probability mask summary, binary mask at the requested threshold,
building pixel percentage and polygons. Polygons are emitted in the raster CRS
when the input is georeferenced; otherwise they are explicitly flagged as pixel
coordinates (`coordinate_space: "pixel"`, `is_pixel_coordinates: true`) and a
warning is attached so they are never mistaken for map coordinates.

## Model 3 — temporal change detection (Siamese ResNet18)

`siamese_resnet18_change_best.pt` plus recorded preprocessing. Both epochs must
have identical raster dimensions; differing CRS or geotransform is reported as
a warning because the change mask is then not pixel-aligned.

Output: changed pixel percentage, change mask summary, change polygons with the
same coordinate-space flagging as Model 2.

## Model 4 — entity resolution (logistic regression)

Required: `model4_entity_resolver_logistic.joblib`,
`model4_feature_scaler.joblib`, `model4_feature_schema.json`. Optional:
isotonic calibrator and threshold.

Features are computed strictly from the saved schema — the adapter never
invents a feature the model was not trained on, and an unknown comparator or an
uninterpretable feature name is a compatibility error rather than a silent
zero. **Owner names are not used**: the training data contains none, so any
`owner_name` supplied by a caller is ignored and a warning is returned.

Evidence lists matched fields, conflicting fields and fields missing from one
or both records, so a reviewer can see why the score is what it is.

## Model 5 — record anomaly detection

Required: `model5_anomaly_classifier.joblib`, `model5_feature_schema.json`.
Optional: scaler, isotonic calibrator, threshold.

The decision is `review_required` / `no_review_required`. The model scores
**record attributes only**: it does not observe geometry, topology or
ownership history, and it cannot establish fraud, illegality or encroachment.
Every result carries a warning stating that geometry and topology anomalies are
out of scope, and the explanation is built from the features that actually
drove the score.
