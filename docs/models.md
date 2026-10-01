# Models

All five adapters share one contract (`app/ml/common.py`):

```python
ModelResult(model, model_version, status, confidence, decision,
            evidence, warnings, provenance, inference_ms)
```

`confidence` is the model's own (optionally calibrated) score. It is never
synthesised: if a model cannot run, the caller gets an error or a
health entry rather than a number.

Adapters are lazy: the artifact is loaded on first use, cached in the registry,
and placed on the device chosen by `SPATIALSHIFT_DEVICE` (`auto` picks CUDA when
available). PyTorch inference runs under `torch.inference_mode()`.

## Statuses

| status | meaning |
| --- | --- |
| `ready` | the artifact was loaded successfully in this process |
| `ready_for_test` | all required files are present, load not yet attempted |
| `artifact_present_but_preprocessing_blocked` | checkpoint exists, but required preprocessing is unavailable |
| `artifact_present_but_inference_blocked` | artifact exists, but architecture/schema/inference contract is incompatible |
| `training_required` | no trained artifact is installed and a reproducible notebook training pipeline exists |
| `missing_artifact` | required file is absent and no training pipeline is declared |
| `error` | an artifact exists but failed to load |
| `unsupported_runtime` | the runtime (e.g. torch) needed for the artifact is unavailable |

Every health response includes `readiness_reason` describing the concrete
blocker or confirming that the artifact loaded successfully.

## Model 1 — parcel correspondence (Siamese CNN)

`parcel_siamese_v2_best.pt`, optional isotonic calibrator, threshold and
preprocessing sidecars.

The recovered state dict confirms Conv1d kernel width 5 for all three layers,
channel sizes `2→64→128→256`, and embedding weight shapes `256→128→64`. The
adapter strictly loads the notebook-derived 64-point `[64, 2]` boundary input
through adaptive max pooling and a 256-to-128-to-64 L2-normalized embedding.
Polygon inputs use equidistant sampling on the largest exterior, centering, and
maximum-radius scaling. Image inference still requires recorded preprocessing;
without it, health reports `artifact_present_but_preprocessing_blocked`.

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

The recovered schema declares 104 ordered features; classifier and scaler
dimensions are checked against all 104 before inference. The recovered
normalization, string-similarity, area, and cross-field feature logic is
implemented in `backend/app/ml/record_features.py` and validated against the
saved schema.

Features are computed strictly from the saved schema — the adapter never
invents a feature the model was not trained on, and an unknown comparator or an
an uninterpretable feature name is a compatibility error rather than a silent
zero. The copied joblib artifacts loaded successfully under the backend's
scikit-learn runtime. **Owner names are not used**: the training data contains
none, so any
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

## Reported Training Benchmarks

These values describe the named training/evaluation sets only. They are not
deployment accuracy for Indian cadastral records. Models 2 and 3 still require
training; their notebook pipelines now package trained checkpoints and exact
preprocessing sidecars into the canonical model directories.

| Model | Evaluation set | Reported metrics |
| --- | --- | --- |
| Model 1 v1 sanity run | 10,000 parcels | ROC-AUC 0.999886; this is v1, not the recovered v2 checkpoint |
| Model 2 | WHU Building Dataset | best validation IoU 0.8395; test IoU 0.842672; Dice/F1 0.914620; precision 0.923784; recall 0.905635 |
| Model 3 | LEVIR-CD Cropped 256 | best validation IoU 0.5950; test IoU 0.5886685; Dice/F1 0.7410842; precision 0.7028147; recall 0.7837613 |
| Model 4 | controlled 3,000-pair test split | accuracy 0.999667; precision 0.999334; recall 1.0; F1 0.999667; ROC-AUC 1.0; PR-AUC 1.0; Brier 0.000112; threshold 0.05 |
| Model 5 | reported benchmark; canonical inference artifacts present | accuracy 0.814167; precision 0.717017; recall 0.833333; F1 0.770812; ROC-AUC 0.907415; PR-AUC 0.823562; Brier 0.115786 |

Model 4's positive pairs are controlled/corrupted variants, not independently
adjudicated real-world duplicates. Model 5 outputs, when its artifact is
available, are record-review signals only and must not be described as fraud,
ownership, title or encroachment determinations.
