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

The recovered state dict confirms Conv1d kernel width 5 for all three layers,
channel sizes `2→64→128→256`, and embedding weight shapes `256→128→64`. The
project manifest documents 64 boundary points, input shape `[64, 2]`, adaptive
max pooling, L2 normalization, and contrastive training. The adapter uses
strict state-dict loading with those kernel dimensions. The checkpoint has no
preprocessing descriptor, and the original feature construction and activation
definitions were not recovered; inference therefore remains unavailable rather
than guessing those details.

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
zero. The recovered 104-feature schema uses names such as
`village__both_empty` but does not define their formulas. Its reference
inference file does not construct the full model feature vector, so this bundle
remains unavailable until that contract is recovered. The joblib bundle was
serialized with scikit-learn 1.6.1 while the backend pins 1.5.2; validate
compatibility with the training runtime before deployment. **Owner names are
not used**: the training data contains none, so any
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
deployment accuracy for Indian cadastral records. Models 2, 3 and 5 have no
trained inference artifacts in the recovered folder, and Model 4's bundle
cannot currently pass this backend's feature-schema contract.

| Model | Evaluation set | Reported metrics |
| --- | --- | --- |
| Model 1 v1 sanity run | 10,000 parcels | ROC-AUC 0.999886; this is v1, not the recovered v2 checkpoint |
| Model 2 | WHU Building Dataset | best validation IoU 0.8395; test IoU 0.842672; Dice/F1 0.914620; precision 0.923784; recall 0.905635 |
| Model 3 | LEVIR-CD Cropped 256 | best validation IoU 0.5950; test IoU 0.5886685; Dice/F1 0.7410842; precision 0.7028147; recall 0.7837613 |
| Model 4 | controlled 3,000-pair test split | accuracy 0.999667; precision 0.999334; recall 1.0; F1 0.999667; ROC-AUC 1.0; PR-AUC 1.0; Brier 0.000112; threshold 0.05 |
| Model 5 | reported benchmark; local model artifact absent | accuracy 0.814167; precision 0.717017; recall 0.833333; F1 0.770812; ROC-AUC 0.907415; PR-AUC 0.823562; Brier 0.115786 |

Model 4's positive pairs are controlled/corrupted variants, not independently
adjudicated real-world duplicates. Model 5 outputs, when its artifact is
available, are record-review signals only and must not be described as fraud,
ownership, title or encroachment determinations.
