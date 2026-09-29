# Model artifacts

Trained artifacts are **not** committed: `.gitignore` excludes `*.pt`, `*.pth`,
`*.pkl`, `*.joblib`, `*.onnx` and `*.h5` under this directory. Only the folder
layout is tracked.

Copy the trained files into these folders (or point `MODEL_ROOT` at another
directory with the same layout):

| directory | required | optional |
| --- | --- | --- |
| `model1_parcel_matcher/` | `parcel_siamese_v2_best.pt` | `parcel_siamese_v2_final.pt`, `parcel_match_isotonic_calibrator.pkl`, `parcel_match_threshold.json`, `parcel_match_preprocessing.json` |
| `model2_building_extractor/` | `building_deeplabv3_resnet50_best.pt` | `building_deeplabv3_resnet50_final.pt`, `building_preprocessing.json` |
| `model3_change_detection/` | `siamese_resnet18_change_best.pt` | `siamese_resnet18_change_final.pt`, `change_preprocessing.json` |
| `model4_entity_resolution/` | `model4_entity_resolver_logistic.joblib`, `model4_feature_scaler.joblib`, `model4_feature_schema.json` | `model4_isotonic_calibrator.joblib`, `model4_threshold.json` |
| `model5_anomaly_detection/` | `model5_anomaly_classifier.joblib`, `model5_feature_schema.json` | `model5_feature_scaler.joblib`, `model5_isotonic_calibrator.joblib`, `model5_threshold.json` |

Verify what the running backend can actually see:

```bash
curl -s localhost:8000/api/models/health | python -m json.tool
```
