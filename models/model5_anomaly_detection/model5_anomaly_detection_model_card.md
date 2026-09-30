
# SpatialShiftAI Model 5
## Parcel Anomaly / Conflict Detection

### Purpose

Model 5 identifies cadastral records whose attribute
patterns are inconsistent or statistically unusual.

The system is intended to create a human-review queue
for multi-source land-record harmonization.

### Architecture

Hybrid anomaly detector:

1. Cadastral attribute feature engineering
2. Field presence and completeness features
3. Identity structure features
4. Administrative frequency features
5. Supervised Logistic Regression
6. Isolation Forest trained on normal records
7. Hybrid anomaly score
8. Isotonic probability calibration
9. Threshold-based review decision
10. Explainable anomaly reasons

### Dataset

Total samples: 8,000

Training samples: 5,600

Validation samples: 1,200

Test samples: 1,200

### Test Metrics

Accuracy: 0.8142

Precision: 0.7170

Recall: 0.8333

F1: 0.7708

ROC-AUC: 0.9074

PR-AUC: 0.8236

Brier score: 0.1158

Decision threshold: 0.3900

### Anomaly Types

- ATTRIBUTE_CONFLICT
- IDENTITY_CONFLICT
- MISSING_CRITICAL_ATTRIBUTE

### Interpretation

The model produces an anomaly/conflict probability.

Records above the configured threshold are sent to
a REVIEW workflow rather than automatically rejected.

### Important Limitations

The anomaly examples are controlled perturbations
of real public cadastral records.

The source does not provide adjudicated anomaly labels.

Therefore the reported benchmark metrics measure the
ability to identify the controlled anomaly patterns,
not real-world anomaly prevalence or production accuracy.

The current benchmark does not include parcel geometry,
topological relationships, building footprints, or
temporal imagery.

The system must NOT be interpreted as determining:

- legal ownership
- land title
- encroachment
- fraud
- property rights
- legal validity

Production deployment requires authoritative
multi-source data, geographic context, target-region
validation and human review.

### SpatialShiftAI Integration

Model 5 is intended to consume evidence from:

- Parcel correspondence
- Entity resolution
- Source attribute comparison
- Building extraction
- Temporal change detection

and produce an explainable review signal for the
harmonization pipeline.
