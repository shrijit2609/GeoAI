"""Artifact manifest for the five trained SpatialShiftAI models.

The manifest is the single source of truth for which files each model needs.
Artifacts are produced outside this repository (training notebooks) and are
installed into ``MODEL_ROOT`` at deployment time; they are intentionally not
committed to Git.
"""

from __future__ import annotations

from app.ml.common import ArtifactSpec, ModelRuntime, ModelSpec

PARCEL_MATCHER = ModelSpec(
    key="parcel_matcher",
    title="Model 1 - Parcel correspondence (Siamese CNN)",
    directory="model1_parcel_matcher",
    runtime=ModelRuntime.TORCH,
    architecture="siamese_cnn",
    task="parcel_correspondence",
    artifacts=(
        ArtifactSpec("parcel_siamese_v2_best.pt", "checkpoint", required=True),
        ArtifactSpec("parcel_siamese_v2_final.pt", "checkpoint_final", required=False),
        ArtifactSpec(
            "parcel_match_isotonic_calibrator.pkl", "calibrator", required=False
        ),
        ArtifactSpec("parcel_match_threshold.json", "threshold", required=False),
        ArtifactSpec("parcel_match_preprocessing.json", "preprocessing", required=False),
    ),
    readiness_requirements=(
        "checkpoint file exists and loads on the configured runtime",
        "required preprocessing information is available either in the checkpoint or sidecar JSON",
        "module architecture must be reconstructable without guessing",
    ),
    notes=(
        "Benchmark performance was measured on a controlled dataset and must not "
        "be presented as real-world accuracy.",
        "Inference requires a preprocessing descriptor (embedded in the checkpoint "
        "or parcel_match_preprocessing.json). Preprocessing is never guessed.",
    ),
)

BUILDING_EXTRACTOR = ModelSpec(
    key="building_extractor",
    title="Model 2 - Building footprint extraction (DeepLabV3-ResNet50)",
    directory="model2_building_extractor",
    runtime=ModelRuntime.TORCH,
    architecture="deeplabv3_resnet50",
    task="building_footprint_segmentation",
    artifacts=(
        ArtifactSpec("building_deeplabv3_resnet50_best.pt", "checkpoint", required=True),
        ArtifactSpec(
            "building_deeplabv3_resnet50_final.pt", "checkpoint_final", required=False
        ),
        ArtifactSpec("building_preprocessing.json", "preprocessing", required=False),
    ),
    readiness_requirements=(
        "checkpoint file exists and loads on the configured runtime",
        "preprocessing metadata is present or embedded in the checkpoint",
        "output head shape must be compatible with the model architecture",
    ),
    notes=(
        "Polygons are returned in pixel coordinates unless the source raster "
        "carries a CRS and geotransform.",
    ),
)

CHANGE_DETECTOR = ModelSpec(
    key="change_detector",
    title="Model 3 - Temporal change detection (Siamese ResNet18)",
    directory="model3_change_detection",
    runtime=ModelRuntime.TORCH,
    architecture="siamese_resnet18_diff_decoder",
    task="temporal_change_detection",
    artifacts=(
        ArtifactSpec("siamese_resnet18_change_best.pt", "checkpoint", required=True),
        ArtifactSpec(
            "siamese_resnet18_change_final.pt", "checkpoint_final", required=False
        ),
        ArtifactSpec("change_preprocessing.json", "preprocessing", required=False),
    ),
    readiness_requirements=(
        "checkpoint file exists and loads on the configured runtime",
        "paired imagery preprocessing matches the checkpoint contract",
        "decoder head dimensions match the saved weights",
    ),
    notes=("Both epochs must share the same CRS, geotransform and raster size.",),
)

ENTITY_RESOLVER = ModelSpec(
    key="entity_resolver",
    title="Model 4 - Land-record entity resolution (logistic regression)",
    directory="model4_entity_resolution",
    runtime=ModelRuntime.SKLEARN,
    architecture="logistic_regression",
    task="record_linkage",
    artifacts=(
        ArtifactSpec("model4_entity_resolver_logistic.joblib", "classifier", required=True),
        ArtifactSpec("model4_feature_scaler.joblib", "scaler", required=True),
        ArtifactSpec("model4_feature_schema.json", "feature_schema", required=True),
        ArtifactSpec("model4_isotonic_calibrator.joblib", "calibrator", required=False),
        ArtifactSpec("model4_threshold.json", "threshold", required=False),
    ),
    readiness_requirements=(
        "classifier, scaler and feature schema files are all present",
        "feature schema can be parsed and matches the expected record layout",
        "optional calibrator and threshold are treated as metadata, not as replacements for required files",
    ),
    notes=(
        "The training data contains no owner names; owner-name matching is not "
        "supported by this model.",
        "Features are computed strictly from model4_feature_schema.json.",
    ),
)

ANOMALY_DETECTOR = ModelSpec(
    key="anomaly_detector",
    title="Model 5 - Parcel record anomaly / conflict detection",
    directory="model5_anomaly_detection",
    runtime=ModelRuntime.SKLEARN,
    architecture="tabular_classifier",
    task="record_anomaly_detection",
    artifacts=(
        ArtifactSpec("model5_anomaly_classifier.joblib", "classifier", required=True),
        ArtifactSpec("model5_feature_schema.json", "feature_schema", required=True),
        ArtifactSpec("model5_feature_scaler.joblib", "scaler", required=False),
        ArtifactSpec("model5_isotonic_calibrator.joblib", "calibrator", required=False),
        ArtifactSpec("model5_threshold.json", "threshold", required=False),
        ArtifactSpec("model5_isolation_forest.joblib", "isolation_forest", required=False),
        ArtifactSpec("model5_inference_config.json", "inference_config", required=False),
    ),
    readiness_requirements=(
        "classifier and feature schema are present and parseable",
        "feature transformation metadata is valid if a scaler or calibrator is supplied",
        "result is a review signal only; the model is never treated as a fraud decision engine",
    ),
    notes=(
        "Scope is attribute/identity/statistical anomalies in land records. The "
        "benchmark does not establish geometry or topology anomaly detection.",
        "Outputs are review signals, never a finding of fraud or encroachment.",
    ),
)

MODEL_SPECS: tuple[ModelSpec, ...] = (
    PARCEL_MATCHER,
    BUILDING_EXTRACTOR,
    CHANGE_DETECTOR,
    ENTITY_RESOLVER,
    ANOMALY_DETECTOR,
)

MODEL_SPECS_BY_KEY: dict[str, ModelSpec] = {spec.key: spec for spec in MODEL_SPECS}
