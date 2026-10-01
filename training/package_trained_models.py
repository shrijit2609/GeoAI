"""Install completed notebook training outputs into canonical model directories.

Run this after the Model 2 and Model 3 training cells in SpatialShiftAI.ipynb.
The script copies only checkpoint files and writes the preprocessing sidecars
required by the backend adapters; it does not copy datasets, caches or metrics.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def install_artifacts(project_root: Path) -> list[Path]:
    """Copy notebook outputs and emit inference preprocessing sidecars."""

    project_root = project_root.resolve()
    artifact_moves = (
        (
            Path("models/building_extraction/building_deeplabv3_best.pt"),
            Path("models/model2_building_extractor/building_deeplabv3_resnet50_best.pt"),
        ),
        (
            Path(
                "models/building_extraction/"
                "building_extraction_deeplabv3_resnet50.pt"
            ),
            Path("models/model2_building_extractor/building_deeplabv3_resnet50_final.pt"),
        ),
        (
            Path("models/change_detection/siamese_resnet18_change_best.pt"),
            Path("models/model3_change_detection/siamese_resnet18_change_best.pt"),
        ),
        (
            Path("models/change_detection/siamese_resnet18_change_final.pt"),
            Path("models/model3_change_detection/siamese_resnet18_change_final.pt"),
        ),
    )

    installed: list[Path] = []
    for source_relative, target_relative in artifact_moves:
        source_path = project_root / source_relative
        target_path = project_root / target_relative
        if not source_path.is_file() and target_path.is_file():
            source_path = target_path
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Expected trained artifact was not produced: {source_path}"
            )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)
        installed.append(target_path)

    preprocessing: dict[Path, dict[str, Any]] = {
        project_root / "models/model2_building_extractor/building_preprocessing.json": {
            "input_size": [256, 256],
            "scale": 1.0 / 255.0,
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "channel_order": "RGB",
            "resize": "OpenCV INTER_LINEAR",
        },
        project_root / "models/model3_change_detection/change_preprocessing.json": {
            "input_size": [256, 256],
            "scale": 1.0 / 255.0,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "channel_order": "RGB",
        },
    }
    for sidecar_path, descriptor in preprocessing.items():
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        if not sidecar_path.is_file():
            sidecar_path.write_text(json.dumps(descriptor, indent=2), encoding="utf-8")
        installed.append(sidecar_path)

    return installed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/content/drive/MyDrive/SpatialShiftAI"),
        help="SpatialShiftAI Drive project directory used by the training notebook",
    )
    arguments = parser.parse_args()

    for artifact_path in install_artifacts(arguments.project_root):
        print(f"Installed {artifact_path.relative_to(arguments.project_root.resolve())}")


if __name__ == "__main__":
    main()
