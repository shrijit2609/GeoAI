"""Exercise the production Model 2 adapter on an image and save its mask."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))

from app.ml.model_registry import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model-root", type=Path, default=Path(__file__).resolve().parents[2] / "models")
    parser.add_argument("--mask-output", type=Path, default=Path("building_mask.png"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    registry = ModelRegistry(args.model_root, device="auto")
    adapter = registry.get("building_extractor")
    probability, binary, _ = adapter.predict_masks(str(args.image), threshold=args.threshold)
    args.mask_output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((binary * 255).astype(np.uint8)).save(args.mask_output)
    print(json.dumps({"status": registry.health("building_extractor").status.value, "mask": str(args.mask_output), "shape": list(binary.shape), "building_pixel_percentage": float(binary.mean() * 100), "probability_mean": float(probability.mean())}, indent=2))


if __name__ == "__main__":
    main()
