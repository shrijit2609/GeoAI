"""Run the production Model 3 adapter on co-registered before/after imagery."""
from __future__ import annotations

import argparse
import base64
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
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--model-root", type=Path, default=Path(__file__).resolve().parents[2] / "models")
    parser.add_argument("--mask-output", type=Path, default=Path("change_mask.png"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    registry = ModelRegistry(args.model_root, device="auto")
    result = registry.get("change_detector").predict(str(args.before), str(args.after), threshold=args.threshold)
    mask = np.asarray(
        Image.open(
            __import__("io").BytesIO(
                base64.b64decode(result.evidence["mask_png_base64"])
            )
        ).convert("L"),
        dtype=np.uint8,
    )
    args.mask_output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask * 255).save(args.mask_output)
    print(json.dumps({"model": result.model, "status": result.status, "decision": result.decision, "changed_pixel_percentage": result.evidence["changed_pixel_percentage"], "mask": str(args.mask_output), "shape": result.evidence["mask_shape"]}, indent=2))


if __name__ == "__main__":
    main()
