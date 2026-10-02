"""Notebook-cell-102 LEVIR-CD Siamese ResNet18 training for Colab/T4."""
from __future__ import annotations

import argparse
import io
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

EPOCHS = 5
MAX_TRAIN = 5000
MAX_VAL = 1000
MAX_TEST = 1000
IMAGE_SIZE = 256


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        array = value
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8) if array.max() <= 1.0 else array.astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)}")


def find_column(candidates: list[str], columns: list[str]) -> str | None:
    lower_map = {column.lower(): column for column in columns}
    return next((lower_map[name.lower()] for name in candidates if name.lower() in lower_map), None)


def detect_columns(frame: pd.DataFrame) -> tuple[str | None, str | None, str | None, str | None]:
    columns = frame.columns.tolist()
    a_col = find_column(["image1", "image_a", "before", "A", "img1", "t1", "image"], columns)
    b_col = find_column(["image2", "image_b", "after", "B", "img2", "t2"], columns)
    label_col = find_column(["label", "mask", "change", "target"], columns)
    pair_col = None
    if b_col is None or label_col is None:
        candidates = [column for column in columns if any(token in column.lower() for token in ("image", "pair", "input"))]
        if len(candidates) >= 2:
            a_col, b_col = candidates[:2]
        elif len(candidates) == 1:
            pair_col = candidates[0]
    if a_col is None or (b_col is None and pair_col is None) or label_col is None:
        raise RuntimeError(f"Cannot identify before/after/mask fields from LEVIR columns {columns}")
    return a_col, b_col, pair_col, label_col


def extract_row(row: pd.Series, columns: tuple[str | None, str | None, str | None, str | None]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_col, b_col, pair_col, label_col = columns
    if a_col is not None and b_col is not None:
        image_a, image_b = decode_image(row[a_col]), decode_image(row[b_col])
    else:
        pair = row[pair_col]
        if isinstance(pair, dict):
            keys = list(pair)
            if len(keys) < 2:
                raise RuntimeError("Pair field contains fewer than two images")
            image_a, image_b = decode_image(pair[keys[0]]), decode_image(pair[keys[1]])
        elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
            image_a, image_b = decode_image(pair[0]), decode_image(pair[1])
        else:
            raise RuntimeError("Cannot decode the LEVIR before/after pair")
    label_value = row[label_col]
    try:
        label = decode_image(label_value).convert("L")
        mask = np.asarray(label, dtype=np.uint8)
    except (TypeError, ValueError):
        mask = np.asarray(label_value)
        if mask.ndim == 3:
            mask = mask.squeeze()
        mask = mask.astype(np.uint8)
    return np.asarray(image_a.convert("RGB"), dtype=np.uint8), np.asarray(image_b.convert("RGB"), dtype=np.uint8), (mask > 0).astype(np.uint8)


def load_splits(dataset_dir: Path) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], ...]:
    parquet_dir = dataset_dir / "data"
    train_path = parquet_dir / "train-00000-of-00001-737f96f51caac8cd.parquet"
    val_path = parquet_dir / "val-00000-of-00001-d09d88a7419f2427.parquet"
    validation_path = parquet_dir / "validation-00000-of-00001-46a87fa6b7b4d293.parquet"
    test_path = parquet_dir / "test-00000-of-00001-31d7c3e3444e5b5d.parquet"
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Expected notebook LEVIR parquet split is missing: {path}")
    val_path = val_path if val_path.is_file() else validation_path
    if not val_path.is_file():
        raise FileNotFoundError(f"Neither LEVIR validation parquet exists under {parquet_dir}")

    splits = []
    for name, path, limit in (("train", train_path, MAX_TRAIN), ("validation", val_path, MAX_VAL), ("test", test_path, MAX_TEST)):
        frame = pd.read_parquet(path, engine="pyarrow")
        columns = detect_columns(frame)
        count = min(len(frame), limit)
        rows = [extract_row(frame.iloc[index], columns) for index in range(count)]
        print(f"Cached {name}: {len(rows)} rows from {path.name}")
        splits.append(rows)
    return tuple(splits)


class ChangeDataset(Dataset):
    def __init__(self, rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]], augment: bool = False):
        self.rows = rows
        self.augment = augment
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        before, after, mask = self.rows[index]
        if self.augment:
            if random.random() < 0.5:
                before, after, mask = np.fliplr(before).copy(), np.fliplr(after).copy(), np.fliplr(mask).copy()
            if random.random() < 0.5:
                before, after, mask = np.flipud(before).copy(), np.flipud(after).copy(), np.flipud(mask).copy()
            if random.random() < 0.5:
                k = random.randint(0, 3)
                before, after, mask = np.rot90(before, k).copy(), np.rot90(after, k).copy(), np.rot90(mask, k).copy()
        before_tensor = (torch.from_numpy(before.transpose(2, 0, 1)).float() / 255.0 - self.mean) / self.std
        after_tensor = (torch.from_numpy(after.transpose(2, 0, 1)).float() / 255.0 - self.mean) / self.std
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        return before_tensor, after_tensor, mask_tensor


class SiameseChangeNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.encoder = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 2, stride=2), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
        )

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        difference = torch.abs(self.encoder(before) - self.encoder(after))
        logits = self.decoder(difference)
        return F.interpolate(logits, size=before.shape[-2:], mode="bilinear", align_corners=False)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1 - ((2 * intersection + eps) / (denominator + eps)).mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return 0.5 * F.binary_cross_entropy_with_logits(logits, targets) + 0.5 * dice_loss(logits, targets)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = intersection = union = pred_sum = true_sum = 0.0
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for before, after, target in loader:
            before, after, target = before.to(device), after.to(device), target.to(device)
            logits = model(before, after)
            total_loss += combined_loss(logits, target).item() * len(before)
            prediction = (torch.sigmoid(logits) >= 0.5).bool()
            truth = target.bool()
            intersection += float((prediction & truth).sum())
            union += float((prediction | truth).sum())
            pred_sum += float(prediction.sum())
            true_sum += float(truth.sum())
            all_predictions.append(prediction.cpu().numpy().reshape(-1).astype(np.uint8))
            all_targets.append(truth.cpu().numpy().reshape(-1).astype(np.uint8))
    predicted = np.concatenate(all_predictions)
    actual = np.concatenate(all_targets)
    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted, zero_division=0)
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "iou": intersection / (union + 1e-9),
        "dice": (2 * intersection) / (pred_sum + true_sum + 1e-9),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
    }


def train(dataset_dir: Path, project_root: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("This training job is intended for a CUDA/T4 Colab runtime. No CPU fallback training is started.")
    seed_everything(42)
    device = torch.device("cuda")
    train_rows, val_rows, test_rows = load_splits(dataset_dir)
    batch_size = 16 if torch.cuda.is_available() else 2
    train_loader = DataLoader(ChangeDataset(train_rows, True), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(ChangeDataset(val_rows), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(ChangeDataset(test_rows), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    model_dir = project_root / "models/model3_change_detection"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "siamese_resnet18_change_best.pt"
    final_path = model_dir / "siamese_resnet18_change_final.pt"
    model = SiameseChangeNet(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    best_iou = -1.0
    history: list[dict[str, Any]] = []

    for epoch in range(1, EPOCHS + 1):
        started = time.time()
        model.train()
        running_loss = 0.0
        for before, after, target in train_loader:
            before, after, target = before.to(device), after.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = combined_loss(model(before, after), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * len(before)
        train_loss = running_loss / max(len(train_loader.dataset), 1)
        validation = evaluate(model, val_loader, device)
        scheduler.step(validation["iou"])
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{key}": value for key, value in validation.items() if key != "confusion_matrix"}, "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - started}
        history.append(row)
        print(f"Epoch {epoch}/{EPOCHS} | train={train_loss:.5f} val_iou={validation['iou']:.5f} val_dice={validation['dice']:.5f}")
        if validation["iou"] > best_iou:
            best_iou = validation["iou"]
            torch.save({"model_state_dict": model.state_dict(), "architecture": "Siamese ResNet18", "epoch": epoch, "val_metrics": validation, "seed": 42}, best_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    result = {"dataset": "ericyu/LEVIRCD_Cropped_256", "training_samples": len(train_rows), "validation_samples": len(val_rows), "test_samples": len(test_rows), "epochs": EPOCHS, "best_epoch": checkpoint["epoch"], "best_validation_iou": best_iou, "test_metrics": test_metrics, "seed": 42}
    torch.save({"model_state_dict": model.state_dict(), "architecture": "Siamese ResNet18", "epochs": EPOCHS, "history": history, "seed": 42}, final_path)
    (model_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (model_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    preprocessing = {"input_size": [IMAGE_SIZE, IMAGE_SIZE], "scale": 1.0 / 255.0, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "channel_order": "RGB", "threshold": 0.5}
    (model_dir / "change_preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    card = f"""# Model 3: Temporal change detection\n\nArchitecture: Siamese ResNet18 shared encoder, absolute feature difference, notebook transposed-convolution decoder. Dataset: `ericyu/LEVIRCD_Cropped_256` parquet. The experiment caches at most 5,000 train / 1,000 validation / 1,000 test examples. Inputs use RGB ImageNet normalization at native 256x256.\n\nAugmentation: paired horizontal/vertical flip and random 0-3 quarter-turn rotations, each with probability .5. Loss: .5 BCEWithLogits + .5 Dice. Optimizer: AdamW lr=1e-4, weight_decay=1e-4. Scheduler: ReduceLROnPlateau(mode=max, factor=.5, patience=1). Epochs: 5. Best checkpoint is maximum validation IoU; test threshold=.5.\n\nTest metrics are saved to `metrics.json` and describe only the LEVIR-CD benchmark, not georeferenced Indian deployment accuracy.\n"""
    (model_dir / "model_card.md").write_text(card, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("/content/drive/MyDrive/SpatialShiftAI"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Select a Google Colab Tesla T4 runtime before downloading data or training.")
    dataset_dir = args.dataset_dir
    if args.download:
        from huggingface_hub import snapshot_download
        dataset_dir = args.project_root / "data/raw/levir_cd"
        snapshot_download(repo_id="ericyu/LEVIRCD_Cropped_256", repo_type="dataset", local_dir=str(dataset_dir))
    if dataset_dir is None:
        raise SystemExit("Pass --dataset-dir or --download. LEVIR-CD data is not included in the repository.")
    print(json.dumps(train(dataset_dir, args.project_root), indent=2))


if __name__ == "__main__":
    main()
