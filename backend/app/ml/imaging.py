"""Image loading and preprocessing shared by the raster models.

Georeferencing is preserved when the source raster carries it. Pixel space and
geographic space are never conflated.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np

from app.core.errors import InvalidInputError, ModelCompatibilityError

ImageInput = Union[str, Path, bytes, "np.ndarray", "LoadedImage"]


@dataclass
class LoadedImage:
    """An image plus whatever georeferencing accompanied it."""

    array: np.ndarray  # HxWxC, float32 or uint8
    crs: str | None = None
    transform: tuple[float, float, float, float, float, float] | None = None
    source: str | None = None

    @property
    def georeferenced(self) -> bool:
        return self.crs is not None and self.transform is not None

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def bands(self) -> int:
        return int(self.array.shape[2]) if self.array.ndim == 3 else 1

    def describe(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "bands": self.bands,
            "dtype": str(self.array.dtype),
            "crs": self.crs,
            "transform": list(self.transform) if self.transform else None,
            "georeferenced": self.georeferenced,
            "source": self.source,
        }


def read_image(value: ImageInput) -> LoadedImage:
    if isinstance(value, LoadedImage):
        return value
    if isinstance(value, np.ndarray):
        return LoadedImage(array=_as_hwc(value), source="array")
    if isinstance(value, (str, Path)):
        return _read_path(Path(value))
    if isinstance(value, (bytes, bytearray)):
        return _read_bytes(bytes(value))
    raise InvalidInputError(f"unsupported image input type: {type(value).__name__}")


def _read_path(path: Path) -> LoadedImage:
    if not path.is_file():
        raise InvalidInputError(f"image file not found: {path}")

    raster = _try_rasterio(path)
    if raster is not None:
        return raster

    return _read_bytes(path.read_bytes(), source=str(path))


def _try_rasterio(path: Path) -> LoadedImage | None:
    try:
        import rasterio
    except ImportError:
        return None
    try:
        with rasterio.open(path) as dataset:
            array = dataset.read()
            crs = str(dataset.crs) if dataset.crs else None
            transform = tuple(dataset.transform)[:6] if dataset.transform else None
            if crs is None and transform is not None and _is_identity(transform):
                transform = None
            return LoadedImage(
                array=_as_hwc(np.moveaxis(array, 0, -1)),
                crs=crs,
                transform=transform,
                source=str(path),
            )
    except Exception:  # noqa: BLE001 - fall back to a plain image reader
        return None


def _is_identity(transform: tuple[float, ...]) -> bool:
    return tuple(round(v, 9) for v in transform[:6]) == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _read_bytes(payload: bytes, source: str = "bytes") -> LoadedImage:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise InvalidInputError("Pillow is required to decode this image") from exc
    with Image.open(io.BytesIO(payload)) as handle:
        array = np.array(handle.convert("RGB"))
    return LoadedImage(array=_as_hwc(array), source=source)


def _as_hwc(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[:, :, None]
    if array.ndim == 3:
        return array
    raise InvalidInputError(f"expected a 2D or 3D image array, got shape {array.shape}")


# ----------------------------------------------------------------------
# preprocessing
# ----------------------------------------------------------------------
def _require(preprocessing: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in preprocessing and preprocessing[key] is not None:
            return preprocessing[key]
    return None


def resolve_input_size(
    model_key: str, preprocessing: dict[str, Any]
) -> tuple[int, int] | None:
    size = _require(preprocessing, ("input_size", "image_size", "img_size", "size"))
    if size is None:
        return None
    if isinstance(size, int):
        return size, size
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[0]), int(size[1])
    raise ModelCompatibilityError(
        model_key, f"unsupported input_size descriptor: {size!r}"
    )


def prepare_array(
    model_key: str,
    array: np.ndarray,
    preprocessing: dict[str, Any],
) -> np.ndarray:
    """Apply the recorded preprocessing and return a CHW float32 array."""

    image = array.astype(np.float32)

    scale = _require(preprocessing, ("scale", "rescale"))
    if scale is None and image.max() > 1.0:
        scale = 1.0 / 255.0
    if scale:
        image = image * float(scale)

    size = resolve_input_size(model_key, preprocessing)
    if size is not None and (image.shape[0], image.shape[1]) != size:
        image = _resize(image, size)

    mean = _require(preprocessing, ("mean", "normalize_mean"))
    std = _require(preprocessing, ("std", "normalize_std"))
    if mean is not None:
        image = image - np.asarray(mean, dtype=np.float32)
    if std is not None:
        image = image / np.asarray(std, dtype=np.float32)

    return np.ascontiguousarray(np.moveaxis(image, -1, 0))


def _resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    height, width = size
    channels = image.shape[2]
    resized = np.empty((height, width, channels), dtype=np.float32)
    for index in range(channels):
        band = Image.fromarray(image[:, :, index], mode="F")
        resized[:, :, index] = np.asarray(
            band.resize((width, height), Image.BILINEAR), dtype=np.float32
        )
    return resized


def prepare_tensor_input(
    array: np.ndarray,
    preprocessing: dict[str, Any],
    device: str,
    model_key: str = "model",
):
    import torch

    prepared = prepare_array(model_key, array, preprocessing)
    return torch.from_numpy(prepared).unsqueeze(0).to(device)
