"""Convert raster masks into polygons.

Polygons are emitted in geographic coordinates only when the source raster
supplied both a CRS and a geotransform. Otherwise they stay in pixel space and
are explicitly labelled as such.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

PIXEL_SPACE = "pixel"
GEOGRAPHIC_SPACE = "crs"


def polygonize_mask(
    binary_mask: np.ndarray,
    transform: Sequence[float] | None = None,
    crs: str | None = None,
    min_pixels: int = 1,
    simplify_tolerance: float | None = None,
) -> dict[str, Any]:
    """Vectorise a binary mask into a FeatureCollection-like payload."""

    from rasterio import features
    from rasterio.transform import Affine, IDENTITY
    from shapely.geometry import mapping, shape

    georeferenced = transform is not None and crs is not None
    affine = Affine(*transform[:6]) if georeferenced else IDENTITY
    mask = binary_mask.astype(np.uint8)

    pixel_area = abs(affine.a * affine.e - affine.b * affine.d) if georeferenced else 1.0
    min_area = float(min_pixels) * pixel_area

    out_features: list[dict[str, Any]] = []
    for geom, value in features.shapes(mask, mask=mask > 0, transform=affine):
        if int(value) == 0:
            continue
        geometry = shape(geom)
        if geometry.is_empty or geometry.area < min_area:
            continue
        if simplify_tolerance:
            geometry = geometry.simplify(simplify_tolerance, preserve_topology=True)
        out_features.append(
            {
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": {
                    "area": float(geometry.area),
                    "area_unit": "crs_units" if georeferenced else "pixels",
                    "pixel_count": None if georeferenced else int(geometry.area),
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "coordinate_space": GEOGRAPHIC_SPACE if georeferenced else PIXEL_SPACE,
        "crs": crs if georeferenced else None,
        "is_pixel_coordinates": not georeferenced,
        "features": out_features,
    }
