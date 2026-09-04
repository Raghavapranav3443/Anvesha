"""Convert a PNG change mask into a georeferenced GeoTIFF."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def png_mask_to_geotiff(png_path: Path, reference) -> Path:
    import rasterio
    from rasterio.transform import from_bounds
    from PIL import Image

    mask = (np.asarray(Image.open(png_path).convert("L")) > 127).astype(np.uint8) * 255
    out = png_path.with_suffix(".tif")
    h, w = mask.shape
    if reference is not None and getattr(reference, "crs", None):
        # Use original dimensions (before downscale) for the transform so the
        # pixel size is correct even when the mask was downscaled for storage.
        orig_h = getattr(reference, "original_height", h) or h
        orig_w = getattr(reference, "original_width", w) or w
        transform = from_bounds(*reference.transform_bounds, orig_w, orig_h) \
            if reference.transform_bounds else from_bounds(0, 0, w * 10, h * 10, w, h)
        crs = reference.crs
    else:
        transform, crs = from_bounds(0, 0, w * 10, h * 10, w, h), None
    with rasterio.open(out, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="uint8", crs=crs, transform=transform,
                       nodata=0) as dst:
        dst.write(mask, 1)
        dst.descriptions = ("change_mask",)
    return out
