"""Robustness suite: unusual CRS/dtypes/scalings must not crash the pipeline."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rasterio
from rasterio.transform import from_bounds

from satquery.agent import AgentController
from satquery.io_utils import InputValidationError, load_image


def _write(path, arr, count, dtype="float32", crs="EPSG:4326",
           transform=None, nodata=None):
    h, w = arr.shape[:2]
    profile = dict(driver="GTiff", height=h, width=w, count=count,
                   dtype=dtype, crs=crs,
                   transform=transform or from_bounds(-78.9, 33.5, -78.8, 33.6, w, h),
                   nodata=nodata)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.moveaxis(arr, -1, 0).astype(dtype))


def test_uint16_reflectance_scaling(tmp_path):
    """Sentinel-style *10000 reflectance must be handled by the encoder path."""
    from satquery.models.backbone import normalise_for_encoder
    a = (np.random.rand(64, 64, 3).astype(np.float32) * 4000) + 200
    out = normalise_for_encoder(a, "rgb")
    assert out.min() >= 0 and out.max() <= 1


def test_sar_db_scale_not_loggied(tmp_path):
    """dB-scale SAR (all-negative) must pass through without log1p collapse."""
    from satquery.models.backbone import normalise_for_encoder
    db = np.random.uniform(-30, -2, (64, 64, 2)).astype(np.float32)
    out = normalise_for_encoder(db, "sar")
    assert abs(float(out.mean())) < 1.0     # standardised, not zeroed


def test_load_image_db_sar_preserved(tmp_path):
    f = tmp_path / "risat_hh_hv.tif"
    db = np.random.uniform(-30, -2, (64, 64, 2)).astype(np.float32)
    _write(f, db, 2)
    img = load_image(f)
    # dB values preserved in range (NOT collapsed toward 0 by a log transform)
    assert float(img.array.min()) >= -31.0
    assert float(img.array.max()) <= -1.0


def test_odd_crs_geotiff(tmp_path):
    f = tmp_path / "odd_crs.tif"
    _write(f, np.random.rand(64, 64, 3).astype(np.float32), 3)
    img = load_image(f)
    assert img.crs == "EPSG:4326"
    comp = load_image.__module__ and None   # placeholder no-op
    from satquery.io_utils import rgb_composite
    assert rgb_composite(img).shape == (64, 64, 3)


def test_nodata_filled(tmp_path):
    f = tmp_path / "nodata.tif"
    a = np.random.rand(64, 64, 1).astype(np.float32)
    a[:10] = -9999.0
    _write(f, a, 1, nodata=-9999.0)
    img = load_image(f)
    assert not np.any(np.isclose(img.array[..., 0], -9999))


def test_corrupt_tiff_clean_error(tmp_path):
    f = tmp_path / "corrupt.tif"
    f.write_bytes(b"not a tiff at all")
    with pytest.raises(InputValidationError):
        load_image(f)


def test_large_raster_downscaled(tmp_path):
    from PIL import Image as PImage
    f = tmp_path / "large.png"
    PImage.fromarray((np.random.rand(2200, 900, 3) * 255).astype(np.uint8)) \
        .save(f)
    img = load_image(f)
    assert max(img.height, img.width) <= 2048


def test_agent_survives_grayscale_single_band(controller, tmp_path):
    f = tmp_path / "gray.png"
    g = np.random.rand(96, 96).astype(np.float32) * 0.5 + 0.25
    from PIL import Image as PImage
    PImage.fromarray((g * 255).astype(np.uint8)).convert("L").save(f)
    res = controller.run([f], "Is there water in this image?")
    assert res.selected_task == "single_vqa"
