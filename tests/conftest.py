"""Shared fixtures: synthetic remote-sensing rasters written as real GeoTIFFs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rasterio
from rasterio.transform import from_bounds


def _write_geotiff(path: Path, arr: np.ndarray, count: int, crs="EPSG:32633"):
    h, w = arr.shape[:2]
    bands = [arr[..., i] if arr.ndim == 3 else arr for i in range(count)]
    profile = dict(driver="GTiff", height=h, width=w, count=count, dtype="float32",
                   crs=crs, transform=from_bounds(0, 0, w * 10, h * 10, w, h))
    data = np.stack([b.astype(np.float32) for b in bands])
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        dst.descriptions = tuple(["S1_VV", "S1_VH"][:count] if "sar" in path.stem
                                 else [f"band{i+1}" for i in range(count)])
    return path


def make_rgb_scene(h=256, w=256, water_box=(150, 150, 230, 230),
                   veg=True) -> np.ndarray:
    """Green vegetated field with a blue water rectangle."""
    img = np.zeros((h, w, 3), dtype=np.float32)
    if veg:
        img[..., 1] = 0.55          # green
        img[..., 0] = 0.20 + np.random.rand(h, w) * 0.05
        img[..., 2] = 0.15
    x0, y0, x1, y1 = water_box
    img[y0:y1, x0:x1, 0] = 0.10
    img[y0:y1, x0:x1, 1] = 0.25
    img[y0:y1, x0:x1, 2] = 0.75     # strong blue -> water
    img += np.random.rand(h, w, 3) * 0.03
    return np.clip(img, 0, 1)


def make_sar(h=256, w=256, bright_boxes=()) -> np.ndarray:
    """2-band SAR-like raster: speckle over smooth background + bright structures."""
    base = 0.15 + 0.05 * np.random.randn(h, w)
    vv = np.clip(np.abs(base), 0.01, None)
    vh = vv * 0.6
    for (x0, y0, x1, y1) in bright_boxes:
        vv[y0:y1, x0:x1] = 0.9      # double-bounce like built-up
        vh[y0:y1, x0:x1] = 0.7
    return np.stack([vv, vh], axis=-1)


def make_multispectral(h=256, w=256) -> np.ndarray:
    """Sentinel-2-ish 4-band stack: B G R NIR; river of water pixels."""
    b = np.full((h, w), 0.08, np.float32)
    g = np.full((h, w), 0.30, np.float32)
    r = np.full((h, w), 0.22, np.float32)
    n = np.full((h, w), 0.60, np.float32)   # healthy vegetation NIR
    yy, xx = np.mgrid[0:h, 0:w]
    river = np.abs(yy - (h * 0.5 + 12 * np.sin(xx / 18))) < 9
    b[river], g[river], r[river], n[river] = 0.28, 0.32, 0.15, 0.04
    return np.stack([b, g, r, n], axis=-1)


@pytest.fixture(scope="session")
def rgb_png(tmp_path_factory):
    p = tmp_path_factory.mktemp("rgb")
    f = p / "scene_rgb.png"
    Image_write(f, make_rgb_scene())
    return f


@pytest.fixture(scope="session")
def ms_geotiff(tmp_path_factory):
    p = tmp_path_factory.mktemp("ms")
    return _write_geotiff(p / "sentinel2_patch.tif", make_multispectral(), 4)


@pytest.fixture(scope="session")
def sar_geotiff(tmp_path_factory):
    p = tmp_path_factory.mktemp("sar")
    arr = make_sar(bright_boxes=[(40, 40, 90, 90)])
    return _write_geotiff(p / "sentinel1_patch_sar.tif", arr, 2)


@pytest.fixture(scope="session")
def bitemporal_pair(tmp_path_factory):
    """Before: vegetation only. After: built-up blocks appear (increase)."""
    p = tmp_path_factory.mktemp("pair")
    before = make_rgb_scene(water_box=(200, 200, 210, 210))
    after = before.copy()
    # new buildings: bright grey boxes in lower-right quadrant
    rng = np.random.RandomState(7)
    for k in range(8):
        y = 160 + rng.randint(0, 70)
        x = 140 + rng.randint(0, 80)
        after[y:y + 14, x:x + 16] = np.array([0.75, 0.75, 0.72])
    fa = _write_geotiff(p / "t2020_optical.tif", before, 3)
    fb = _write_geotiff(p / "t2024_optical.tif", after, 3)
    return fa, fb


@pytest.fixture(scope="session")
def opt_sar_pair(tmp_path_factory):
    p = tmp_path_factory.mktemp("optsar")
    opt = _write_geotiff(p / "optical.tif", make_multispectral(), 4)
    sar = _write_geotiff(p / "sar.tif",
                         make_sar(bright_boxes=[(40, 40, 90, 90)]), 2)
    return opt, sar


@pytest.fixture(scope="session")
def bad_format_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("bad")
    f = p / "not_an_image.bmp"
    f.write_bytes(b"BM\x00\x00fake")
    return f


def Image_write(path: Path, arr: np.ndarray):
    from PIL import Image
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


# silence heavy model loading during most tests
@pytest.fixture(scope="session", autouse=True)
def _fast_env():
    import os
    os.environ.setdefault("SATQUERY_SKIP_TRAINING_DOWNLOADS", "1")


@pytest.fixture(scope="session")
def controller():
    from satquery.agent import AgentController
    return AgentController()
