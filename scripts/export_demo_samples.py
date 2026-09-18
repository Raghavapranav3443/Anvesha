"""Export bundled demo samples so the GUI works out of the box.

Creates GeoTIFF/PNG samples for all four input configurations:
single optical, bi-temporal pair, optical-SAR pair.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from anvesha.config import CONFIG  # noqa: E402

import rasterio
from rasterio.transform import from_bounds


def write_tif(path: Path, arr: np.ndarray, count: int):
    h, w = arr.shape[:2]
    profile = dict(driver="GTiff", height=h, width=w, count=count,
                   dtype="float32", crs="EPSG:32633",
                   transform=from_bounds(445000, 5330000, 446200, 5331200, w, h))
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.stack([arr[..., i].astype(np.float32) for i in range(count)]))
        descs = ([f"band{i+1}" for i in range(count)])
        dst.descriptions = tuple(descs)


def main():
    from conftest import make_rgb_scene, make_sar, make_multispectral
    out = CONFIG.samples_dir
    out.mkdir(exist_ok=True)

    # single optical scene (PNG, RSVQA-style benchmark format)
    rgb = make_rgb_scene()
    save = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    rasterio_png = None
    from PIL import Image
    Image.fromarray(save).save(out / "demo_single_optical.png")

    # single multispectral GeoTIFF with a river
    ms = make_multispectral()
    write_tif(out / "demo_single_multispectral.tif", ms, 4)

    # bi-temporal pair: built-up grows between dates
    before = rgb.copy()
    after = rgb.copy()
    rng = np.random.RandomState(3)
    for k in range(10):
        y, x = rng.randint(150, 230), rng.randint(30, 110)
        after[y:y + 16, x:x + 20] = np.array([0.78, 0.76, 0.72])
    write_tif(out / "demo_change_2020.tif", before, 3)
    write_tif(out / "demo_change_2024.tif", after, 3)

    # optical + SAR pair of the same area (built-up bright in SAR)
    sar = make_sar(bright_boxes=[(40, 40, 90, 90), (160, 60, 200, 100)])
    write_tif(out / "demo_pair_optical.tif", ms, 4)
    write_tif(out / "demo_pair_sar.tif", sar, 2)

    print("demo samples written to", out)


if __name__ == "__main__":
    main()
