"""Generate ISRO/SAC-style demonstration inputs: a co-registered
Cartosat-2S (optical) + RISAT (SAR, dB-scale HH/HV) pair with sensor-named
files so the full input pipeline (modality inference -> agent routing ->
fusion analysis) is exercised exactly as the SAC evaluation set would.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.config import CONFIG  # noqa: E402

import rasterio
from rasterio.transform import from_bounds


def cartosat_style(h=256, w=256) -> np.ndarray:
    """Optical RGB look: urban block + fields + river, ~0.5 m GSD feel."""
    img = np.zeros((h, w, 3), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    urban = (xx > 150) & (yy < 110)
    img[urban] = [0.55, 0.53, 0.50]
    img[~urban] = [0.34, 0.45, 0.20]
    stripe = (np.sin(yy / 7.0) > 0.4) & ~urban
    img[stripe] = [0.42, 0.55, 0.24]          # crop-row variation
    road = np.abs(yy - (h * 0.62)) < 3
    img[road] = [0.42, 0.42, 0.44]
    img += np.random.rand(h, w, 1).astype(np.float32) * 0.05
    return np.clip(img, 0, 1)


def risat_style(h=256, w=256) -> np.ndarray:
    """RISAT-like SAR in dB: speckled field texture, bright urban doubles,
    dark smooth river; values negative like real SAR products."""
    base_db = -18 - 6 * np.abs(np.random.randn(h, w))
    urban = (np.mgrid[0:h, 0:w][1] > 150) & (np.mgrid[0:h, 0:w][0] < 110)
    base_db[urban] += 14                      # strong double-bounce
    river = np.abs(np.mgrid[0:h, 0:w][0] - h * 0.62) < 4
    base_db[river] -= 12                      # specular smooth surface
    hh = np.clip(base_db, -38, -3)
    hv = hh - 8 - 3 * np.random.rand(h, w)
    return np.stack([hh, hv], axis=-1)


def write_tif(path: Path, arr: np.ndarray, count: int, descs):
    h, w = arr.shape[:2]
    profile = dict(driver="GTiff", height=h, width=w, count=count,
                   dtype="float32", crs="EPSG:7781",   # WGS84 / India NSF LCC
                   transform=from_bounds(77.0, 28.4, 77.12, 28.52, w, h))
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.moveaxis(arr, -1, 0).astype(np.float32))
        dst.descriptions = descs


def main():
    out = CONFIG.samples_dir
    out.mkdir(exist_ok=True)
    opt = cartosat_style()
    sar = risat_style()
    write_tif(out / "isro_cartosat2s_optical.tif", opt, 3,
              ("cartosat_red", "cartosat_green", "cartosat_blue"))
    write_tif(out / "isro_risat_sar.tif", sar, 2,
              ("risat_hh", "risat_hv"))
    print("ISRO-style samples written to", out)


if __name__ == "__main__":
    main()
