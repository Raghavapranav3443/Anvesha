"""Image loading, modality inference, normalisation and input validation.

Supported inputs
----------------
* GeoTIFF / TIFF  (rasterio): multispectral (Sentinel-2 style), SAR
  (Sentinel-1 style VV/VH) or RGB, georeferenced or not.
* PNG / JPEG      (accepted for prescribed public benchmark datasets such as
  RSVQA / VRSBench / LEVIR-CD).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import rasterio
    HAS_RASTERIO = True
except Exception:  # pragma: no cover - rasterio is a hard dep but keep safe
    HAS_RASTERIO = False

from PIL import Image

GEOTIFF_EXTS = {".tif", ".tiff"}
BENCH_EXTS = {".png", ".jpg", ".jpeg"}
ALLOWED_EXTS = GEOTIFF_EXTS | BENCH_EXTS

SAR_HINTS = ("s1", "sentinel-1", "sentinel_1", "sar", "vv", "vh",
             "risat", "rsat", "hh", "hv")
OPTICAL_HINTS = ("cartosat", "s2", "sentinel-2", "sentinel_2", "optical")
OPTICAL_MS_BANDS = ["blue", "green", "red", "nir"]


class InputValidationError(ValueError):
    """Raised when user input violates the supported input scope."""


@dataclass
class RSImage:
    """A loaded remote-sensing raster."""
    array: np.ndarray                 # HxWxC float32
    format: str                       # 'geotiff' | 'png' | 'jpeg'
    path: Optional[Path] = None
    modality: str = "rgb"             # 'sar' | 'multispectral' | 'rgb' | 'grayscale'
    band_names: List[str] = field(default_factory=list)
    crs: Optional[str] = None
    transform_bounds: Optional[Tuple[float, float, float, float]] = None
    acquired: str = ""                # free-form date string if known

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def bands(self) -> int:
        return int(self.array.shape[2])

    def summary(self) -> dict:
        return {
            "file": self.path.name if self.path else "<array>",
            "format": self.format,
            "modality": self.modality,
            "size": [self.width, self.height],
            "bands": self.bands,
            "band_names": self.band_names[:6],
            "georeferenced": self.crs is not None,
            "crs": self.crs,
            "bounds": list(self.transform_bounds) if self.transform_bounds else None,
        }


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #

def _infer_modality(name: str, bands: int, descriptions: Sequence[str]) -> Tuple[str, List[str]]:
    low = name.lower()
    desc = " ".join(str(d).lower() for d in descriptions)
    if any(h in low or h in desc for h in SAR_HINTS) and \
            not any(h in low for h in OPTICAL_HINTS):
        pols = [p for p in ("HH", "HV", "VV", "VH")
                if p.lower() in low or p in desc]
        names = pols[:2] if len(pols) >= 2 else (["HH", "HV"] if bands >= 2 else ["intensity"])
        return "sar", (names + [f"ch{i}" for i in range(bands)])[:bands]
    if bands == 1:
        return "grayscale", ["gray"]
    if bands == 2:
        return "sar", ["VV", "VH"]
    if bands == 3:
        return "rgb", ["red", "green", "blue"]
    # >=4 bands -> Sentinel-2 style ordering B02,B03,B04,B08,...
    base = ["B02 blue", "B03 green", "B04 red", "B08 nir",
            "B05 rededge1", "B06 rededge2", "B07 rededge3",
            "B8A nir-narrow", "B11 swir1", "B12 swir2",
            "B01 coastal", "B09 wvapor", "B10 cirrus"]
    return "multispectral", (base + [f"ch{i}" for i in range(bands)])[:bands]


def _downscale(arr: np.ndarray, max_px: int) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = max(h, w) / float(max_px)
    if scale <= 1.0:
        return arr
    nh, nw = max(16, int(h / scale)), max(16, int(w / scale))
    img = Image.fromarray(_to_uint8_display(arr))
    img = img.resize((nw, nh), Image.BILINEAR)
    out = np.asarray(img, dtype=np.float32) / 255.0
    if out.ndim == 2:
        out = out[..., None]
    return out


def _to_uint8_display(arr: np.ndarray) -> np.ndarray:
    """Percentile stretch to 0..255 for PIL-based resampling."""
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, 1.0), np.percentile(a, 99.5)
    if hi <= lo:
        hi = lo + 1e-6
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    return (a * 255).astype(np.uint8)


def load_image(path: str | Path, max_px: Optional[int] = None,
               modality_override: Optional[str] = None) -> RSImage:
    """Load a raster.

    ``modality_override``: "sar" | "optical" | None. When set, it forces the
    modality label instead of the filename/statistics heuristic — removing
    the single point of failure where a power-scale SAR product with an
    optical-sounding name (or vice versa) is misrouted. The dB-vs-power
    normalisation check still runs on the actual pixel statistics.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise InputValidationError(
            f"Unsupported file format '{ext}' for '{path.name}'. "
            f"Use GeoTIFF/TIFF for geospatial imagery; PNG/JPEG only for the "
            f"prescribed public benchmark datasets."
        )
    if not path.exists():
        raise InputValidationError(f"File not found: {path}")

    if ext in GEOTIFF_EXTS:
        if not HAS_RASTERIO:
            raise InputValidationError("rasterio is required for GeoTIFF input.")
        try:
            with rasterio.open(path) as src:
                raw = src.read().astype(np.float32)          # CxHxW
                arr = np.moveaxis(raw, 0, -1)
                crs = str(src.crs) if src.crs else None
                b = src.bounds
                bounds = (b.left, b.bottom, b.right, b.top) if b else None
                descriptions = list(src.descriptions or [])
                nodata = src.nodata
        except InputValidationError:
            raise
        except Exception as e:
            raise InputValidationError(
                f"Could not read '{path.name}' as a valid GeoTIFF/TIFF raster "
                f"({type(e).__name__}). Re-export the file or verify it is a "
                f"valid geospatial raster.") from e
        if nodata is not None:
            mask = np.isclose(arr, float(nodata))
            if mask.any():
                fill = np.nanmedian(np.where(mask, np.nan, arr))
                fill = 0.0 if not np.isfinite(fill) else float(fill)
                arr[mask] = fill
        modality, band_names = _infer_modality(path.name, arr.shape[2], descriptions)
        fmt = "geotiff"
    else:
        pil = Image.open(path).convert("RGB")
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        crs, bounds, descriptions = None, None, []
        modality, band_names = _infer_modality(path.name, 3, [])
        fmt = "jpeg" if ext in {".jpg", ".jpeg"} else "png"

    limit = max_px or _config_max_px()
    if max(arr.shape[0], arr.shape[1]) > limit:
        arr = _downscale(arr, limit)

    # Explicit user/judge override wins over the naming heuristic
    if modality_override:
        ov = modality_override.strip().lower()
        if ov in ("sar", "optical"):
            if ov == "sar":
                modality = "sar"
                band_names = (["VV", "VH"] + [f"ch{i}" for i in range(arr.shape[2])])[:arr.shape[2]] \
                    if arr.ndim == 3 and arr.shape[2] >= 2 else ["intensity"]
            else:
                modality = "multispectral" if (arr.ndim == 3 and arr.shape[2] >= 4) else "rgb"
                base = ["red", "green", "blue"]
                band_names = (base + [f"ch{i}" for i in range(arr.shape[2])])[:arr.shape[2]] \
                    if arr.ndim == 3 else ["gray"]

    # SAR amplitude normalisation (log-scale); dB-scale products are kept as-is
    if modality == "sar" and arr.size and float(np.median(arr)) >= 0:
        arr = np.log1p(arr)

    return RSImage(array=arr, format=fmt, path=path, modality=modality,
                   band_names=band_names, crs=crs, transform_bounds=bounds)


def _config_max_px() -> int:
    from .config import CONFIG
    return CONFIG.max_image_px


def rgb_composite(img: RSImage) -> np.ndarray:
    """Return an HxWx3 display composite in [0,1]."""
    a = img.array
    c = a.shape[2]
    if c >= 4:  # assume S2-ish order blue,green,red,nir...
        idx = [2, 1, 0]
    elif c == 3:
        idx = [0, 1, 2]
    elif c == 2:
        idx = [0, 1, 0]
    else:
        idx = [0, 0, 0]
    comp = a[..., idx]
    lo = np.percentile(comp, 1.0)
    hi = np.percentile(comp, 99.5)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((comp - lo) / (hi - lo), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Validation of input configurations
# --------------------------------------------------------------------------- #

VALID_CONFIGS = {
    1: "single image (optical/multispectral or SAR)",
    2: "image pair (bi-temporal or co-registered optical-SAR)",
}


def validate_inputs(images: Sequence[RSImage]) -> dict:
    """Validate number/modality/geometry compatibility. Returns configuration."""
    n = len(images)
    if n == 0:
        raise InputValidationError("No images supplied. Upload at least one image.")
    if n > 2:
        raise InputValidationError(
            "At most 2 images are supported (single image, bi-temporal pair "
            "or co-registered optical-SAR pair)."
        )

    cfg: dict[str, Any] = {"num_images": n}

    if n == 1:
        img = images[0]
        cfg["configuration"] = "single"
        cfg["modalities"] = [img.modality]
        return cfg

    a, b = images
    if (a.width, a.height) != (b.width, b.height):
        raise InputValidationError(
            f"Image pair is not spatially compatible: {a.path.name if a.path else 'img1'} "
            f"is {a.width}x{a.height}, but {b.path.name if b.path else 'img2'} is "
            f"{b.width}x{b.height}. Co-registration requires identical pixel grids."
        )
    if a.crs and b.crs and a.crs != b.crs:
        raise InputValidationError(
            f"CRS mismatch between pair ({a.crs} vs {b.crs}). Re-project to a common CRS first."
        )

    modalities = {a.modality, b.modality}
    sar_count = sum(1 for m in (a.modality, b.modality) if m == "sar")

    if sar_count == 2:
        raise InputValidationError(
            "Two SAR images detected. Cross-modal analysis requires one optical/"
            "multispectral and one SAR image; use change mode for two same-modality dates."
        )
    if sar_count == 1 and len(modalities & {"rgb", "multispectral", "grayscale"}) == 1:
        cfg["configuration"] = "optical_sar_pair"
    else:
        cfg["configuration"] = "bitemporal_pair"
    cfg["modalities"] = [a.modality, b.modality]
    return cfg


def describe_configuration(images: Sequence[RSImage], cfg: dict) -> str:
    if cfg["configuration"] == "single":
        return f"Single {cfg['modalities'][0]} image."
    if cfg["configuration"] == "optical_sar_pair":
        return "Co-registered optical/multispectral + SAR pair."
    return "Bi-temporal image pair (same modality, different dates)."
