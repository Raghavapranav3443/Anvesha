"""Windowed Cloud-Optimized GeoTIFF reads onto a shared analysis grid.

What this module guarantees
---------------------------
Two dates fetched for the same place land on **one grid**: same CRS, same
transform, same pixel count. Pixel ``(row, col)`` therefore means the same patch
of ground in both images, which is the precondition for every change number the
pipeline produces. Getting this wrong does not crash -- it silently compares
different ground and reports confident, meaningless change.

Two traps this code exists to close, both found by testing rather than reading
---------------------------------------------------------------------------
1. **Degrees into a projected transform.** ``rasterio.windows.from_bounds``
   takes coordinates in the *dataset's* CRS. Passing a lon/lat AOI into a
   Sentinel-2 UTM tile yields a sub-pixel window that reads as an empty array
   (``shape (0, 0)``) and then fails on a reduction. The observed result of the
   naive version was a window 0.025 pixels wide. Every read here converts
   through :func:`rasterio.warp.transform_bounds` first, and the result is
   validated before use.

2. **Raw digital numbers.** The offline pipeline's own sample tiles are
   ``float32`` surface reflectance in ``[0, 1]`` (measured: 4-band, median
   0.2). Publishing raw Sentinel ``uint16`` DNs (0-10000) would put fetched
   imagery on a different radiometric scale from every benchmark tile the
   models were validated on, changing the statistics that
   ``modality_certainty`` votes on and the thresholds the demo suite pins. So
   conversion uses the ``scale``/``offset`` the STAC catalogue *declares* for
   each band, never a hardcoded constant, and clamps to the valid range.

Grid construction
-----------------
The target grid is a UTM zone chosen from the AOI's centre, covering the AOI
window at native 10 m ground sampling, capped at a maximum pixel count. UTM
keeps areas meaningful for the hectare figures the decision layer quotes;
``EPSG:4326`` would make pixels non-square in metres.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .errors import FetchFailed, ScopeTooLarge
from .providers.stac import S2_ASSET_ORDER, SceneRef

BBox = Tuple[float, float, float, float]

# Sentinel-2 L2A native ground sample distance, metres.
S2_GSD_M = 10.0
DEFAULT_MAX_PX = 1024
MIN_PX = 128
# Declared fallback if a catalogue omits raster:bands. Sentinel-2 L2A DNs are
# reflectance * 10000, so this is the documented relationship, not a guess.
FALLBACK_SCALE = 0.0001
FALLBACK_OFFSET = 0.0
# Below this fraction of valid pixels a fetch is refused rather than analysed.
MIN_VALID_FRACTION = 0.60
# Share of pixels clamped to exactly zero that means the radiometric convention
# is wrong rather than the scene simply being dark.
MAX_DEAD_FRACTION = 0.60
# Key the catalogues use to say the BOA offset is already baked into the pixels.
BOA_OFFSET_APPLIED_KEY = "earthsearch:boa_offset_applied"

# GDAL needs to be told it may open a remote COG and that directory listing is
# pointless (and slow) for object storage.
_VSI_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": str(32 * 1024 * 1024),
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
}


def apply_vsi_env() -> None:
    for key, value in _VSI_ENV.items():
        os.environ.setdefault(key, value)


@dataclass
class TargetGrid:
    """The shared analysis grid both dates are resampled onto."""

    epsg: int
    transform: Any                      # affine.Affine
    width: int
    height: int
    source: str = ""

    @property
    def crs(self):
        from rasterio.crs import CRS

        return CRS.from_epsg(self.epsg)

    @property
    def pixel_size_m(self) -> float:
        return abs(float(self.transform.a))

    @property
    def area_ha(self) -> float:
        return (self.width * self.pixel_size_m) * (self.height * self.pixel_size_m) / 10_000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epsg": self.epsg,
            "width": self.width,
            "height": self.height,
            "pixel_size_m": round(self.pixel_size_m, 2),
            "area_ha": round(self.area_ha, 1),
            "transform": list(self.transform)[:6],
            "source": self.source,
        }


def utm_epsg(lat: float, lon: float) -> int:
    """UTM zone EPSG for a coordinate. India is entirely in the northern hemisphere."""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = min(60, max(1, zone))
    return (32600 if lat >= 0 else 32700) + zone


def plan_grid(bbox: BBox, *, px: Optional[int] = None,
              max_px: int = DEFAULT_MAX_PX,
              gsd_m: float = S2_GSD_M) -> TargetGrid:
    """Build the analysis grid covering ``bbox`` (WGS84 lon/lat)."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    if px is None:
        from .mode import load_settings

        try:
            px = int((load_settings().get("acquire") or {}).get("window_px", DEFAULT_MAX_PX))
        except (TypeError, ValueError):
            px = DEFAULT_MAX_PX
    target_px = max(MIN_PX, min(int(max_px), int(px)))

    west, south, east, north = bbox
    lat, lon = (south + north) / 2.0, (west + east) / 2.0
    epsg = utm_epsg(lat, lon)
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", f"EPSG:{epsg}", west, south, east, north, densify_pts=21)

    width_m = max(1.0, right - left)
    height_m = max(1.0, top - bottom)
    longest = max(width_m, height_m)
    if longest > gsd_m * 20_000:                       # >200 km on a side
        raise ScopeTooLarge(
            f"analysis window is {longest/1000:.0f} km across; refusing a "
            f"windowed read at that extent")

    pixel = longest / float(target_px)
    pixel = max(gsd_m, pixel)                          # never upsample past native
    width = max(MIN_PX, int(round(width_m / pixel)))
    height = max(MIN_PX, int(round(height_m / pixel)))
    # Re-align so the request covers the AOI exactly with the chosen pixel size.
    transform = from_origin(left, top, pixel, pixel)
    return TargetGrid(epsg=epsg, transform=transform, width=width, height=height,
                      source="utm-from-aoi")


def read_band_to_grid(url: str, grid: TargetGrid, *, band: int = 1,
                      resampling: str = "bilinear",
                      timeout_s: float = 180.0
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Reproject one band of a remote COG onto ``grid``.

    Returns ``(digital_numbers, valid_mask)`` -- **raw** DNs, unconverted, plus
    an explicit boolean validity mask.

    Why the two are returned separately, and why that matters
    --------------------------------------------------------
    The obvious implementation returns converted reflectance and treats
    ``value > 0`` as "has data". That is wrong, and was wrong here in a way
    that produced a real failure: this catalogue declares
    ``scale=0.0001, offset=-0.1`` with ``earthsearch:boa_offset_applied: True``
    (processing baseline 05.12). After atmospheric correction, dark surfaces
    genuinely map to slight negative reflectance and are clamped to zero. So
    ``value > 0`` is False for perfectly good water and shadow pixels.

    Measured on the scene that first exposed this: the window reported **23%**
    valid under the value-based test while the tile actually covers the area
    **100%**. Had that threshold been lowered to make it pass, the app would
    have analysed imagery in which most real pixels were silently treated as
    missing. Validity comes from the nodata mask; value never decides it.

    The reproject path guarantees both dates share a grid even when two scenes
    fall in different UTM zones -- which happens near zone boundaries and is
    otherwise a silent misalignment.
    """
    apply_vsi_env()
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    method = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }.get(resampling, Resampling.bilinear)

    destination = np.zeros((grid.height, grid.width), dtype="float32")
    try:
        with rasterio.open(url) as src:
            if src.crs is None:
                raise FetchFailed(f"{url} has no CRS; cannot place it on a grid")
            try:
                src_band = rasterio.band(src, band)
            except Exception as exc:
                raise FetchFailed(f"{url}: band {band} unavailable ({exc})") from exc
            reproject(
                source=src_band,
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata if src.nodata is not None else 0,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                dst_nodata=0.0,
                resampling=method,
            )
    except FetchFailed:
        raise
    except Exception as exc:
        raise FetchFailed(f"reading {url} failed: {type(exc).__name__}: {exc}") from exc

    # The COG declares nodata=0, so a reprojected zero means "no source pixel
    # here". This mask is the only validity signal used anywhere downstream.
    valid = destination != 0
    return destination, valid


def to_reflectance(dn: np.ndarray, valid: Optional[np.ndarray] = None, *,
                   scale: Optional[float],
                   offset: Optional[float]) -> np.ndarray:
    """Convert digital numbers to surface reflectance in [0, 1].

    Uses the catalogue-declared scale/offset, falling back to the documented
    Sentinel-2 relationship only when the catalogue is silent. Clamping to
    ``[0, 1]`` is deliberate and matches both the physics (reflectance is
    non-negative) and the offline pipeline's own sample tiles, which are
    ``float32`` in ``[0, 1]``.

    ``valid`` is applied *after* conversion so that a pixel clamped to zero for
    being dark is not confused with a pixel that had no source data.
    """
    s = FALLBACK_SCALE if scale is None else float(scale)
    o = FALLBACK_OFFSET if offset is None else float(offset)
    out = dn.astype("float32") * s + o
    np.clip(out, 0.0, 1.0, out=out)
    if valid is not None:
        out[~valid] = 0.0
    return out


@dataclass(frozen=True)
class Conversion:
    """One radiometric convention: ``reflectance = dn * scale + offset``."""

    scale: float
    offset: float
    reason: str

    def __str__(self) -> str:
        return f"x{self.scale} + {self.offset} ({self.reason})"


def reflectance_conversion(scene: SceneRef) -> Conversion:
    """Choose the DN -> reflectance convention for a scene.

    The subtlety that made this necessary
    -------------------------------------
    Sentinel-2 L2A products from processing baseline 04.00 onwards carry
    ``BOA_ADD_OFFSET = -1000``, so ``reflectance = (dn - 1000) / 10000``. But the
    AWS Earth Search COGs advertise ``raster:bands.offset = -0.1`` **and** an
    item property ``earthsearch:boa_offset_applied: True``, meaning the offset is
    already baked into the pixel values. Subtracting it a second time drives a
    large share of otherwise ordinary pixels negative, and clamping then
    reports them as reflectance zero.

    Measured on ``S2B_46RFR_20260417`` over Assam, red band: median DN 901.
    Applying the offset gives a median red reflectance of **-0.01** for a
    landscape that is mostly tea garden, town and river -- physically
    impossible, and it forces NDVI above 1, which is also impossible. Using the
    scale alone gives 0.090, which is exactly what that surface should be.

    So the catalogue's own declaration wins, and the result is validated
    against physics in :func:`assess_radiometry` rather than trusted.
    """
    scale: Optional[float] = None
    offset: Optional[float] = None
    for logical in S2_ASSET_ORDER:
        if logical in scene.scales:
            scale = scene.scales[logical]
            offset = scene.offsets.get(logical)
            break
    if scale is None:
        scale = FALLBACK_SCALE
    boa_applied = scene.properties.get(BOA_OFFSET_APPLIED_KEY)
    if boa_applied is True and offset:
        return Conversion(scale, 0.0,
                          "catalogue reports the BOA offset is already applied")
    return Conversion(scale, float(offset or 0.0),
                      "catalogue-declared scale and offset applied")


def alternate_conversion(base: Conversion) -> Conversion:
    """The other convention, for the case where physics rejects the first."""
    if base.offset != 0.0:
        return Conversion(base.scale, 0.0, "alternate: offset dropped")
    return Conversion(base.scale, -0.1 * (base.scale / FALLBACK_SCALE),
                      "alternate: offset applied")


def assess_radiometry(stack: np.ndarray, valid: np.ndarray,
                      bands: Sequence[str]) -> Dict[str, Any]:
    """Check converted reflectance against what is physically possible.

    Two independent detectors, because the failure they catch is silent:

    * **NDVI range.** ``(nir-red)/(nir+red)`` with non-negative inputs cannot
      leave ``[-1, 1]``. If it does, a channel has gone negative and the
      convention is wrong.
    * **Dead-pixel share.** An over-applied offset drives most pixels to exactly
      zero after clamping. A genuinely dark scene does not do this uniformly
      across bands.
    """
    result: Dict[str, Any] = {"ok": True, "reason": "", "ndvi_median": None,
                              "dead_fraction": 0.0}
    lower = [b.lower() for b in bands]
    if stack.size == 0 or valid.size == 0 or not valid.any():
        result.update(ok=False, reason="no valid pixels to assess")
        return result

    dead = 0.0
    for index in range(stack.shape[0]):
        band = stack[index][valid]
        if band.size:
            dead = max(dead, float((band == 0.0).mean()))
    result["dead_fraction"] = round(dead, 4)

    if "red" in lower and "nir" in lower:
        red = stack[lower.index("red")][valid].astype("float64")
        nir = stack[lower.index("nir")][valid].astype("float64")
        denom = nir + red
        safe = np.abs(denom) > 1e-9
        if safe.any():
            ndvi = (nir[safe] - red[safe]) / denom[safe]
            median = float(np.median(ndvi))
            result["ndvi_median"] = round(median, 4)
            if not np.isfinite(median) or abs(median) > 1.0:
                result.update(ok=False,
                              reason=(f"NDVI median {median:.3f} is outside "
                                      f"[-1, 1], which non-negative "
                                      f"reflectance cannot produce"))
                return result

    if dead > MAX_DEAD_FRACTION:
        result.update(ok=False,
                      reason=(f"{dead*100:.0f}% of pixels clamped to zero, "
                              f"indicating an over-applied offset"))
    return result


@dataclass
class FetchedScene:
    """One date, on the shared grid, written to disk."""

    path: Path
    scene_id: str
    date: str
    provider: str
    epsg: int
    width: int
    height: int
    bands: List[str] = field(default_factory=list)
    valid_fraction: float = 0.0
    cloud_pct: Optional[float] = None
    asset_urls: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    conversion: str = ""
    radiometry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": str(self.path.name),
            "scene_id": self.scene_id,
            "date": self.date,
            "provider": self.provider,
            "epsg": self.epsg,
            "size": [self.width, self.height],
            "bands": list(self.bands),
            "valid_fraction": round(self.valid_fraction, 3),
            "cloud_pct": self.cloud_pct,
            "assets": dict(self.asset_urls),
            "warnings": list(self.warnings),
            "conversion": self.conversion,
            "radiometry": dict(self.radiometry),
        }


def fetch_scene(scene: SceneRef, grid: TargetGrid, out_path: Path, *,
                bands: Sequence[str] = S2_ASSET_ORDER,
                keep_scl: bool = True, min_valid: float = MIN_VALID_FRACTION
                ) -> FetchedScene:
    """Fetch one scene's bands onto ``grid`` and write a 4-band GeoTIFF.

    Band order matches the offline pipeline's own convention for a >=4-band
    raster (B02 blue, B03 green, B04 red, B08 nir), and the acquisition date is
    written both into the filename and into an ``acquisition_date`` tag so
    ``satquery.geodate`` recovers the real dates instead of the placeholders
    ``T1``/``T2``.

    Raw digital numbers are read **once** and kept, so that testing a second
    radiometric convention costs no additional network traffic -- the check is
    only useful if it is cheap enough to always run.
    """
    apply_vsi_env()
    missing = [b for b in bands if b not in scene.assets]
    if missing:
        raise FetchFailed(
            f"scene {scene.scene_id} is missing band(s) {missing}; "
            f"available: {sorted(scene.assets)}")

    raws: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for logical in bands:
        raw, valid = read_band_to_grid(scene.assets[logical], grid, band=1)
        raws.append(raw)
        masks.append(valid)

    # Valid only where EVERY band has source data: a pixel missing one band
    # cannot be scored, and averaging per-band fractions overstates coverage.
    combined = np.all(masks, axis=0) if masks else np.zeros(raws[0].shape, bool)
    valid_all = float(combined.mean()) if combined.size else 0.0
    if valid_all < min_valid:
        raise FetchFailed(
            f"scene {scene.scene_id} covers only {valid_all*100:.0f}% of the "
            f"analysis window (need {min_valid*100:.0f}%). Pick a different "
            f"scene or a smaller window.")

    def build(conv: Conversion) -> np.ndarray:
        return np.stack(
            [to_reflectance(raws[i], combined, scale=conv.scale, offset=conv.offset)
             for i in range(len(bands))], axis=0).astype("float32")

    warnings: List[str] = []
    conversion = reflectance_conversion(scene)
    stack = build(conversion)
    radiometry = assess_radiometry(stack, combined, bands)

    if not radiometry["ok"]:
        # The declared convention failed a physics check. Try the other one
        # before refusing, and say so loudly either way: shipping reflectance
        # that cannot exist would quietly corrupt every downstream number.
        alternative = alternate_conversion(conversion)
        alt_stack = build(alternative)
        alt_radiometry = assess_radiometry(alt_stack, combined, bands)
        if alt_radiometry["ok"]:
            warnings.append(
                f"radiometry: catalogue-declared conversion was rejected "
                f"({radiometry['reason']}); used {alternative}")
            conversion, stack, radiometry = alternative, alt_stack, alt_radiometry
        else:
            raise FetchFailed(
                f"scene {scene.scene_id}: no radiometric convention produced "
                f"physically possible reflectance. Declared: "
                f"{radiometry['reason']}. Alternate: {alt_radiometry['reason']}.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import rasterio

    descriptions = ["blue (B02)", "green (B03)", "red (B04)", "nir (B08)"][:len(bands)]
    tags = {
        "acquisition_date": scene.date,
        "scene_id": scene.scene_id,
        "provider": scene.provider,
        "collection": scene.collection,
        "product": "surface-reflectance",
        "reflectance_scale": str(conversion.scale),
        "reflectance_offset": str(conversion.offset),
        "reflectance_conversion": conversion.reason,
        "valid_fraction": f"{valid_all:.4f}",
        "ndvi_median": str(radiometry.get("ndvi_median")),
    }
    with rasterio.open(
        out_path, "w", driver="GTiff", height=grid.height, width=grid.width,
        count=len(bands), dtype="float32", crs=grid.crs, transform=grid.transform,
        nodata=0.0, compress="deflate", predictor=3, tiled=True,
        blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(stack)
        for index, name in enumerate(descriptions, start=1):
            dst.set_band_description(index, name)
        dst.update_tags(**tags)

    if scene.cloud_pct is not None and scene.cloud_pct > 20:
        warnings.append(f"scene cloud cover is {scene.cloud_pct:.0f}%")
    if valid_all < 0.995:
        missing = 100.0 * (1.0 - valid_all)
        warnings.append(
            f"{missing:.1f}% of this window has no source pixels (the scene "
            f"edge falls inside the analysis area)")
    if radiometry.get("dead_fraction", 0.0) > 0.25:
        warnings.append(
            f"{radiometry['dead_fraction']*100:.0f}% of pixels are at zero "
            f"reflectance (dark scene, or clipped atmospheric correction)")

    return FetchedScene(
        path=out_path, scene_id=scene.scene_id, date=scene.date,
        provider=scene.provider, epsg=grid.epsg, width=grid.width,
        height=grid.height, bands=list(bands),        valid_fraction=valid_all,
        cloud_pct=scene.cloud_pct,
        asset_urls={b: scene.assets[b] for b in bands if b in scene.assets},
        warnings=warnings,
        conversion=str(conversion),
        radiometry=dict(radiometry),
    )



def pick_pair(scenes: Sequence[SceneRef], *, min_gap_days: int = 7
              ) -> Tuple[Optional[SceneRef], Optional[SceneRef]]:
    """Choose the most-separated low-cloud pair from a set of scenes.

    Change detection needs two dates far enough apart that real change can
    occur. Picking the two best scenes by cloud alone often selects two days in
    the same week, which reports "no change" about a period in which nothing
    could have changed -- a true statement that means nothing.
    """
    usable = [s for s in scenes if s.has_bands and s.date]
    if not usable:
        return None, None
    usable.sort(key=lambda s: (s.date, s.cloud_pct if s.cloud_pct is not None else 999))
    newest = usable[-1]
    older_candidates = []
    for scene in usable[:-1]:
        gap = _days_between(scene.date, newest.date)
        if gap >= min_gap_days:
            older_candidates.append((gap, scene))
    if not older_candidates:
        # Nothing satisfies the gap; refuse rather than fabricate a comparison.
        return None, None
    # Prefer the largest gap, breaking ties on clearer sky.
    older_candidates.sort(key=lambda pair: (-pair[0],
                                            pair[1].cloud_pct if pair[1].cloud_pct is not None else 999))
    return older_candidates[0][1], newest


def _days_between(a: str, b: str) -> int:
    from datetime import date as _date

    try:
        ya, ma, da = (int(x) for x in a.split("-")[:3])
        yb, mb, db = (int(x) for x in b.split("-")[:3])
        return abs((_date(yb, mb, db) - _date(ya, ma, da)).days)
    except Exception:
        return 0


def write_provenance(path: Path, payload: Dict[str, Any]) -> Path:
    """Write a provenance sidecar next to the fetched imagery."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return sidecar
