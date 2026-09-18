"""B8 — pixel→lonlat projection.

Affine math reuses the pattern already proven in
``anvesha/server/rasterout.py::png_mask_to_geotiff``: the transform spans
``transform_bounds`` over the ORIGINAL (pre-downscale) pixel grid, so current
coordinates must be rescaled by ``original_size / current_size`` before
projection. Without a CRS the answer is honest pixel-space — never guessed.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from .extract import AcquiredInfo  # noqa: F401  (re-export convenience)

__all__ = ["GeoBox", "pixel_to_lonlat", "polygon_geojson", "quadrant_of"]


class GeoBox:                                    # plan Types: plain record
    def __init__(self, lonlat_polygon: List[List[float]], crs: Optional[str],
                 gsd_m: Optional[float], note: str) -> None:
        self.lonlat_polygon = lonlat_polygon
        self.crs = crs
        self.gsd_m = gsd_m
        self.note = note

    def to_dict(self) -> dict:
        return {"lonlat_polygon": self.lonlat_polygon, "crs": self.crs,
                "gsd_m": self.gsd_m, "note": self.note}


def quadrant_of(box: Sequence[float], w: int, h: int) -> str:
    """Cardinal position of a box centre (same vocabulary as change maps)."""
    x0, y0, x1, y1 = box
    dx = (x0 + x1) / 2.0 / max(w, 1) - 0.5
    dy = (y0 + y1) / 2.0 / max(h, 1) - 0.5
    ns = "north" if dy < -0.05 else ("south" if dy > 0.05 else "")
    ew = "west" if dx < -0.05 else ("east" if dx > 0.05 else "")
    return "-".join(p for p in (ns, ew) if p) or "centre"


def _affine(img):
    """(sx, sy, left, top, fx, fy) or None when not georeferenced.

    sx/sy: metres per ORIGINAL pixel; fx/fy: original-per-current scale.
    """
    bounds = getattr(img, "transform_bounds", None)
    crs = getattr(img, "crs", None)
    if not bounds or not crs:
        return None
    left, bottom, right, top = bounds
    orig_w = getattr(img, "original_width", None) or img.width
    orig_h = getattr(img, "original_height", None) or img.height
    sx = (right - left) / float(orig_w)
    sy = (top - bottom) / float(orig_h)
    fx = orig_w / float(img.width)
    fy = orig_h / float(img.height)
    return sx, sy, left, top, fx, fy


def pixel_to_lonlat(img, box: Sequence[float]) -> GeoBox:
    """Project a pixel box [x0, y0, x1, y1] to a lon/lat closed polygon."""
    aff = _affine(img)
    if aff is None:
        return GeoBox(lonlat_polygon=[], crs=None, gsd_m=None,
                      note="pixel-space (no CRS) — coordinates not guessed")
    sx, sy, left, top, fx, fy = aff
    x0, y0, x1, y1 = box
    gx0, gy0 = x0 * fx, y0 * fy
    gx1, gy1 = x1 * fx, y1 * fy
    lon0, lon1 = left + gx0 * sx, left + gx1 * sx
    lat0, lat1 = top - gy0 * sy, top - gy1 * sy
    poly = [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1],
            [lon0, lat0]]
    quad = quadrant_of(box, img.width, img.height)
    centre_lon, centre_lat = (lon0 + lon1) / 2.0, (lat0 + lat1) / 2.0
    note = (f"{quad} quadrant (~lat {centre_lat:.4f}, lon {centre_lon:.4f})")
    return GeoBox(lonlat_polygon=poly, crs=str(img.crs),
                  gsd_m=round((sx + sy) / 2.0, 3), note=note)


def polygon_geojson(boxes: Sequence[Sequence[float]], img) -> dict:
    """GeoJSON FeatureCollection for pixel boxes (empty when pixel-space)."""
    aff = _affine(img)
    features = []
    for box in boxes or []:
        gb = pixel_to_lonlat(img, box)
        if not gb.lonlat_polygon:
            continue
        features.append({
            "type": "Feature",
            "properties": {"kind": "box", "note": gb.note},
            "geometry": {"type": "Polygon", "coordinates": [gb.lonlat_polygon]},
        })
    return {
        "type": "FeatureCollection",
        "crs": str(img.crs) if aff else None,
        "pixel_bounds": list(img.transform_bounds) if aff else None,
        "features": features,
        "note": ("georeferenced" if aff else
                 "pixel-space (no CRS) — coordinates not guessed"),
    }
