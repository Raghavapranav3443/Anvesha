"""Impact Analysis engine: turns a bi-temporal change mask into quantified,
spatially-contextualised findings (area, distance-to-water, ranked zones,
analyst actions). Deterministic GIS math over existing specialist outputs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------- #
# Geospatial helpers (numpy-only, no scipy)
# --------------------------------------------------------------------- #

def chamfer_distance(target: np.ndarray, max_px: int = 512) -> np.ndarray:
    """Approximate Euclidean distance transform (two-pass chamfer) toward the
    nearest True pixel of `target`. Distance in pixels."""
    t = target.astype(bool)
    if max_px and max(t.shape) > max_px:
        scale = max(t.shape) / max_px
        small_t = _resize_bool(t, max_px)
        d_small = _chamfer(small_t)
        return _resize_float(d_small, t.shape) * scale
    return _chamfer(t)


def _resize_bool(t: np.ndarray, max_px: int) -> np.ndarray:
    from PIL import Image
    h, w = t.shape
    s = max_px / max(h, w)
    im = Image.fromarray((t * 255).astype(np.uint8))
    im = im.resize((max(8, int(w * s)), max(8, int(h * s))), Image.NEAREST)
    return np.asarray(im) > 127


def _resize_float(a: np.ndarray, shape) -> np.ndarray:
    from PIL import Image
    im = Image.fromarray(np.clip(a * 16, 0, 255).astype(np.uint8))
    im = im.resize((shape[1], shape[0]), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 16.0


def _chamfer(t: np.ndarray) -> np.ndarray:
    INF = float(t.size)
    d = np.where(t, 0.0, INF).astype(np.float32)
    a, b = 1.0, 1.41421356
    # forward pass
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if d[i, j] == 0:
                continue
            best = d[i, j]
            if i > 0:
                best = min(best, d[i - 1, j] + a)
            if j > 0:
                best = min(best, d[i, j - 1] + a)
            if i > 0 and j > 0:
                best = min(best, d[i - 1, j - 1] + b)
            if i > 0 and j < d.shape[1] - 1:
                best = min(best, d[i - 1, j + 1] + b)
            d[i, j] = best
    # backward pass
    for i in range(d.shape[0] - 1, -1, -1):
        for j in range(d.shape[1] - 1, -1, -1):
            best = d[i, j]
            if i < d.shape[0] - 1:
                best = min(best, d[i + 1, j] + a)
            if j < d.shape[1] - 1:
                best = min(best, d[i, j + 1] + a)
            if i < d.shape[0] - 1 and j > 0:
                best = min(best, d[i + 1, j - 1] + b)
            if i < d.shape[0] - 1 and j < d.shape[1] - 1:
                best = min(best, d[i + 1, j + 1] + b)
            d[i, j] = best
    return d


def gsd_meters(img) -> float:
    """Ground sample distance in metres/pixel. Uses geotransform when
    available; geographic CRS converted via latitude; otherwise labelled
    10 m assumption (Sentinel-like default)."""
    try:
        tb = img.transform_bounds
        if tb:
            x0, y0, x1, y1 = tb
            w_m = x1 - x0
            lat = (y0 + y1) / 2
            if abs(w_m) <= 1.0:                      # degrees (EPSG:4326-ish)
                w_m *= 111_320.0 * max(abs(np.cos(np.radians(lat))), 0.01)
            return max(abs(w_m) / img.width, 0.05)
    except Exception:
        pass
    return 10.0                                      # labelled assumption


def px_to_ha(px_count: int, gsd_m: float) -> float:
    return round(px_count * gsd_m * gsd_m / 10_000.0, 3)


# --------------------------------------------------------------------- #
# Concept masks (reuse grounder score maps)
# --------------------------------------------------------------------- #

def concept_mask(img, concept: str, thr_pct: float = 82) -> np.ndarray:
    from .models.grounder import _score_map
    sm = _score_map(img, concept)
    thr = max(float(np.percentile(sm, thr_pct)), 0.55)
    return sm >= thr


def frac(mask_a: np.ndarray, region: np.ndarray) -> float:
    r = region.sum()
    return float((mask_a & region).sum() / r) if r else 0.0


# --------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------- #

ZONES = [("NW", "NE"), ("SW", "SE")]


def analyse_impact(a, b, change_mask: np.ndarray,
                   query: str = "") -> Dict[str, Any]:
    """Quantify + contextualise `change_mask` between images a→b."""
    H, W = change_mask.shape
    total_px = H * W
    gsd = gsd_meters(b)

    changed = change_mask.astype(bool)
    changed_px = int(changed.sum())
    area_ha = px_to_ha(changed_px, gsd)

    # --- concept masks -------------------------------------------------
    water_b = concept_mask(b, "water")
    water_a = concept_mask(a, "water")
    water_any = water_a | water_b
    veg_a = concept_mask(a, "vegetation")
    veg_b = concept_mask(b, "vegetation")
    built_a = concept_mask(a, "built-up")
    built_b = concept_mask(b, "built-up")

    dist_to_water = chamfer_distance(water_any, max_px=512) * gsd  # metres

    # --- spatial context ----------------------------------------------
    near = {
        "within_250m": float((changed & (dist_to_water <= 250)).sum() /
                             max(changed_px, 1)),
        "within_500m": float((changed & (dist_to_water <= 500)).sum() /
                             max(changed_px, 1)),
        "within_1000m": float((changed & (dist_to_water <= 1000)).sum() /
                              max(changed_px, 1)),
    }

    # --- concept transitions inside changed area -----------------------
    veg_lost_px = int((veg_a & ~veg_b & changed).sum())
    veg_gained_px = int((~veg_a & veg_b & changed).sum())
    built_new_px = int((built_b & ~built_a & changed).sum())
    built_lost_px = int((built_a & ~built_b & changed).sum())

    veg_lost_ha = px_to_ha(veg_lost_px, gsd)
    built_new_ha = px_to_ha(built_new_px, gsd)

    # --- zone ranking (4x4) --------------------------------------------
    gh, gw = H // 4, W // 4
    zones = []
    row_names = ["N", "N", "S", "S"]
    col_names = ["W", "", "E"]
    for zi in range(4):
        for zj in range(4):
            cell = changed[zi * gh:(zi + 1) * gh, zj * gw:(zj + 1) * gw]
            cpx = int(cell.sum())
            if cpx == 0:
                continue
            dw = dist_to_water[zi * gh:(zi + 1) * gh,
                               zj * gw:(zj + 1) * gw][cell]
            near_w = float((dw <= 500).mean()) if dw.size else 0.0
            col_tag = "W" if zj == 0 else ("E" if zj >= 2 else "C")
            row_tag = row_names[zi]
            zones.append({
                "zone": f"{row_tag}-{col_tag}{zj % 2 + 1}",
                "box": [zj * gw, zi * gh, (zj + 1) * gw, (zi + 1) * gh],
                "area_ha": px_to_ha(cpx, gsd),
                "near_water_frac": round(near_w, 3),
                "priority": round(cpx / total_px *
                                  (1.0 + near_w), 5),
            })
    zones.sort(key=lambda z: -z["priority"])

    # --- findings --------------------------------------------------------
    findings: List[Dict[str, str]] = []
    delta_bu = built_new_px - built_lost_px
    if built_new_px > 0 and near["within_500m"] > 0.25:
        findings.append({
            "what": f"New built-up development (~{built_new_ha} ha)",
            "where": (f"{near['within_500m'] * 100:.0f}% lies within 500 m "
                      "of a water body"),
            "action": "Possible encroachment near water — recommend field "
                      "verification and records check.",
        })
    elif delta_bu > 0:
        findings.append({"what": f"Built-up growth (~{built_new_ha} ha net)",
                         "where": "across the changed region",
                         "action": "Routine monitoring."})
    if veg_lost_ha > max(0.05, area_ha * 0.15):
        findings.append({
            "what": f"Vegetation cleared (~{veg_lost_ha} ha)",
            "where": "inside the detected change region",
            "action": "Verify potential land clearance / deforestation.",
        })
    if not findings:
        findings.append({"what": "No significant thematic transition",
                         "where": "across the scene",
                         "action": "Continue routine monitoring."})

    return {
        "gsd_m": round(gsd, 2),
        "gsd_assumed": not bool(getattr(b, "transform_bounds", None)),
        "changed_area_ha": area_ha,
        "changed_fraction": round(changed_px / total_px, 4),
        "near_water": {k: round(v, 3) for k, v in near.items()},
        "transitions": {
            "vegetation_lost_ha": veg_lost_ha,
            "vegetation_gained_ha": px_to_ha(veg_gained_px, gsd),
            "built_up_new_ha": built_new_ha,
            "built_up_lost_ha": px_to_ha(built_lost_px, gsd),
        },
        "zones_top": zones[:6],
        "findings": findings,
        "priority_zone": zones[0]["zone"] if zones else "",
        "distance_model": "chamfer EDT to combined water extent",
    }
