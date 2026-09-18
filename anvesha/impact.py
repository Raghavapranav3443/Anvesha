"""Impact Analysis engine: turns a bi-temporal change mask into quantified,
spatially-contextualised findings (area, distance-to-water, ranked zones,
analyst actions). Deterministic GIS math over existing specialist outputs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _scipy_edt_available() -> bool:
    try:
        from scipy.ndimage import distance_transform_edt  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------- #
# Geospatial helpers
# --------------------------------------------------------------------- #

def chamfer_distance(target: np.ndarray, max_px: int = 512) -> np.ndarray:
    """True Euclidean distance transform toward the nearest True pixel of
    *target*.  Distance in pixels.

    Uses ``scipy.ndimage.distance_transform_edt`` when available (correct,
    fast C implementation).  Falls back to a two-pass chamfer approximation
    when scipy is absent (air-gapped edge case).
    """
    t = target.astype(bool)
    if _scipy_edt_available():
        from scipy.ndimage import distance_transform_edt as _edt
        return _edt(~t).astype(np.float32)
    # Fallback: chamfer approximation (not metric-accurate)
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
    """Two-pass chamfer approximation (fallback when scipy is unavailable)."""
    INF = float(t.size)
    d = np.where(t, 0.0, INF).astype(np.float32)
    a, b = 1.0, 1.41421356
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


def label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """8-connected components of a boolean mask, 0 meaning background.

    scipy does this in C when it is available. Without it, an iterative flood
    fill over the True pixels only -- exactly as correct, and it pays for the
    foreground rather than the whole scene, which is the right trade because
    the masks this is used on are a small fraction of the image.
    """
    m = mask.astype(bool)
    if not m.any():
        return np.zeros(m.shape, dtype=np.int32), 0
    try:
        from scipy.ndimage import label as _label
        labels, n = _label(m, structure=np.ones((3, 3), dtype=int))
        return labels.astype(np.int32), int(n)
    except ImportError:
        pass
    labels = np.zeros(m.shape, dtype=np.int32)
    remaining = {(int(r), int(c)) for r, c in np.argwhere(m)}
    n = 0
    while remaining:
        n += 1
        stack = [remaining.pop()]
        labels[stack[0]] = n
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    p = (r + dr, c + dc)
                    if p in remaining:
                        remaining.discard(p)
                        labels[p] = n
                        stack.append(p)
    return labels, n


def region_stats(mask: np.ndarray, gsd: float) -> Dict[str, Any]:
    """How a mask is *arranged*: component count, and how much of it is in the
    single largest component.

    Area alone cannot tell construction from a change of surface. Four months
    of building work tends to appear as many small parcels, roof-sized patches
    and linear road or bund features, while a sediment bar, an exposed river
    bed, an embankment or one large earthworks site produces one compact block.
    ``largest_share`` is the fact that separates those two readings.
    """
    m = mask.astype(bool)
    total = int(m.sum())
    if total == 0:
        return {"regions": 0, "largest_ha": 0.0, "largest_share": 0.0}
    labels, n = label_components(m)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(np.int64)
    largest = int(sizes.max()) if sizes.size else 0
    return {
        "regions": int(n),
        "largest_ha": px_to_ha(largest, gsd),
        "largest_share": round(largest / total, 4),
    }


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
    # The arrangement of the new built-up ground, not just its area: the
    # decision layer needs both to avoid calling a change of surface
    # "construction" (Decisions.md D27/D29).
    built_new_regions = region_stats(built_b & ~built_a & changed, gsd)

    # --- zone ranking (4x4) --------------------------------------------
    # Use a grid that covers the full image. Integer division of H/W by 4
    # can leave a gap at the right/bottom edge when H or W is not divisible
    # by 4, so we compute cell boundaries explicitly and let the last cell in
    # each row/column absorb the remainder.
    zones = []
    row_names = ["N", "N", "S", "S"]
    col_names = ["W", "", "E"]
    row_bounds = [i * H // 4 for i in range(5)]  # 0, H//4, 2H//4, 3H//4, H
    col_bounds = [j * W // 4 for j in range(5)]  # 0, W//4, 2W//4, 3W//4, W
    for zi in range(4):
        for zj in range(4):
            r0, r1 = row_bounds[zi], row_bounds[zi + 1]
            c0, c1 = col_bounds[zj], col_bounds[zj + 1]
            cell = changed[r0:r1, c0:c1]
            cpx = int(cell.sum())
            if cpx == 0:
                continue
            dw = dist_to_water[r0:r1, c0:c1][cell]
            near_w = float((dw <= 500).mean()) if dw.size else 0.0
            col_tag = "W" if zj == 0 else ("E" if zj >= 2 else "C")
            row_tag = row_names[zi]
            zones.append({
                "zone": f"{row_tag}-{col_tag}{zj % 2 + 1}",
                "box": [c0, r0, c1, r1],
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
            "built_up_new_regions": built_new_regions["regions"],
            "built_up_new_largest_ha": built_new_regions["largest_ha"],
            "built_up_new_largest_share": built_new_regions["largest_share"],
        },
        "zones_top": zones[:6],
        "findings": findings,
        "priority_zone": zones[0]["zone"] if zones else "",
        "distance_model": "chamfer EDT to combined water extent",
    }


# --------------------------------------------------------------------- #
# Confidence for impact findings
# --------------------------------------------------------------------- #
#
# This formula used to be written out twice -- once in the tool that builds the
# impact output (anvesha/tools_impl.py) and once in the investigation path in
# anvesha/agent.py -- against two *different* dict shapes. Only one copy could
# see ``changed_fraction``; the other read the tool's return dict, which does not
# carry it, so its signal term was always 0 and its reported confidence could
# only ever be 0.4, 0.5 or 0.6. A number that looks evidence-derived but cannot
# vary with the evidence is worse than an honest constant -- it invites a
# decision to be made on a figure that never measured anything.
#
# So the formula lives here once, and it returns the terms it used. That makes
# the figure auditable after the fact, and makes a missing input *visible* in the
# output instead of silently collapsing the result to a constant.

IMPACT_CONF_BASE = 0.50
IMPACT_CONF_MAX = 0.95
IMPACT_CONF_MIN = 0.10
_NO_TRANSITION = "No significant thematic transition"


def impact_confidence(impact: Dict[str, Any]) -> Dict[str, Any]:
    """Confidence for an impact analysis, together with the arithmetic used.

    ``changed_fraction`` drives the signal term. A caller that passes a dict
    without it does not get a plausible-looking number: ``missing_inputs``
    names the absent key and the runtime records it.
    """
    has_cf = impact.get("changed_fraction") is not None
    signal = min(float(impact.get("changed_fraction") or 0.0) * 5.0, 0.20)
    findings = impact.get("findings") or []
    first = findings[0] if findings else {}
    specific = 0.10 if str(first.get("what", "")) not in ("", _NO_TRANSITION) \
        else 0.0
    gsd_penalty = 0.10 if impact.get("gsd_assumed") else 0.0
    raw = IMPACT_CONF_BASE + signal + specific - gsd_penalty
    value = max(IMPACT_CONF_MIN, min(IMPACT_CONF_MAX, raw))
    return {
        "value": round(float(value), 3),
        "terms": {"base": IMPACT_CONF_BASE, "signal": round(signal, 4),
                  "specificity": specific, "gsd_penalty": -gsd_penalty},
        "equation": ("base + signal + specificity - gsd_penalty "
                     "(clipped 0.10-0.95)"),
        "missing_inputs": [] if has_cf else ["changed_fraction"],
    }
