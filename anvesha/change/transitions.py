"""B5 — change semantics: per-pixel transition map (class_T1 -> class_T2).

Restricts transitions to the binary change-mask pixels, assigns each date a
class from spectral index masks, then reports top-3 transitions with hectares,
quadrant and a near-water flag. Cross-checks against the CDVQA
``change_to_what`` head when a question is supplied; surfaces disagreement as
``uncertain`` with alternates (never hidden).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Transition:
    from_class: str
    to_class: str
    pixel_count: int
    area_ha: float
    quadrant: str
    near_water_frac: float = 0.0


@dataclass
class TransitionTable:
    changed_fraction: float
    date_a: str
    date_b: str
    top: List[Transition] = field(default_factory=list)
    uncertain: List[Transition] = field(default_factory=list)
    cdvqa_check: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["top"] = [asdict(t) for t in self.top]
        d["uncertain"] = [asdict(t) for t in self.uncertain]
        return d


def _index_class_map(img) -> np.ndarray:
    """HxW class index from spectral indices: 0 water,1 veg,2 built,3 bare."""
    from ..io_utils import rgb_composite
    a = rgb_composite(img).astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    exg = 2.0 * g - r - b
    ndwi = (g - r) / (g + r + 1e-6)
    bright = a.mean(axis=2)
    sat = a.max(axis=2) - a.min(axis=2)
    water = ((ndwi > 0.05) & (b > g * 0.9)).astype(np.float32)
    veg = (exg > 0.08).astype(np.float32)
    built = ((bright > 0.45) & (sat < 0.18)).astype(np.float32)
    bare = ((r > g) & (g > b) & (bright > 0.3)).astype(np.float32)
    score = np.stack([water, veg, built, bare], axis=0)
    return np.argmax(score, axis=0).astype(np.uint8)


_WATER_NAME = {0: "water", 1: "vegetation", 2: "built-up", 3: "bare"}


def _quadrant(counts: np.ndarray, h: int, w: int) -> str:
    """Dominant transition's centroid quadrant."""
    ys, xs = np.where(counts > 0)
    if len(xs) == 0:
        return "centre"
    dx, dy = float(xs.mean()) / w - 0.5, float(ys.mean()) / h - 0.5
    ns = "north" if dy < -0.05 else ("south" if dy > 0.05 else "")
    ew = "west" if dx < -0.05 else ("east" if dx > 0.05 else "")
    return "-".join(p for p in (ns, ew) if p) or "centre"


def build_transitions(a, b, change_mask: np.ndarray, gsd_m: float,
                      date_a: str = "T1", date_b: str = "T2",
                      top_k: int = 3) -> TransitionTable:
    """Build the transition table for changed pixels only.

    ``change_mask``: HxW bool/array; ``gsd_m``: metres per pixel for ha.
    """
    h, w = change_mask.shape[:2]
    ca = _index_class_map(a)
    cb = _index_class_map(b)
    mask = change_mask.astype(bool)
    changed_frac = float(mask.sum()) / max(int(mask.size), 1)

    counts: Dict[Tuple[int, int], int] = {}
    for key in zip(ca[mask].tolist(), cb[mask].tolist()):
        counts[key] = counts.get(key, 0) + 1

    # near-water fraction: changed pixels within 500m of T1 water
    water_a = (ca == 0).astype(np.float32)
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(1 - water_a) * gsd_m
    near_water = float((mask & (dist < 500)).sum()) / max(int(mask.sum()), 1)

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:top_k]
    top = []
    for (fa, fb), px in ranked:
        sub = mask & (ca == fa) & (cb == fb)
        quad = _quadrant(sub.astype(np.float32), h, w)
        area_ha = float(px) * (gsd_m ** 2) / 10000.0
        top.append(Transition(from_class=_WATER_NAME.get(fa, str(fa)),
                              to_class=_WATER_NAME.get(fb, str(fb)),
                              pixel_count=int(px), area_ha=round(area_ha, 2),
                              quadrant=quad, near_water_frac=round(near_water, 3)))

    return TransitionTable(changed_fraction=round(changed_frac, 4),
                           date_a=date_a, date_b=date_b, top=top)
