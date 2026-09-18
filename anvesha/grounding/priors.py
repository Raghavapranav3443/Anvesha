"""B3 — grounding priors: shape/dispersion features per concept.

Cheap geometric priors to break ties among spectral-index regions:
    water   -> large + compact
    road    -> elongated (high eccentricity)
    building -> mid-size clusters
    vegetation/bare -> dispersed
Computed from a connected-component mask, so they never touch the spectral
threshold — they only re-rank / break ties among regions the spectrals already
found.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _components(mask: np.ndarray):
    """Return (labels, n) using scipy when available, else None."""
    try:
        from scipy.ndimage import label as _label
        labels, n = _label(mask.astype(np.int32))
        return labels.astype(np.int32), int(n)
    except ImportError:
        return None, 0


def shape_priors(box_or_mask, mask: Optional[np.ndarray] = None) -> dict:
    """Geometric priors for a region.

    Accepts either a pixel ``mask`` (HxW bool) or an oriented bbox + component
    mask. Returns eccentricity / elongation / compactness / area_frac.
    """
    if mask is None:
        return {"eccentricity": 0.0, "elongation": 1.0,
                "compactness": 0.0, "area_frac": 0.0}
    ys, xs = np.where(mask)
    if len(xs) < 4:
        return {"eccentricity": 0.0, "elongation": 1.0,
                "compactness": 0.0, "area_frac": float(mask.mean())}
    # second-moment ellipse
    cy, cx = float(ys.mean()), float(xs.max() + xs.min()) / 2.0
    dy, dx = ys - cy, xs - cx
    mu20, mu02, mu11 = float((dx ** 2).mean()), float((dy ** 2).mean()), \
        float((dx * dy).mean())
    det = max(mu20 * mu02 - mu11 ** 2, 0.0)
    tr = mu20 + mu02
    # eigenvalues of the covariance-ish matrix
    gap = math.sqrt(max((mu20 - mu02) ** 2 + 4 * mu11 ** 2, 0.0))
    lam1 = max((tr + gap) / 2.0, 1e-9)
    lam2 = max((tr - gap) / 2.0, 1e-9)
    eccentricity = math.sqrt(max(1.0 - lam2 / lam1, 0.0))
    elongation = math.sqrt(lam1 / lam2)
    perimeter = float(_perimeter(mask))
    area = float(mask.sum())
    compactness = 4 * math.pi * area / max(perimeter ** 2, 1e-9)
    return {"eccentricity": round(eccentricity, 3),
            "elongation": round(elongation, 2),
            "compactness": round(min(compactness, 1.0), 3),
            "area_frac": round(float(mask.mean()), 4)}


def _perimeter(mask: np.ndarray) -> int:
    """Count boundary pixels (4-neighbourhood)."""
    m = mask.astype(bool)
    if not m.any():
        return 0
    up = np.zeros_like(m); up[1:, :] = m[1:, :]
    down = np.zeros_like(m); down[:-1, :] = m[:-1, :]
    left = np.zeros_like(m); left[:, 1:] = m[:, 1:]
    right = np.zeros_like(m); right[:, :-1] = m[:, :-1]
    boundary = m & ~(up & down & left & right)
    return int(boundary.sum())


def prior_score(concept: str, priors: dict) -> float:
    """Higher = better match between the concept's prior and the region shape.

    Tuned order-of-magnitude; only used to break ties among same-threshold
    spectral regions, so exact weights do not change the primary (spectral)
    ranking.
    """
    ecc, elon, comp = priors.get("eccentricity", 0.0), \
        priors.get("elongation", 1.0), priors.get("compactness", 0.0)
    area = priors.get("area_frac", 0.0)
    if concept == "water":
        return 0.5 * comp + 0.3 * min(area * 6, 1.0) + 0.2 * (1 - ecc)
    if concept == "road":
        return 0.6 * min(elon / 4.0, 1.0) + 0.4 * ecc
    if concept == "building":
        mid = max(1.0 - abs(area * 12 - 0.5), 0.0)
        return 0.5 * mid + 0.3 * comp + 0.2 * (1 - elon / 4.0)
    if concept == "vegetation":
        return 0.4 * min(area * 4, 1.0) + 0.3 * (1 - comp) + 0.3 * ecc
    # bare
    return 0.4 * (1 - comp) + 0.3 * (1 - min(elon / 3.0, 1.0)) + \
        0.3 * min(area * 3, 1.0)
