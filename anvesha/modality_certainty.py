"""B1 — Modality certainty: stats-first disambiguation of SAR vs optical.

Kills the filename-dependent misclassification failure (`area42_patch_002.tif`
class): the filename vote (weight 1) is combined with a pixel-statistics vote
(weight 2) into an auditable ``modality_certainty`` record that rides on every
``RSImage`` and lands in the trace, the report and the UI badge.

Rules (frozen, from the masterplan):
* dB product  — bands <= 2 AND median < 0 AND pct_neg > 15%  -> SAR (decisive)
* 2-band      — band-corr > 0.85 AND low texture             -> SAR (suspect;
                  only flips the label when the band-count fallback already
                  says SAR and the name carries no optical hint)
* bands >= 4  -> multispectral (decisive over a SAR-sounding name)
* bands == 3  -> RGB (decisive over a SAR-sounding name)
* explicit user override always wins (recorded as a vote).

Standalone module: no imports from ``io_utils`` (avoids circulars); operates
on arrays + strings only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Multi-band SAR products beyond 2 polarisations are vanishingly rare, and
# SAR never ships 3 (RGB) or >=4 (MS) bands — those flips are safe by physics.
_MS_BASE = ["red", "green", "blue", "nir",
            "rededge1", "rededge2", "rededge3",
            "nir-narrow", "swir1", "swir2",
            "coastal", "wvapor", "cirrus"]


def _stats_evidence(arr: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(arr, dtype=np.float32)
    c = int(a.shape[2]) if a.ndim == 3 else 1
    flat = a.ravel()
    ev: Dict[str, Any] = {
        "bands": c,
        "median": float(np.median(flat)),
        "pct_neg": float((flat < 0).mean()),
    }
    if c == 2:
        b0, b1 = a[..., 0].ravel(), a[..., 1].ravel()
        s0, s1 = float(np.std(b0)), float(np.std(b1))
        ev["band_corr"] = (float(np.corrcoef(b0, b1)[0, 1])
                           if s0 > 1e-9 and s1 > 1e-9 else None)
        ev["texture"] = _texture(a[..., 0])
    return ev


def _texture(g: np.ndarray, stride: int = 4) -> float:
    """Coarse high-frequency energy: mean |x - 9px box blur| on a subsample."""
    small = g[::stride, ::stride]
    return float(np.abs(small - _box_blur(small, 9)).mean())


def _box_blur(x: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    p = np.pad(x, pad, mode="edge")
    cs = np.cumsum(np.cumsum(p, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    h, w = x.shape
    return (cs[k:k + h, k:k + w] - cs[:-k, k:k + w] - cs[k:k + h, :-k]
            + cs[:-k, :-k]) / float(k * k)


def stats_vote(evidence: Dict[str, Any]) -> Optional[str]:
    """Pixel-statistics modality vote: 'sar' | 'optical' | None."""
    c = int(evidence.get("bands", 0))
    if c <= 2 and evidence.get("median", 0.0) < 0 and \
            evidence.get("pct_neg", 0.0) > 0.15:
        return "sar"                      # dB-scale product: decisive
    if c >= 4 or c == 3:
        return "optical"
    if c == 2:
        corr = evidence.get("band_corr")
        if corr is not None and corr > 0.85 and \
                evidence.get("texture", 1.0) < 0.15:
            return "sar"                  # smooth correlated 2-band: SAR-like
    return None                           # 1-band grayscale: ambiguous


def _band_names(label: str, bands: int) -> List[str]:
    if label == "sar":
        if bands >= 2:
            return (["VV", "VH"] + [f"ch{i}" for i in range(bands)])[:bands]
        return ["intensity"]
    if label == "multispectral":
        return (_MS_BASE + [f"ch{i}" for i in range(bands)])[:bands]
    if label == "rgb":
        return (["red", "green", "blue"] + [f"ch{i}" for i in range(bands)])[:bands]
    return ["gray"]


def refine_modality(name: str, arr: np.ndarray,
                    modality: str, band_names: List[str],
                    descriptions: Optional[List[str]] = None,
                    ) -> Tuple[str, List[str], Dict[str, Any]]:
    """Combine the filename/band-count inference with the stats vote.

    Returns ``(modality, band_names, certainty_dict)``. The stats vote
    (weight 2) overrides the filename/band-count vote (weight 1) only on the
    decisive rules above; agreement raises confidence, conflict lowers it and
    is recorded verbatim in ``reason`` (auditable, never hidden).
    """
    ev = _stats_evidence(arr)
    sv = stats_vote(ev)
    votes: Dict[str, Any] = {"filename": modality, "stats": sv}
    label, confidence, reason = modality, 0.5, "band-count inference only"

    if sv is not None and sv == modality:
        label, confidence = modality, 0.9
        reason = f"filename and pixel statistics agree ({sv})"
    elif sv is not None and sv != modality:
        # Decisive stats flips: dB SAR evidence, or band counts a SAR product
        # cannot have (>=4 MS / 3 RGB bands).
        decisive = (sv == "sar" and ev.get("median", 0) < 0
                    and ev.get("pct_neg", 0) > 0.15) or ev["bands"] >= 3
        if decisive:
            # stats_vote returns the coarse "optical" label; map back to the
            # band-count-specific label the rest of the codebase expects.
            if sv == "optical":
                label = "multispectral" if ev["bands"] >= 4 else "rgb"
            else:
                label = sv
            confidence = 0.85
            reason = (f"pixel statistics override filename: stats says "
                      f"{label} (bands={ev['bands']}, median={ev['median']:.3g}, "
                      f"pct_neg={ev['pct_neg']:.2f})")
        else:
            confidence = 0.45
            reason = (f"stats suggest {sv} but filename says {modality}; "
                      f"keeping {modality} (non-decisive evidence)")
    elif sv is None and modality == "sar":
        confidence = 0.7
        reason = "filename/band-count SAR inference; 1-band stats ambiguous"

    if label != modality:
        band_names = _band_names(label, ev["bands"])
    certainty = {"label": label, "votes": votes, "confidence": round(confidence, 2),
                 "stats": {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in ev.items()},
                 "reason": reason}
    return label, band_names, certainty
