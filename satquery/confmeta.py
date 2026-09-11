"""B7 — confidence-method metadata.

Reads ``weights/calibration.json`` (produced by ``scripts/calibrate_all.py``) and
attaches an auditable ``{value, method, n_cal}`` stamp to every confidence
surface. Specialists whose confidence is a closed-form blend (grounding,
change, fusion, impact) are labelled ``formula`` with their equation; VQA and
counting carry ``temp``/``platt`` when calibrated, falling back to
``formula`` otherwise. Never invents a number — absent calibration => method
``formula`` and ``n_cal`` None.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .config import CONFIG

_LOCK = threading.Lock()
_CACHE: Optional[Dict[str, Any]] = None


@dataclass
class ConfidenceMeta:
    value: float
    method: str        # "temp" | "platt" | "formula"
    n_cal: Optional[int]
    component: str


def load_calibration(refresh: bool = False) -> Dict[str, Any]:
    """Load + cache ``weights/calibration.json`` (missing => {})."""
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE
    with _LOCK:
        if _CACHE is not None and not refresh:
            return _CACHE
        p = CONFIG.weights_dir / "calibration.json"
        try:
            _CACHE = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except Exception:
            _CACHE = {}
        return _CACHE


def get(component: str, value: float) -> ConfidenceMeta:
    """Calibrated confidence meta for a specialist, or formula fallback."""
    cal = load_calibration().get(component)
    if cal:
        return ConfidenceMeta(
            value=float(value),
            method=str(cal.get("method", "formula")),
            n_cal=int(cal["n_cal"]) if cal.get("n_cal") else None,
            component=component)
    return ConfidenceMeta(value=float(value), method="formula",
                          n_cal=None, component=component)


# Specialist -> (calibration_component, human equation for formula method)
_FORMULA = {
    "grounding": ("grounding", "0.35*lex + 0.45*purity + 0.2*area"),
    "change_analysis": ("change",
                        "thresholded prob-map F1 on LEVIR-CD val"),
    "change_vqa": ("cdvqa", "change-conditioned head over diff features"),
    "optical_sar": ("fusion", "0.6*optical + 0.4*SAR evidence blend"),
    "impact_analysis": ("change",
                        "base + signal + specificity - gsd_penalty"),
    "captioning": ("caption", "evidence-derived; BLEU 0.283 bench-matched"),
}


def formula(component: str) -> str:
    """Human-readable equation for a formula-blended specialist."""
    return _FORMULA.get(component, (component, "evidence blend"))[1]


def stamp(outputs: Dict[str, Any], component: str) -> Dict[str, Any]:
    """Add a ``confidence_meta`` entry to an output dict (additive only).

    Safe no-op if the output has no ``confidence`` key.
    """
    val = outputs.get("confidence")
    if val is None:
        return outputs
    meta = get(component, float(val))
    out = dict(outputs)
    out["confidence_meta"] = {
        "value": meta.value, "method": meta.method,
        "n_cal": meta.n_cal, "component": meta.component}
    if meta.method == "formula":
        out["confidence_meta"]["equation"] = formula(component)
    return out
