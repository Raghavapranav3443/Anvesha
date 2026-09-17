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

from .calib_metrics import lookup_band
from .config import CONFIG

_LOCK = threading.Lock()
_CACHE: Optional[Dict[str, Any]] = None
_REPORT_CACHE: Optional[Dict[str, Any]] = None


@dataclass
class ConfidenceMeta:
    value: float
    method: str        # "temp" | "identity" | "inherited" | "formula" | ...
    n_cal: Optional[int]
    component: str
    # ``calibrated`` is earned, never asserted: it is True only when the
    # checkpoint really applies a non-identity temperature AND the fit recorded
    # how many samples it used. A label like "temp" with no sample count (or
    # with T=1.0, which is the identity) is not calibration and must not read
    # as such -- that exact mismatch shipped once and is why this exists.
    calibrated: bool = False
    why: str = ""
    measured_ece: Optional[float] = None


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


def load_report(refresh: bool = False) -> Dict[str, Any]:
    """Load ``weights/calibration_report.json`` (measured reliability tables).

    Written by ``scripts/eval_calibration.py``. Missing => {} and every
    measurement-based field degrades to None rather than to a guess.
    """
    global _REPORT_CACHE
    if _REPORT_CACHE is not None and not refresh:
        return _REPORT_CACHE
    with _LOCK:
        if _REPORT_CACHE is not None and not refresh:
            return _REPORT_CACHE
        p = CONFIG.weights_dir / "calibration_report.json"
        try:
            data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except Exception:
            data = {}
        comp = data.get("component")
        _REPORT_CACHE = {comp: data} if comp else {}
        return _REPORT_CACHE


def get(component: str, value: float) -> ConfidenceMeta:
    """Calibrated confidence meta for a specialist, or formula fallback."""
    cal = load_calibration().get(component)
    if cal:
        method = str(cal.get("method", "formula"))
        n_cal = int(cal["n_cal"]) if cal.get("n_cal") else None
        return ConfidenceMeta(
            value=float(value), method=method, n_cal=n_cal,
            component=component,
            calibrated=bool(method == "temp" and n_cal),
            why=str(cal.get("why", "")),
            measured_ece=cal.get("measured_ece"))
    return ConfidenceMeta(value=float(value), method="formula",
                          n_cal=None, component=component,
                          why="no calibration record on file")


def measured_band(component: str, confidence: float) -> Optional[Dict[str, Any]]:
    """What was actually achieved when this component claimed this confidence?

    Reads the reliability table measured by ``scripts/eval_calibration.py``.
    Returns None when nothing has been measured for this component or the band
    has too few samples -- in that case a caller must not substitute the claim
    for the measurement.
    """
    payload = load_report().get(component)
    if not payload:
        return None
    table = (payload.get("checkpoint") or {}).get("reliability_table")
    return lookup_band(float(confidence), table)


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


# --------------------------------------------------------------------------- #
# Method reliability — the missing half of "confidence"
# --------------------------------------------------------------------------- #
#
# A reported confidence is the model's *self-assessment*. It says nothing about
# how often that component is actually right on held-out data. Those are two
# different things, and conflating them lets a confident-looking answer from a
# weak component read as strong evidence.
#
# ``grounding`` is the clearest case: its formula can return a high value while
# the learned variants were gated off at IoU@0.5 ~ 0.126. Reporting 0.9 there
# would be dishonest, so trust is the *product* of the two factors, which caps
# the influence of a low-reliability component regardless of how confident it
# sounds.
#
# Every number below is a measurement already recorded in this repository
# (README scorecard / MODEL_CARDS.md / calibration.json). None are invented.
# ``captioning`` is the one weak proxy: BLEU is not a probability, so it is the
# least defensible entry and is documented as such rather than hidden.
RELIABILITY: Dict[str, Dict[str, Any]] = {
    "vqa": {"reliability": 0.700,
            "evidence": "RSVQA-LR aggregate exact-match 0.700 (full test, n=9,491)"},
    "counting": {"reliability": 0.440,
                 "evidence": "counting head exact-match 0.44 (ordinal soft-CE v4)"},
    "cdvqa": {"reliability": 0.683,
              "evidence": "CDVQA test answer accuracy 0.683"},
    "change": {"reliability": 0.818,
               "evidence": "LEVIR-CD full test F1 0.818 (IoU 0.692, n=1,500)"},
    "change_analysis": {"reliability": 0.818,
                        "evidence": "inherits the change detector (LEVIR-CD F1 0.818)"},
    "fusion": {"reliability": 0.850,
               "evidence": "BigEarthNet v2 co-registered val label recall 0.85"},
    "optical_sar": {"reliability": 0.850,
                    "evidence": "inherits the optical-SAR fusion net (val recall 0.85)"},
    "grounding": {"reliability": 0.126,
                  "evidence": "IoU@0.5 ~0.126; learned variants gated off (D15.1) -- "
                              "weak evidence by design"},
    "impact_analysis": {"reliability": 0.818,
                        "evidence": "derived GIS maths over the change mask (LEVIR-CD "
                                    "F1 0.818); inherits the detector's reliability"},
    "captioning": {"reliability": 0.320,
                   "evidence": "BigEarthNet.txt multi-ref BLEU 0.32 (weak proxy: BLEU is "
                               "not a probability)"},
}

# Plain-English bands so a non-expert can read trust without knowing the numbers.
TRUST_BANDS = (
    (0.60, "high", "The evidence supports acting on this."),
    (0.40, "moderate", "Indicative. Worth acting on after a quick check."),
    (0.25, "low", "A lead, not a finding. Verify before acting."),
    (0.00, "very_low", "Not reliable enough to base a decision on."),
)


def reliability_of(component: str) -> Dict[str, Any]:
    """Measured reliability of a component, or an explicit ``unknown`` marker.

    Unknown components are reported as unknown rather than defaulted to a
    comfortable number.
    """
    entry = RELIABILITY.get(component)
    if entry:
        return dict(entry)
    return {"reliability": None, "evidence": "no measured reliability on file"}


def trust_band(trust: float) -> Dict[str, str]:
    for threshold, label, advice in TRUST_BANDS:
        if trust >= threshold:
            return {"band": label, "advice": advice}
    return {"band": "very_low", "advice": TRUST_BANDS[-1][2]}


def effective_trust(component: str, confidence: float) -> Dict[str, Any]:
    """Combine self-reported confidence with measured method reliability.

    Returns both factors *separately* alongside their product, so a reader (and
    the decision layer) can see which one is doing the limiting. When a
    component's reliability is unknown, trust is capped at the confidence rather
    than assumed to be fine, and ``reliability`` is reported as None.

    Where a *measured* reliability table exists for this component, the band the
    claim falls in is attached as ``band_evidence``: the confidence we showed
    next to how often we were right when we showed it. That is the number a
    non-expert should act on, and it is reported alongside -- never silently
    replacing -- the model's own opinion.
    """
    rel = reliability_of(component).get("reliability")
    conf = max(0.0, min(1.0, float(confidence)))
    known = isinstance(rel, (int, float))
    model_trust = conf * float(rel) if known else conf
    band_ev = measured_band(component, conf)
    observed = band_ev.get("observed_accuracy") if band_ev else None

    # Prefer measurement over the model's opinion. ``band_ev`` is P(this answer
    # is right | we showed roughly this confidence), measured on held-out data --
    # it is a direct answer to the question a user is really asking. The
    # claim-times-method-quality product remains the fallback (and is always
    # reported) because it generalises to components nobody has measured yet.
    if isinstance(observed, (int, float)):
        trust, source = float(observed), "measured_band"
        limiting = "confidence" if conf < trust else "measured_accuracy"
    elif known:
        trust, source = float(model_trust), "confidence_x_reliability"
        limiting = "reliability" if float(rel) < conf else "confidence"
    else:
        trust, source = float(conf), "unverified"
        limiting = "unknown reliability"

    out: Dict[str, Any] = {
        "confidence": round(conf, 3),
        "reliability": round(float(rel), 3) if known else None,
        "model_opinion_trust": round(float(model_trust), 3),
        "trust": round(float(trust), 3),
        "trust_source": source,
        "limiting_factor": limiting,
    }
    if band_ev:
        out["band_evidence"] = {
            "band": band_ev.get("band"),
            "claimed": band_ev.get("claimed_mean"),
            "observed_accuracy": band_ev.get("observed_accuracy"),
            "n": band_ev.get("n"),
            "optimism": band_ev.get("optimism"),
            "source": "measured on held-out data (weights/calibration_report.json)",
        }
    if known or source == "measured_band":
        out.update(trust_band(float(trust)))
    else:
        # No measured reliability on file. Do NOT present this as high trust on
        # the strength of the model's own opinion -- an unverified method is an
        # unverified method, however confident it sounds.
        out.update({"band": "unverified",
                    "advice": "No measured reliability for this method. Treat the "
                              "result as unverified."})
    return out


def stamp(outputs: Dict[str, Any], component: str) -> Dict[str, Any]:
    """Add a ``confidence_meta`` entry to an output dict (additive only).

    Safe no-op if the output has no ``confidence`` key. The entry now carries
    the *two* halves of trust -- the reported confidence and the component's
    measured reliability -- plus their product, because a confidence without
    method reliability is not something a user should act on.
    """
    val = outputs.get("confidence")
    if val is None:
        return outputs
    meta = get(component, float(val))
    out = dict(outputs)
    trust = effective_trust(component, meta.value)
    out["confidence_meta"] = {
        "value": meta.value, "method": meta.method,
        "n_cal": meta.n_cal, "component": meta.component,
        "calibrated": meta.calibrated,
        "calibration_note": meta.why,
        "reliability": trust["reliability"],
        "reliability_evidence": reliability_of(component)["evidence"],
        "trust": trust["trust"],
        "trust_band": trust["band"],
        "trust_advice": trust["advice"],
        "limiting_factor": trust["limiting_factor"],
    }
    if meta.measured_ece is not None:
        out["confidence_meta"]["measured_ece"] = meta.measured_ece
    out["confidence_meta"]["model_opinion_trust"] = trust["model_opinion_trust"]
    out["confidence_meta"]["trust_source"] = trust["trust_source"]
    if trust.get("band_evidence"):
        out["confidence_meta"]["band_evidence"] = trust["band_evidence"]
    if meta.method == "formula":
        out["confidence_meta"]["equation"] = formula(component)
    return out


def claim_problems() -> Dict[str, str]:
    """Audit the sidecar for labels the evidence does not support.

    Returns ``{component: problem}``; empty means every recorded label is
    consistent with what it claims. Used by ``tests/test_confidence_trust.py``
    so a hand-edited or stale ``calibration.json`` cannot quietly re-introduce
    the "calibrated" claim without a fit behind it.
    """
    problems: Dict[str, str] = {}
    for component, cal in load_calibration().items():
        if not isinstance(cal, dict):
            problems[component] = "record is not an object"
            continue
        method = str(cal.get("method", "formula"))
        param = cal.get("param")
        n_cal = cal.get("n_cal")
        if method == "temp":
            if param in (None, 1.0, 1):
                problems[component] = (
                    "labelled 'temp' but the applied temperature is the identity, "
                    "which carries no calibration")
            elif not n_cal:
                problems[component] = (
                    "labelled 'temp' with no fitted sample count; the fit is not "
                    "auditable")
        elif method in ("identity", "inherited", "unverified", "unreadable"):
            if param is not None and method == "identity" and float(param) != 1.0:
                problems[component] = (
                    "labelled 'identity' but records a non-identity temperature")
        elif method not in ("formula", "platt"):
            problems[component] = f"unknown method label {method!r}"
    return problems
