"""Calibration metrics -- is a reported confidence *true*?

A confidence number is only useful to a non-expert if it has been checked
against outcomes. These functions are the check: they compare what the system
claimed with what it got right on held-out data.

Everything here is pure NumPy and free of model imports, so it is cheap to test
and safe to call from the runtime (the decision layer reads a reliability table
produced by these functions rather than trusting a model's self-assessment).

Terminology used throughout the project:

  gap        mean(confidence) - accuracy. Positive means overconfident.
  ECE        expected calibration error -- mean |confidence - accuracy| within
             confidence bins, weighted by bin size. This is the number that
             matters when the confidence is shown to a user, because it measures
             whether the *displayed* number means anything.
  NLL        negative log-likelihood. A proper scoring rule, but dominated by
             the probability assigned to the true class, so it rewards sharpening
             the whole distribution. It is NOT the right objective for a
             confidence that is displayed as a decision aid -- see
             ``scripts/calibrate.py`` for the measurement that showed this.

The distinction between ECE and NLL is not academic: on this project's VQA head
the NLL-optimal temperature (1.7) is *miscalibrated* (ECE 0.076) while the
ECE-optimal temperature (0.8) is well calibrated (ECE 0.017). Optimising the
wrong one would have shipped worse confidence while looking like an improvement.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Bands chosen to match how a decision is actually phrased to a user:
# "unlikely" / "more likely than not" / "likely" / "very likely".
DEFAULT_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9),
    (0.9, 0.95), (0.95, 1.01),
)


def expected_calibration_error(confidence: Sequence[float],
                              correct: Sequence[bool],
                              n_bins: int = 15) -> float:
    """Mean |confidence - accuracy| within equal-width bins, weighted by bin size.

    Returns 0.0 for an empty sample (nothing claimed, nothing wrong).
    """
    conf = np.asarray(confidence, dtype=float).ravel()
    hit = np.asarray(correct, dtype=bool).ravel()
    if conf.size == 0 or conf.size != hit.size:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = conf.size
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        ece += (m.sum() / total) * abs(float(conf[m].mean()) - float(hit[m].mean()))
    return float(ece)


def brier_score(confidence: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean squared error of the confidence against the 0/1 outcome. Lower is better."""
    conf = np.asarray(confidence, dtype=float).ravel()
    hit = np.asarray(correct, dtype=bool).ravel()
    if conf.size == 0 or conf.size != hit.size:
        return 0.0
    return float(np.mean((conf - hit.astype(float)) ** 2))


def accuracy_confidence_gap(confidence: Sequence[float],
                            correct: Sequence[bool]) -> float:
    """Positive => overconfident (claims more than it delivers)."""
    conf = np.asarray(confidence, dtype=float).ravel()
    hit = np.asarray(correct, dtype=bool).ravel()
    if conf.size == 0:
        return 0.0
    return float(conf.mean() - hit.mean())


def judge(confidence: Sequence[float], correct: Sequence[bool],
          tolerance: float = 0.02) -> Dict[str, object]:
    """One-word verdict on a confidence surface, with the numbers behind it."""
    gap = accuracy_confidence_gap(confidence, correct)
    if gap > tolerance:
        verdict = "overconfident"
    elif gap < -tolerance:
        verdict = "underconfident"
    else:
        verdict = "well_calibrated"
    return {"gap": round(gap, 4), "verdict": verdict}


def reliability_table(confidence: Sequence[float], correct: Sequence[bool],
                      bands: Sequence[Tuple[float, float]] = DEFAULT_BANDS,
                      min_n: int = 30) -> List[Dict[str, object]]:
    """Per-band: what we claimed, what we observed, and how many samples.

    ``optimism`` is positive when the band claimed more than it delivered.

    A band never observed in the data is reported with ``observed_accuracy``
    None rather than a guessed value -- absence of evidence is not evidence.
    ``sufficient`` marks bands with enough samples to base a decision on.
    """
    conf = np.asarray(confidence, dtype=float).ravel()
    hit = np.asarray(correct, dtype=bool).ravel()
    rows: List[Dict[str, object]] = []
    for lo, hi in bands:
        m = (conf >= lo) & (conf < hi)
        n = int(m.sum())
        row: Dict[str, object] = {
            "band": f"{lo:.2f}-{min(hi, 1.0):.2f}",
            "lo": float(lo),
            "hi": float(min(hi, 1.0)),
            "n": n,
            "claimed_mean": round(float(conf[m].mean()), 3) if n else None,
            "observed_accuracy": round(float(hit[m].mean()), 3) if n else None,
            "sufficient": bool(n >= min_n),
        }
        if n:
            row["optimism"] = round(float(conf[m].mean() - hit[m].mean()), 3)
        rows.append(row)
    return rows


def lookup_band(confidence: float,
                table: Optional[Sequence[Dict[str, object]]],
                min_n: int = 30) -> Optional[Dict[str, object]]:
    """What did we actually achieve when we claimed roughly this confidence?

    Returns None when no sufficiently-populated band covers the value, so a
    caller can say "unmeasured" instead of substituting the claim.
    """
    if not table:
        return None
    value = float(confidence)
    for row in table:
        lo, hi = float(row.get("lo", 0.0)), float(row.get("hi", 1.0))
        if lo <= value < hi:
            if int(row.get("n", 0)) < min_n or row.get("observed_accuracy") is None:
                return None
            return dict(row)
    # Above the top edge (e.g. exactly 1.0) -- fall back to the last band.
    last = table[-1]
    if float(last.get("hi", 0.0)) >= value and int(last.get("n", 0)) >= min_n:
        return None if last.get("observed_accuracy") is None else dict(last)
    return None


def select_temperature(candidates: Sequence[float],
                       confidence_at: Dict[float, Sequence[float]],
                       correct: Sequence[bool],
                       identity: float = 1.0) -> Dict[str, object]:
    """Pick the temperature that minimises ECE, never worse than the identity.

    The fallback is the point: if no fitted temperature beats doing nothing, the
    honest answer is to keep the identity and say so, rather than to record a
    "calibration" that degrades the number users are shown.
    """
    if identity not in confidence_at:
        raise ValueError("identity temperature must be among the candidates")
    scores = {t: expected_calibration_error(confidence_at[t], correct)
              for t in candidates}
    best_t = min(scores, key=lambda t: scores[t])
    if scores[best_t] >= scores[identity]:
        return {"temperature": float(identity), "ece": scores[identity],
                "identity_ece": scores[identity], "improved": False,
                "all_scores": {str(k): round(v, 4) for k, v in scores.items()}}
    return {"temperature": float(best_t), "ece": scores[best_t],
            "identity_ece": scores[identity], "improved": True,
            "all_scores": {str(k): round(v, 4) for k, v in scores.items()}}
