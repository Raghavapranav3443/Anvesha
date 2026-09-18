"""Sensitivity -- how much does the conclusion depend on one arbitrary number?

Every change decision rests on a threshold. A conclusion that holds only at the
exact threshold we happened to pick is not a conclusion, it is a coincidence of
our own configuration. This module re-runs the decision's own predicate across a
range of thresholds and reports the range over which the answer is unchanged, so
a non-expert is told *how much margin* the finding has instead of a bare verdict.

It also states a detection floor: what this imagery physically cannot show. At
10 m per pixel a 0.25 ha patch is 25 pixels; below that, individual detections
are dominated by edge noise rather than real ground cover change. Telling a user
what the data cannot reveal is part of answering honestly.

Pure functions over arrays already on disk -- offline by construction.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

# The detector's measured quality, used to justify the floor below rather than
# to decorate the output.
DETECTOR_F1 = 0.818          # LEVIR-CD full test, n=1,500
DETECTOR_SOURCE = "LEVIR-CD full-test F1 0.818 (IoU 0.692)"

# A coherent patch must span this many pixels before we treat it as ground
# change rather than threshold noise. 25 px is a 5x5 block: a deliberately
# conservative operating choice, NOT a measured quantity, and labelled as such
# wherever it reaches a user.
MIN_COHERENT_PIXELS = 25

DEFAULT_THRESHOLDS: Sequence[float] = tuple(
    round(float(t), 3) for t in np.arange(0.50, 1.001, 0.02))


def _nominal_threshold() -> Dict[str, Any]:
    """Resolve the configured change threshold, recording where it came from."""
    for module, attr in (("anvesha.config", "CHANGE_MASK_PROB_THRESHOLD"),
                         ("anvesha.tools_impl", "CHANGE_MASK_PROB_THRESHOLD")):
        try:
            mod = __import__(module, fromlist=[attr])
            val = getattr(mod, attr, None)
            if isinstance(val, (int, float)):
                return {"value": float(val), "source": f"{module}.{attr}"}
        except Exception:
            continue
    return {"value": 0.5, "source": "fallback default (configured threshold "
                                   "could not be resolved)"}


def pixel_area_ha(gsd_m: Optional[float]) -> Optional[float]:
    """Ground area of one pixel, in hectares."""
    if not gsd_m or float(gsd_m) <= 0:
        return None
    return (float(gsd_m) ** 2) / 10_000.0


def detection_floor_ha(gsd_m: Optional[float]) -> Optional[float]:
    """Smallest change area this imagery can be relied on to show."""
    px = pixel_area_ha(gsd_m)
    return None if px is None else round(MIN_COHERENT_PIXELS * px, 3)


def detection_floor_text(gsd_m: Optional[float]) -> str:
    """Plain-English statement of the data's limit, for a non-expert."""
    floor = detection_floor_ha(gsd_m)
    if floor is None:
        return ("We do not know the ground resolution of these images, so we "
                "cannot say what size of change they would miss.")
    return (f"These images show roughly {float(gsd_m):.0f} metres of ground "
            f"per dot. Anything smaller than about {floor:.2f} hectares cannot "
            f"be told apart from ordinary speckle in the picture, so changes "
            f"below that size may exist without these images showing them.")


def threshold_sweep(prob_map: np.ndarray,
                    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
                    max_pixels: int = 4_000_000) -> Dict[str, Any]:
    """Changed-area fraction as a function of the probability threshold.

    ``max_pixels`` subsamples very large fields so the sweep stays cheap enough
    to run inline with a request; the sampling is recorded when it happens.
    """
    arr = np.asarray(prob_map, dtype=np.float32)
    sampled = False
    if arr.size > max_pixels:
        step = int(np.ceil(np.sqrt(arr.size / max_pixels)))
        arr = arr[::step, ::step]
        sampled = True
    total = float(arr.size)
    points = []
    for t in thresholds:
        frac = float((arr >= float(t)).sum()) / total if total else 0.0
        points.append({"threshold": float(t), "changed_fraction": round(frac, 5)})
    return {
        "points": points,
        "sampled": sampled,
        "n_pixels": int(total),
        "detector_f1": DETECTOR_F1,
        "detector_source": DETECTOR_SOURCE,
    }


def stability_range(sweep: Dict[str, Any], predicate: Callable[[float], bool],
                    nominal: Optional[float] = None) -> Dict[str, Any]:
    """Threshold range over which ``predicate`` keeps returning the same answer.

    ``predicate`` receives a changed-area fraction and returns the decision's
    own yes/no. Reporting the contiguous range around the nominal threshold
    turns "4.9 hectares changed" into "that holds for thresholds 0.72-0.94".
    """
    nom = _nominal_threshold() if nominal is None else {
        "value": float(nominal), "source": "caller"}
    points = sweep.get("points") or []
    if not points:
        return {"available": False, "reason": "no probability field to sweep"}

    ref = min(points, key=lambda p: abs(p["threshold"] - nom["value"]))
    baseline = predicate(ref["changed_fraction"])

    lo = hi = ref["threshold"]
    for point in sorted(points, key=lambda p: p["threshold"]):
        if point["threshold"] > ref["threshold"]:
            if predicate(point["changed_fraction"]) == baseline:
                hi = point["threshold"]
            else:
                break
    for point in sorted(points, key=lambda p: -p["threshold"]):
        if point["threshold"] < ref["threshold"]:
            if predicate(point["changed_fraction"]) == baseline:
                lo = point["threshold"]
            else:
                break

    holds = (lo <= sweep["points"][0]["threshold"]
             and hi >= sweep["points"][-1]["threshold"])
    return {
        "available": True,
        "nominal_threshold": nom["value"],
        "nominal_source": nom["source"],
        "stable_from": lo,
        "stable_to": hi,
        "holds_across_full_range": bool(holds),
        "conclusion_at_nominal": bool(baseline),
        "reverses_outside": not holds,
        "detector_f1": sweep.get("detector_f1"),
        "detector_source": sweep.get("detector_source"),
        "sampled": sweep.get("sampled", False),
    }


def stability_text(stability: Dict[str, Any], n_settings: Optional[int] = None) -> str:
    """Plain-English margin statement.

    Deliberately avoids naming the setting being varied. "How strict we were
    about what counts as changed" is what it means to the person reading this;
    the parameter name is not.
    """
    if not stability.get("available"):
        return ("We could not check how much this conclusion depends on how strict "
                "we were about what counts as changed, because this analysis did "
                "not keep the record it would need.")
    lo, hi = stability["stable_from"], stability["stable_to"]
    tested = f"{n_settings} " if n_settings else "every "
    if stability.get("holds_across_full_range"):
        return (f"We tried {tested}different settings for how big a difference "
                f"must be before it counts as change, and the conclusion came out "
                f"the same every time. It does not depend on that choice.")
    return (f"This conclusion holds only while we count fairly small differences "
            f"as change. If we counted only larger differences, it would change, "
            f"so treat it as a margin rather than a certainty.")
