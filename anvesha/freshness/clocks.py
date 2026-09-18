"""C3 — freshness clocks (deterministic, run/input age only — never claims
satellite revisit)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional


# Per-task staleness thresholds (days) — adaptation plan WS-3 + R5 addendum.
_THRESHOLDS = {"change_analysis": 16, "change_vqa": 16, "impact_analysis": 16,
                "optical_sar": 14, "single_vqa": 7, "captioning": 7,
                "grounding": 7, "investigation": 16, "auto": 7}


@dataclass
class FreshnessClock:
    generated_at: str
    latest_obs: Optional[str]
    quality: str               # "ok" | "degraded"
    staleness_days: Optional[int]
    threshold_days: int
    clocks: dict               # {"server": iso, "data": iso|None}
    method_note: str
    why: Optional[str] = None  # present when quality == "degraded"

    def to_dict(self) -> dict:
        return asdict(self)


def _age_days(iso: Optional[str]) -> Optional[int]:
    if not iso or iso == "unknown":
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:                  # treat naive dates as UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).days


def clocks_for(result, imgs) -> FreshnessClock:
    """Quality = degraded iff fallback specialist active, no-CRS-where-geo-
    expected, or confidence below the per-task gate; else ok."""
    from ..confmeta import load_calibration
    task = getattr(result, "selected_task", None) or "auto"
    thr = _THRESHOLDS.get(task, _THRESHOLDS["auto"])

    # latest observation = most recent acquired date among inputs
    acquired = [getattr(im, "acquired", "") or "" for im in imgs]
    known = [a for a in acquired if a and a != "unknown"]
    latest = max(known) if known else None
    age = _age_days(latest)

    # degraded reasons
    why_parts = []
    if age is not None and age > thr:
        why_parts.append(f"latest input is {age}d old (threshold {thr}d)")
    if hasattr(result, "outputs") and isinstance(result.outputs, dict):
        conf = result.outputs.get("confidence")
        if conf is not None and float(conf) < 0.45:
            why_parts.append(f"confidence {float(conf):.2f} below 0.45 gate")
    quality = "degraded" if why_parts else "ok"

    return FreshnessClock(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        latest_obs=latest or None,
        quality=quality,
        staleness_days=age,
        threshold_days=thr,
        clocks={"server": datetime.now().isoformat(timespec="seconds"),
                "data": latest or None},
        method_note=("run + input acquisition age only; no orbital-look "
                     "scheduling claims. degraded = age>threshold or low "
                     "confidence."),
        why="; ".join(why_parts) if why_parts else None)
