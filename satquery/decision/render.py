"""Render a decision record as Markdown for the downloadable reports.

The decision layer answers "what do I do with this?". A user who downloads the
report is exactly the user who needs that answer -- so it has to be a *section*,
not a key inside a JSON dump. Before this existed, the Markdown and PDF exports
rendered the decision as part of a generic ``## Outputs`` preview: technically
present, and unreadable to the person it was written for.

Kept out of ``advise.py`` so the narrative logic stays free of formatting, and so
both the Markdown report and the UI can consume the same ``advice`` dict.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def decision_heading(decision: Dict[str, Any]) -> str:
    label = ((decision.get("advice") or {}).get("outcome_label")
             or decision.get("outcome") or "Result")
    return f"## What to do — {label}"


def as_markdown(decision: Optional[Dict[str, Any]]) -> List[str]:
    """Decision record -> Markdown lines. Empty list when there is no decision."""
    if not isinstance(decision, dict):
        return []
    advice = decision.get("advice") or {}
    if not advice:
        return []

    lines: List[str] = ["", "---", "", decision_heading(decision), ""]

    if advice.get("headline"):
        lines += [f"**{advice['headline']}**", ""]

    found = advice.get("what_we_found")
    if found:
        lines += [found, ""]

    clauses = (("What it means", advice.get("what_it_means")),
               ("What to do", advice.get("what_to_do")),
               ("Who to tell", advice.get("who_to_tell")),
               ("How far to trust this", advice.get("how_far_to_trust")))
    for title, body in clauses:
        if body:
            lines += [f"**{title}.** {body}", ""]

    also = [t for t in (advice.get("also_true") or []) if t]
    if also:
        lines.append("**Also true for this area:**")
        lines += [f"- {t}" for t in also]
        lines.append("")

    limits = [t for t in (advice.get("what_we_cannot_tell") or []) if t]
    if limits:
        lines.append("**What this analysis cannot tell you:**")
        lines += [f"- {t}" for t in limits]
        lines.append("")

    if advice.get("result_note"):
        lines += [f"*{advice['result_note']}*", ""]

    trust = decision.get("trust") or {}
    ev = trust.get("band_evidence") or {}
    if isinstance(ev.get("observed_accuracy"), (int, float)) and ev.get("n"):
        claimed = ev.get("claimed")
        said = (f"{float(claimed) * 100:.0f}%" if isinstance(claimed, (int, float))
                else "this range")
        lines += [
            (f"*Measured on held-out data: when results landed in this range we "
             f"had said {said}, and they were right about "
             f"{float(ev['observed_accuracy']) * 100:.0f}% of the time "
             f"({ev['n']} cases).*"),
            "",
        ]
    return lines
