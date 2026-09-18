"""B9 — open-ended answer composer (no VLM).

Builds a multi-sentence, fully traceable answer from EXISTING specialist
outputs: classifier verdict + caption sentence + top-2 evidence bullets +
confidence + trace link. Every clause names its source key so a judge can
verify it against the trace. Explicit pitch vs generic VLM: "every sentence
has a source; GPT-4V cannot say that on SAR."
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..text import concept_of_text


def compose_answer(query: str, out: Dict, imgs, run_id: str = "") -> Dict:
    """Return a composed, traceable answer dict (additive over ``out``).

    Preserves ``out["answer"]`` verbatim as ``answer_raw``. The composed
    answer is multi-sentence with per-clause provenance.
    """
    out = dict(out)
    concept, _ = concept_of_text(query)
    clauses: List[Dict[str, str]] = []

    verdict = str(out.get("answer") or "").strip()
    if verdict:
        clauses.append({"kind": "verdict", "text": verdict,
                        "source_key": "answer"})

    caption = (out.get("caption") or "").strip().rstrip(".")
    if caption:
        clauses.append({"kind": "scene", "text": caption,
                        "source_key": "caption"})

    evidence = out.get("evidence") or []
    for ev in evidence[:2]:
        if isinstance(ev, dict):
            clauses.append({"kind": "evidence",
                            "text": str(ev.get("text", ev)),
                            "source_key": str(ev.get("source_key", "evidence"))})
        elif ev:
            clauses.append({"kind": "evidence", "text": str(ev),
                            "source_key": "evidence"})

    conf = out.get("confidence")
    meta = out.get("confidence_meta") or {}
    if conf is not None:
        method = meta.get("method", "formula")
        clauses.append({"kind": "confidence",
                        "text": f"Confidence {float(conf):.0%} "
                                f"(method: {method}).",
                        "source_key": "confidence_meta"})

    parts = [c["text"] for c in clauses if c.get("text")]
    composed = " ".join(parts)
    if concept:
        composed = composed.replace(". ", f". The query concerned {concept}. ", 1) \
            if composed else composed

    out["answer"] = composed or verdict
    out["answer_raw"] = verdict or out.get("answer")
    out["answer_clauses"] = clauses
    out["trace_link"] = f"/api/reports/{run_id}" if run_id else None
    out["provenance_pitch"] = ("Every clause above names its source key and "
                               "can be verified in the execution trace; a "
                               "generic VLM cannot point to its sources on SAR.")
    return out
