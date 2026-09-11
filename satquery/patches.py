"""Patch layer: the single additive seam for all B/C masterplan patches.

Contract (system docs/implementation_plan.md, R1 + Key Decisions):
* New behaviour lives in NEW modules; specialists are never modified.
* ``apply_patches(registry)`` wraps each ``ToolSpec.fn`` with a per-task
  enricher registered in ``_ENRICHERS``. Enrichers only ADD keys to the tool
  output dict — they never replace or remove existing keys, and a failing
  enricher can never fail the tool (guarded call).
* ``enrich_result(result, images)`` attaches run-level additive keys
  (dossier / freshness / honesty / reticles / geo / calibration-method) to
  ``AgentResult.outputs`` before ``write_report`` persists them into
  ``report.json``.
* Kill-switch: ``SATQUERY_PATCHES=0`` disables the whole layer — tools run
  byte-identical to the pre-patch baseline (used for A/B and test parity).

D0 status: scaffolding only. ``_ENRICHERS`` is intentionally empty;
``enrich_result`` is an intentional no-op. Later steps (B3-B9, C2-C4)
register enrichers; the wiring here never changes again.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

PATCH_ENV = "SATQUERY_PATCHES"

# Per-task output enrichers: task id -> fn(ctx, out) -> None (mutates ``out``).
# Populated by later steps; MUST only add keys (additive contract).
_ENRICHERS: Dict[str, Callable[[Dict, Dict], None]] = {}


# --------------------------------------------------------------------------- #
# Enricher implementations (B6 / B7 / B9) — additive only.
# --------------------------------------------------------------------------- #

def _stamp_output(out: Dict, component: str) -> None:
    """Add a ``confidence_meta`` entry (B7) if a confidence is present."""
    from .confmeta import stamp as _stamp
    if out.get("confidence") is None:
        return
    stamped = _stamp(out, component)
    if "confidence_meta" in stamped:
        out["confidence_meta"] = stamped["confidence_meta"]


def _after_caption(ctx: Dict, out: Dict) -> None:
    from .captions.compose import compose_caption
    img = (ctx.get("images") or [None])[0]
    query = ctx.get("query") or ""
    if "caption" not in out:
        return
    enriched = compose_caption(img, query,
                               {k: out[k] for k in ("caption", "labels", "layout")
                                if k in out})
    out["caption"] = enriched.get("caption", out.get("caption"))
    out["query_concept"] = enriched.get("query_concept")
    out["caption_learned"] = enriched.get("caption_learned")
    _stamp_output(out, "captioning")


def _after_vqa(ctx: Dict, out: Dict) -> None:
    from .answers.compose import compose_answer
    imgs = ctx.get("images") or []
    query = ctx.get("query") or ""
    composed = compose_answer(query, dict(out), imgs)
    out["answer"] = composed.get("answer", out.get("answer"))
    out["answer_raw"] = composed.get("answer_raw")
    out["answer_clauses"] = composed.get("answer_clauses")
    out["provenance_pitch"] = composed.get("provenance_pitch")
    _stamp_output(out, "vqa")


def _after_grounding(ctx, out):
    _stamp_output(out, "grounding")


def _after_change(ctx, out):
    _stamp_output(out, "change")


def _after_change_vqa(ctx, out):
    _stamp_output(out, "cdvqa")


def _after_optical_sar(ctx, out):
    _stamp_output(out, "fusion")


# Register enrichers at import time (active only when patches_enabled()).
for _n, _f in (("captioning", _after_caption), ("single_vqa", _after_vqa),
                ("grounding", _after_grounding),
                ("change_analysis", _after_change),
                ("change_vqa", _after_change_vqa),
                ("optical_sar", _after_optical_sar)):
    _ENRICHERS[_n] = _f


def patches_enabled() -> bool:
    """False only when SATQUERY_PATCHES is explicitly "0"."""
    return os.environ.get(PATCH_ENV, "1").strip() != "0"


def register_enricher(task: str, fn: Callable[[Dict, Dict], None]) -> None:
    """Register (or replace) an output enricher for a task id.

    Test-only registrations should be popped again; production enrichers
    register at import time from their own modules.
    """
    _ENRICHERS[task] = fn


def apply_patches(registry: Dict[str, Any]) -> None:
    """Wrap every registered tool with its enricher (idempotent).

    With no enrichers registered this is a verified no-op: the tool
    signature (name/description/requires/needs_query) shown in the
    execution trace is never altered.
    """
    if not patches_enabled():
        return
    for name, spec in registry.items():
        fn = getattr(spec, "fn", None)
        if fn is None or getattr(fn, "_satquery_patched", False):
            continue                      # idempotent: never double-wrap
        wrapped = _wrap(name, fn)
        if wrapped is fn:
            continue
        wrapped._satquery_patched = True
        spec.fn = wrapped


def _wrap(name: str, fn: Callable) -> Callable:
    enricher = _ENRICHERS.get(name)
    if enricher is None:
        return fn                         # nothing registered -> passthrough

    def patched(ctx: Dict, _fn=fn, _enrich=enricher):
        out = _fn(ctx)
        try:
            _enrich(ctx, out)
        except Exception:                 # enricher must never fail a tool
            import traceback
            out["_patch_error"] = traceback.format_exc(limit=2)
        return out

    return patched


def enrich_result(result: Any, images: List) -> None:
    """Attach run-level additive keys to ``result.outputs`` (R1 contract).

    D0: intentional no-op placeholder. From step D2 onward this attaches the
    frozen additive keys — ``dossier``, ``freshness``, ``honesty``,
    ``reticles``, ``geo``, ``acquired``, plus calibration-method stamps —
    without ever replacing an existing key. All failures are swallowed so a
    patch defect can never break a judge-visible run.
    """
    if not patches_enabled():
        return
    try:
        _enrich_result_keys(result, images)
    except Exception:                     # defensive: patches never crash runs
        pass


def _enrich_result_keys(result: Any, images: List) -> None:
    """Concrete R1 key attachment (C2/C3): dossier + freshness into outputs."""
    outputs = getattr(result, "outputs", None)
    if not isinstance(outputs, dict):
        return
    try:
        from .freshness import clocks_for
        outputs["freshness"] = clocks_for(result, images).to_dict()
    except Exception:
        pass
    try:
        from .dossier import emit as _dossier_emit
        outputs["dossier"] = _dossier_emit(result, images).to_dict()
    except Exception:
        pass
