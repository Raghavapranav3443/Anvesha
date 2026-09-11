# -*- coding: utf-8 -*-
"""B2 — router hardening: keyword expansion + optional CLIP text re-rank.

Re-ranks the top-k intents from ``classify_task`` using the CLIP text tower
cosine similarity vs auditable task-description sentences. Fully offline and
deterministic when the tower is cached; degrades gracefully to keyword-only
when ``SATQUERY_RERANK=0`` or the tower is unavailable — never raises.
"""
from __future__ import annotations

import os
import threading
from typing import Dict, List, Tuple

from .text import content_tokens

RERANK_ENV = "SATQUERY_RERANK"
_DIM = 512

_LOCK = threading.Lock()
_EXTENSION: Dict[str, List[str]] = {}

TASK_SENTENCES = {
    "single_vqa": "answer a question about what is present or how many",
    "captioning": "describe the scene land cover and major objects",
    "grounding": "locate highlight and outline a named region",
    "change_analysis": "detect map and measure changes between two dates",
    "change_vqa": "answer questions about what changed between two dates",
    "optical_sar": "fuse optical and radar evidence for the same area",
    "impact_analysis": "quantify the impact of changes area and proximity",
    "investigation": "run a full multi-step investigation with impact",
}


def _load_extension() -> Dict[str, List[str]]:
    global _EXTENSION
    if _EXTENSION:
        return _EXTENSION
    with _LOCK:
        if _EXTENSION:
            return _EXTENSION
        import json
        from .config import CONFIG
        p = CONFIG.weights_dir / "task_keywords.json"
        try:
            data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            _EXTENSION = {k: list(v) for k, v in data.items()} if isinstance(data, dict) else {}
        except Exception:
            _EXTENSION = {}
        return _EXTENSION


def expanded_keywords() -> Dict[str, List[str]]:
    return dict(_load_extension())


def _short_circuit(configuration: str) -> str | None:
    if configuration == "optical_sar_pair":
        return "optical_sar"
    return None


def re_rank(query: str, ranked: List[Tuple[str, float]], top_k: int = 2,
            configuration: str = "single"
            ) -> Tuple[List[Tuple[str, float]], str]:
    if os.environ.get(RERANK_ENV, "1").strip() == "0":
        return ranked[:top_k], "keyword-only (rerank disabled)"

    sc = _short_circuit(configuration)
    if sc and sc in dict(ranked):
        return [(sc, dict(ranked).get(sc, 0.0))], "pair-mode short-circuit"

    # blank query: deterministic default instead of zero-signal rescoring
    # (single image with no question -> describe it; pair -> compare it)
    if not query.strip():
        default = ("change_analysis" if configuration == "bitemporal_pair"
                   else "captioning")
        return [(default, 0.35)], "blank-query default"

    from .text import hashed_bow
    q_bow = hashed_bow(query.lower(), dim=_DIM)
    extension = _load_extension()
    candidates = list(dict(ranked).keys())[: max(top_k * 2, 4)]

    def keyword_score(task: str) -> float:
        toks = set(content_tokens(query.lower()))
        kws = list(TASK_SENTENCES.get(task, task).split()) + extension.get(task, [])
        if not kws:
            return 0.0
        # Absolute capped hit count — NOT density (hits/len(kws)). Density
        # rewarded *short* keyword lists, so a single shared token ("water")
        # out-scored a task with a richer vocabulary and flipped near-ties
        # ("can you see a water body" misrouted single_vqa -> captioning).
        # Capping at 3 keeps multiple distinct hits dominant over one.
        return min(sum(1.0 for kw in kws if kw in toks), 3.0) / 3.0

    clip = None
    try:
        from .clip_text import get_clip_text
        clip = get_clip_text()
    except Exception:
        clip = None

    rescored = []
    for task in candidates:
        base = float(dict(ranked).get(task, 0.0))
        kw = keyword_score(task)
        clip_sim = 0.0
        if clip is not None and hasattr(clip, "embed"):
            try:
                sent = TASK_SENTENCES.get(task, task)
                emb = clip.embed([sent])
                if emb is not None and len(emb):
                    v = emb[0].astype("float32")
                    n = (v / (float((v ** 2).sum()) ** 0.5 + 1e-9)).astype("float32")
                    clip_sim = max(0.0, float(q_bow @ n))
            except Exception:
                clip_sim = 0.0
        rescored.append((task, 0.5 * base + 0.3 * kw + 0.2 * clip_sim))

    rescored.sort(key=lambda kv: -kv[1])
    method = ("keyword+CLIP rerank" if clip is not None
              else "keyword-only rerank (CLIP unavailable)")
    return rescored[:top_k], method


def best_intent(query: str, configuration: str = "single"
                ) -> Tuple[str, float, str]:
    from .agent import classify_task
    info = classify_task(query, configuration)
    ranked = [(t, s) for t, s in info.get("ranked_candidates", [])]
    if not ranked:
        ranked = [(info.get("selected_task", "single_vqa"),
                   info.get("confidence", 0.0))]
    top, method = re_rank(query, ranked, top_k=1, configuration=configuration)
    task = top[0][0] if top else info.get("selected_task", "single_vqa")
    return task, float(top[0][1]) if top else 0.35, method
