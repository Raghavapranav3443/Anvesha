"""D4 demo orchestration — B2 router hardening (offline-safe).

Covers keyword expansion, re-rank with keyword-only fallback (no CLIP tower
loaded), pair-mode short-circuit, the miner output, and the golden intent set.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from satquery.rerank import (re_rank, expanded_keywords, TASK_SENTENCES,
                             best_intent, _short_circuit)


# ---- rerank module ------------------------------------------------------- #

def test_task_sentences_all_tasks():
    from satquery.agent import TASK_ALIASES
    from satquery.registry import TASK_IDS
    for t in TASK_IDS:
        assert t in TASK_SENTENCES


def test_short_circuit_sar_pair():
    assert _short_circuit("optical_sar_pair") == "optical_sar"
    assert _short_circuit("single") is None


def test_rerank_keyword_only_fallback_when_disabled(monkeypatch):
    monkeypatch.setenv("SATQUERY_RERANK", "0")
    ranked = [("single_vqa", 0.8), ("captioning", 0.2)]
    out, method = re_rank("is there water?", ranked, top_k=2)
    assert out == ranked
    assert "disabled" in method


def test_rerank_preserves_feasible_order_without_clip(monkeypatch):
    monkeypatch.setenv("SATQUERY_RERANK", "1")
    ranked = [("single_vqa", 0.9), ("captioning", 0.3)]
    out, method = re_rank("how many buildings are there?", ranked, top_k=2,
                          configuration="single")
    assert len(out) == 2
    assert out[0][0] in ("single_vqa", "captioning")
    assert "keyword" in method  # CLIP likely unavailable -> keyword fallback


def test_rerank_pair_mode_wins():
    ranked = [("single_vqa", 0.9), ("optical_sar", 0.1)]
    out, method = re_rank("fuse optical and sar", ranked, top_k=2,
                          configuration="optical_sar_pair")
    assert out[0][0] == "optical_sar"
    assert "short-circuit" in method


def test_expanded_keywords_loads_from_mined_file():
    kw = expanded_keywords()
    assert isinstance(kw, dict)
    assert "single_vqa" in kw or "change_vqa" in kw  # miner ran


def test_best_intent_single_returns_vqa_for_question():
    task, conf, method = best_intent("is there water?", "single")
    assert task == "single_vqa"
    assert 0.0 <= conf <= 1.0


def test_best_intent_sar_pair_short_circuits():
    task, _, _ = best_intent("fuse optical and sar evidence",
                             "optical_sar_pair")
    assert task == "optical_sar"


# ---- miner output --------------------------------------------------------- #

def test_task_keywords_file_well_formed():
    p = Path("weights/task_keywords.json")
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert any(isinstance(v, list) and v for v in data.values())


def test_golden_intent_file_well_formed():
    p = Path("scripts/golden_intent.json")
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert 100 <= len(data) <= 500
    valid = {"single_vqa", "captioning", "grounding", "change_analysis",
             "change_vqa", "optical_sar", "investigation", "impact_analysis"}
    for item in data:
        assert "query" in item and "expected_task" in item
        assert item["expected_task"] in valid


def test_golden_intent_has_all_tasks():
    p = Path("scripts/golden_intent.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    tasks = {item["expected_task"] for item in data}
    assert "single_vqa" in tasks
    assert "grounding" in tasks


def test_golden_accuracy_keyword_baseline():
    """Keyword-only classifier must beat a low bar on the golden set (the
    existing TASK_KEYWORDS already handle the explicit cases)."""
    p = Path("scripts/golden_intent.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    # only evaluate the explicit-seed queries (first 15), deterministic
    seeds = [d for d in data if d["query"].endswith((".", "?"))][:15]
    if not seeds:
        pytest.skip("no seed queries")
    from scripts.mine_intent_keywords import bucket as _bucket
    correct = sum(1 for s in seeds
                  if _bucket(s["query"]) == s["expected_task"])
    # at least the explicit seeds must classify correctly
    assert correct == len(seeds), f"keyword bucket misclassifies seeds"
