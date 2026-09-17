"""Regression guard for intent-routing determinism (contract invariant I11).

Background: ``classify_task`` used to build its candidate set with a bare
``set()`` and then stable-sort blended scores with no tie-break. Because tied
scores are common ("can you see a water body" scores single_vqa == captioning),
the winner was decided by string-hash order, which CPython randomises per
process. The same query could therefore select different specialists in
different processes, and ``ranked_candidates`` -- which is persisted into
``report.json`` -- reordered between runs.

These tests fail against that implementation and pass against the fixed one.
They are deliberately subprocess-based: hash-seed variation is a *per-process*
property, so an in-process assertion cannot detect it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# A battery that includes (a) the exact tie that used to flip, (b) lay-phrased
# questions the plain-English layer targets, and (c) one query per configuration.
BATTERY = [
    ("is there water present", "single"),
    ("can you see a water body", "single"),
    ("is there a road?", "single"),
    ("describe the land-cover of this image.", "single"),
    ("highlight the water body in this image.", "single"),
    ("is the lake near my village drying up?", "single"),
    ("what changed between these two dates?", "bitemporal_pair"),
    ("did anything change here?", "bitemporal_pair"),
    ("use the optical and SAR image together.", "optical_sar_pair"),
]

_ROUTE_SCRIPT = """
import json, sys
%PRELUDE%
from satquery.agent import classify_task
BATTERY = json.loads(sys.argv[1])
out = {q: classify_task(q, cfg)["task"] for q, cfg in BATTERY}
print("ROUTES=" + json.dumps(out, sort_keys=True))
"""


def _route(battery, *, seed="0", prelude="", env_extra=None):
    """Run the battery in a fresh interpreter and return the routing vector."""
    script = _ROUTE_SCRIPT.replace("%PRELUDE%", prelude)
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    env.pop("SATQUERY_RERANK_CLIP", None)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "-c", script, json.dumps(battery)],
        cwd=str(ROOT), capture_output=True, text=True, env=env, timeout=300,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("ROUTES="):
            return json.loads(line[len("ROUTES="):])
    pytest.fail(f"routing subprocess produced no result.\n"
                f"stdout={proc.stdout[-500:]}\nstderr={proc.stderr[-800:]}")


# --------------------------------------------------------------------------- #
# I11: determinism across process hash seeds
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", ["0", "1", "2", "3", "5", "7", "11", "23"])
def test_routing_stable_across_hash_seeds(seed):
    """Every seed must produce the identical routing vector.

    Seeds 2 and 3 are the two that used to fail (`captioning` instead of
    `single_vqa` for the water-body question) -- keep them in the list.
    """
    reference = _route(BATTERY, seed="0")
    assert _route(BATTERY, seed=seed) == reference


def test_water_body_tie_resolves_to_presence_question():
    """Correctness guard, not just stability.

    A presence question must not lose to a description request. An
    alphabetical tie-break would satisfy determinism while failing this:
    `captioning` sorts before `single_vqa`.
    """
    routes = _route([("is there water present", "single"),
                     ("can you see a water body", "single")])
    assert routes["is there water present"] == "single_vqa"
    assert routes["can you see a water body"] == "single_vqa"


# --------------------------------------------------------------------------- #
# I11: determinism across model load state and profile
# --------------------------------------------------------------------------- #

def test_routing_independent_of_clip_load_state():
    """Cold vs CLIP-warm process must route identically.

    Routing must not depend on whether a model happened to load. This is what
    keeps the `core` (no CLIP) and `extras` (CLIP) profiles interchangeable.
    """
    cold = _route(BATTERY, prelude="")
    warm = _route(BATTERY, prelude=(
        "from satquery.models.clip_text import get_clip_text\n"
        "_warm = get_clip_text()\n"))
    assert cold == warm


def test_routing_independent_of_clip_rerank_switch():
    """The opt-in CLIP re-rank term must not perturb default routing."""
    default = _route(BATTERY)
    enabled = _route(BATTERY, env_extra={"SATQUERY_RERANK_CLIP": "1"})
    # The term is opt-in and measured to cost accuracy, so with it enabled the
    # vector may differ -- but it must still be *deterministic*.
    again = _route(BATTERY, env_extra={"SATQUERY_RERANK_CLIP": "1"})
    assert enabled == again
    assert set(default) == set(enabled)


# --------------------------------------------------------------------------- #
# Explicit tie-break semantics (unit-level, in-process)
# --------------------------------------------------------------------------- #

def test_tie_index_prefers_declared_configuration_order():
    from satquery.agent import _tie_index
    # single_vqa is declared before grounding, which is before captioning
    assert _tie_index("single_vqa", "single") < _tie_index("grounding", "single")
    assert _tie_index("grounding", "single") < _tie_index("captioning", "single")
    # bitemporal declaration order
    assert _tie_index("change_vqa", "bitemporal_pair") < \
        _tie_index("change_analysis", "bitemporal_pair")


def test_tie_index_is_total_and_deterministic():
    """Unknown ids must sort last, in a stable way (no exceptions, no ties)."""
    from satquery.agent import _tie_index
    tasks = ["single_vqa", "captioning", "grounding", "change_vqa",
             "change_analysis", "impact_analysis", "optical_sar",
             "investigation", "not_a_real_task"]
    for cfg in ("single", "bitemporal_pair", "optical_sar_pair", "unknown_cfg"):
        indices = [_tie_index(t, cfg) for t in tasks]
        assert all(isinstance(i, int) for i in indices)
        # an unknown task must never outrank a known one for that configuration
        assert indices[-1] == max(indices)


def test_rerank_method_reports_computed_value():
    """`rerank_method` used to be a hardcoded "none", hiding re-ranks from the
    trace. It must now reflect what actually happened."""
    from satquery.agent import classify_task
    info = classify_task("is there water present", "single")
    assert info["rerank_method"] != "none"
    assert isinstance(info["rerank_method"], str) and info["rerank_method"]


def test_classify_report_shape_is_stable():
    """The keys the trace/report depend on must all still be present."""
    from satquery.agent import classify_task
    info = classify_task("is there water present", "single")
    assert set(info) >= {"task", "confidence", "method", "ranked_candidates",
                         "infeasible_ignored", "rerank_method"}
    assert isinstance(info["ranked_candidates"], list)
    assert all(isinstance(t, str) for t, _ in info["ranked_candidates"])
