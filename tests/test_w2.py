"""W2 test debt: coverage for impact engine, investigation chain,
suggestions, count-head routing and the TorchScript CPU path."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anvesha.agent import AgentController
from anvesha.io_utils import load_image
from anvesha.suggestions import suggest


# ------------------------------------------------------------------- #
# Impact engine
# ------------------------------------------------------------------- #

def test_impact_analysis_tool(controller, bitemporal_pair):
    from anvesha.tools_impl import impact_analysis_tool
    a, b = [load_image(p) for p in bitemporal_pair]
    out = impact_analysis_tool({"images": [a, b], "query": "", "params": {}})
    assert "findings" in out and out["findings"]
    assert out["changed_area_ha"] >= 0
    assert set(out["near_water"]) >= {"within_250m", "within_500m", "within_1000m"}
    assert "transitions" in out and "zones_top" in out
    # georeferenced fixtures → real GSD from the transform, no assumption
    assert out["gsd_assumed"] is False and out["gsd_m"] == pytest.approx(10.0)
    assert out["answer"]  # synthesised narrative exists


def test_impact_gsd_assumed_for_non_geo(rgb_png):
    from anvesha.impact import gsd_meters
    img = load_image(rgb_png)
    assert gsd_meters(img) == 10.0       # labelled assumption


# ------------------------------------------------------------------- #
# Investigation chain
# ------------------------------------------------------------------- #

def test_investigation_chain(controller, bitemporal_pair):
    fa, fb = bitemporal_pair
    res = controller.run([fa, fb],
                         "Investigate urban expansion around the water body.")
    assert res.selected_task == "investigation"
    names = [s["name"] for s in res.trace]
    assert "plan_investigation" in names
    execs = [n for n in names if n.startswith("execute:")]
    # Query-conditioned routing selects 'urban' plan for 'urban expansion' query
    assert execs == ["execute:change_analysis", "execute:grounding_built_up",
                     "execute:impact_analysis"]
    inv = res.outputs["investigation"]
    assert "impact" in inv and "findings" in inv["impact"]
    assert res.outputs.get("suggestions")
    assert res.answer  # synthesised narrative


def test_investigation_step_error_captured(controller, bitemporal_pair, monkeypatch):
    """A failing mid-chain step must be recorded, not silently swallowed."""
    fa, fb = bitemporal_pair

    def boom(ctx):
        raise RuntimeError("boom")

    # patch the controller's OWN registry entry (the planner reads spec.fn)
    orig_fn = controller.registry["impact_analysis"].fn
    controller.registry["impact_analysis"].fn = boom
    try:
        res = controller.run([fa, fb], "Investigate the change.")
        names = [s["name"] for s in res.trace]
        assert "execute:impact_analysis" in names
        step = next(s for s in res.trace
                    if s["name"] == "execute:impact_analysis")
        assert step["status"] == "error"
        assert res.answer  # synthesis still completes around the failure
    finally:
        controller.registry["impact_analysis"].fn = orig_fn


# ------------------------------------------------------------------- #
# Suggestions
# ------------------------------------------------------------------- #

def test_suggestions_conditional_rules():
    base = {"selected_task": "impact_analysis", "outputs": {}}
    s0 = suggest(base)
    assert isinstance(s0, list) and s0

    veg = {"selected_task": "impact_analysis",
           "outputs": {"investigation": {"impact": {
               "transitions": {"vegetation_lost_ha": 2.5}}}}}
    s1 = suggest(veg)
    assert any("vegetation" in q.lower() for q in s1)

    water = {"selected_task": "impact_analysis",
             "outputs": {"investigation": {"impact": {
                 "near_water": {"within_500m": 0.6}}}}}
    s2 = suggest(water)
    assert "How much of the change lies within 500 m of water?" == s2[0]

    # dedupe
    assert len(s2) == len(set(x.lower() for x in s2))


# ------------------------------------------------------------------- #
# Count-head routing
# ------------------------------------------------------------------- #

def test_count_head_routing(controller, rgb_png):
    from anvesha.config import CONFIG
    if not (CONFIG.weights_dir / "count_head.pt").exists():
        pytest.skip("count head not trained")
    if not CONFIG.vqa_weights.exists():
        pytest.skip("VQA weights not present (count head loads via the "
                    "trained VQA model; CI has no vqa_head.pt)")
    from anvesha.models.vqa import get_vqa_model
    m = get_vqa_model()
    res = m.answer(load_image(rgb_png), "How many roads are visible in this image?")
    assert res["source"].startswith("dedicated counting head")
    assert res["answer"].strip().isdigit()


# ------------------------------------------------------------------- #
# Per-type specialist head routing
# ------------------------------------------------------------------- #

def test_type_head_routing(controller, rgb_png):
    from anvesha.config import CONFIG
    if not (CONFIG.weights_dir / "type_heads.pt").exists():
        pytest.skip("type heads not trained")
    from anvesha.models.vqa import get_vqa_model
    m = get_vqa_model()
    assert m.th is not None
    # presence question routes to the presence specialist, decodes a string
    res = m.answer(load_image(rgb_png), "Is there a road present in this image?")
    assert res["source"].startswith("per-type specialist head (presence)")
    assert isinstance(res["answer"], str) and res["answer"] in ("yes", "no")
    # rural/urban question routes to its specialist with valid vocab
    res2 = m.answer(load_image(rgb_png), "Is it a rural or an urban area?")
    assert res2["source"].startswith("per-type specialist head (rural_urban)")
    assert res2["answer"] in ("rural", "urban")


# ------------------------------------------------------------------- #
# TorchScript CPU path
# ------------------------------------------------------------------- #

def test_torchscript_cpu_path():
    from anvesha.config import CONFIG
    if not (CONFIG.weights_dir / "ts" / "vqa_encoder_int8.ts").exists():
        pytest.skip("TorchScript export not present")
    if not CONFIG.vqa_weights.exists():
        pytest.skip("VQA weights not present (TorchScript path activates "
                    "only on a trained model; CI has no vqa_head.pt)")
    from anvesha.models.vqa import RSVQAModel
    m = RSVQAModel(device="cpu")
    assert m.trained
    import torch.jit as jit
    assert isinstance(m.encoder, jit.ScriptModule)
    assert isinstance(m.head, jit.ScriptModule)


# ------------------------------------------------------------------- #
# Clarification loop (low-intent-confidence -> "did you mean" options)
# ------------------------------------------------------------------- #

def test_clarification_on_ambiguous_query(controller, rgb_png):
    from anvesha.agent import build_clarification, classify_task
    intent = classify_task("tell me about this place", "single")
    clar = build_clarification("tell me about this place", intent, "single")
    if clar is not None:
        assert clar["needed"] and clar["options"]
        assert all(o["task"] != clar["chosen"]["task"] for o in clar["options"])
        assert all(o.get("label") for o in clar["options"])


def test_clarification_skips_strong_intent_and_empty_query():
    from anvesha.agent import build_clarification, classify_task
    strong = classify_task("Highlight the water body referred to in the query.",
                           "single")
    assert strong["confidence"] >= 0.55
    assert build_clarification("any query", strong, "single") is None
    assert build_clarification("", classify_task("", "single"), "single") is None


def test_clarification_surfaces_in_run_outputs(controller, rgb_png):
    # Use a deliberately ambiguous query that has weak keyword + embedding signal.
    # NOTE: the clarification gate keys off the *intent* confidence, not the
    # answer confidence (which comes from the specialist tool output).
    from anvesha.agent import CLARIFY_THRESHOLD, classify_task
    res = controller.run([rgb_png], "something")
    intent_conf = classify_task("something", "single")["confidence"]
    clar = res.outputs.get("clarification")
    if intent_conf < CLARIFY_THRESHOLD:
        assert clar and clar["needed"] and clar["options"]
    else:
        assert clar is None
    # the run must also surface the intent confidence for auditability
    assert res.outputs.get("intent_confidence") == intent_conf
