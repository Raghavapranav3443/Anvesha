"""W2 test debt: coverage for impact engine, investigation chain,
suggestions, count-head routing and the TorchScript CPU path."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.agent import AgentController
from satquery.io_utils import load_image
from satquery.suggestions import suggest


# ------------------------------------------------------------------- #
# Impact engine
# ------------------------------------------------------------------- #

def test_impact_analysis_tool(controller, bitemporal_pair):
    from satquery.tools_impl import impact_analysis_tool
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
    from satquery.impact import gsd_meters
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
    assert execs == ["execute:change_analysis", "execute:grounding_water",
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
    from satquery.config import CONFIG
    if not (CONFIG.weights_dir / "count_head.pt").exists():
        pytest.skip("count head not trained")
    from satquery.models.vqa import get_vqa_model
    m = get_vqa_model()
    res = m.answer(load_image(rgb_png), "How many roads are visible in this image?")
    assert res["source"].startswith("dedicated counting head")
    assert res["answer"].strip().isdigit()


# ------------------------------------------------------------------- #
# TorchScript CPU path
# ------------------------------------------------------------------- #

def test_torchscript_cpu_path():
    from satquery.config import CONFIG
    if not (CONFIG.weights_dir / "ts" / "vqa_encoder_int8.ts").exists():
        pytest.skip("TorchScript export not present")
    from satquery.models.vqa import RSVQAModel
    m = RSVQAModel(device="cpu")
    assert m.trained
    import torch.jit as jit
    assert isinstance(m.encoder, jit.ScriptModule)
    assert isinstance(m.head, jit.ScriptModule)
