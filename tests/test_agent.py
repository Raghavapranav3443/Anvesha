from pathlib import Path

import numpy as np
import pytest

from satquery.agent import classify_task
from satquery.io_utils import InputValidationError, load_image


def test_intent_routing_single(controller, rgb_png):
    for query, expected in [
        ("Describe the land-cover and major objects visible in this image.", "captioning"),
        ("Highlight the water body referred to in the query.", "grounding"),
        ("Is there water in this image?", "single_vqa"),
    ]:
        intent = classify_task(query, configuration := "single")
        assert intent["task"] == expected, f"{query!r} -> {intent}"


def test_intent_routing_pairs():
    assert classify_task("What changed between these two dates?",
                         "bitemporal_pair")["task"] in ("change_vqa", "change_description")
    assert classify_task("Has the built-up area increased, decreased, or "
                         "remained unchanged?", "bitemporal_pair")["task"] == \
        "change_vqa"
    assert classify_task("Use the optical and SAR images together",
                         "optical_sar_pair")["task"] == "optical_sar"


def test_infeasible_tasks_ignored():
    intent = classify_task("Use the optical and SAR images together", "single")
    assert intent["task"] != "optical_sar"
    assert "optical_sar" in "".join(intent.get("infeasible_ignored", []))


def test_single_vqa_run(controller, ms_geotiff):
    res = controller.run([ms_geotiff], "Is there vegetation in this image?")
    assert res.selected_task == "single_vqa"
    assert res.answer.lower().startswith(("yes", "no"))
    assert 0 < res.confidence <= 1
    names = [s["name"] for s in res.trace]
    assert names[:3] == ["validate_inputs", "classify_task", "select_tool"]
    assert any(str(n).startswith("execute:") for n in names)


def test_captioning_run(controller, rgb_png):
    res = controller.run([rgb_png], "")
    assert res.selected_task == "captioning"
    assert len(res.answer.split()) > 6
    assert res.outputs["labels"]


def test_grounding_run_finds_water(controller, rgb_png):
    res = controller.run([rgb_png],
                         "Highlight the water body referred to in the query.")
    assert res.selected_task == "grounding"
    boxes = res.outputs["boxes"]
    assert boxes, "expected at least one grounded region"
    x0, y0, x1, y1 = boxes[0]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2          # lake centre is (190,190)
    assert 140 <= cx <= 240 and 140 <= cy <= 240
    assert res.visuals.get("overlay") is not None


def test_change_analysis_and_change_vqa(controller, bitemporal_pair):
    fa, fb = bitemporal_pair
    res = controller.run([fa, fb], "")
    assert res.selected_task == "change_analysis"
    assert "description" in res.outputs
    assert res.visuals.get("change_overlay") is not None

    res2 = controller.run(
        [fa, fb], "Has the built-up area increased, decreased, or remained unchanged?")
    assert res2.selected_task == "change_vqa"
    assert "increased" in res2.answer.lower()
    assert res2.confidence >= 0.0


def test_optical_sar_run(controller, opt_sar_pair):
    fo, fs = opt_sar_pair
    res = controller.run([fo, fs],
                         "Use the optical and SAR images together to identify "
                         "built-up and water-covered regions.")
    assert res.selected_task == "optical_sar"
    out = res.outputs
    assert set(out["optical_evidence"]) >= {"water", "built-up"}
    assert out["notes"] and isinstance(out["notes"], list)
    assert "agreement" in out


def test_wrong_config_for_task_raises(controller, rgb_png):
    with pytest.raises(InputValidationError):
        controller.run([rgb_png], "", task_override="optical_sar")


def test_scene_encoder_loads_when_weights_exist():
    """Regression: a swallowed NameError in _load() used to silently disable
    the fine-tuned scene encoder."""
    import pytest as _pytest
    from satquery.config import CONFIG
    if not CONFIG.scene_encoder_weights.exists():
        _pytest.skip("no trained scene encoder in weights/")
    from satquery.models.scene import SceneClassifier
    sc = SceneClassifier()
    assert sc.trained is True
    assert sc.encoder is not None


def test_report_written(controller, rgb_png):
    from satquery.config import CONFIG
    res = controller.run([rgb_png], "Is there water in this image?")
    assert Path(res.report_paths["json"]).exists()
    assert Path(res.report_paths["markdown"]).exists()
    import json
    payload = json.loads(Path(res.report_paths["json"]).read_text(encoding="utf-8"))
    for key in ("selected_task", "final_answer", "confidence", "execution_summary"):
        assert key in payload
