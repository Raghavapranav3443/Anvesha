"""End-to-end demonstration suite + report integrity checks."""
import json
from pathlib import Path

import numpy as np


MANDATORY_DEMOS = [
    # (images fixture name, query, expected task)
    ("rgb_png", "Describe the land-cover and major objects visible in this image.", "captioning"),
    ("ms_geotiff", "Is there vegetation in this image?", "single_vqa"),
    ("rgb_png", "Highlight the water body referred to in the query.", "grounding"),
]


def test_mandatory_single_image_demos(controller, request):
    for fixture, query, expected in MANDATORY_DEMOS:
        path = request.getfixturevalue(fixture)
        res = controller.run([path], query)
        assert res.selected_task == expected, f"{fixture}: {res.selected_task}"
        assert res.answer
        assert 0 < res.confidence <= 1


def test_mandatory_multi_image_demos(controller, bitemporal_pair, opt_sar_pair):
    fa, fb = bitemporal_pair
    res = controller.run([fa, fb],
                         "What changed between these two dates, and where did "
                         "the change occur?")
    assert res.selected_task in ("change_vqa", "change_analysis")
    assert res.visuals.get("change_overlay") is not None

    fo, fs = opt_sar_pair
    res2 = controller.run([fo, fs],
                          "Use the optical and SAR images together to identify "
                          "built-up and water-covered regions.")
    assert res2.selected_task == "optical_sar"


def test_execution_summary_auditable(controller, rgb_png):
    res = controller.run([rgb_png],
                         "Highlight the water body referred to in the query.")
    names = [s["name"] for s in res.trace]
    # observable execution trace must contain task routing + tool selection +
    # execution with parameters and outputs
    assert "validate_inputs" in names
    assert "classify_task" in names
    assert "select_tool" in names
    exec_step = next(s for s in res.trace if str(s["name"]).startswith("execute"))
    assert exec_step["status"] == "ok"
    assert "duration_ms" in exec_step

    payload = json.loads(Path(res.report_paths["json"]).read_text(encoding="utf-8"))
    assert payload["selected_task"] == res.selected_task
    tools_used = payload["extra"]["tools_used"]
    assert tools_used and all(t for t in tools_used)


def test_confidence_present_everywhere(controller, ms_geotiff):
    res = controller.run([ms_geotiff], "How many buildings are visible?")
    assert isinstance(res.confidence, float) and res.confidence > 0
