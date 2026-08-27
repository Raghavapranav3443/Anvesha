"""Integration tests using real ISRO sample images.

These tests verify the full pipeline works end-to-end with actual satellite
data, not just synthetic fixtures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLES = ROOT / "samples"


@pytest.fixture(scope="module")
def controller():
    from satquery.agent import AgentController
    return AgentController()


@pytest.mark.skipif(
    not (SAMPLES / "isro_cartosat2s_optical.tif").exists(),
    reason="ISRO sample not available"
)
class TestISROIntegration:
    def test_single_image_vqa(self, controller):
        """Full VQA pipeline on ISRO Cartosat-2S optical image."""
        from satquery.io_utils import load_image
        img = load_image(SAMPLES / "isro_cartosat2s_optical.tif")
        result = controller.run([img], "Is there water in this image?")
        assert result.selected_task in ("single_vqa", "grounding", "captioning")
        assert result.answer
        assert result.confidence > 0
        assert len(result.trace) >= 3, "Trace should have at least validate + classify + execute"
        # Verify trace has timing info (recorded, not necessarily >0 — a
        # sub-millisecond rule-reasoner step can round to 0 on fast machines)
        exec_steps = [s for s in result.trace if s.get("name", "").startswith("execute")]
        assert len(exec_steps) >= 1, "At least one execute step in trace"
        assert "duration_ms" in exec_steps[0], "Execute step should record timing"

    def test_single_image_captioning(self, controller):
        """Full captioning pipeline on ISRO sample."""
        from satquery.io_utils import load_image
        img = load_image(SAMPLES / "isro_cartosat2s_optical.tif")
        result = controller.run([img], "Describe the scene")
        assert result.selected_task in ("captioning", "single_vqa")
        assert result.answer
        assert result.confidence > 0

    def test_trace_completeness(self, controller):
        """Verify the execution trace has all required pipeline steps."""
        from satquery.io_utils import load_image
        img = load_image(SAMPLES / "isro_cartosat2s_optical.tif")
        result = controller.run([img], "What is shown in this image?")
        step_names = [s["name"] for s in result.trace]
        assert "validate_inputs" in step_names, "Missing validate step"
        assert "classify_task" in step_names, "Missing classify step"
        assert any(n.startswith("execute:") for n in step_names), "Missing execute step"
        assert step_names[-1] in ("finish", "controller"), "Missing finish step"


@pytest.mark.skipif(
    not (SAMPLES / "demo_change_2020.tif").exists(),
    reason="Change demo samples not available"
)
class TestChangeDetectionIntegration:
    def test_bitemporal_change(self, controller):
        """Full change detection pipeline on demo bi-temporal pair."""
        from satquery.io_utils import load_image
        a = load_image(SAMPLES / "demo_change_2020.tif")
        b = load_image(SAMPLES / "demo_change_2024.tif")
        result = controller.run([a, b], "What changed between these dates?")
        assert result.selected_task in ("change_analysis", "change_vqa")
        assert result.answer
        assert result.confidence > 0
        # Should have outputs with change info
        assert result.outputs is not None
