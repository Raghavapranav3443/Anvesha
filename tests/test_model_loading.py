"""Regression tests: trained weights must actually load.

Mirrors test_agent.py::test_scene_encoder_loads_when_weights_exist for the
remaining specialists. Guards against the silent-fallback failure mode where
a corrupt/moved/incompatible checkpoint quietly degrades a specialist to
heuristic mode (the exact bug class that hit SceneEncoder historically).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anvesha.config import CONFIG


def _require(path: Path, what: str):
    if not path.exists():
        pytest.skip(f"no trained {what} in weights/")


def test_vqa_model_loads_when_weights_exist():
    _require(CONFIG.vqa_weights, "VQA head")
    from anvesha.models.vqa import RSVQAModel
    m = RSVQAModel(device="cpu")
    assert m.trained is True, (
        "VQA weights exist but the model silently fell back to the "
        "rule-based reasoner — check the load path / checkpoint format")
    assert m.encoder is not None and m.head is not None


def test_change_detector_loads_when_weights_exist():
    _require(CONFIG.change_weights, "change detector")
    from anvesha.models.change import ChangeDetectorNet
    d = ChangeDetectorNet(device="cpu")
    assert d.trained is True, (
        "change weights exist but the detector silently fell back to "
        "smoothed differencing")
    assert d.encoder is not None and d.head is not None


def test_fusion_loads_when_weights_exist():
    _require(CONFIG.fusion_weights, "optical-SAR fusion")
    from anvesha.models.optical_sar import FusionNet
    f = FusionNet(device="cpu")
    # FusionNet refuses synthetic-mode checkpoints by design (stays heuristic
    # until trained on real pairs) — trained=True is the only acceptable
    # state when a real checkpoint is present.
    assert f.trained is True, (
        "fusion weights exist but FusionNet silently fell back to the "
        "heuristic analyser (synthetic-mode refusal or load error)")
    assert f.head is not None and f.opt_encoder is not None \
        and f.sar_encoder is not None


def test_model_status_reports_trained():
    """The degradation-transparency helper must reflect real load state."""
    from anvesha.models.status import model_status
    status = model_status()
    assert set(status) >= {"scene_encoder", "vqa", "change", "fusion"}
    if CONFIG.scene_encoder_weights.exists():
        assert status["scene_encoder"] == "trained"
    if CONFIG.vqa_weights.exists():
        assert status["vqa"] == "trained"
    if CONFIG.change_weights.exists():
        assert status["change"] == "trained"
    if CONFIG.fusion_weights.exists():
        assert status["fusion"] == "trained"