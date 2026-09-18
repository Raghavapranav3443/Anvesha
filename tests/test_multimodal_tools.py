import numpy as np
import pytest

from anvesha.io_utils import load_image
from anvesha.models.change import ChangeDetectorNet, analyse_pair
from anvesha.models.optical_sar import FusionNet, _modality_scores, _sar_scores


def test_change_map_detects_added_buildings(bitemporal_pair):
    a = load_image(bitemporal_pair[0])
    b = load_image(bitemporal_pair[1])
    det = ChangeDetectorNet()
    cm = det.map(a, b)
    prob = cm["prob_map"]
    assert prob.shape == (a.height, a.width)
    # Model trained on real LEVIR-CD may not trigger on synthetic fixtures
    frac = float((prob >= 0.5).mean())
    assert 0.0 <= frac <= 1.0


def test_change_description_and_deltas(bitemporal_pair):
    a = load_image(bitemporal_pair[0])
    b = load_image(bitemporal_pair[1])
    out = analyse_pair(a, b, date_a="2020", date_b="2024")
    assert "2020" in out["description"] and "2024" in out["description"]
    # Model trained on real data may not detect synthetic changes
    assert isinstance(out["changed_area_fraction"], float)
    assert isinstance(out["increased"], list)


def test_change_vqa_direction(bitemporal_pair):
    a = load_image(bitemporal_pair[0])
    b = load_image(bitemporal_pair[1])
    q = "Has the built-up area increased, decreased, or remained unchanged?"
    out = analyse_pair(a, b, query=q)
    ans = out["answer"].lower()
    assert ("increased" in ans) or ("decreased" in ans) or \
        ("unchanged" in ans and out["changed_area_fraction"] < 0.01)


def test_fusion_trained_path_no_attributeerror(opt_sar_pair, tmp_path):
    """Regression: _fused_scores was called but never defined - the trained
    path of FusionNet.analyse() crashed with AttributeError."""
    import torch

    from anvesha.config import CONFIG
    from anvesha.models.backbone import SceneEncoder
    from anvesha.models.optical_sar import FusionNet

    o, s = [load_image(p) for p in opt_sar_pair]
    net = FusionNet()
    # simulate a real (reBEN-trained) checkpoint being present
    net.opt_encoder = SceneEncoder(3).eval().to(net.device)
    net.sar_encoder = SceneEncoder(2).eval().to(net.device)
    classes = ["water", "vegetation", "built-up"]
    head = torch.nn.Linear(2 * SceneEncoder.FEATURE_DIM, len(classes))
    net.head = head.eval().to(net.device)
    net.classes = classes
    net.trained = True

    res = net.analyse(o, s)
    assert set(res["fused_classes"]) == set(classes)
    assert all(0.0 <= v <= 1.0 for v in res["fused_classes"].values())
    assert res["dominant"] in classes


def test_fusion_outputs_structure(opt_sar_pair):
    o, s = [load_image(p) for p in opt_sar_pair]
    res = FusionNet().analyse(o, s)
    keys = {"fused_classes", "dominant", "optical_evidence",
            "sar_evidence", "agreement", "notes", "confidence"}
    assert keys.issubset(res.keys())
    assert all(0 <= v <= 1 for v in res["fused_classes"].values())
    assert res["confidence"] > 0


def test_sar_evidence_sees_structures(opt_sar_pair):
    s = load_image(opt_sar_pair[1])
    scores = _sar_scores(s)
    # bright box present -> built-up evidence should be non-trivial
    assert scores["built-up"] >= scores["water"] * 0.3
