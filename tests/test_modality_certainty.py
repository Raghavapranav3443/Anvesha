"""B1 modality certainty: name-independence + override precedence.

Proves the `area42_patch_002.tif` failure class is dead: content, not the
name, decides the modality; an explicit override always wins.
"""
from __future__ import annotations

import numpy as np
import pytest

from anvesha.io_utils import load_image
from anvesha.modality_certainty import refine_modality, stats_vote, _stats_evidence


def _write_tif(path, arr, crs="EPSG:32633"):
    import rasterio
    from rasterio.transform import from_bounds
    h, w, c = arr.shape
    profile = dict(driver="GTiff", height=h, width=w, count=c, dtype="float32",
                   crs=crs, transform=from_bounds(0, 0, w * 10, h * 10, w, h))
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.moveaxis(arr, -1, 0))


def _sar_dB(h=64, w=64):
    """dB-scale SAR-like raster: negative backscatter + speckle."""
    base = -8.0 - np.abs(np.random.randn(h, w)) * 2.0
    vh = base * 0.6
    return np.stack([base, vh], axis=-1)


def test_dB_stats_vote_decisive():
    ev = _stats_evidence(_sar_dB())
    assert stats_vote(ev) == "sar"
    assert ev["median"] < 0 and ev["pct_neg"] > 0.15


def test_name_independence_sar_content_neutral_name(tmp_path):
    # SAR dB content with a completely neutral name (the failure class).
    f = tmp_path / "area42_patch_002.tif"
    _write_tif(f, _sar_dB())
    img = load_image(f)
    assert img.modality == "sar"
    cert = img.modality_certainty
    assert cert["label"] == "sar"
    assert "stats" in cert["votes"]


def test_multispectral_content_beats_sar_name(tmp_path):
    # 4-band MS content named like SAR -> stats (weight 2) must win.
    f = tmp_path / "sentinel1_sar_scene.tif"
    b = np.full((64, 64), 0.08, np.float32)
    g = np.full((64, 64), 0.30, np.float32)
    r = np.full((64, 64), 0.22, np.float32)
    n = np.full((64, 64), 0.60, np.float32)
    _write_tif(f, np.stack([b, g, r, n], axis=-1))
    img = load_image(f)
    assert img.modality == "multispectral"
    cert = img.modality_certainty
    assert cert["votes"]["filename"] == "sar"      # the old failure recorded
    assert cert["votes"]["stats"] == "optical"
    assert "override" not in cert or cert["override"] != "stats"


def test_rgb_content_beats_sar_name(tmp_path):
    f = tmp_path / "s1_3band_sar.tif"
    img3 = np.zeros((64, 64, 3), np.float32)
    img3[..., 1] = 0.55
    _write_tif(f, img3)
    loaded = load_image(f)
    assert loaded.modality == "rgb"


def test_explicit_override_wins_and_is_recorded(tmp_path):
    f = tmp_path / "area42_patch_002.tif"
    _write_tif(f, _sar_dB())
    img = load_image(f, modality_override="optical")
    assert img.modality == "rgb"                    # 2-band optical -> rgb
    cert = img.modality_certainty
    assert cert["override"] == "optical"
    assert cert["confidence"] == 0.99
    assert cert["reason"] == "explicit user/judge override"


def test_refine_agreement_raises_confidence():
    arr = _sar_dB()
    label, names, cert = refine_modality("s1_vh_vv.tif", arr, "sar",
                                         ["VV", "VH"], [])
    assert label == "sar"
    assert cert["confidence"] >= 0.85
    assert "agree" in cert["reason"]


def test_summary_carries_additive_keys(tmp_path):
    f = tmp_path / "area42_patch_002.tif"
    _write_tif(f, _sar_dB())
    img = load_image(f)
    s = img.summary()
    assert "modality_certainty" in s
    assert s["modality_certainty"]["label"] == "sar"
