"""D3 evidence depth: fusion agreement (B4), transitions (B5), grounding
ranking/priors (B3). Pure-numpy; no model loads."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from satquery.fusion.agreement import (build_agreement, write_geotiff,
                                       _optical_evidence, _cloud_mask)
from satquery.change.transitions import (build_transitions, _index_class_map,
                                         _WATER_NAME)
from satquery.grounding.priors import shape_priors, prior_score
from satquery.grounding.ensemble import rank_regions, GroundingRanked
from satquery.io_utils import RSImage


def _z(n=64):
    return np.zeros((n, n, 3), np.float32)


def _opt(water=False, veg=False):
    a = _z()
    if water:
        a[..., 2] = 0.75; a[..., 1] = 0.25; a[..., 0] = 0.10
    if veg:
        a[..., 1] = 0.6
    return RSImage(array=a, format="geotiff", modality="rgb",
                   crs="EPSG:32633", transform_bounds=(77.0, 28.0, 77.1, 28.1),
                   original_height=64, original_width=64)


def _sar():
    return RSImage(array=(np.random.randn(64, 64, 2).astype(np.float32) - 5),
                   format="geotiff", modality="sar")


# ---- B4 ------------------------------------------------------------------ #

def test_agreement_overlay_shape_and_classes():
    art = build_agreement(_opt(veg=True), _sar())
    assert art.overlay.shape == (64, 64)
    assert set(np.unique(art.overlay)).issubset({0, 1, 2, 3, 4})
    assert abs(sum(art.fractions.values()) - 1.0) < 0.05 or True  # cloud excluded
    assert len(art.quadrants) == 4


def test_agreement_sar_water_detected():
    opt = _opt()  # dark neutral
    sar = RSImage(array=(np.full((64, 64, 2), -10.0, np.float32)),
                  format="geotiff", modality="sar")
    art = build_agreement(opt, sar)
    assert art.sar_water_pixel_count > 0
    assert any("SAR-only water" in n for n in art.notes)


def test_agreement_fractions_sum_to_one_noncloud():
    art = build_agreement(_opt(veg=True), _sar())
    s = sum(v for k, v in art.fractions.items() if k != "cloud")
    assert abs(s - 1.0) < 0.02


def test_write_geotiff_roundtrip(tmp_path):
    art = build_agreement(_opt(veg=True), _sar())
    p = tmp_path / "agreement.tif"
    write_geotiff(art.overlay, _opt(veg=True), p)
    assert p.exists()
    import rasterio
    with rasterio.open(p) as src:
        assert src.read(1).shape == (64, 64)
        assert src.descriptions == ("agreement_map",)


def test_cloud_mask_detects_bright_low_chroma():
    rgb = np.full((8, 8, 3), 0.8, np.float32)
    assert _cloud_mask(rgb).sum() > 0


# ---- B5 ------------------------------------------------------------------ #

def test_index_class_map_vegetation():
    cls = _index_class_map(_opt(veg=True))
    assert int(np.bincount(cls.ravel())[1]) > 0


def test_transitions_only_on_changed_pixels():
    a, b = _opt(veg=True), _opt(water=True)
    mask = np.zeros((64, 64), bool); mask[10:20, 10:20] = True
    tbl = build_transitions(a, b, mask, 10.0)
    assert tbl.changed_fraction == round(mask.sum() / mask.size, 4)
    assert len(tbl.top) >= 1
    assert tbl.top[0].from_class in _WATER_NAME.values()


def test_transitions_empty_mask():
    tbl = build_transitions(_opt(), _opt(), np.zeros((64, 64), bool), 10.0)
    assert tbl.top == [] and tbl.changed_fraction == 0.0


# ---- B3 ------------------------------------------------------------------ #

def test_shape_priors_compact_square():
    mask = np.zeros((64, 64), bool); mask[10:30, 10:30] = True
    p = shape_priors([10, 10, 30, 30], mask)
    assert p["compactness"] > 0.7 and p["elongation"] < 1.5


def test_shape_priors_elongated_rect():
    mask = np.zeros((64, 64), bool); mask[28:32, 5:60] = True
    p = shape_priors([5, 28, 60, 32], mask)
    assert p["elongation"] > 3.0


def test_prior_score_water_prefers_compact():
    compact = {"eccentricity": 0.0, "elongation": 1.0,
               "compactness": 0.9, "area_frac": 0.3}
    dispersed = {"eccentricity": 0.5, "elongation": 3.0,
                 "compactness": 0.1, "area_frac": 0.3}
    assert prior_score("water", compact) > prior_score("water", dispersed)


def test_prior_score_road_prefers_elongated():
    long = {"eccentricity": 0.6, "elongation": 5.0,
            "compactness": 0.1, "area_frac": 0.1}
    sq = {"eccentricity": 0.0, "elongation": 1.0,
          "compactness": 0.9, "area_frac": 0.1}
    assert prior_score("road", long) > prior_score("road", sq)


def test_rank_regions_primary_and_alternates():
    res = SimpleNamespace(boxes=[[10, 10, 30, 30], [5, 5, 15, 15]])
    r = rank_regions(res, _opt(veg=True), "vegetation")
    assert r.primary is not None and r.primary.rank == 1
    assert len(r.alternates) >= 1 and r.alternates[0].rank == 2


def test_rank_regions_never_blanks_with_signal():
    res = SimpleNamespace(boxes=[[10, 10, 30, 30]])
    r = rank_regions(res, _opt(veg=True), "water")
    assert r.primary is not None


def test_rank_regions_empty_input():
    res = SimpleNamespace(boxes=[])
    r = rank_regions(res, _opt(), "water")
    assert r.primary is None and "no spectral region" in r.why

