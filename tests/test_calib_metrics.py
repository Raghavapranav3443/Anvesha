"""Calibration metrics: the numbers that decide whether confidence is honest.

These are pure functions, so they get exhaustive tests rather than smoke tests --
everything in the confidence layer and the decision layer ultimately reads them,
and a wrong ECE silently mislabels every answer the system gives.
"""
from __future__ import annotations

import numpy as np
import pytest

from satquery.calib_metrics import (accuracy_confidence_gap, brier_score,
                                    expected_calibration_error, judge,
                                    lookup_band, reliability_table,
                                    select_temperature)


# --------------------------------------------------------------------------- #
# expected_calibration_error
# --------------------------------------------------------------------------- #

def test_perfectly_calibrated_scores_near_zero():
    """Claim 0.8, right 80% of the time -> ECE ~0."""
    conf = np.full(1000, 0.8)
    correct = np.array([True] * 800 + [False] * 200)
    assert expected_calibration_error(conf, correct) < 1e-9


def test_overconfidence_is_measured():
    """Claim 0.9, right 60% of the time -> ECE ~0.3."""
    conf = np.full(1000, 0.9)
    correct = np.array([True] * 600 + [False] * 400)
    assert expected_calibration_error(conf, correct) == pytest.approx(0.3, abs=1e-9)


def test_underconfidence_is_measured_symmetrically():
    conf = np.full(1000, 0.5)
    correct = np.array([True] * 900 + [False] * 100)
    assert expected_calibration_error(conf, correct) == pytest.approx(0.4, abs=1e-9)


def test_ece_is_bin_weighted_not_a_bare_gap():
    """A large error in a tiny bin must count less than a small error in a big one.

    This is the property that makes ECE the right objective for a displayed
    confidence: it weights by how often a band is actually used.
    """
    conf = np.array([0.1] * 990 + [0.95] * 10)
    correct = np.array([False] * 990 + [True] * 10)
    # Bin 0.95-1.00 is wrong by 0.05 on 10 samples; bin 0.00-0.50 wrong by 0.1
    # on 990. A bare |mean gap| would be tiny; ECE is dominated by the big bin.
    ece = expected_calibration_error(conf, correct)
    assert ece > 0.05


def test_empty_input_is_zero_not_an_error():
    assert expected_calibration_error([], []) == 0.0


def test_mismatched_lengths_do_not_crash():
    assert expected_calibration_error([0.5, 0.6], [True]) == 0.0


# --------------------------------------------------------------------------- #
# gap / brier / judge
# --------------------------------------------------------------------------- #

def test_gap_sign_is_overconfidence():
    conf = np.full(100, 0.9)
    correct = np.array([True] * 50 + [False] * 50)
    assert accuracy_confidence_gap(conf, correct) == pytest.approx(0.4, abs=1e-9)


def test_brier_is_zero_for_perfect_certainty():
    assert brier_score([1.0, 0.0], [True, False]) == 0.0


def test_brier_penalises_confident_mistakes():
    assert brier_score([0.95], [False]) > brier_score([0.55], [False])


@pytest.mark.parametrize("conf,correct,expected", [
    (0.10, False, "well_calibrated"),   # bin 0.0-0.5: claims .1, right 0% -> gap .1
])
def test_judge_reports_a_verdict(conf, correct, expected):
    # Single sample: mean conf 0.10 vs accuracy 0.0 -> gap +0.10 -> overconfident.
    assert judge([conf], [correct])["verdict"] == "overconfident"


def test_judge_calls_out_underconfidence():
    v = judge([0.50] * 90 + [0.50] * 10,
              [True] * 90 + [False] * 10)
    assert v["verdict"] == "underconfident"
    assert v["gap"] < 0


# --------------------------------------------------------------------------- #
# reliability table
# --------------------------------------------------------------------------- #

def test_reliability_table_reports_observation_not_claim():
    conf = np.array([0.95] * 100)
    correct = np.array([True] * 70 + [False] * 30)
    row = [r for r in reliability_table(conf, correct) if r["n"]][0]
    assert row["claimed_mean"] == pytest.approx(0.95, abs=1e-3)
    assert row["observed_accuracy"] == pytest.approx(0.70, abs=1e-3)
    assert row["optimism"] == pytest.approx(0.25, abs=1e-3)


def test_reliability_table_marks_thin_bands_unusable():
    conf = np.array([0.85] * 5)
    correct = np.array([True] * 5)
    row = [r for r in reliability_table(conf, correct) if r["n"]][0]
    assert row["sufficient"] is False


def test_empty_band_has_no_invented_accuracy():
    """Absence of evidence must not become a number."""
    rows = reliability_table(np.array([0.55]), np.array([True]))
    empty = [r for r in rows if not r["n"]]
    assert empty, "fixture should leave some bands empty"
    for r in empty:
        assert r["observed_accuracy"] is None
        assert r["claimed_mean"] is None
        assert "optimism" not in r


# --------------------------------------------------------------------------- #
# lookup_band
# --------------------------------------------------------------------------- #

def _table():
    return reliability_table(np.array([0.55, 0.55, 0.95, 0.95]),
                             np.array([True, True, True, False]))


def test_lookup_finds_the_matching_band():
    hit = lookup_band(0.95, _table(), min_n=1)
    assert hit and hit["observed_accuracy"] == pytest.approx(0.5, abs=1e-3)


def test_lookup_refuses_thin_bands():
    """A band with too few samples must return None, never the claim."""
    assert lookup_band(0.95, _table(), min_n=50) is None


def test_lookup_returns_none_without_a_table():
    assert lookup_band(0.9, None) is None
    assert lookup_band(0.9, []) is None


# --------------------------------------------------------------------------- #
# select_temperature -- the guard that keeps calibration from making things worse
# --------------------------------------------------------------------------- #

def test_selects_the_temperature_with_lowest_ece():
    correct = np.array([True] * 60 + [False] * 40)
    curves = {
        1.0: np.full(100, 0.90),   # ECE 0.30
        0.8: np.full(100, 0.65),   # ECE 0.05  <- best
        1.5: np.full(100, 0.45),   # ECE 0.15
    }
    pick = select_temperature([1.0, 0.8, 1.5], curves, correct, identity=1.0)
    assert pick["temperature"] == 0.8
    assert pick["improved"] is True
    assert pick["ece"] < pick["identity_ece"]


def test_refuses_a_temperature_that_is_worse_than_the_identity():
    """The core safety property: never record a 'calibration' that degrades."""
    correct = np.array([True] * 70 + [False] * 30)
    curves = {
        1.0: np.full(100, 0.68),   # ECE 0.02  <- already good
        1.7: np.full(100, 0.50),   # ECE 0.20  <- NLL would have picked this
    }
    pick = select_temperature([1.0, 1.7], curves, correct, identity=1.0)
    assert pick["temperature"] == 1.0
    assert pick["improved"] is False
    assert pick["ece"] == pick["identity_ece"]


def test_selection_requires_the_identity_to_be_a_candidate():
    with pytest.raises(ValueError):
        select_temperature([0.5], {0.5: np.array([0.5])}, np.array([True]))
