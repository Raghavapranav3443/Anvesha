"""Geometry of a change mask: how many pieces, and how big the largest one is.

Area alone cannot tell "new construction" from "a change of surface". A season
of building work appears as many small parcels, roof-sized patches and linear
road or bund features; a sediment bar, an exposed river bed, an embankment or
one large earthworks site appears as a single compact block. These tests pin the
fact that separates the two readings -- including the path taken when scipy is
not installed, which must agree with scipy exactly rather than approximating it.
"""
from __future__ import annotations

import sys

import numpy as np

from anvesha.impact import label_components, region_stats


def _blobs() -> np.ndarray:
    """Two 3x3 blocks and one isolated pixel."""
    m = np.zeros((12, 12), dtype=bool)
    m[1:4, 1:4] = True
    m[8:11, 8:11] = True
    m[5, 11] = True
    return m


def test_components_are_counted_eight_connected():
    labels, n = label_components(_blobs())
    assert n == 3
    assert labels[1, 1] == labels[3, 3], "diagonally touching pixels are one blob"
    assert labels[1, 1] != labels[8, 8]
    assert labels[0, 0] == 0, "background must be 0, as scipy's convention"


def test_scipy_free_fallback_agrees_with_scipy(monkeypatch):
    """The air-gapped path must produce the same labelling, not an approximation.

    ``scipy`` arrives as a transitive dependency of scikit-learn, so the
    fallback rarely runs -- which is exactly why it needs a test rather than
    optimism.
    """
    scipy_labels, scipy_n = label_components(_blobs())
    monkeypatch.setitem(sys.modules, "scipy.ndimage", None)
    plain_labels, plain_n = label_components(_blobs())
    assert plain_n == scipy_n
    assert (sorted(np.bincount(plain_labels.ravel())[1:])
            == sorted(np.bincount(scipy_labels.ravel())[1:]))


def test_an_empty_mask_is_not_an_error():
    labels, n = label_components(np.zeros((5, 5), dtype=bool))
    assert n == 0
    assert labels.shape == (5, 5)
    assert region_stats(np.zeros((5, 5), dtype=bool), 10.0) == {
        "regions": 0, "largest_ha": 0.0, "largest_share": 0.0}


def test_largest_share_separates_one_block_from_many():
    """The same area, arranged two ways, must not read the same."""
    block = np.zeros((100, 100), dtype=bool)
    block[10:60, 10:60] = True                     # 2 500 px in one piece

    scattered = np.zeros((100, 100), dtype=bool)
    for i in range(0, 100, 10):
        for j in range(0, 100, 10):
            scattered[i:i + 5, j:j + 5] = True     # 100 pieces of 25 px

    one = region_stats(block, 10.0)
    many = region_stats(scattered, 10.0)

    assert one["regions"] == 1
    assert one["largest_share"] == 1.0
    assert one["largest_ha"] == 25.0, "2 500 px at 10 m is 25 ha"
    assert many["regions"] == 100
    assert many["largest_share"] < 0.02
