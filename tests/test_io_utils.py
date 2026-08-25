from pathlib import Path

import numpy as np
import pytest

from satquery.io_utils import (RSImage, InputValidationError, load_image,
                               rgb_composite, validate_inputs)
from satquery.text import concept_of_text, hashed_bow


def test_load_rgb_png(rgb_png):
    img = load_image(rgb_png)
    assert isinstance(img, RSImage)
    assert img.modality == "rgb" and img.bands == 3
    assert 0.0 <= float(img.array.min()) and float(img.array.max()) <= 1.0 + 1e-6


def test_load_multispectral_geotiff(ms_geotiff):
    img = load_image(ms_geotiff)
    assert img.format == "geotiff"
    assert img.modality == "multispectral" and img.bands == 4
    assert img.crs == "EPSG:32633"
    comp = rgb_composite(img)
    assert comp.shape[2] == 3


def test_load_sar_geotiff_log_scaled(sar_geotiff):
    img = load_image(sar_geotiff)
    assert img.modality == "sar"
    assert np.isclose(img.array.max(), np.log1p(0.9), rtol=0.05)


def test_unsupported_format_rejected(bad_format_file):
    with pytest.raises(InputValidationError, match="Unsupported file format"):
        load_image(bad_format_file)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(InputValidationError, match="not found"):
        load_image(tmp_path / "ghost.tif")


def test_pair_validation_bitemporal(bitemporal_pair):
    a, b = [load_image(p) for p in bitemporal_pair]
    cfg = validate_inputs([a, b])
    assert cfg["configuration"] == "bitemporal_pair"


def test_pair_validation_crossmodal(opt_sar_pair):
    a, b = [load_image(p) for p in opt_sar_pair]
    cfg = validate_inputs([a, b])
    assert cfg["configuration"] == "optical_sar_pair"


def test_pair_size_mismatch_rejected(tmp_path, rgb_png):
    from PIL import Image
    small = tmp_path / "small.png"
    Image.fromarray((make_small())).save(small)
    a, b = load_image(rgb_png), load_image(small)
    with pytest.raises(InputValidationError, match="not spatially compatible"):
        validate_inputs([a, b])


def test_pair_height_mismatch_rejected(tmp_path):
    """Regression: geometry check must compare b.height against b.height."""
    from PIL import Image
    from satquery.io_utils import load_image, validate_inputs
    a_f, b_f = tmp_path / "a.png", tmp_path / "b.png"
    Image.fromarray((np.random.rand(128, 256, 3) * 255).astype(np.uint8)).save(a_f)
    Image.fromarray((np.random.rand(96, 256, 3) * 255).astype(np.uint8)).save(b_f)
    a, b = load_image(a_f), load_image(b_f)          # same width, diff height
    with pytest.raises(InputValidationError, match="not spatially compatible"):
        validate_inputs([a, b])


def test_two_sar_rejected(tmp_path, sar_geotiff):
    a = load_image(sar_geotiff)
    b = load_image(sar_geotiff)
    with pytest.raises(InputValidationError, match="Two SAR"):
        validate_inputs([a, b])


def make_small():
    return (np.random.rand(64, 64, 3) * 255).astype(np.uint8)


def test_text_utils():
    c, s = concept_of_text("Highlight the water body in this scene")
    assert c == "water" and s > 0
    v = hashed_bow("river near the city")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5
