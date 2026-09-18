"""B8 geodate: tag + filename extraction, pixel→lonlat affine, quadrants."""
from __future__ import annotations

import numpy as np
import pytest

from anvesha.geodate import (GeoBox, extract_acquired, extract_from_name,
                              extract_from_tags, pixel_to_lonlat,
                              polygon_geojson, quadrant_of)
from anvesha.io_utils import RSImage, load_image


def _write_tif_with_tags(path, arr, tags, crs="EPSG:4326"):
    import rasterio
    from rasterio.transform import from_bounds
    h, w, c = arr.shape
    # explicit bounds: lon 77.0..77.10, lat 28.0..28.10 (Delhi-ish), 0.10 deg
    left, bottom, right, top = 77.0, 28.0, 77.10, 28.10
    profile = dict(driver="GTiff", height=h, width=w, count=c, dtype="float32",
                   crs=crs, transform=from_bounds(left, bottom, right, top, w, h))
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.moveaxis(arr, -1, 0))
        dst.update_tags(**tags)


def test_tag_datetime_tiff_format():
    info = extract_from_tags({"TIFFTAG_DATETIME": "2024:05:01 10:00:00"})
    assert info.date == "2024-05-01"
    assert info.source == "rasterio_tags"


def test_filename_regexes():
    assert extract_from_name("S2A_20240501T053621_proc.tif").date == "2024-05-01"
    assert extract_from_name("siteA_2024-07-12_dem.tif").date == "2024-07-12"
    assert extract_from_name("cartosat_20240301.tif").date == "2024-03-01"
    assert extract_from_name("no_date_here.tif") is None
    # calendar-invalid compact dates must not parse
    assert extract_from_name("x_20241301.tif") is None


def test_unknown_when_nothing_matches():
    info = extract_acquired(name="plain.tif", tags={"OTHER": "x"})
    assert info.date == "unknown" and info.source == "unknown"


def test_load_image_wires_acquired_and_tags_win(tmp_path):
    f = tmp_path / "s2_2024-05-01.tif"
    arr = np.zeros((32, 32, 3), np.float32)
    _write_tif_with_tags(f, arr, {"TIFFTAG_DATETIME": "2023:12:25 08:30:00"})
    img = load_image(f)
    assert img.acquired == "2023-12-25"        # tags beat the filename
    assert img.summary()["acquired"] == "2023-12-25"


def test_load_image_filename_fallback(tmp_path):
    f = tmp_path / "plain_2024-05-01.tif"
    _write_tif_with_tags(f, np.zeros((32, 32, 3), np.float32), {})
    img = load_image(f)
    assert img.acquired == "2024-05-01"


def test_quadrant_vocabulary():
    assert quadrant_of([65, 15, 75, 25], 100, 100) == "north-east"
    assert quadrant_of([30, 60, 40, 70], 100, 100) == "south-west"
    assert quadrant_of([45, 45, 55, 55], 100, 100) == "centre"


def _geo_img(w=100, h=100, ow=200, oh=200, crs="EPSG:4326"):
    bounds = (77.0, 28.0, 77.10, 28.10)
    return RSImage(array=np.zeros((h, w, 3), np.float32), format="geotiff",
                   modality="rgb", crs=crs, transform_bounds=bounds,
                   original_height=oh, original_width=ow)


def test_pixel_to_lonlat_affine_roundtrip():
    img = _geo_img()
    # current 100x100 grid; original 200x200 -> scale factor 2
    gb = pixel_to_lonlat(img, [50, 25, 60, 35])
    assert gb.crs == "EPSG:4326"
    poly = gb.lonlat_polygon
    assert len(poly) == 5 and poly[0] == poly[-1]      # closed ring
    # box [50,25,60,35] -> original [100,50,120,70]
    # lon = 77 + 100*0.0005 = 77.05 .. 77 + 120*0.0005 = 77.06
    # lat = 28.10 - 50*0.0005 = 28.075 .. 28.10 - 70*0.0005 = 28.065
    assert abs(poly[0][0] - 77.05) < 1e-9 and abs(poly[0][1] - 28.075) < 1e-9
    assert abs(poly[2][0] - 77.06) < 1e-9 and abs(poly[2][1] - 28.065) < 1e-9
    assert "quadrant" in gb.note


def test_pixel_space_when_no_crs():
    img = RSImage(array=np.zeros((50, 50, 3), np.float32), format="geotiff")
    gb = pixel_to_lonlat(img, [1, 1, 2, 2])
    assert gb.lonlat_polygon == [] and "pixel-space" in gb.note
    gj = polygon_geojson([[1, 1, 2, 2]], img)
    assert gj["features"] == [] and "pixel-space" in gj["note"]


def test_polygon_geojson_features():
    img = _geo_img()
    gj = polygon_geojson([[10, 10, 20, 20], [30, 30, 40, 40]], img)
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 2
    ring = gj["features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]
