"""Offline tests for the online acquisition layer.

Every test here runs with no network. That is a requirement, not a preference:
the project's air-gap gate runs the whole suite with outbound sockets blocked,
so anything in ``tests/`` that reached the internet would make the gate fail --
or, worse, pass vacuously.

Live behaviour is covered by ``scripts/acquire_smoke.py``, which is run
deliberately and reports against the real services.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

from anvesha.acquire import aoi as aoi_mod
from anvesha.acquire import fetch as fetch_mod
from anvesha.acquire import http as http_mod
from anvesha.acquire.cache import DiskCache, key_for
from anvesha.acquire.errors import FetchFailed, NetworkBlockedAirgap
from anvesha.acquire.providers.bhuvan import (CONTROL_BBOX, BhuvanProvider,
                                               state_codes_for)
from anvesha.acquire.providers.stac import (SceneRef, StacProvider,
                                             default_providers,
                                             search_with_fallback)


# --------------------------------------------------------------------------- #
# Fixtures / fakes
# --------------------------------------------------------------------------- #

class FakeTransport:
    """Transport that answers from a table. Records every call."""

    def __init__(self, *, json_map: Optional[Dict[str, Any]] = None,
                 bytes_map: Optional[Dict[str, bytes]] = None,
                 fail: Optional[Exception] = None) -> None:
        self.json_map = json_map or {}
        self.bytes_map = bytes_map or {}
        self.fail = fail
        self.calls: list = []

    def _route(self, url: str):
        self.calls.append(url)
        if self.fail is not None:
            raise self.fail
        if url in self.json_map:
            return self.json_map[url]
        if url in self.bytes_map:
            return self.bytes_map[url]
        raise AssertionError(f"FakeTransport has no answer for {url}")

    def get_bytes(self, url, **kw): return self._route(url)
    def get_json(self, url, **kw):
        value = self._route(url)
        return json.loads(value) if isinstance(value, (bytes, str)) else value
    def post_json(self, url, payload, **kw): return self._route(url)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch):
    """Keep every test off the real disk cache.

    Without this, a live smoke run populates ``data/acquire_cache`` and tests
    expecting a geocoder round trip silently start passing on a cache hit --
    which is exactly what happened before this fixture existed.
    """
    cache = DiskCache(tmp_path / "cache")
    monkeypatch.setattr(aoi_mod, "CACHE", cache)
    monkeypatch.setattr(http_mod, "CACHE", cache)
    return cache


def _scene(**over) -> SceneRef:
    base = dict(
        provider="stac_earth_search", scene_id="S2B_TILE_20260417_0_L2A",
        datetime="2026-04-17T04:41:23Z", collection="sentinel-2-l2a",
        cloud_pct=3.1, epsg=32646,
        assets={b: f"https://example.invalid/{b}.tif"
                for b in ("blue", "green", "red", "nir")},
    )
    base.update(over)
    return SceneRef(**base)


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

def test_cache_key_is_order_independent_for_dicts():
    assert key_for("a", {"x": 1, "y": 2}) == key_for("a", {"y": 2, "x": 1})


def test_cache_key_changes_when_a_filter_changes():
    a = key_for("search", {"bbox": [0, 0, 1, 1], "cloud": 20})
    b = key_for("search", {"bbox": [0, 0, 1, 1], "cloud": 30})
    assert a != b, "a changed filter must not reuse a cached answer"


def test_cache_roundtrip_and_metadata(tmp_path: Path):
    cache = DiskCache(tmp_path)
    key = key_for("thing")
    assert cache.get(key) is None
    cache.put(key, b"hello", kind="test", url="https://example.invalid/x")
    assert cache.get(key) == b"hello"
    meta = cache.get_meta(key)
    assert meta["url"] == "https://example.invalid/x"
    assert meta["bytes"] == 5
    assert len(meta["sha256"]) == 64


def test_cache_eviction_drops_oldest_first(tmp_path: Path):
    cache = DiskCache(tmp_path)
    for name in ("old", "new"):
        cache.put(key_for(name), b"x" * 2048)
        time.sleep(0.01)
    old_key, new_key = key_for("old"), key_for("new")
    freed = cache.evict(limit=2048)
    assert freed >= 2048
    assert not cache.has(old_key), "oldest entry should be evicted first"
    assert cache.has(new_key), "newest entry should survive"


# --------------------------------------------------------------------------- #
# Transport / air gap
# --------------------------------------------------------------------------- #

def test_transport_refuses_when_not_online(monkeypatch):
    monkeypatch.setattr(http_mod, "current_mode", lambda: "airgap")
    transport = http_mod.RequestsTransport()
    with pytest.raises(NetworkBlockedAirgap) as excinfo:
        transport.get_bytes("https://example.invalid/x")
    assert "air-gap" in str(excinfo.value).lower()


def test_airgap_serves_a_warm_cache_hit(_isolated_cache, monkeypatch):
    """Air-gap mode replays a fetch it already made instead of refusing.

    The refusal used to run *before* the cache was consulted, so a populated
    cache was unreachable exactly when it was needed: offline, at the demo. No
    socket is involved in a hit, so serving it cannot violate the air gap.
    """
    monkeypatch.setattr(http_mod, "current_mode", lambda: "airgap")
    url = "https://example.invalid/scene.tif"
    _isolated_cache.put(key_for("GET", url, {}, ""), b"tile-bytes")

    transport = http_mod.RequestsTransport(cache=_isolated_cache)
    assert transport.get_bytes(url) == b"tile-bytes"


def test_airgap_still_refuses_when_nothing_is_cached(_isolated_cache,
                                                    monkeypatch):
    """The air gap still holds: a miss is refused before any I/O, and says so.

    "No cached copy" and "not attempted" are different situations and the error
    has to distinguish them, or the refusal reads as a bug in the network stack.
    """
    monkeypatch.setattr(http_mod, "current_mode", lambda: "airgap")
    transport = http_mod.RequestsTransport(cache=_isolated_cache)
    with pytest.raises(NetworkBlockedAirgap) as excinfo:
        transport.get_bytes("https://example.invalid/never-fetched.tif")
    assert "no cached copy" in str(excinfo.value)


def test_cached_scene_search_is_replayed_offline(_isolated_cache, monkeypatch):
    """A STAC search is a POST; its cached answer must be readable offline.

    It was written and never read -- ``post_json`` passed ``use_cache=False``
    down -- so re-running scene discovery for an area already searched had no
    offline path at all, however warm the cache was.
    """
    url = "https://example.invalid/search"
    payload = {"collections": ["sentinel-2-l2a"], "bbox": [0, 0, 1, 1]}
    answer = {"features": [{"id": "S2B_TILE_20260417_0_L2A"}]}
    _isolated_cache.put(key_for("POST", url, {}, payload),
                        json.dumps(answer).encode("utf-8"), kind="stac",
                        url=url)
    monkeypatch.setattr(http_mod, "current_mode", lambda: "airgap")

    transport = http_mod.RequestsTransport(cache=_isolated_cache)
    assert transport.post_json(url, payload, kind="stac") == answer


def test_env_mode_override_does_not_become_a_stored_preference(
        monkeypatch, tmp_path: Path):
    """An env var must not rewrite the saved mode.

    It did: running the smoke script with ``ANVESHA_MODE=online`` wrote
    ``"mode": "online"`` into ``data/settings.json``, silently replacing the
    air-gap default for every subsequent launch.
    """
    from anvesha.acquire import mode as mode_mod

    settings = tmp_path / "settings.json"
    monkeypatch.setattr(mode_mod, "settings_path", lambda: settings)
    monkeypatch.setattr(mode_mod, "_settings", None)
    monkeypatch.setattr(mode_mod, "_mode", None)

    mode_mod.set_mode("airgap")                      # a deliberate choice
    assert json.loads(settings.read_text(encoding="utf-8"))["mode"] == "airgap"

    monkeypatch.setenv(mode_mod.ENV_MODE, "online")
    mode_mod.enforce_from_env()

    assert mode_mod.current_mode() == "online", "the override applies now"
    assert json.loads(settings.read_text(encoding="utf-8"))["mode"] == "airgap", \
        "the override must not be persisted"


def test_offline_transport_always_refuses():
    transport = http_mod.OfflineTransport()
    with pytest.raises(NetworkBlockedAirgap):
        transport.get_json("https://example.invalid/x")
    assert transport.calls, "the refused call should be recorded"


# --------------------------------------------------------------------------- #
# AOI
# --------------------------------------------------------------------------- #

def test_parse_coordinates_point_expands_to_a_window():
    bbox = aoi_mod.parse_coordinates("27.48, 94.90")
    assert bbox is not None
    west, south, east, north = bbox
    assert west < 94.90 < east and south < 27.48 < north
    frame = aoi_mod.bbox_size_km(bbox)
    assert max(frame) == pytest.approx(aoi_mod.DEFAULT_WINDOW_KM, rel=0.05)


def test_parse_coordinates_bbox_is_taken_as_given():
    assert aoi_mod.parse_coordinates("91.6,26.05,91.85,26.2") == \
        (91.6, 26.05, 91.85, 26.2)


def test_parse_coordinates_rejects_inverted_bbox():
    with pytest.raises(aoi_mod._AmbiguousCoordinates):
        aoi_mod.parse_coordinates("91.85,26.2,91.6,26.05")


def test_parse_coordinates_ignores_plain_text():
    assert aoi_mod.parse_coordinates("Dibrugarh, Assam") is None


def test_gazetteer_resolves_a_state_offline():
    hits = aoi_mod.gazetteer_lookup("Assam")
    assert hits, "the bundled index should cover Assam with no network"
    assert hits[0].source == "gazetteer"
    west, south, east, north = hits[0].bbox
    assert west < east and south < north


def test_gazetteer_ignores_trailing_country():
    assert aoi_mod.gazetteer_lookup("Kerala, India")
    assert aoi_mod.gazetteer_lookup("kerala")


def test_resolve_prefers_explicit_coordinates():
    aoi = aoi_mod.resolve("26.15, 91.75")
    assert aoi.source == "explicit"
    assert aoi.confidence == 1.0


def test_analysis_window_is_clipped_to_the_requested_extent():
    aoi = aoi_mod.analysis_window("Assam")
    width, height = aoi_mod.bbox_size_km(aoi.bbox)
    assert max(width, height) <= aoi_mod.DEFAULT_WINDOW_KM + 0.5


def test_resolve_refuses_unknown_place_without_network():
    with pytest.raises(aoi_mod.PlaceNotFound):
        aoi_mod.resolve("Nowhereville-in-the-sea", allow_network=False)


def test_resolve_uses_geocoder_when_available():
    payload = [{
        "display_name": "Dibrugarh, Assam, India",
        "boundingbox": ["27.43", "27.54", "94.84", "94.96"],
        "address": {"city": "Dibrugarh", "state": "Assam"},
    }]
    transport = FakeTransport(json_map={aoi_mod.NOMINATIM_URL: payload})
    aoi = aoi_mod.resolve("Dibrugarh", transport)
    assert aoi.source == "nominatim"
    assert aoi.admin["state"] == "Assam"


def test_geocoded_place_is_cached_so_it_works_offline_next_time():
    payload = [{
        "display_name": "Sonitpur, Assam, India",
        "boundingbox": ["26.60", "26.75", "92.70", "92.85"],
        "address": {"county": "Sonitpur", "state": "Assam"},
    }]
    live = FakeTransport(json_map={aoi_mod.NOMINATIM_URL: payload})
    first = aoi_mod.resolve("Sonitpur", live)
    assert first.source == "nominatim"

    offline = FakeTransport(fail=NetworkBlockedAirgap("blocked"))
    second = aoi_mod.resolve("Sonitpur", offline)
    assert second.source == "cache", \
        "a place resolved once must keep working with no network"


def test_multiple_matches_are_surfaced_not_silently_chosen(monkeypatch):
    """Two places with the same name must produce a choice, not a coin flip."""
    index = {
        "places": [
            {"name": "Aurangabad", "aliases": [],
             "bbox": [75.20, 19.80, 75.45, 20.00]},
            {"name": "Aurangabad", "aliases": [],
             "bbox": [84.30, 24.70, 84.45, 24.85]},
        ]
    }
    monkeypatch.setattr(aoi_mod, "load_gazetteer", lambda: index)
    aoi = aoi_mod.resolve("Aurangabad", allow_network=False)
    assert aoi.name == "Aurangabad"
    assert len(aoi.alternatives) == 1, \
        "the second match must be offered rather than dropped"


def test_state_name_is_clipped_and_flagged_rather_than_refused():
    aoi = aoi_mod.analysis_window("Assam", allow_network=False)
    width, height = aoi_mod.bbox_size_km(aoi.bbox)
    assert max(width, height) <= aoi_mod.DEFAULT_WINDOW_KM + 0.5
    assert "window_note" in aoi.admin, \
        "clipping a state to a centroid square must be disclosed"


def test_resolve_still_refuses_a_whole_state_when_the_area_matters():
    with pytest.raises(Exception):
        aoi_mod.resolve("Assam", allow_network=False, must_fit=True)


# --------------------------------------------------------------------------- #
# STAC
# --------------------------------------------------------------------------- #

def _feature(cloud, scene_id="S2_TEST", assets=None, props=None):
    props = dict(props or {})
    props.update({"datetime": "2026-04-17T04:41:23Z", "eo:cloud_cover": cloud,
                  "proj:epsg": 32646})
    return {
        "id": scene_id, "bbox": [94.8, 27.4, 94.9, 27.5],
        "properties": props,
        "assets": assets if assets is not None else {
            "blue": {"href": "https://example.invalid/blue.tif"},
            "green": {"href": "https://example.invalid/green.tif"},
            "red": {"href": "https://example.invalid/red.tif"},
            "nir": {"href": "https://example.invalid/nir.tif"},
        },
    }


def test_cloud_filter_is_applied_client_side():
    """The catalogue's own filter returned a false empty; ours must not."""
    provider = StacProvider(name="t", endpoint="https://example.invalid/v1")
    transport = FakeTransport(json_map={
        provider._search_url(): {"features": [_feature(5), _feature(60)]}
    })
    scenes = provider.search((94.8, 27.4, 94.9, 27.5), start="2026-04-01",
                             end="2026-04-30", transport=transport,
                             max_cloud_pct=20)
    assert len(scenes) == 1
    assert scenes[0].cloud_pct == 5


def test_unknown_cloud_cover_is_kept_not_dropped():
    provider = StacProvider(name="t", endpoint="https://example.invalid/v1")
    transport = FakeTransport(json_map={
        provider._search_url(): {"features": [_feature(None)]}
    })
    scenes = provider.search((94.8, 27.4, 94.9, 27.5), start="2026-04-01",
                             end="2026-04-30", transport=transport,
                             max_cloud_pct=20)
    assert len(scenes) == 1, "unknown cloud must be surfaced, not filtered away"


def test_asset_picking_captures_declared_scale_and_offset():
    provider = StacProvider(name="t", endpoint="https://example.invalid/v1")
    assets = {
        "blue": {"href": "https://example.invalid/b.tif",
                 "raster:bands": [{"scale": 0.0001, "offset": -0.1}]},
        "red": {"href": "https://example.invalid/r.tif",
                "raster:bands": [{"scale": 0.0001, "offset": -0.1}]},
        "nir": {"href": "https://example.invalid/n.tif",
                "raster:bands": [{"scale": 0.0001, "offset": -0.1}]},
        "visual-jp2": {"href": "https://example.invalid/visual.jp2"},
    }
    ref = provider._to_ref(_feature(5, assets=assets))
    assert ref.scales["red"] == pytest.approx(0.0001)
    assert ref.offsets["red"] == pytest.approx(-0.1)
    assert not any(str(u).endswith(".jp2") for u in ref.assets.values())


def test_default_providers_puts_primary_first_and_skips_unknown():
    providers = default_providers("stac_earth_search", ("stac_cdse", "nope"))
    assert [p.name for p in providers] == ["stac_earth_search", "stac_cdse"]


def test_signed_provider_is_known_but_never_used_automatically():
    from anvesha.acquire.providers.stac import all_providers

    assert all_providers()["stac_bhoonidhi"].requires_signing is True
    # Asking for it explicitly must still not put an unusable catalogue in the
    # search path; it would fail every request and mask the working ones.
    names = [p.name for p in default_providers("stac_bhoonidhi", ())]
    assert names == ["stac_earth_search"]


def test_search_fallback_reports_empty_separately_from_failure():
    """'no imagery here' and 'could not reach the catalogue' are different facts."""
    failing = StacProvider(name="broken", endpoint="https://example.invalid/v1")
    empty = StacProvider(name="empty", endpoint="https://example.invalid/v2")
    transport = FakeTransport(json_map={empty._search_url(): {"features": []}})
    scenes, errors = search_with_fallback([failing, empty], (0, 0, 1, 1),
                                          start="2026-01-01", end="2026-02-01",
                                          transport=transport)
    assert scenes == []
    messages = " ".join(e["error"] for e in errors)
    assert "no scenes matched" in messages
    assert "AssertionError" in messages or len(errors) == 2


def test_cloud_limit_is_named_when_it_rejected_every_scene():
    """Scenes the cloud limit rejected must not read as "no scenes matched".

    Live case: a monsoon window over Assam returned 21 scenes, the clearest at
    24% cloud against a 20% limit, and the plan told the user that no scenes
    matched in the date range -- naming the one lever that could not help.
    """
    provider = StacProvider(name="monsoon", endpoint="https://example.invalid/v1")
    transport = FakeTransport(json_map={
        provider._search_url(): {"features": [_feature(24), _feature(88)]}
    })
    scenes, errors = search_with_fallback([provider], (94.8, 27.4, 94.9, 27.5),
                                          start="2026-06-20", end="2026-09-18",
                                          transport=transport, max_cloud_pct=20)
    assert scenes == []
    message = errors[0]["error"]
    assert "no scenes matched" not in message, message
    assert "2 scene(s) matched" in message
    assert "20% cloud limit" in message
    assert "24%" in message, "the clearest scene's cloud cover is the useful number"
    assert "widening the dates will not help" in message


def test_the_advice_in_that_message_actually_works():
    """Raising the cloud limit returns the scenes the message named."""
    provider = StacProvider(name="monsoon", endpoint="https://example.invalid/v1")
    transport = FakeTransport(json_map={
        provider._search_url(): {"features": [_feature(24), _feature(88)]}
    })
    scenes, errors = search_with_fallback([provider], (94.8, 27.4, 94.9, 27.5),
                                          start="2026-06-20", end="2026-09-18",
                                          transport=transport, max_cloud_pct=30)
    assert [s.cloud_pct for s in scenes] == [24]
    assert errors == []


# --------------------------------------------------------------------------- #
# Fetch: grid
# --------------------------------------------------------------------------- #

def test_plan_grid_uses_a_utm_zone_and_square_pixels():
    grid = fetch_mod.plan_grid((94.84, 27.43, 94.96, 27.54))
    assert grid.epsg == 32646
    # North-up rasters have a positive x step and a negative y step.
    assert grid.transform.a == pytest.approx(-grid.transform.e, rel=1e-6)
    assert fetch_mod.MIN_PX <= grid.width <= 1024


def test_plan_grid_area_matches_the_requested_window():
    grid = fetch_mod.plan_grid((94.84, 27.43, 94.96, 27.54))
    assert 100 < grid.area_ha / 100 < 200        # ~12 km square = ~14,400 ha


def test_utm_zone_picks_east_and_west_correctly():
    assert fetch_mod.utm_epsg(27.5, 94.9) == 32646
    assert fetch_mod.utm_epsg(19.0, 72.9) == 32643
    assert fetch_mod.utm_epsg(-33.9, 151.2) == 32756


# --------------------------------------------------------------------------- #
# Fetch: radiometry (the bug that mattered)
# --------------------------------------------------------------------------- #

def test_offset_flag_survives_the_property_filter():
    """The conversion depends on this property, so it must be carried through.

    It was dropped by the property whitelist, which meant the physics guard had
    to catch the resulting error on every single fetch instead of the primary
    path simply being right.
    """
    provider = StacProvider(name="t", endpoint="https://example.invalid/v1")
    ref = provider._to_ref(_feature(5, props={
        "earthsearch:boa_offset_applied": True}))
    assert ref.properties.get("earthsearch:boa_offset_applied") is True
    assert fetch_mod.reflectance_conversion(ref).offset == 0.0


def test_conversion_honours_the_already_applied_offset_flag():
    """earth-search declares offset -0.1 *and* that it is already applied."""
    scene = _scene(scales={"red": 0.0001}, offsets={"red": -0.1},
                   properties={"earthsearch:boa_offset_applied": True})
    conv = fetch_mod.reflectance_conversion(scene)
    assert conv.offset == 0.0, "offset must not be applied twice"
    assert "already applied" in conv.reason


def test_conversion_applies_declared_offset_when_flag_absent():
    scene = _scene(scales={"red": 0.0001}, offsets={"red": -0.1},
                   properties={})
    conv = fetch_mod.reflectance_conversion(scene)
    assert conv.offset == pytest.approx(-0.1)


def test_alternate_conversion_flips_the_offset():
    base = fetch_mod.Conversion(0.0001, -0.1, "declared")
    alt = fetch_mod.alternate_conversion(base)
    assert alt.offset == 0.0
    back = fetch_mod.alternate_conversion(alt)
    assert back.offset == pytest.approx(-0.1)


def test_validity_is_independent_of_value():
    """A dark pixel clamped to zero is data, not a hole."""
    dn = np.array([[0, 500, 3000]], dtype="float32")      # 0 == nodata
    valid = dn != 0
    refl = fetch_mod.to_reflectance(dn, valid, scale=0.0001, offset=-0.1)
    assert refl[0, 0] == 0.0, "nodata stays zero"
    assert refl[0, 1] == 0.0, "dark pixel clamps to zero but is NOT flagged missing"
    assert valid[0, 1], "the clamped pixel must remain valid"
    assert refl[0, 2] == pytest.approx(0.2)


def test_assess_radiometry_accepts_a_plausible_scene():
    red = np.full((4, 4), 0.09, dtype="float32")
    nir = np.full((4, 4), 0.30, dtype="float32")
    stack = np.stack([red * 0.6, red * 0.8, red, nir])
    report = fetch_mod.assess_radiometry(stack, np.ones((4, 4), bool),
                                         ["blue", "green", "red", "nir"])
    assert report["ok"] is True
    assert -1.0 <= report["ndvi_median"] <= 1.0
    assert 0.4 < report["ndvi_median"] < 0.7        # vegetation


def test_assess_radiometry_rejects_impossible_ndvi():
    """Negative channels force NDVI outside [-1, 1], which cannot happen."""
    red = np.full((4, 4), -0.05, dtype="float32")
    nir = np.full((4, 4), 0.02, dtype="float32")
    stack = np.stack([red, red, red, nir])
    report = fetch_mod.assess_radiometry(stack, np.ones((4, 4), bool),
                                         ["blue", "green", "red", "nir"])
    assert report["ok"] is False
    assert "NDVI" in report["reason"]


def test_assess_radiometry_flags_an_over_applied_offset():
    """Clamping most pixels to zero is the signature of subtracting twice."""
    red = np.zeros((10, 10), dtype="float32")
    nir = np.full((10, 10), 0.02, dtype="float32")
    stack = np.stack([red, red, red, nir])
    report = fetch_mod.assess_radiometry(stack, np.ones((10, 10), bool),
                                         ["blue", "green", "red", "nir"])
    assert report["ok"] is False
    assert report["dead_fraction"] >= fetch_mod.MAX_DEAD_FRACTION


def test_reflectance_clamps_and_keeps_nodata_zero():
    dn = np.array([[-5, 0, 5000, 20000]], dtype="float32")
    out = fetch_mod.to_reflectance(dn, dn != 0, scale=0.0001, offset=0.0)
    assert out[0, 0] == 0.0            # negative clamped
    assert out[0, 1] == 0.0            # nodata preserved
    assert out[0, 2] == pytest.approx(0.5)
    assert out[0, 3] == 1.0            # clamped high


# --------------------------------------------------------------------------- #
# Fetch: pairing
# --------------------------------------------------------------------------- #

def test_pick_pair_requires_a_real_time_gap():
    same_week = [_scene(datetime="2026-04-10T00:00:00Z", scene_id="a"),
                 _scene(datetime="2026-04-12T00:00:00Z", scene_id="b")]
    older, newer = fetch_mod.pick_pair(same_week, min_gap_days=7)
    assert older is None and newer is None, \
        "two dates a fortnight apart cannot evidence change; refuse"


def test_pick_pair_prefers_the_largest_gap():
    scenes = [_scene(datetime="2026-01-05T00:00:00Z", scene_id="jan"),
              _scene(datetime="2026-03-05T00:00:00Z", scene_id="mar"),
              _scene(datetime="2026-08-10T00:00:00Z", scene_id="aug")]
    older, newer = fetch_mod.pick_pair(scenes, min_gap_days=7)
    assert older.scene_id == "jan"
    assert newer.scene_id == "aug"


def test_pick_pair_ignores_scenes_without_bands():
    broken = _scene(datetime="2026-01-05T00:00:00Z", scene_id="broken",
                    assets={})
    good = _scene(datetime="2026-08-10T00:00:00Z", scene_id="good")
    older, newer = fetch_mod.pick_pair([broken, good])
    assert older is None and newer is None


def test_days_between_is_symmetric_and_safe():
    assert fetch_mod._days_between("2026-01-01", "2026-01-31") == 30
    assert fetch_mod._days_between("2026-01-31", "2026-01-01") == 30
    assert fetch_mod._days_between("garbage", "2026-01-01") == 0


# --------------------------------------------------------------------------- #
# Bhuvan / ISRO
# --------------------------------------------------------------------------- #

def _png(opaque_fraction: float, size: int = 40, sample_step: int = 4) -> bytes:
    """PNG whose *measured* opacity equals ``opaque_fraction``.

    Coverage is measured by subsampling, so opacity is painted at the same
    sampled positions. Painting whole rows instead would produce a very
    different measured fraction and make the test assert the wrong thing.
    """
    from PIL import Image

    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    pixels = image.load()
    positions = [(x, y) for x in range(0, size, sample_step)
                 for y in range(0, size, sample_step)]
    count = int(round(opaque_fraction * len(positions)))
    for x, y in positions[:count]:
        pixels[x, y] = (255, 0, 0, 255)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_bhuvan_index_is_bundled_and_covers_key_states():
    provider = BhuvanProvider(FakeTransport())
    for state in ("Assam", "Kerala", "Punjab", "Karnataka"):
        assert provider.layers_for_state(state), f"no ISRO layers for {state}"


def test_bhuvan_state_alias_resolution():
    assert "as" in state_codes_for("Assam")
    assert "pj" in state_codes_for("Punjab") or "pb" in state_codes_for("Punjab")
    assert state_codes_for("Atlantis") == ()


def test_bhuvan_blank_tile_is_not_reported_as_present():
    """HTTP 200 with a transparent PNG means 'no data here', not 'forest'."""
    # The live service draws ~3% of ink even when it has nothing to draw, so a
    # blank tile is "same as the control", not "fully transparent".
    provider = BhuvanProvider(FakeTransport())
    provider.render = lambda layer, bbox, **kw: _png(0.03)  # type: ignore[assignment]
    layer = provider.assess("nuis:AS_DI_forest", (94.0, 27.0, 94.1, 27.1),
                            theme="forest", state="AS")
    assert layer.coverage == pytest.approx(0.03, abs=0.005)
    assert layer.evidence == pytest.approx(0.0, abs=0.005)
    assert BhuvanProvider.is_present(layer) is False, \
        "framing ink must never be reported as an ISRO finding"


def test_bhuvan_populated_tile_is_reported_as_present():
    provider = BhuvanProvider(FakeTransport())
    provider.render = lambda layer, bbox, **kw: _png(     # type: ignore[assignment]
        0.015 if bbox == CONTROL_BBOX else 0.15)
    layer = provider.assess("nuis:AS_DI_builtup_urban", (94.0, 27.0, 94.1, 27.1),
                            theme="builtup_urban", state="AS")
    assert BhuvanProvider.is_present(layer) is True
    assert layer.evidence > 0.10


def test_bhuvan_wms_exception_body_raises_instead_of_returning_a_picture():
    body = (b'<?xml version="1.0"?><ServiceExceptionReport>'
            b'<ServiceException code="LayerNotDefined">'
            b'Could not find layer</ServiceException></ServiceExceptionReport>')
    provider = BhuvanProvider(FakeTransport(bytes_map={
        "https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms": body}))
    with pytest.raises(Exception) as excinfo:
        provider.render("nuis:BOGUS_layer", (94.0, 27.0, 94.1, 27.1))
    assert "LayerNotDefined" in str(excinfo.value)


def test_bhuvan_describe_is_plain_english():
    provider = BhuvanProvider(FakeTransport())
    provider.render = lambda layer, bbox, **kw: _png(      # type: ignore[assignment]
        0.015 if bbox == CONTROL_BBOX else 0.15)
    layer = provider.assess("nuis:AS_DI_builtup_urban", (94.0, 27.0, 94.1, 27.1),
                            theme="builtup_urban")
    line = BhuvanProvider.describe([layer])[0]
    assert "ISRO" in line and "%" in line
    assert "built-up" in line
