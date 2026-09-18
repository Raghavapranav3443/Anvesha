"""Resolve a plain-English place into an analysis window.

Design decision worth stating explicitly
---------------------------------------
An AOI is treated as a *place*, and we analyse a bounded window around it
rather than everything inside its administrative boundary. Asking for "Assam"
and then reading a state-sized COG window would mean gigabytes of transfer and
minutes of waiting, on a machine chosen for being modest. So the AOI resolves
to a point, and the fetch takes a fixed ground extent around it (default 12 km,
configurable). That keeps every request small, comparable between runs, and
explainable to a non-expert ("roughly a 12 km square around the place you
named").

Resolution order -- offline first, and it never guesses
------------------------------------------------------
1. **Explicit coordinates** ("26.1, 91.7" or a 4-number bbox). Always wins.
2. **Bundled gazetteer** of Indian states and union territories
   (``data/india_states.json``), generated from OpenStreetMap once by
   ``scripts/build_place_index.py``. Works with no network at all.
3. **Previously resolved places**, cached to disk. This is what makes the
   *second* look at a district work offline.
4. **Nominatim**, online mode only -- free-text, including districts and
   villages.

If a query matches several places, all candidates are returned for the user to
choose. Silently picking one of three "Aurangabad"s is exactly the kind of
quiet wrongness this project has been systematically removing.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import CONFIG
from .cache import CACHE, key_for
from .errors import AcquireError
from .http import Transport

BBox = Tuple[float, float, float, float]        # (west, south, east, north)

DATA_DIR = Path(__file__).resolve().parent / "data"
GAZETTEER_PATH = DATA_DIR / "india_states.json"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Hard guard: refuse an AOI wider than this even if the caller insists, because
# a window that big is not a windowed read any more.
MAX_AOI_KM = 300.0
# Default ground extent of the analysis window, in kilometres.
DEFAULT_WINDOW_KM = 12.0


class PlaceNotFound(AcquireError):
    """No place could be resolved, or several matched and a choice is needed."""


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def bbox_size_km(bbox: BBox) -> Tuple[float, float]:
    """(width_km, height_km) of a lon/lat bbox, including latitude convergence."""
    west, south, east, north = bbox
    width = _haversine_km((south + north) / 2, west, (south + north) / 2, east)
    height = _haversine_km(south, west, north, west)
    return width, height


def bbox_area_km2(bbox: BBox) -> float:
    w, h = bbox_size_km(bbox)
    return w * h


def bbox_centroid(bbox: BBox) -> Tuple[float, float]:
    west, south, east, north = bbox
    return ((south + north) / 2.0, (west + east) / 2.0)   # (lat, lon)


def clip_to_extent(bbox: BBox, max_km: float = DEFAULT_WINDOW_KM) -> BBox:
    """Shrink a bbox to at most ``max_km`` on each side, keeping its centre."""
    west, south, east, north = bbox
    width, height = bbox_size_km(bbox)
    lat, lon = bbox_centroid(bbox)
    if width <= max_km and height <= max_km:
        return bbox
    half_lat = min(max_km, height) / 2.0 / 110.574
    km_per_deg_lon = 111.320 * max(0.15, math.cos(math.radians(lat)))
    half_lon = min(max_km, width) / 2.0 / km_per_deg_lon
    return (lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)


def bbox_around_point(lat: float, lon: float, km: float = DEFAULT_WINDOW_KM) -> BBox:
    half_lat = km / 2.0 / 110.574
    km_per_deg_lon = 111.320 * max(0.15, math.cos(math.radians(lat)))
    half_lon = km / 2.0 / km_per_deg_lon
    return (lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)


# --------------------------------------------------------------------------- #
# Explicit coordinates
# --------------------------------------------------------------------------- #

_COORD_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


class _AmbiguousCoordinates(AcquireError):
    """A coordinate-looking string that cannot be interpreted safely."""


def parse_coordinates(text: str) -> Optional[BBox]:
    """Parse explicit coords. Returns None if the text is not coordinate-like.

    Accepts:
    * ``26.15, 91.75``            -> a point, expanded to a default window
    * ``91.6,26.05,91.85,26.20``  -> an explicit bbox (west,south,east,north)
    * ``26.15N 91.75E``           -> a point with hemisphere suffixes

    Raises :class:`_AmbiguousCoordinates` when the numbers are coordinate-like
    but nonsensical (e.g. a latitude past the poles), rather than clamping and
    quietly analysing the wrong place.
    """
    if not text or not re.search(r"\d", text):
        return None
    cleaned = text.replace("°", " ").replace("—", "-")
    numbers = [float(m.group(0)) for m in _COORD_RE.finditer(cleaned)]
    if len(numbers) not in (2, 4):
        return None
    # A number followed by N/S must be a latitude, so reorder if needed.
    if len(numbers) == 2:
        first, second = numbers
        low = cleaned.upper()
        if re.search(rf"{re.escape(str(first)):s}\s*[NS]", low) or \
           re.search(rf"{re.escape(str(second)):s}\s*[EW]", low):
            lat, lon = first, second
        else:
            lat, lon = (first, second) if first < second else (second, first) \
                if abs(first) <= 90 and abs(second) <= 180 else (first, second)
        if not (-90 <= lat <= 90):
            raise _AmbiguousCoordinates(
                f"{text!r}: interpreted latitude {lat} is outside -90..90")
        if not (-180 <= lon <= 180):
            raise _AmbiguousCoordinates(
                f"{text!r}: interpreted longitude {lon} is outside -180..180")
        return bbox_around_point(lat, lon)
    west, south, east, north = numbers
    if not (west < east and south < north):
        raise _AmbiguousCoordinates(
            f"{text!r}: expected west < east and south < north "
            f"(got W={west}, S={south}, E={east}, N={north})")
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise _AmbiguousCoordinates(f"{text!r}: longitudes out of range")
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise _AmbiguousCoordinates(f"{text!r}: latitudes out of range")
    return (west, south, east, north)


# --------------------------------------------------------------------------- #
# Gazetteer
# --------------------------------------------------------------------------- #

@dataclass
class Aoi:
    """A resolved area of interest."""

    name: str
    bbox: BBox
    source: str = "unknown"
    confidence: float = 0.5
    admin: Dict[str, str] = field(default_factory=dict)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def centroid(self) -> Tuple[float, float]:
        return bbox_centroid(self.bbox)

    @property
    def area_km2(self) -> float:
        return bbox_area_km2(self.bbox)

    def window(self, max_km: float = DEFAULT_WINDOW_KM) -> BBox:
        return clip_to_extent(self.bbox, max_km)

    def to_dict(self) -> Dict[str, Any]:
        lat, lon = self.centroid
        w, h = bbox_size_km(self.bbox)
        return {
            "name": self.name,
            "bbox": list(self.bbox),
            "centroid": {"lat": round(lat, 6), "lon": round(lon, 6)},
            "source": self.source,
            "confidence": self.confidence,
            "admin": dict(self.admin),
            "area_km2": round(self.area_km2, 1),
            "size_km": [round(w, 2), round(h, 2)],
            "alternatives": list(self.alternatives),
        }


_gazetteer_cache: Optional[Dict[str, Any]] = None


def load_gazetteer() -> Dict[str, Any]:
    """Read the bundled state index. A missing file yields an empty index."""
    global _gazetteer_cache
    if _gazetteer_cache is not None:
        return _gazetteer_cache
    try:
        _gazetteer_cache = json.loads(GAZETTEER_PATH.read_text(encoding="utf-8"))
    except Exception:
        _gazetteer_cache = {"places": [], "generated": None}
    return _gazetteer_cache


def _normalise(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    cleaned = re.sub(r"\b(india|state|district|ut|union territory)\b", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def gazetteer_lookup(query: str) -> List[Aoi]:
    """Offline matches from the bundled index. Exact beats prefix beats substring."""
    needle = _normalise(query)
    if not needle:
        return []
    exact, prefix, partial = [], [], []
    for place in load_gazetteer().get("places", []):
        names = [place.get("name", "")] + list(place.get("aliases") or [])
        norm_names = [_normalise(n) for n in names if n]
        if needle in norm_names:
            exact.append(place)
        elif any(n.startswith(needle) or needle.startswith(n) for n in norm_names if n):
            prefix.append(place)
        elif any(needle in n for n in norm_names if n):
            partial.append(place)
    ordered = exact + prefix + partial
    out: List[Aoi] = []
    for place in ordered:
        bbox = place.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        out.append(Aoi(
            name=place.get("name", query),
            bbox=tuple(float(x) for x in bbox),   # type: ignore[arg-type]
            source="gazetteer",
            confidence=0.9 if place in exact else 0.7,
            admin={"state": place.get("name", ""), "country": "India"},
        ))
    return out


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #

def _cached_lookup(query: str) -> Optional[Aoi]:
    key = key_for("aoi", _normalise(query))
    raw = CACHE.get(key)
    if raw is None:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    bbox = data.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    return Aoi(name=data.get("name", query), bbox=tuple(bbox),  # type: ignore[arg-type]
               source="cache", confidence=0.8,
               admin=data.get("admin") or {})


def _cache_lookup(query: str, aoi: Aoi) -> None:
    key = key_for("aoi", _normalise(query))
    payload = {"name": aoi.name, "bbox": list(aoi.bbox), "admin": aoi.admin}
    CACHE.put(key, json.dumps(payload).encode("utf-8"), kind="aoi",
              url="gazetteer", extra={"query": query})


def nominatim_lookup(query: str, transport: Transport,
                     limit: int = 5) -> List[Aoi]:
    """Free-text geocode via OpenStreetMap. Online mode only (raises otherwise)."""
    text = str(query).strip()
    if not text:
        return []
    if "india" not in text.lower():
        text = f"{text}, India"
    payload = transport.get_json(
        NOMINATIM_URL,
        params={"q": text, "format": "jsonv2", "limit": int(limit),
                "countrycodes": "in", "addressdetails": 1},
        kind="geocode")
    if not isinstance(payload, list):
        return []
    out: List[Aoi] = []
    for item in payload:
        raw = item.get("boundingbox") or []
        if len(raw) != 4:
            continue
        south, north, west, east = (float(x) for x in raw)
        addr = item.get("address") or {}
        name = (addr.get("city") or addr.get("county")
                or addr.get("state_district") or item.get("display_name")
                or query)
        out.append(Aoi(
            name=str(name),
            bbox=(west, south, east, north),
            source="nominatim",
            confidence=0.75,
            admin={"state": str(addr.get("state", "")),
                   "district": str(addr.get("state_district", "")),
                   "country": str(addr.get("country", "India"))},
        ))
    return out


def resolve(query: str, transport: Optional[Transport] = None,
            *, allow_network: bool = True,
            window_km: float = DEFAULT_WINDOW_KM,
            must_fit: bool = True) -> Aoi:
    """Resolve a query to a single AOI, or raise :class:`PlaceNotFound`.

    ``PlaceNotFound`` carries the candidate list so the caller can offer a
    choice. Multi-match is a refusal, not a silent pick.

    ``must_fit`` guards the case where the caller needs the *whole* named area
    (and would otherwise silently analyse a small part of it). Callers that are
    going to clip to a fixed window anyway -- :func:`analysis_window` -- pass
    ``False``, because there the window is the unit of work and a state name is
    a legitimate way to say "somewhere in this state".
    """
    text = str(query or "").strip()
    if not text:
        raise PlaceNotFound("no place given")

    explicit = None
    try:
        explicit = parse_coordinates(text)
    except _AmbiguousCoordinates as exc:
        raise PlaceNotFound(str(exc)) from exc
    if explicit is not None:
        return Aoi(name=text, bbox=explicit, source="explicit", confidence=1.0,
                   admin={})

    candidates = gazetteer_lookup(text)
    if not candidates:
        cached = _cached_lookup(text)
        if cached is not None:
            candidates = [cached]
    if not candidates and allow_network and transport is not None:
        candidates = nominatim_lookup(text, transport)
        if candidates:
            _cache_lookup(text, candidates[0])

    if not candidates:
        raise PlaceNotFound(
            f"could not find a place matching {text!r}. Try "
            f"'<district>, <state>' with coordinates, or give a lat,lon pair.")

    # Prefer the most specific match, but surface the others rather than hiding them.
    primary = candidates[0]
    if len(candidates) > 1:
        primary.alternatives = [
            {"name": c.name, "bbox": list(c.bbox), "source": c.source}
            for c in candidates[1:6]
        ]

    width, height = bbox_size_km(primary.bbox)
    if must_fit and max(width, height) > MAX_AOI_KM:
        raise AcquireError(
            f"{primary.name!r} spans {max(width, height):.0f} km, which is too "
            f"large to analyse as one window (limit {MAX_AOI_KM:.0f} km). "
            f"Name a district or a town instead of a whole state.")
    return primary


def analysis_window(query: str, transport: Optional[Transport] = None,
                    *, window_km: float = DEFAULT_WINDOW_KM,
                    allow_network: bool = True) -> Aoi:
    """Resolve a place and return the window that will actually be analysed.

    Large areas are clipped rather than refused, because for a non-expert "show
    me Assam" is a reasonable thing to type. But clipping a state to a 12 km
    square around its centroid analyses an arbitrary field, so the AOI carries
    an explicit note saying so -- the honest fix is to tell the user to name a
    town, not to pretend the answer covers a state.
    """
    aoi = resolve(query, transport, allow_network=allow_network,
                  window_km=window_km, must_fit=False)
    original_km = max(bbox_size_km(aoi.bbox))
    aoi.bbox = aoi.window(window_km)
    if original_km > MAX_AOI_KM:
        note = (f"{aoi.name} is about {original_km:.0f} km across, so this "
                f"analysis covers a {window_km:.0f} km square at its centre. "
                f"Name a town or district for a precise result.")
        aoi.admin["window_note"] = note
        aoi.confidence = min(aoi.confidence, 0.5)
    elif original_km > window_km:
        aoi.admin["window_note"] = (
            f"{aoi.name} is about {original_km:.0f} km across; this analysis "
            f"covers a {window_km:.0f} km square at its centre.")
    return aoi
