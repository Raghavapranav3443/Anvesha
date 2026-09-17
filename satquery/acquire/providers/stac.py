"""STAC imagery providers: credential-free Sentinel-2 L2A (and Sentinel-1 GRD).

Why STAC on open endpoints rather than Bhoonidhi
-----------------------------------------------
The ISRO Bhoonidhi catalogue is the natural first choice on paper, but access
requires an emailed credential request with a lead time measured in days to
weeks. Blocking the whole online layer on that is not acceptable, and it is not
necessary either: Bhoonidhi redistributes Copernicus Sentinel data, and the
same products are published as public Cloud-Optimized GeoTIFFs by the
Copernicus/AWS open-data programmes. Same sensor family, same processing
baseline, no account, available now.

Bhoonidhi therefore becomes one more entry in the provider list -- the
interface is identical -- so the ISRO-native provenance story can be switched
on later without touching a single caller.

The cloud-filter trap, and why filtering is client-side
-------------------------------------------------------
Measured directly against ``earth-search.aws.element84.com/v1``: issuing a
search **with** a ``query: {"eo:cloud_cover": {"lt": 25}}`` clause returned
**zero** features for an AOI that returned four without it. HTTP 200, a
well-formed empty result, no error. A server-side filter that silently fails
to a false negative is the single most dangerous failure mode for this layer,
because the user is told "no imagery exists for your area" when in fact plenty
does.

So every search is issued **unfiltered** server-side and filtered here, in code
we control and can test. Cloud cover is still requested as a sort hint, never
as a correctness filter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..errors import NetworkBlockedAirgap, NoCatalogAvailable, NoSceneFound
from ..http import Transport

BBox = Tuple[float, float, float, float]

# Sentinel-2 L2A band assets we care about, in the order the offline pipeline
# expects for a >=4-band GeoTIFF (B02, B03, B04, B08).
S2_ASSET_ORDER = ("blue", "green", "red", "nir")

ASSET_ALIASES = {
    "blue": ("blue", "B02", "B2"),
    "green": ("green", "B03", "B3"),
    "red": ("red", "B04", "B4"),
    "nir": ("nir", "B08", "B8", "nir08"),
    "scl": ("scl", "SCL"),
    "visual": ("visual", "TCI"),
}


@dataclass
class SceneRef:
    """One imagery granule, resolved to directly-readable asset URLs."""

    provider: str
    scene_id: str
    datetime: str
    collection: str
    cloud_pct: Optional[float]
    epsg: Optional[int]
    assets: Dict[str, str] = field(default_factory=dict)
    platform: str = ""
    bbox: Optional[List[float]] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    # Radiometric conversion declared by the catalogue (STAC ``raster:bands``).
    # Sentinel-2 L2A publishes digital numbers; reflectance = dn*scale + offset.
    # Carrying it here means the fetch layer never has to guess a magic factor.
    scales: Dict[str, float] = field(default_factory=dict)
    offsets: Dict[str, float] = field(default_factory=dict)

    @property
    def date(self) -> str:
        return (self.datetime or "")[:10]

    @property
    def has_bands(self) -> bool:
        return all(a in self.assets for a in S2_ASSET_ORDER)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "id": self.scene_id,
            "date": self.date,
            "datetime": self.datetime,
            "collection": self.collection,
            "cloud_pct": self.cloud_pct,
            "epsg": self.epsg,
            "platform": self.platform,
            "bands": sorted(self.assets),
        }


@dataclass
class StacProvider:
    """A STAC API endpoint plus the asset mapping for one collection."""

    name: str
    endpoint: str
    collection: str = "sentinel-2-l2a"
    label: str = ""
    asset_order: Sequence[str] = S2_ASSET_ORDER
    requires_signing: bool = False
    notes: str = ""

    # -- internals ------------------------------------------------------ #

    def _search_url(self) -> str:
        return self.endpoint.rstrip("/") + "/search"

    def _pick_assets(self, raw_assets: Dict[str, Any]
                     ) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, float]]:
        """Map logical band -> COG href, plus any declared scale/offset.

        Aliases deliberately never resolve to the ``-jp2`` variants these
        catalogues also publish: the JPEG2000 copies are larger and markedly
        slower to window-read, and the COG is the asset we want.
        """
        hrefs: Dict[str, str] = {}
        scales: Dict[str, float] = {}
        offsets: Dict[str, float] = {}
        lowered = {str(k).lower(): v for k, v in (raw_assets or {}).items()}
        for logical, aliases in ASSET_ALIASES.items():
            for alias in aliases:
                entry = lowered.get(alias.lower())
                if not entry:
                    continue
                href = entry.get("href") if isinstance(entry, dict) else entry
                if not href:
                    continue
                hrefs[logical] = str(href)
                bands = (entry.get("raster:bands") or []) if isinstance(entry, dict) else []
                if bands and isinstance(bands[0], dict):
                    try:
                        scales[logical] = float(bands[0]["scale"])
                    except (KeyError, TypeError, ValueError):
                        pass
                    try:
                        offsets[logical] = float(bands[0]["offset"])
                    except (KeyError, TypeError, ValueError):
                        pass
                break
        return hrefs, scales, offsets

    def _to_ref(self, feature: Dict[str, Any]) -> SceneRef:
        props = feature.get("properties") or {}
        assets, scales, offsets = self._pick_assets(feature.get("assets") or {})
        epsg = props.get("proj:epsg")
        try:
            epsg = int(epsg) if epsg is not None else None
        except (TypeError, ValueError):
            epsg = None
        cloud = props.get("eo:cloud_cover")
        try:
            cloud = float(cloud) if cloud is not None else None
        except (TypeError, ValueError):
            cloud = None
        return SceneRef(
            provider=self.name,
            scene_id=str(feature.get("id", "")),
            datetime=str(props.get("datetime") or ""),
            collection=self.collection,
            cloud_pct=cloud,
            epsg=epsg,
            assets=assets,
            platform=str(props.get("platform") or props.get("constellation") or ""),
            bbox=list(feature.get("bbox") or []) or None,
            # Only the properties that change a downstream decision are kept.
            # `earthsearch:boa_offset_applied` is here because it decides the
            # DN -> reflectance convention: if the catalogue has already applied
            # the BOA offset, subtracting the declared offset again drives most
            # pixels negative. See fetch.reflectance_conversion.
            properties={k: v for k, v in props.items()
                        if k in ("proj:shape", "proj:transform", "grid:code",
                                 "eo:cloud_cover", "s2:degraded_msi_data_percentage",
                                 "earthsearch:boa_offset_applied",
                                 "s2:processing_baseline",
                                 "s2:nodata_pixel_percentage",
                                 "s2:water_percentage",
                                 "s2:vegetation_percentage")},
            scales=scales,
            offsets=offsets,
        )

    # -- public --------------------------------------------------------- #

    def search(self, bbox: BBox, *, start: str, end: str,
               transport: Transport, limit: int = 40,
               max_cloud_pct: Optional[float] = None,
               stats: Optional[Dict[str, Any]] = None) -> List[SceneRef]:
        """Search this catalogue. Server-side filtering is deliberately absent.

        ``stats``, when passed, is filled with what the catalogue actually
        returned. A caller that ends up with nothing needs it to explain itself:
        "21 scenes matched and the cloud limit rejected all of them" and "no
        scenes matched" lead the user to different levers, and only one of them
        (widening the dates) is useless for the first.
        """
        if self.requires_signing:
            raise NoCatalogAvailable(
                f"{self.name} requires credentials; falling back is the caller's job")
        payload: Dict[str, Any] = {
            "collections": [self.collection],
            "bbox": list(bbox),
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "limit": int(limit),
        }
        # No server-side `sortby` or `query`: support for both varies between
        # STAC implementations, and a rejected or misinterpreted clause is
        # indistinguishable from "no imagery here". Ordering and filtering are
        # done in Python, where a mistake is visible and testable.
        data = transport.post_json(self._search_url(), payload, kind="stac")
        features = (data or {}).get("features") or []
        refs = [self._to_ref(f) for f in features if isinstance(f, dict)]
        refs = [r for r in refs if r.scene_id]
        kept = self.filter_cloud(refs, max_cloud_pct)
        if stats is not None:
            known = [r.cloud_pct for r in refs if r.cloud_pct is not None]
            stats.clear()
            stats.update({
                "provider": self.name,
                "returned": len(features),
                "with_assets": len(refs),
                "kept": len(kept),
                "lowest_cloud_pct": min(known) if known else None,
                "unknown_cloud": sum(1 for r in refs if r.cloud_pct is None),
                "max_cloud_pct": (None if max_cloud_pct is None
                                  else float(max_cloud_pct)),
            })
        return kept

    @staticmethod
    def filter_cloud(refs: Iterable[SceneRef],
                     max_cloud_pct: Optional[float]) -> List[SceneRef]:
        """Client-side cloud filter. Unknown cloud cover is kept, not dropped.

        Dropping an unknown is how a filter turns into a silent "no data".
        Unknown cloud is surfaced to the user instead, so they can decide.
        """
        if max_cloud_pct is None:
            return list(refs)
        limit = float(max_cloud_pct)
        return [r for r in refs if r.cloud_pct is None or r.cloud_pct <= limit]


# --------------------------------------------------------------------------- #
# Catalogue endpoints
# --------------------------------------------------------------------------- #

def all_providers() -> Dict[str, StacProvider]:
    """Every catalogue the app knows about, configured or not.

    Kept separate from :func:`default_providers` so the UI can show the full
    picture -- including that an ISRO-native Bhoonidhi catalogue exists and is
    one credential away -- rather than silently omitting what is unconfigured.
    """
    from ..mode import load_settings

    settings = load_settings().get("acquire", {})
    return {
        "stac_earth_search": StacProvider(
            name="stac_earth_search",
            endpoint="https://earth-search.aws.element84.com/v1",
            collection=settings.get("default_collection", "sentinel-2-l2a"),
            label="AWS Earth Search (Sentinel-2 L2A)",
            notes="Public COGs on S3. No account needed. Primary.",
        ),
        "stac_cdse": StacProvider(
            name="stac_cdse",
            endpoint="https://catalogue.dataspace.copernicus.eu/stac",
            collection="sentinel-2-l2a",
            label="Copernicus Data Space (Sentinel-2 L2A)",
            notes="Copernicus-operated mirror; useful when Earth Search is down.",
        ),
        "stac_bhoonidhi": StacProvider(
            name="stac_bhoonidhi",
            endpoint="https://bhoonidhi.nrsc.gov.in/bhoonidhi/stac",
            collection="sentinel-2-l2a",
            label="ISRO Bhoonidhi",
            requires_signing=True,
            notes=("ISRO-native imagery archive. Requires a credential request; "
                   "not configured. Sentinel-2 is the same sensor family, so "
                   "results are comparable when it is enabled."),
        ),
    }


def default_providers(primary: Optional[str] = None,
                      fallbacks: Optional[Sequence[str]] = None
                      ) -> List[StacProvider]:
    """Provider list in priority order, from settings. Signed ones excluded."""
    from ..mode import load_settings

    settings = load_settings().get("acquire", {})
    primary = primary or settings.get("primary_catalog", "stac_earth_search")
    fallbacks = tuple(fallbacks if fallbacks is not None
                      else (settings.get("fallback_catalogs") or ()))
    known = all_providers()

    ordered: List[StacProvider] = []
    for key in (primary, *fallbacks):
        provider = known.get(key)
        if provider is None or provider.requires_signing:
            continue
        if provider not in ordered:
            ordered.append(provider)
    if not ordered:
        ordered.append(known["stac_earth_search"])
    return ordered


def empty_search_reason(stats: Dict[str, Any], *, start: str, end: str) -> str:
    """Why a search produced nothing, in the terms the user can act on.

    "No scenes matched in this date range" is true only when the catalogue
    returned nothing. When scenes *did* match and the cloud limit rejected every
    one of them, saying "no scenes matched" is false and actively unhelpful: it
    points at the date range, so the user widens the dates, gets the same
    answer, and never learns that the lever was the cloud threshold.
    """
    returned = int(stats.get("returned") or 0)
    usable = int(stats.get("with_assets") or 0)
    if returned == 0:
        return "no scenes matched in this date range"
    if usable == 0:
        return (f"{returned} scene(s) matched {start}..{end}, but none carried "
                f"the bands this analysis needs")
    limit = stats.get("max_cloud_pct")
    if limit is None:
        return (f"{returned} scene(s) matched {start}..{end}, but none survived "
                f"filtering")
    lowest = stats.get("lowest_cloud_pct")
    unknown = int(stats.get("unknown_cloud") or 0)
    if lowest is None:                                   # pragma: no cover
        detail = (f"cloud cover is unknown for all {usable} of them, which is "
                  f"surfaced rather than dropped")
    else:
        detail = f"the clearest was {lowest:.0f}%"
        if unknown:
            detail += f" and {unknown} had unknown cloud cover"
    return (f"{returned} scene(s) matched {start}..{end}, but all were rejected "
            f"by the {limit:.0f}% cloud limit ({detail}). Raise the cloud limit, "
            f"or expect a clearer season than this one -- widening the dates "
            f"will not help.")


def search_with_fallback(providers: Sequence[StacProvider], bbox: BBox, *,
                         start: str, end: str, transport: Transport,
                         limit: int = 40,
                         max_cloud_pct: Optional[float] = None
                         ) -> Tuple[List[SceneRef], List[Dict[str, str]]]:
    """Try each provider in order. Returns (scenes, errors).

    An empty result is reported as an empty result; only a raised failure is
    recorded as an error. Conflating the two is how "no imagery" gets confused
    with "could not reach the catalogue", and the user needs to tell those
    apart.
    """
    errors: List[Dict[str, str]] = []
    for provider in providers:
        stats: Dict[str, Any] = {}
        try:
            found = provider.search(bbox, start=start, end=end,
                                    transport=transport, limit=limit,
                                    max_cloud_pct=max_cloud_pct, stats=stats)
        except NetworkBlockedAirgap:
            # Air-gap is a policy decision, not a catalogue problem. Treating it
            # as a provider failure made "blocked by policy" indistinguishable
            # from "no imagery exists here", and the API then answered 200 with
            # an empty plan instead of refusing.
            raise
        except Exception as exc:
            errors.append({"provider": provider.name,
                           "error": f"{type(exc).__name__}: {exc}"})
            continue
        if found:
            return found, errors
        errors.append({"provider": provider.name,
                       "error": empty_search_reason(stats, start=start,
                                                   end=end)})
    return [], errors
