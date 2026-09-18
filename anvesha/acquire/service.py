"""Online orchestration: place name in, analysis-ready image pair out.

This is the layer that answers the question the offline pipeline could not:
*"where do I get the data?"* It is deliberately a two-step API -- plan, then
execute -- because fetching is slow and costs bandwidth, and a non-expert
deserves to see what will be downloaded, from which satellite passes and how
cloudy they are, before it happens.

What the result is not
----------------------
Acquisition is kept **outside** ``AgentController.run``. The audited pipeline's
first step is ``validate_inputs`` and its trace is pinned by tests; hiding a
network fetch inside it would change the trace's meaning and make the same
query non-reproducible. So acquisition produces files and a provenance record,
and the ordinary offline pipeline analyses them exactly as it would an upload.

That split is also what makes the offline promise concrete: **fetching needs the
network; re-analysing what you already fetched does not.** The GeoTIFFs and the
provenance sidecar stay on disk under the run directory.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..config import CONFIG
from .aoi import DEFAULT_WINDOW_KM, Aoi, analysis_window
from .errors import AcquireError, NetworkBlockedAirgap, NoSceneFound
from .fetch import (TargetGrid, FetchedScene, fetch_scene, pick_pair, plan_grid,
                    write_provenance)
from .http import Transport, default_transport
from .mode import current_mode, load_settings
from .providers.bhuvan import BhuvanProvider, ContextLayer
from .providers.stac import SceneRef, default_providers, search_with_fallback

DEFAULT_DAYS = 240


@dataclass
class AcquirePlan:
    """What would be downloaded, before anything is."""

    aoi: Aoi
    grid: TargetGrid
    scenes: List[SceneRef] = field(default_factory=list)
    providers: List[str] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
    start: str = ""
    end: str = ""
    max_cloud_pct: float = 20.0
    window_km: float = DEFAULT_WINDOW_KM
    pair: Tuple[Optional[SceneRef], Optional[SceneRef]] = (None, None)

    @property
    def has_pair(self) -> bool:
        return self.pair[0] is not None and self.pair[1] is not None

    def to_dict(self) -> Dict[str, Any]:
        older, newer = self.pair
        return {
            "place": self.aoi.to_dict(),
            "grid": self.grid.to_dict(),
            "date_range": {"start": self.start, "end": self.end},
            "max_cloud_pct": self.max_cloud_pct,
            "window_km": self.window_km,
            "providers": list(self.providers),
            "scene_count": len(self.scenes),
            "candidates": [s.to_dict() for s in self.scenes[:40]],
            "pair": {
                "before": older.to_dict() if older else None,
                "after": newer.to_dict() if newer else None,
            },
            "errors": list(self.errors),
        }


@dataclass
class AcquiredPair:
    """Two dates fetched onto a shared grid, plus verified ISRO context."""

    acquire_id: str
    plan: AcquirePlan
    before: FetchedScene
    after: FetchedScene
    context: List[ContextLayer] = field(default_factory=list)
    context_errors: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    provenance_path: Optional[Path] = None
    elapsed_s: float = 0.0

    @property
    def paths(self) -> Tuple[Path, Path]:
        return self.before.path, self.after.path

    def date_labels(self) -> Tuple[str, str]:
        return self.before.date, self.after.date

    def to_dict(self) -> Dict[str, Any]:
        before_date, after_date = self.date_labels()
        return {
            "acquire_id": self.acquire_id,
            "plan": self.plan.to_dict(),
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "dates": {"before": before_date, "after": after_date},
            "context": [layer.to_dict() for layer in self.context],
            "context_plain": BhuvanProvider.describe(self.context),
            "context_errors": list(self.context_errors),
            "warnings": list(self.warnings),
            "provenance": str(self.provenance_path) if self.provenance_path else None,
            "elapsed_s": round(self.elapsed_s, 1),
        }


def _settings_defaults() -> Dict[str, Any]:
    acq = load_settings().get("acquire") or {}
    return {
        "days": int(acq.get("lookback_days", DEFAULT_DAYS) or DEFAULT_DAYS),
        "max_cloud": float(acq.get("max_cloud_pct", 20) or 20),
        "window_px": int(acq.get("window_px", 1024) or 1024),
    }


def acquire_status() -> Dict[str, Any]:
    """Everything the UI needs to render an honest online-mode state."""
    from .cache import CACHE

    from .providers.stac import all_providers

    defaults = _settings_defaults()
    providers = default_providers()
    active = {p.name for p in providers}
    return {
        "mode": current_mode(),
        "defaults": defaults,
        "providers": [{"name": p.name, "label": p.label,
                       "requires_signing": p.requires_signing,
                       "configured": p.name in active,
                       "notes": p.notes} for p in all_providers().values()],
        "active_providers": sorted(active),        "cache": CACHE.stats(),
        "isro": {
            "service": "ISRO Bhuvan WMS",
            "credential_free": True,
            "endpoint": "https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms",
            "note": ("ISRO's own thematic layers are served without an account, "
                     "so context and corroboration work today. Bhoonidhi "
                     "imagery needs credentials and is optional."),
        },
    }


def plan(query: str, *, days: Optional[int] = None,
         max_cloud_pct: Optional[float] = None,
         window_km: float = DEFAULT_WINDOW_KM,
         transport: Optional[Transport] = None,
         allow_network: bool = True) -> AcquirePlan:
    """Resolve a place and search the catalogues. No imagery is downloaded."""
    transport = transport or default_transport()
    defaults = _settings_defaults()
    days = defaults["days"] if days is None else int(days)
    max_cloud_pct = defaults["max_cloud"] if max_cloud_pct is None else float(max_cloud_pct)

    aoi = analysis_window(query, transport, window_km=window_km,
                          allow_network=allow_network)
    grid = plan_grid(aoi.bbox, px=defaults["window_px"])

    end = time.strftime("%Y-%m-%d")
    start = time.strftime("%Y-%m-%d", time.gmtime(time.time() - max(1, days) * 86400))
    providers = default_providers()
    scenes, errors = search_with_fallback(
        providers, aoi.bbox, start=start, end=end, transport=transport,
        limit=60, max_cloud_pct=max_cloud_pct)

    plan_obj = AcquirePlan(
        aoi=aoi, grid=grid, scenes=scenes,
        providers=[p.name for p in providers], errors=errors,
        start=start, end=end, max_cloud_pct=max_cloud_pct, window_km=window_km)
    plan_obj.pair = pick_pair(scenes)
    return plan_obj


def execute(the_plan: AcquirePlan, *, transport: Optional[Transport] = None,
            out_root: Optional[Path] = None, acquire_id: Optional[str] = None,
            include_context: bool = True) -> AcquiredPair:
    """Fetch the planned pair and gather verified ISRO context.

    Refuses rather than substitutes: if the two dates cannot be chosen, or a
    scene fails its coverage or physics checks, the caller is told, because a
    quietly swapped scene produces a confident comparison of two different
    things.
    """
    transport = transport or default_transport()
    started = time.time()
    older, newer = the_plan.pair
    if older is None or newer is None:
        raise NoSceneFound(
            f"no usable image pair for {the_plan.aoi.name!r} between "
            f"{the_plan.start} and {the_plan.end} at or under "
            f"{the_plan.max_cloud_pct:.0f}% cloud. Try a longer date range or a "
            f"higher cloud allowance.")

    acquire_id = acquire_id or f"acq-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    root = Path(out_root) if out_root is not None else (CONFIG.runs_dir / acquire_id / "acquired")
    root.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    fetched: List[FetchedScene] = []
    # `before` first so the naming stays chronological in the directory listing.
    for label, scene in (("before", older), ("after", newer)):
        target = root / f"s2_{scene.date}.tif"
        got = fetch_scene(scene, the_plan.grid, target)
        warnings.extend(f"{scene.date}: {w}" for w in got.warnings)
        fetched.append(got)

    context: List[ContextLayer] = []
    context_errors: List[Dict[str, str]] = []
    if include_context:
        state = (the_plan.aoi.admin or {}).get("state") or ""
        if state:
            try:
                bhuvan = BhuvanProvider(transport)
                context, context_errors = bhuvan.context_for(
                    state, the_plan.aoi.window(the_plan.window_km))
            except NetworkBlockedAirgap:
                # Never degrade a policy refusal into "no context available".
                raise
            except Exception as exc:
                context_errors.append({"layer": "*",
                                       "error": f"{type(exc).__name__}: {exc}"})
        else:
            context_errors.append({
                "layer": "*",
                "error": (f"no Indian state resolved for {the_plan.aoi.name!r}, "
                          f"so ISRO context layers could not be selected")})

    result = AcquiredPair(
        acquire_id=acquire_id, plan=the_plan, before=fetched[0], after=fetched[1],
        context=context, context_errors=context_errors, warnings=warnings,
        elapsed_s=time.time() - started)

    provenance = {
        "acquire_id": acquire_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requested_place": the_plan.aoi.name,
        "resolved_by": the_plan.aoi.source,
        "aoi": the_plan.aoi.to_dict(),
        "grid": the_plan.grid.to_dict(),
        "date_range": {"start": the_plan.start, "end": the_plan.end},
        "max_cloud_pct": the_plan.max_cloud_pct,
        "scenes": {"before": fetched[0].to_dict(), "after": fetched[1].to_dict()},
        "isro_context": [layer.to_dict() for layer in context],
        "isro_context_errors": context_errors,
        "warnings": warnings,
        "elapsed_s": round(result.elapsed_s, 1),
        "note": ("Imagery from open Copernicus/AWS Cloud-Optimized GeoTIFF "
                 "catalogues. ISRO context from Bhuvan WMS."),
    }
    result.provenance_path = write_provenance(root / "provenance", provenance)
    return result


def acquire(query: str, *, transport: Optional[Transport] = None,
            out_root: Optional[Path] = None, include_context: bool = True,
            **plan_kwargs: Any) -> Tuple[AcquirePlan, AcquiredPair]:
    """Plan and execute in one call. Returns both so the UI can show the reasoning."""
    the_plan = plan(query, transport=transport, **plan_kwargs)
    return the_plan, execute(the_plan, transport=transport, out_root=out_root,
                             include_context=include_context)


def provenance_for(acquire_id: str) -> Optional[Dict[str, Any]]:
    """Read a stored provenance record, or None."""
    safe = Path(str(acquire_id)).name
    path = CONFIG.runs_dir / safe / "acquired" / "provenance.provenance.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


ALLOWED_ACQUIRED_SUFFIXES: FrozenSet[str] = frozenset({".tif", ".tiff"})


def acquired_images(acquire_id: str) -> List[Path]:
    """The imagery for an acquisition, oldest first. Traversal-hardened.

    Paths are rebuilt from the run directory rather than trusted from the
    request, and only GeoTIFFs are returned, so an ``acquire_id`` can never be
    used to read an arbitrary file.
    """
    safe = Path(str(acquire_id)).name
    folder = CONFIG.runs_dir / safe / "acquired"
    if not folder.is_dir():
        return []
    files = [p for p in sorted(folder.iterdir())
             if p.is_file() and p.suffix.lower() in ALLOWED_ACQUIRED_SUFFIXES]
    return files


def dates_for_images(paths: Sequence[Path]) -> Tuple[str, str]:
    """Acquisition dates recovered from the files themselves."""
    from ..geodate import extract_acquired

    found: List[str] = []
    for path in paths:
        try:
            import rasterio

            with rasterio.open(path) as src:
                tags = dict(src.tags() or {})
        except Exception:
            tags = {}
        info = extract_acquired(name=Path(path).name, tags=tags)
        found.append(info.date or "")
    if len(found) >= 2:
        return found[0], found[1]
    return (found[0] if found else "", "")
