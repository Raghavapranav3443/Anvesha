"""Live end-to-end smoke test for the online acquisition layer.

Proves, against the real services, that Anvesha can go from a place name to a
pair of analysis-ready GeoTIFFs and a verified ISRO context reading.

Run it in online mode::

    SATQUERY_MODE=online python scripts/acquire_smoke.py
    SATQUERY_MODE=online python scripts/acquire_smoke.py --place "Dibrugarh, Assam"

It is deliberately a *script* rather than a test: the test suite must pass with
the network provably cut, so nothing in ``tests/`` touches the internet. This
is the thing you run to demonstrate that the online half genuinely works.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from satquery.acquire import mode as acquire_mode          # noqa: E402
from satquery.acquire.fetch import fetch_scene, pick_pair, plan_grid  # noqa: E402
from satquery.acquire.http import default_transport        # noqa: E402
from satquery.acquire.providers import BhuvanProvider, default_providers  # noqa: E402
from satquery.acquire.providers.stac import search_with_fallback  # noqa: E402


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n  {text}\n{'=' * 72}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--place", default="Dibrugarh, Assam")
    parser.add_argument("--days", type=int, default=240)
    parser.add_argument("--max-cloud", type=float, default=35.0)
    parser.add_argument("--window-km", type=float, default=12.0)
    parser.add_argument("--out", default="runs/acquire_smoke")
    args = parser.parse_args()

    banner("0. MODE")
    status = acquire_mode.guard_status()
    print(f"  mode={status['mode']}  guard_installed={status['guard_installed']}")
    if status["mode"] != "online":
        print("\n  Refusing to run: start with SATQUERY_MODE=online.")
        return 2

    transport = default_transport()

    banner("1. RESOLVE PLACE -> AOI")
    from satquery.acquire.aoi import analysis_window, gazetteer_lookup

    offline_hits = gazetteer_lookup(args.place)
    print(f"  offline gazetteer matches: {[a.name for a in offline_hits]}")
    started = time.time()
    aoi = analysis_window(args.place, transport, window_km=args.window_km)
    print(f"  resolved in {time.time()-started:.1f}s -> {aoi.name!r} "
          f"via {aoi.source}")
    print(f"  bbox={aoi.bbox}  size_km={aoi.to_dict()['size_km']}")
    print(f"  centre lat/lon = {aoi.centroid}")

    banner("2. SEARCH (STAC)")
    end = time.strftime("%Y-%m-%d")
    start = time.strftime("%Y-%m-%d", time.gmtime(time.time() - args.days * 86400))
    providers = default_providers()
    print(f"  window {start} .. {end}  cloud<={args.max_cloud}%")
    print(f"  providers: {[p.name for p in providers]}")
    started = time.time()
    scenes, errors = search_with_fallback(
        providers, aoi.bbox, start=start, end=end, transport=transport,
        limit=40, max_cloud_pct=args.max_cloud)
    print(f"  {len(scenes)} scene(s) in {time.time()-started:.1f}s")
    for err in errors:
        print(f"    ! {err['provider']}: {err['error']}")
    for scene in scenes[:6]:
        cloud = "?" if scene.cloud_pct is None else f"{scene.cloud_pct:.0f}%"
        print(f"    - {scene.date}  cloud {cloud:>4}  {scene.scene_id}")

    if not scenes:
        print("\n  No scenes; nothing further to fetch.")
        return 1

    banner("3. PICK A DATE PAIR")
    older, newer = pick_pair(scenes)
    if older is None or newer is None:
        print("  Could not find two dates far enough apart. Try --days larger.")
        return 1
    print(f"  A: {older.date}  cloud={older.cloud_pct}  {older.scene_id}")
    print(f"  B: {newer.date}  cloud={newer.cloud_pct}  {newer.scene_id}")

    banner("4. WINDOWED COG FETCH -> SHARED GRID")
    grid = plan_grid(aoi.bbox)
    print(f"  grid: EPSG:{grid.epsg} {grid.width}x{grid.height} "
          f"@ {grid.pixel_size_m:.1f} m  ({grid.area_ha:.0f} ha)")
    out_dir = ROOT / args.out
    fetched = []
    for label, scene in (("A", older), ("B", newer)):
        started = time.time()
        name = f"s2_{scene.date}.tif"
        got = fetch_scene(scene, grid, out_dir / name)
        fetched.append(got)
        print(f"  {label} {scene.date}: {name}  {got.width}x{got.height}  "
              f"valid={got.valid_fraction*100:.0f}%  in {time.time()-started:.1f}s")
        for warning in got.warnings:
            print(f"      ! {warning}")

    import numpy as np
    import rasterio

    print("  reflectance sanity (should be ~0..1 with median near 0.1-0.3):")
    for got in fetched:
        with rasterio.open(got.path) as ds:
            data = ds.read()
            print(f"    {got.path.name}: min={data.min():.3f} "
                  f"max={data.max():.3f} p50={np.percentile(data, 50):.3f} "
                  f"crs={ds.crs}")

    banner("5. ISRO CONTEXT (Bhuvan, verified presence)")
    state = aoi.admin.get("state") or args.place.split(",")[-1].strip()
    bhuvan = BhuvanProvider(transport)
    from satquery.acquire.providers.bhuvan import state_codes_for

    print(f"  state={state!r} -> Bhuvan codes {state_codes_for(state)}")
    started = time.time()
    layers, ctx_errors = bhuvan.context_for(state, aoi.window(args.window_km))
    print(f"  {len(layers)} verified layer(s) in {time.time()-started:.1f}s")
    for layer in layers:
        print(f"    - {layer.name:32} coverage={layer.coverage*100:5.1f}% "
              f"control={layer.control_coverage*100:4.1f}% "
              f"evidence={layer.evidence*100:.1f}%")
    for err in ctx_errors:
        print(f"    ! {err['layer']}: {err['error']}")
    if not layers:
        print("  (no ISRO layer verified as present for this window)")

    banner("RESULT")
    print(f"  two GeoTIFFs written under {out_dir}")
    print("  online layer works end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
