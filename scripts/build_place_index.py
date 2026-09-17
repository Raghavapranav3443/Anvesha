"""Generate the offline place index used by ``satquery.acquire.aoi``.

Why this exists
---------------
The app has to answer "where?" for a non-expert with no network and no GIS
skills. A geocoding API alone would make offline mode unable to find any place
at all, which would gut the air-gap claim.

So we resolve places from a bundled index, and build that index once here from
OpenStreetMap's Nominatim. states and union territories are included because
they are few, stable, and hand-checkable -- and because anything a district
falls inside is already covered. Districts and villages are resolved online at
runtime and then *cached to disk*, so a place resolved once keeps working
offline afterwards.

Nominatim's usage policy requires a descriptive User-Agent and at most one
request per second. Both are honoured below; do not lower ``SLEEP_S``.

Usage::

    python scripts/build_place_index.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "satquery" / "acquire" / "data" / "india_states.json"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
SLEEP_S = 1.1
USER_AGENT = "Anvesha-place-index-builder/1.0 (SIH project; contact: team@anvesha.local)"

# 28 states + 8 union territories. `aliases` cover the spellings people actually
# type, including the older names still in wide use.
PLACES = [
    ("Andhra Pradesh", ["AP"]),
    ("Arunachal Pradesh", []),
    ("Assam", []),
    ("Bihar", []),
    ("Chhattisgarh", []),
    ("Goa", []),
    ("Gujarat", []),
    ("Haryana", []),
    ("Himachal Pradesh", ["HP"]),
    ("Jharkhand", []),
    ("Karnataka", ["Karnatak"]),
    ("Kerala", []),
    ("Madhya Pradesh", ["MP"]),
    ("Maharashtra", []),
    ("Manipur", []),
    ("Meghalaya", []),
    ("Mizoram", []),
    ("Nagaland", []),
    ("Odisha", ["Orissa"]),
    ("Punjab", []),
    ("Rajasthan", []),
    ("Sikkim", []),
    ("Tamil Nadu", ["TN"]),
    ("Telangana", []),
    ("Tripura", []),
    ("Uttar Pradesh", ["UP"]),
    ("Uttarakhand", ["Uttaranchal"]),
    ("West Bengal", ["Bengal", "WB"]),
    # Union territories
    ("Andaman and Nicobar Islands", ["Andaman", "Nicobar"]),
    ("Chandigarh", []),
    ("Dadra and Nagar Haveli and Daman and Diu", ["Daman", "Diu", "Dadra"]),
    ("Delhi", ["New Delhi", "NCT of Delhi"]),
    ("Jammu and Kashmir", ["Jammu", "Kashmir"]),
    ("Ladakh", []),
    ("Lakshadweep", []),
    ("Puducherry", ["Pondicherry"]),
]


def fetch_bbox(name: str) -> tuple | None:
    import requests

    resp = requests.get(
        NOMINATIM_URL,
        params={"q": f"{name}, India", "format": "jsonv2", "limit": 1,
                "countrycodes": "in", "featuretype": "state"},
        headers={"User-Agent": USER_AGENT}, timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return None
    raw = rows[0].get("boundingbox") or []
    if len(raw) != 4:
        return None
    south, north, west, east = (float(x) for x in raw)
    return (west, south, east, north), rows[0].get("display_name", "")


def main() -> int:
    places = []
    failures = []
    for index, (name, aliases) in enumerate(PLACES, start=1):
        try:
            result = fetch_bbox(name)
        except Exception as exc:
            failures.append(f"  {name:44} {type(exc).__name__}: {exc}")
            time.sleep(SLEEP_S)
            continue
        if not result:
            failures.append(f"  {name:44} no result")
            time.sleep(SLEEP_S)
            continue
        bbox, display = result
        places.append({
            "name": name,
            "aliases": aliases,
            "bbox": [round(v, 5) for v in bbox],
            "display_name": display[:120],
        })
        print(f"  [{index:2}/{len(PLACES)}] {name:44} "
              f"{bbox[0]:8.3f} {bbox[1]:7.3f} {bbox[2]:8.3f} {bbox[3]:7.3f}")
        time.sleep(SLEEP_S)

    if failures:
        print("\n  === failures ===")
        for line in failures:
            print(line)
    if len(places) < len(PLACES) * 0.8:
        raise SystemExit(
            f"only {len(places)}/{len(PLACES)} places resolved; refusing to "
            f"write a partial offline index")

    document = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "OpenStreetMap Nominatim",
        "license": "ODbL 1.0 (c) OpenStreetMap contributors",
        "count": len(places),
        "places": places,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(f"\n  wrote {OUT_PATH.relative_to(ROOT)} "
          f"({OUT_PATH.stat().st_size // 1024} KB, {len(places)} places)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
