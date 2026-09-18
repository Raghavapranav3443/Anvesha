"""Build a compact, offline index of ISRO Bhuvan WMS layers.

Why a generated index instead of querying live every time:

Bhuvan's WMS capabilities document is ~7.5 MB and takes tens of seconds to
fetch. We do not want the app to depend on that round trip to know what
context layers exist, and we cannot ship the 7.5 MB document in a repository
that is meant to stay compact.

So: fetch once, distil to the few thousand entries we might actually use, and
ship *that*. The runtime can still refresh from the live service (online mode,
TTL-cached) but it never has to.

Two things this script deliberately refuses to do:

1. It never invents a layer name. Bhuvan's naming is genuinely unpredictable
   (Assam's districts are coded ``AS_DI``, ``AS_NA``, ``AS_SI``, ``AS_TE``,
   ``AS_TI`` -- Kamrup is *not* ``AS_KM``). Guessing produces a WMS
   ServiceExceptionReport returned with HTTP 200, which is exactly the kind of
   silent failure this project keeps getting bitten by. Names come from the
   capabilities document or they do not exist.

2. It never records a layer it did not see. If the fetch fails, the script
   fails loudly rather than writing a partial index that would look complete.

Usage::

    python scripts/build_bhuvan_catalog.py            # use cached doc if fresh
    python scripts/build_bhuvan_catalog.py --refresh  # force a live fetch
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CAPABILITIES_URL = (
    "https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms"
    "?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetCapabilities"
)
CACHE_PATH = ROOT / ".cache" / "bhuvan_wms.xml"
OUT_PATH = ROOT / "anvesha" / "acquire" / "data" / "bhuvan_layers.json"
CACHE_TTL_S = 7 * 24 * 3600

# Indian state / UT codes as they appear in Bhuvan NUIS layer names. Derived
# empirically from the live capabilities document, then named. Anything not in
# this table keeps its raw code so an unknown state is still usable.
STATE_CODES = {
    "AND": ("Andaman & Nicobar Islands", "AN"),
    "AP": ("Andhra Pradesh", "AP"),
    "AR": ("Arunachal Pradesh", "AR"),
    "AS": ("Assam", "AS"),
    "BR": ("Bihar", "BR"),
    "CG": ("Chhattisgarh", "CG"),
    "CH": ("Chandigarh", "CH"),
    "DD": ("Daman & Diu", "DD"),
    "DN": ("Dadra & Nagar Haveli", "DN"),
    "GA": ("Goa", "GA"),
    "GJ": ("Gujarat", "GJ"),
    "HP": ("Himachal Pradesh", "HP"),
    "HR": ("Haryana", "HR"),
    "JH": ("Jharkhand", "JH"),
    "JK": ("Jammu & Kashmir", "JK"),
    "KA": ("Karnataka", "KA"),
    "KAR": ("Karnataka", "KA"),
    "KL": ("Kerala", "KL"),
    "LD": ("Lakshadweep", "LD"),
    "MH": ("Maharashtra", "MH"),
    "ML": ("Meghalaya", "ML"),
    "MN": ("Manipur", "MN"),
    "MP": ("Madhya Pradesh", "MP"),
    "MZ": ("Mizoram", "MZ"),
    "NL": ("Nagaland", "NL"),
    "OR": ("Odisha", "OR"),
    "PJ": ("Punjab", "PB"),
    "PY": ("Puducherry", "PY"),
    "RJ": ("Rajasthan", "RJ"),
    "SK": ("Sikkim", "SK"),
    "TG": ("Telangana", "TG"),
    "TN": ("Tamil Nadu", "TN"),
    "TR": ("Tripura", "TR"),
    "UP": ("Uttar Pradesh", "UP"),
    "UT": ("Uttarakhand", "UT"),
    "WB": ("West Bengal", "WB"),
}

# Context themes that can change what a non-expert should do. Ordered: the
# first substring that matches wins, so longer/more specific keys come first.
THEMES = [
    ("builtup_urban", "built-up area (urban)"),
    ("builtup_rural", "built-up area (rural)"),
    ("builtup", "built-up area"),
    ("wasteland", "wasteland"),
    ("wetland", "wetland"),
    ("forest", "forest"),
    ("drainage_line", "drainage line"),
    ("drainage", "drainage"),
    ("agriculture", "agriculture"),
    ("cropland", "cropland"),
    ("admin_boundary", "administrative boundary"),
    ("watershed", "watershed"),
    ("waterbody", "water body"),
    ("river", "river"),
    ("slope", "slope"),
    ("soil", "soil"),
    ("lulc", "land use / land cover"),
    ("ul10k", "land use 1:10,000"),
]

# Layer prefixes worth carrying. `nuis:` holds the NRIS district thematics that
# matter here; the rest are surfaced by name so a caller can opt in.
WANTED_PREFIXES = ("nuis:", "cite:", "basemap:")


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def fetch_capabilities(refresh: bool) -> bytes:
    if CACHE_PATH.exists() and not refresh:
        age = time.time() - CACHE_PATH.stat().st_mtime
        if age < CACHE_TTL_S:
            print(f"  cache hit  {CACHE_PATH}  ({age/3600:.1f}h old)")
            return CACHE_PATH.read_bytes()
    print(f"  fetching   {CAPABILITIES_URL[:70]}...")
    import requests

    started = time.time()
    resp = requests.get(
        CAPABILITIES_URL, timeout=60, headers={"User-Agent": "anvesha-catalog"}
    )
    resp.raise_for_status()
    if len(resp.content) < 100_000:
        raise SystemExit(
            f"refusing to index a {len(resp.content)}-byte capabilities document; "
            "that is a service error page, not capabilities"
        )
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_bytes(resp.content)
    print(f"  fetched    {len(resp.content)} bytes in {time.time()-started:.1f}s")
    return resp.content


def extract_layers(xml_bytes: bytes):
    """Return [(name, title)] for real <Layer><Name> entries.

    Styles also contain a <Name> element, which is how it is possible to
    "find" thousands of layer names that the service will then reject with
    LayerNotDefined. Only direct children of a <Layer> count.
    """
    root = ET.fromstring(xml_bytes)
    out = []
    for el in root.iter():
        if _local(el.tag) != "Layer":
            continue
        name = None
        title = None
        for child in el:
            tag = _local(child.tag)
            if tag == "Name" and name is None:
                name = (child.text or "").strip()
            elif tag == "Title" and title is None:
                title = (child.text or "").strip()
        if name:
            out.append((name, title or name))
    return out


def classify(name: str):
    """(state_code, state_name, theme_key, theme_label) for a layer name."""
    body = name.split(":", 1)[1] if ":" in name else name
    state_code = None
    match = re.match(r"^([A-Z]{2,4})[_A-Z]", body)
    if match:
        state_code = match.group(1)
    # Longest state code that is actually known wins, so KAR beats KA.
    if state_code and state_code not in STATE_CODES:
        for size in (3, 2):
            cand = body[:size]
            if cand in STATE_CODES:
                state_code = cand
                break
        else:
            state_code = state_code if len(state_code) <= 3 else None
    state_name = STATE_CODES.get(state_code, (state_code, state_code))[0] if state_code else None
    lowered = body.lower()
    for key, label in THEMES:
        if key in lowered:
            return state_code, state_name, key, label
    return state_code, state_name, "other", "other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="force a live fetch")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    xml_bytes = fetch_capabilities(args.refresh)
    layers = extract_layers(xml_bytes)
    print(f"  parsed     {len(layers)} real layers")

    entries = []
    theme_counts: Counter = Counter()
    state_counts: Counter = Counter()
    for name, title in layers:
        if not name.startswith(WANTED_PREFIXES):
            continue
        state_code, state_name, theme, theme_label = classify(name)
        theme_counts[theme] += 1
        if state_code:
            state_counts[state_code] += 1
        entries.append(
            {
                "name": name,
                "title": title[:120],
                "state": state_code,
                "theme": theme,
            }
        )

    # Keep `other` out of the shipped index: it is noise that would bloat the
    # bundle without giving the decision layer anything actionable.
    entries = [e for e in entries if e["theme"] != "other"]
    document = {
        "source": CAPABILITIES_URL,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "layer_count": len(entries),
        "state_codes": {
            code: STATE_CODES.get(code, (code, code))[0]
            for code in sorted(state_counts)
        },
        "themes": {
            key: label for key, label in THEMES
        },
        "layers": entries,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    size_kb = OUT_PATH.stat().st_size / 1024

    if not args.quiet:
        print(f"\n  === themes ({len(theme_counts)} distinct) ===")
        for key, label in THEMES:
            if theme_counts.get(key):
                print(f"    {key:16} {theme_counts[key]:5}   {label}")
        print(f"\n  === states ({len(state_counts)}) ===")
        print("   ", ", ".join(sorted(state_counts)))
        unmapped = sorted(state_counts)
        print(f"\n  wrote {OUT_PATH.relative_to(ROOT)}  ({size_kb:.0f} KB, {len(entries)} layers)")
    else:
        print(f"  wrote {OUT_PATH.relative_to(ROOT)}  ({size_kb:.0f} KB, {len(entries)} layers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
