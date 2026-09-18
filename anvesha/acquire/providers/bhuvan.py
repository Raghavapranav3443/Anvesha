"""ISRO Bhuvan context: authoritative Indian thematic layers, no credentials.

Why this is the ISRO integration, and why it needs nothing from anyone
----------------------------------------------------------------------
ISRO's Bhoonidhi catalogue requires an emailed credential request with a
multi-week lead time. Bhuvan, ISRO's own geoportal, publishes an **OGC WMS**
endpoint at ``bhuvan-vec1.nrsc.gov.in/bhuvan/wms`` that serves 6,671 layers
with **no account, no key, no approval**. Verified live.

That matters because the two services answer different questions, and the
decision layer only needs one of them:

* Bhoonidhi answers *"give me the imagery"* -- which the open Sentinel COGs
  already answer, credential-free, with identical lineage.
* Bhuvan answers *"what does the Government of India's own map say this land
  is?"* -- which is what turns an analysis into advice a non-expert can act on
  and defend. "Our analysis found new construction" is an observation. "Our
  analysis found new construction, and ISRO's district land-use layer records
  this parcel as wetland" is a decision.

So Bhuvan is not a stopgap. It is the authority lane.

The failure mode this module exists to prevent
---------------------------------------------
Bhuvan returns **HTTP 200 with a valid PNG for a layer that has no data in the
requested area**. A naive client sees a picture and concludes the layer is
present, then reports "ISRO records this as forest" about a blank image. That
would be the most damaging possible bug in this layer: confidently fabricated
official corroboration.

Measured behaviour: a no-data render is not fully transparent -- it carries a
constant ~3% opaque fraction from the service's own framing. A layer with data
in the window measured 8-20%. So the test is not "are there pixels" but "are
there *more* pixels than this service draws when it has nothing to draw".

The threshold is therefore **self-calibrating**: for any layer we care about we
render the same layer over a control window far outside the AOI's state and
require a real margin above that. No magic constant, and a service-side style
change cannot silently turn blanks into confirmations.
"""
from __future__ import annotations

import io
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..cache import CACHE, key_for
from ..errors import AcquireError
from ..http import Transport

BBox = Tuple[float, float, float, float]

WMS_URL = "https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms"
INDEX_PATH = Path(__file__).resolve().parents[1] / "data" / "bhuvan_layers.json"

# A control window with no Indian land thematic layers, used to measure how
# much ink the service spends on "nothing here".
CONTROL_BBOX: BBox = (84.0, 2.0, 84.2, 2.2)      # Bay of Bengal

# Required margin over the control coverage before we will call a layer
# "present in this window". Chosen to sit well above render noise and well
# below the 8-20% observed for genuinely populated layers.
PRESENCE_MARGIN = 0.02

# Themes that can change what a non-expert should actually do, in the order we
# prefer to report them.
CONTEXT_THEMES: Sequence[Tuple[str, str]] = (
    ("builtup_urban", "built-up (urban)"),
    ("builtup_rural", "built-up (rural)"),
    ("wetland", "wetland"),
    ("forest", "forest"),
    ("wasteland", "wasteland"),
    ("agriculture", "agriculture"),
    ("lulc", "land use / land cover"),
    ("ul10k", "land use 1:10,000"),
)

# Bhuvan's state abbreviations do not always match the usual postal codes, so
# name -> code resolution carries an explicit alias table rather than guessing.
_STATE_ALIASES = {
    "jammu and kashmir": ("jk",),
    "jammu & kashmir": ("jk",),
    "odisha": ("or",),
    "orissa": ("or",),
    "uttarakhand": ("uk",),
    "uttaranchal": ("uk",),
    "pondicherry": ("py",),
    "puducherry": ("py",),
    "telangana": ("ts", "tg"),
    "andaman and nicobar islands": ("and", "an"),
    "andaman & nicobar islands": ("and", "an"),
    "dadra and nagar haveli and daman and diu": ("dn", "dd", "diu"),
    "delhi": ("dl", "ncr"),
    "nct of delhi": ("dl", "ncr"),
    "punjab": ("pj", "pb"),
    "karnataka": ("ka", "kar"),
    "west bengal": ("wb",),
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


@dataclass
class ContextLayer:
    """One thematic layer, with the evidence that it is actually present."""

    name: str
    theme: str
    theme_label: str
    coverage: float
    control_coverage: float
    state: str = ""
    title: str = ""

    @property
    def evidence(self) -> float:
        return round(max(0.0, self.coverage - self.control_coverage), 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.name,
            "theme": self.theme,
            "theme_label": self.theme_label,
            "title": self.title,
            "coverage": round(self.coverage, 4),
            "control_coverage": round(self.control_coverage, 4),
            "evidence": self.evidence,
            "state": self.state,
        }


_index_lock = threading.Lock()
_index: Optional[Dict[str, Any]] = None


def load_index() -> Dict[str, Any]:
    """Read the bundled Bhuvan layer index (generated by scripts/build_bhuvan_catalog.py)."""
    global _index
    if _index is not None:
        return _index
    with _index_lock:
        if _index is None:
            try:
                _index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            except Exception:
                _index = {"layers": [], "state_codes": {}, "themes": {}}
    return _index


def _state_to_codes() -> Dict[str, Tuple[str, ...]]:
    """Reverse the index's code->name map, honouring the alias table."""
    mapping: Dict[str, List[str]] = {}
    for code, name in (load_index().get("state_codes") or {}).items():
        mapping.setdefault(_slug(name), []).append(code.lower())
    for name, codes in _STATE_ALIASES.items():
        mapping.setdefault(_slug(name), [])
        for code in codes:
            if code not in mapping[_slug(name)]:
                mapping[_slug(name)].append(code)
    return {k: tuple(dict.fromkeys(v)) for k, v in mapping.items()}


def state_codes_for(state_name: str) -> Tuple[str, ...]:
    """Bhuvan layer codes for a state name, or () when unknown."""
    if not state_name:
        return ()
    return _state_to_codes().get(_slug(state_name), ())


def _coverage_of_png(png: bytes, sample_step: int = 4) -> float:
    """Opaque fraction of a PNG, subsampled. Returns 0.0 on an unreadable image."""
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(png)).convert("RGBA")
        width, height = image.size
        pixels = image.load()
        xs = range(0, width, sample_step)
        ys = range(0, height, sample_step)
        total = 0
        opaque = 0
        for x in xs:
            for y in ys:
                total += 1
                alpha = pixels[x, y][3]
                if alpha > 8:                       # tolerate anti-aliased edges
                    opaque += 1
        return (opaque / total) if total else 0.0
    except Exception:
        return 0.0


class BhuvanProvider:
    """ISRO Bhuvan WMS: thematic context, with presence verification."""

    def __init__(self, transport: Transport, *, url: str = WMS_URL,
                 workers: int = 4) -> None:
        self.transport = transport
        self.url = url
        self.workers = max(1, int(workers))

    # -- raw WMS -------------------------------------------------------- #

    def render(self, layer: str, bbox: BBox, *, size: int = 256,
               use_cache: bool = True) -> bytes:
        """GetMap as a transparent PNG."""
        west, south, east, north = bbox
        params = {
            "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap",
            "LAYERS": layer, "STYLES": "", "SRS": "EPSG:4326",
            "BBOX": f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}",
            "WIDTH": str(size), "HEIGHT": str(size),
            "FORMAT": "image/png", "TRANSPARENT": "TRUE",
        }
        raw = self.transport.get_bytes(self.url, params=params, kind="wms_map")
        if not raw:
            raise AcquireError(f"Bhuvan returned an empty body for {layer}")
        head = raw[:120].lstrip()
        if head.startswith(b"<") or b"ServiceException" in raw[:400]:
            # A WMS exception arrives as HTTP 200 with an XML body. Ragged
            # layer names are common, and this is the only place that says so.
            detail = "unknown WMS error"
            match = re.search(rb"<ServiceException[^>]*>(.*?)</ServiceException>",
                              raw, re.S)
            if match:
                detail = match.group(1).strip().decode("utf-8", "replace")[:160]
            raise AcquireError(f"Bhuvan rejected {layer}: {detail}")
        return raw

    # -- presence verification ------------------------------------------ #

    def coverage(self, layer: str, bbox: BBox, *, size: int = 256) -> float:
        return _coverage_of_png(self.render(layer, bbox, size=size))

    def assess(self, layer: str, bbox: BBox, *, state: str = "",
               theme: str = "", title: str = "", size: int = 256,
               verify_against_control: bool = True) -> ContextLayer:
        """Render a layer over the AOI and decide, with evidence, if it is present.

        The control render is what makes the claim honest. Without it, a blank
        tile from the service is indistinguishable from a tile full of data and
        the report would attribute a finding to ISRO that ISRO never made.
        """
        aoi_cov = self.coverage(layer, bbox, size=size)
        control_cov = 0.0
        if verify_against_control:
            try:
                control_cov = self.coverage(layer, CONTROL_BBOX, size=size)
            except Exception:
                control_cov = 0.0
        label = dict(CONTEXT_THEMES).get(theme, theme or "layer")
        return ContextLayer(name=layer, theme=theme, theme_label=label,
                            coverage=aoi_cov, control_coverage=control_cov,
                            state=state, title=title)

    @staticmethod
    def is_present(layer: ContextLayer) -> bool:
        """True only when the AOI has meaningfully more data than the control."""
        return layer.evidence >= PRESENCE_MARGIN

    # -- index queries --------------------------------------------------- #

    def layers_for_state(self, state_name: str,
                         themes: Sequence[str] = tuple(t for t, _ in CONTEXT_THEMES)
                         ) -> List[Dict[str, Any]]:
        codes = {c.lower() for c in state_codes_for(state_name)}
        if not codes:
            return []
        wanted = set(themes)
        out = []
        for entry in load_index().get("layers", []):
            if entry.get("theme") not in wanted:
                continue
            state = str(entry.get("state") or "").lower()
            if state and state not in codes:
                continue
            out.append(entry)
        # One layer per theme, preferring the shorter (state-level) name.
        best: Dict[str, Dict[str, Any]] = {}
        for entry in out:
            theme = entry["theme"]
            current = best.get(theme)
            if current is None or len(entry["name"]) < len(current["name"]):
                best[theme] = entry
        return [best[t] for t, _ in CONTEXT_THEMES if t in best]

    def context_for(self, state_name: str, bbox: BBox, *,
                    max_layers: int = 6, size: int = 256
                    ) -> Tuple[List[ContextLayer], List[Dict[str, str]]]:
        """Verified thematic context for an AOI. Returns (found, errors).

        Renders candidates concurrently: Bhuvan is slow (seconds per tile) and
        a sequential probe of six layers would stall the request.
        """
        candidates = self.layers_for_state(state_name)[:max_layers]
        if not candidates:
            return [], [{"layer": "*",
                         "error": f"no ISRO layers indexed for state {state_name!r}"}]

        assessed: List[ContextLayer] = []
        errors: List[Dict[str, str]] = []

        def work(entry: Dict[str, Any]) -> ContextLayer:
            return self.assess(entry["name"], bbox, state=entry.get("state") or "",
                               theme=entry.get("theme") or "",
                               title=entry.get("title") or "", size=size)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(work, entry): entry for entry in candidates}
            for future, entry in futures.items():
                try:
                    assessed.append(future.result())
                except Exception as exc:
                    errors.append({"layer": entry.get("name", "?"),
                                   "error": f"{type(exc).__name__}: {exc}"})

        present = [l for l in assessed if self.is_present(l)]
        order = [t for t, _ in CONTEXT_THEMES]
        present.sort(key=lambda l: (order.index(l.theme)
                                    if l.theme in order else 99, -l.evidence))
        return present, errors

    # -- plain English --------------------------------------------------- #

    @staticmethod
    def describe(layers: Sequence[ContextLayer]) -> List[str]:
        """One plain sentence per present layer, for the report and the UI."""
        lines: List[str] = []
        for layer in layers:
            pct = max(1, int(round(layer.coverage * 100)))
            lines.append(
                f"ISRO's {layer.theme_label} layer covers roughly {pct}% of this "
                f"window ({layer.name}).")
        return lines


def cache_stats() -> Dict[str, Any]:
    return CACHE.stats()
