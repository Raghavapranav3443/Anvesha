"""B8 — acquisition-date extraction.

Order: rasterio dataset tags first (``TIFFTAG_DATETIME``,
``acquisition_date``, ``sensing_time``, ...) then filename regexes
(``YYYY-MM-DD``, ``YYYYMMDD``, Sentinel ``S2x_YYYYMMDDTHHMMSS``).
Never guesses: when nothing matches, ``date`` is "unknown" and the source
records it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

__all__ = ["AcquiredInfo", "extract_acquired", "extract_from_name",
           "extract_from_tags"]


@dataclass
class AcquiredInfo:
    date: str          # ISO "YYYY-MM-DD" or "unknown"
    source: str        # "rasterio_tags" | "filename" | "unknown"
    raw: Optional[str] = None


_TAG_KEYS = ("tifftag_datetime", "datetime", "acquisition_date",
             "acquisitiondate", "acquisition_time", "sensing_time",
             "sensingtime", "date", "obs_date", "imagery_date")

_MONTH_DAYS = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _valid_ymd(y: int, m: int, d: int) -> bool:
    return 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= _MONTH_DAYS[m - 1]


def _to_iso(y: int, m: int, d: int) -> str:
    return f"{y:04d}-{m:02d}-{d:02d}"


def _parse_loose(text: str) -> Optional[str]:
    """Parse 'YYYY:MM:DD HH:MM:SS' (TIFF), ISO, or compact dates."""
    t = str(text).strip()
    m = re.match(r"(\d{4})[-:/](\d{1,2})[-:/](\d{1,2})", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _valid_ymd(y, mo, d):
            return _to_iso(y, mo, d)
    m = re.match(r"(\d{4})(\d{2})(\d{2})", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _valid_ymd(y, mo, d):
            return _to_iso(y, mo, d)
    return None


def extract_from_tags(tags: Optional[Dict[str, Any]]) -> Optional[AcquiredInfo]:
    """First matching acquisition-ish tag, parsed to ISO."""
    if not tags:
        return None
    for key, value in sorted(tags.items()):
        if key.lower() in _TAG_KEYS and value:
            iso = _parse_loose(value)
            if iso:
                return AcquiredInfo(date=iso, source="rasterio_tags",
                                    raw=str(value))
    return None


def extract_from_name(name: str) -> Optional[AcquiredInfo]:
    """Filename date regexes (validated calendar dates only)."""
    s = str(name)
    for pat in (r"(\d{4})-(\d{2})-(\d{2})",
                r"S2[A-C]_(\d{4})(\d{2})(\d{2})T\d{6}",
                r"(?:^|[_\-])(\d{4})(\d{2})(\d{2})(?:[_\-.]|$)"):
        m = re.search(pat, s)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if _valid_ymd(y, mo, d):
                return AcquiredInfo(date=_to_iso(y, mo, d),
                                    source="filename", raw=m.group(0))
    return None


def extract_acquired(name: Optional[str] = None,
                     tags: Optional[Dict[str, Any]] = None) -> AcquiredInfo:
    """Tags first, filename second; ``unknown`` when neither matches."""
    hit = extract_from_tags(tags) or extract_from_name(name or "")
    if hit:
        return hit
    return AcquiredInfo(date="unknown", source="unknown", raw=None)

