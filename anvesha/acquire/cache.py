"""Content-addressed disk cache for the acquisition layer.

Why a cache is load-bearing rather than an optimisation
------------------------------------------------------
Anvesha is positioned as "works offline, online is an add-on". That claim is
only useful if the *second* look at an area works with no network. So every
byte we fetch is addressed by its request and written to disk with a
provenance sidecar. Re-asking the same question offline then succeeds, and the
report can state exactly when the imagery behind it was retrieved.

Two design notes
----------------
* **Content-addressed, not URL-addressed.** The cache key hashes the full
  request (endpoint + collection + bbox + dates + filters), so two different
  AOIs that happen to share a URL cannot collide, and changing a filter
  invalidates honestly instead of serving a stale answer.
* **The budget is enforced, not advisory.** ``acquire.cache_gb`` is a hard
  ceiling: after every write the oldest entries are dropped until the tree
  fits. A cache that silently grows without bound is a disk-space incident
  waiting to happen on a machine chosen for being modest.

Everything here is safe to call in air-gap mode: it never touches the network
and a missing cache entry is a normal ``None``, not an error.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..config import CONFIG
from .mode import load_settings

DEFAULT_BUDGET_GB = 0.5
_META_SUFFIX = ".meta.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def cache_root() -> Path:
    """Location of the acquisition cache (inside the app's data dir)."""
    root = Path(CONFIG.data_dir) / "acquire_cache"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return root


def budget_bytes() -> int:
    settings = load_settings()
    acq = settings.get("acquire") or {}
    try:
        gb = float(acq.get("cache_gb", DEFAULT_BUDGET_GB))
    except (TypeError, ValueError):
        gb = DEFAULT_BUDGET_GB
    return max(16 * 1024 * 1024, int(gb * 1024 ** 3))


def key_for(*parts: Any) -> str:
    """Stable 32-char key for a set of request components.

    Dicts are serialised with sorted keys so that two logically identical
    requests always hash the same regardless of construction order.
    """
    chunks = []
    for part in parts:
        if isinstance(part, dict):
            chunks.append(json.dumps(part, sort_keys=True, default=str))
        elif isinstance(part, (list, tuple)):
            chunks.append(json.dumps(list(part), default=str))
        else:
            chunks.append(str(part))
    digest = hashlib.sha256("\x1f".join(chunks).encode("utf-8")).hexdigest()
    return digest[:32]


@dataclass
class CacheEntry:
    key: str
    path: Path
    meta: Dict[str, Any]

    @property
    def bytes(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0


def blob_path(key: str) -> Path:
    return cache_root() / key[:2] / key


def meta_path(key: str) -> Path:
    return cache_root() / key[:2] / (key + _META_SUFFIX)


class DiskCache:
    """Small LRU disk cache. Thread-safe; never raises on a cache miss."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self._root = Path(root) if root is not None else None
        self._lock = threading.RLock()

    # -- paths ---------------------------------------------------------- #

    @property
    def root(self) -> Path:
        return self._root if self._root is not None else cache_root()

    def _blob(self, key: str) -> Path:
        return self.root / key[:2] / key

    def _meta(self, key: str) -> Path:
        return self.root / key[:2] / (key + _META_SUFFIX)

    # -- reads ---------------------------------------------------------- #

    def get(self, key: str) -> Optional[bytes]:
        p = self._blob(key)
        try:
            data = p.read_bytes()
        except OSError:
            return None
        # Refresh recency for LRU without rewriting the payload.
        try:
            os.utime(p, None)
        except OSError:
            pass
        return data

    def get_meta(self, key: str) -> Dict[str, Any]:
        try:
            return json.loads(self._meta(key).read_text(encoding="utf-8"))
        except Exception:
            return {}

    def get_entry(self, key: str) -> Optional[CacheEntry]:
        p = self._blob(key)
        if not p.exists():
            return None
        return CacheEntry(key=key, path=p, meta=self.get_meta(key))

    def has(self, key: str) -> bool:
        return self._blob(key).exists()

    # -- writes --------------------------------------------------------- #

    def put(self, key: str, data: bytes, *, kind: str = "",
            url: str = "", content_type: str = "",
            extra: Optional[Dict[str, Any]] = None) -> Path:
        """Write a payload + provenance sidecar, then enforce the budget.

        Written via a temp file + ``os.replace`` so a crash mid-write cannot
        leave a half-file that would later be served as a valid response.
        """
        with self._lock:
            blob = self._blob(key)
            blob.parent.mkdir(parents=True, exist_ok=True)
            tmp = blob.with_suffix(blob.suffix + f".{os.getpid()}.tmp")
            try:
                tmp.write_bytes(data)
                os.replace(tmp, blob)
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
            meta = {
                "key": key,
                "kind": kind,
                "url": url,
                "content_type": content_type,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "fetched_at": _now(),
            }
            if extra:
                meta["extra"] = extra
            try:
                self._meta(key).write_text(json.dumps(meta, indent=2),
                                           encoding="utf-8")
            except OSError:
                pass
            self.evict()
            return blob

    # -- maintenance ---------------------------------------------------- #

    def iter_entries(self):
        for meta_file in self.root.glob(f"*/*{_META_SUFFIX}"):
            key = meta_file.name[: -len(_META_SUFFIX)]
            blob = self._blob(key)
            if blob.exists():
                yield CacheEntry(key=key, path=blob, meta={})

    def total_bytes(self) -> int:
        total = 0
        for entry in self.iter_entries():
            total += entry.bytes
        return total

    def evict(self, limit: Optional[int] = None) -> int:
        """Drop least-recently-used entries until the tree fits. Returns freed bytes."""
        limit = budget_bytes() if limit is None else limit
        with self._lock:
            entries = []
            for entry in self.iter_entries():
                try:
                    entries.append((entry.path.stat().st_mtime, entry))
                except OSError:
                    continue
            total = sum(e.bytes for _, e in entries)
            if total <= limit:
                return 0
            entries.sort(key=lambda pair: pair[0])      # oldest first
            freed = 0
            for _, entry in entries:
                if total <= limit:
                    break
                size = entry.bytes
                for p in (entry.path, self._meta(entry.key)):
                    try:
                        p.unlink()
                    except OSError:
                        pass
                total -= size
                freed += size
            return freed

    def clear(self) -> int:
        freed = self.total_bytes()
        for entry in list(self.iter_entries()):
            for p in (entry.path, self._meta(entry.key)):
                try:
                    p.unlink()
                except OSError:
                    pass
        return freed

    def stats(self) -> Dict[str, Any]:
        entries = list(self.iter_entries())
        used = sum(e.bytes for e in entries)
        limit = budget_bytes()
        return {
            "root": str(self.root),
            "entries": len(entries),
            "used_bytes": used,
            "used_mb": round(used / 1024 ** 2, 1),
            "limit_bytes": limit,
            "limit_mb": round(limit / 1024 ** 2, 1),
            "used_pct": round(100.0 * used / limit, 1) if limit else 0.0,
        }


CACHE = DiskCache()
