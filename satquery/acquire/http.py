"""HTTP transport with retries, caching and an air-gap refusal that serves the cache first.

The transport is a seam on purpose
---------------------------------
Every provider takes a ``Transport`` rather than importing ``requests``
directly. That gives three things that matter for this project:

* tests exercise real provider logic with **zero network**, so the suite stays
  fast and offline (the air-gap gate runs the whole suite with the network
  provably cut);
* the air-gap refusal is enforced in *one* place, with a readable message,
  instead of relying on every caller to remember;
* a provider cannot accidentally bypass the audit hook, because it never
  holds a socket itself.

Air-gap ordering
----------------
:meth:`RequestsTransport._request` resolves the cache key and serves a **cache
hit first**; only a miss reaches the mode check that raises
:class:`NetworkBlockedAirgap`. The order is the point. A hit performs no I/O, so
air-gap mode can still replay a fetch the app already made -- a scene search or
an ISRO map render included -- instead of reading "offline" as "nothing works".
This was the reverse, and the refusal ran first: a fully populated cache was
unreachable the moment the app went offline, which is exactly the mode the
demo ships in. A miss still refuses *before* any socket call, and the appended
hint distinguishes "not cached" from "not attempted". The installed audit hook
in :mod:`satquery.acquire.mode` remains the backstop for any code path that does
reach a socket -- this is belt and braces, and the early check exists to produce
a clear error rather than a deep GDAL traceback.
"""
from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

from .cache import CACHE, DiskCache, key_for
from .errors import FetchFailed, NetworkBlockedAirgap
from .mode import current_mode

DEFAULT_TIMEOUT = 20.0
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 0.6
DEFAULT_UA = ("Anvesha/1.0 (offline-first satellite analysis; "
              "+https://github.com/anvesha)")

# Statuses worth retrying: transient server-side or throttling responses.
_RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class TransportError(FetchFailed):
    """A request failed after exhausting retries."""


@runtime_checkable
class Transport(Protocol):
    """The seam. Implementations must raise on failure, never return None."""

    def get_bytes(self, url: str, *, params: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None,
                  timeout: Optional[float] = None, use_cache: bool = True,
                  kind: str = "") -> bytes: ...

    def get_json(self, url: str, *, params: Optional[Dict[str, Any]] = None,
                 headers: Optional[Dict[str, str]] = None,
                 timeout: Optional[float] = None,
                 use_cache: bool = True, kind: str = "json") -> Any: ...

    def post_json(self, url: str, payload: Any, *,
                  headers: Optional[Dict[str, str]] = None,
                  timeout: Optional[float] = None,
                  use_cache: bool = True, kind: str = "json") -> Any: ...


def _require_online(target: str) -> None:
    """Refuse outbound I/O in air-gap mode.

    Called only after the cache has missed, so the message can say the honest
    thing: there is no local copy of this request, which is why it would need the
    network. Saying "refused" without that distinction reads as a bug in the
    network stack rather than a policy the user can act on.
    """
    if current_mode() != "online":
        raise NetworkBlockedAirgap(
            f"air-gap mode: refusing to contact {target} -- no cached copy of "
            f"this request exists. Switch to online mode to fetch imagery, or "
            f"re-run work whose data was already fetched.",
            event="acquire.http", target=target)


class RequestsTransport:
    """Real transport. Retries idempotent GETs; POSTs retry only on connection errors."""

    def __init__(self, *, cache: Optional[DiskCache] = None,
                 retries: int = DEFAULT_RETRIES,
                 backoff: float = DEFAULT_BACKOFF,
                 user_agent: str = DEFAULT_UA) -> None:
        self.cache = cache if cache is not None else CACHE
        self.retries = max(1, int(retries))
        self.backoff = max(0.0, float(backoff))
        self.user_agent = user_agent

    # -- internals ------------------------------------------------------ #

    def _headers(self, extra: Optional[Dict[str, str]]) -> Dict[str, str]:
        head = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, identity"}
        if extra:
            head.update({k: v for k, v in extra.items() if v is not None})
        return head

    @staticmethod
    def _sleep_for(attempt: int, backoff: float, retry_after: Optional[str]) -> None:
        """Exponential backoff with jitter, honouring Retry-After when sane."""
        if retry_after:
            try:
                wait = float(retry_after)
                if 0 < wait <= 30:
                    time.sleep(wait)
                    return
            except (TypeError, ValueError):
                pass
        delay = backoff * (2 ** attempt)
        time.sleep(delay + random.uniform(0, delay * 0.25))

    def _key(self, method: str, url: str, params: Any, body: Any) -> str:
        return key_for(method, url, params or {}, body if body is not None else "")

    def _request(self, method: str, url: str, *,
                 params: Optional[Dict[str, Any]] = None,
                 headers: Optional[Dict[str, str]] = None,
                 timeout: Optional[float] = None,
                 body: Any = None,
                 use_cache: bool = True,
                 kind: str = "") -> Tuple[bytes, Dict[str, str], str]:
        key = self._key(method, url, params, body)
        # Cache first, mode second. A hit does no I/O, so air-gap mode can still
        # serve it; only a miss is refused below. Both methods read the cache for
        # the same reason: a STAC search is a deterministic read expressed as a
        # POST, and its result was already being written and never read back.
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                return cached, {"X-Anvesha-Cache": "hit"}, key
        _require_online(url)
        import requests  # imported lazily so air-gap paths never import the stack

        last_error: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                resp = requests.request(
                    method, url, params=params, headers=self._headers(headers),
                    timeout=timeout or DEFAULT_TIMEOUT,
                    json=body if body is not None else None,
                )
            except NetworkBlockedAirgap:
                raise
            except Exception as exc:                    # connection-level failure
                if isinstance(exc, OSError) and _looks_like_airgap(exc):
                    raise
                last_error = exc
                if attempt + 1 < self.retries:
                    self._sleep_for(attempt, self.backoff, None)
                continue

            if resp.status_code in _RETRY_STATUS and attempt + 1 < self.retries:
                self._sleep_for(attempt, self.backoff,
                                resp.headers.get("Retry-After"))
                continue

            if resp.status_code >= 400:
                raise TransportError(
                    f"{method} {url} -> HTTP {resp.status_code}: "
                    f"{resp.text[:200] if resp.content else ''}")
            return resp.content, dict(resp.headers), key
        raise TransportError(
            f"{method} {url} failed after {self.retries} attempts: {last_error}")

    # -- protocol ------------------------------------------------------- #

    def get_bytes(self, url: str, *, params: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None,
                  timeout: Optional[float] = None, use_cache: bool = True,
                  kind: str = "") -> bytes:
        data, _hdr, key = self._request("GET", url, params=params,
                                        headers=headers, timeout=timeout,
                                        use_cache=use_cache, kind=kind)
        if use_cache:
            self.cache.put(key, data, kind=kind, url=url)
        return data

    def get_json(self, url: str, *, params: Optional[Dict[str, Any]] = None,
                 headers: Optional[Dict[str, str]] = None,
                 timeout: Optional[float] = None,
                 use_cache: bool = True, kind: str = "json") -> Any:
        head = dict(headers or {})
        head.setdefault("Accept", "application/json")
        raw = self.get_bytes(url, params=params, headers=head, timeout=timeout,
                             use_cache=use_cache, kind=kind)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise TransportError(
                f"{url} did not return JSON ({type(exc).__name__})") from exc

    def post_json(self, url: str, payload: Any, *,
                  headers: Optional[Dict[str, str]] = None,
                  timeout: Optional[float] = None,
                  use_cache: bool = True, kind: str = "json") -> Any:
        head = dict(headers or {})
        head.setdefault("Accept", "application/json")
        raw, hdr, key = self._request(
            "POST", url, headers=head, timeout=timeout, body=payload,
            use_cache=use_cache)
        # Cache POST results too, and *read* them on the way in (see _request): a
        # STAC search is deterministic for our purposes, so re-running it offline
        # reuses the answer rather than failing on the missing network.
        if use_cache and hdr.get("X-Anvesha-Cache") != "hit":
            self.cache.put(key, raw, kind=kind, url=url)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise TransportError(
                f"{url} did not return JSON ({type(exc).__name__})") from exc


def _looks_like_airgap(exc: BaseException) -> bool:
    """True when an OSError is really our air-gap refusal in disguise."""
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, NetworkBlockedAirgap):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


class OfflineTransport:
    """Refuses everything. Used by tests and as a fail-closed default."""

    def __init__(self, reason: str = "offline transport: network access disabled") -> None:
        self.reason = reason
        self.calls: list = []

    def _refuse(self, url: str):
        self.calls.append(url)
        raise NetworkBlockedAirgap(self.reason, event="acquire.offline",
                                   target=url)

    def get_bytes(self, url, **kw): return self._refuse(url)          # noqa: D102
    def get_json(self, url, **kw): return self._refuse(url)           # noqa: D102
    def post_json(self, url, payload, **kw): return self._refuse(url)  # noqa: D102


def default_transport() -> Transport:
    """Transport for production callers."""
    return RequestsTransport()
