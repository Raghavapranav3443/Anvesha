"""Air-gap mode + an outbound-network guard that cannot be quietly removed.

Why an audit hook
-----------------
The obvious way to enforce "no network" is to monkeypatch
``socket.socket.connect``. That is weaker than it looks: it misses socket use
inside C extensions that bypass the Python-level method, and any code can
simply rebind it back. ``sys.addaudithook`` is the right primitive -- CPython
raises ``socket.*`` audit events inside the socket implementation itself, and
**audit hooks cannot be removed once installed**.

The trap that design creates
----------------------------
Because hooks cannot be removed, a naive implementation makes the
``airgap <-> online`` toggle one-way: the first time you switch to ``airgap``,
that process is blocked forever, even after switching back to ``online``.

So the hook is installed **once, permanently**, and reads the *current mode* on
every call. The hook is the mechanism; the mode is the policy. Verified working
in both directions within a single process.

What is blocked, and what is not
--------------------------------
Blocked to non-loopback destinations: ``socket.connect``, ``socket.sendto``
(UDP egress is otherwise an unguarded hole), ``socket.getaddrinfo``,
``socket.gethostbyname``, ``socket.gethostbyaddr`` (closing DNS-based
exfiltration, which a connect-only guard leaves open).

Allowed: loopback (the server binds and health-checks over ``127.0.0.1``),
``bind``/``listen``/``accept`` (inbound, and the app is a server), and AF_UNIX
connect (local by construction).

Escape hatch
------------
``ANVESHA_AIRGAP_GUARD=0`` disables enforcement. It is for diagnosis only, is
reported by :func:`guard_status`, and is surfaced in ``/healthz`` so a disabled
guard can never be presented as an enforced one.
"""
from __future__ import annotations

import collections
import json
import os
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional

from ..config import CONFIG
from .errors import NetworkBlockedAirgap

Mode = Literal["airgap", "online"]

ENV_MODE = "ANVESHA_MODE"
ENV_DISABLE = "ANVESHA_AIRGAP_GUARD"

VALID_MODES = ("airgap", "online")

# Audit events that constitute outbound network activity.
_NETWORK_EVENTS = frozenset({
    "socket.connect",
    "socket.sendto",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyaddr",
})

# Events whose args are (host_or_addr, ...) rather than (socket, address).
_HOST_FIRST_EVENTS = frozenset({
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyaddr",
})

_LOOPBACK_NAMES = frozenset({
    "localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback", "",
})
_LOOPBACK_ADDRS = frozenset({"127.0.0.1", "::1", "0.0.0.0", "::"})

_BLOCK_LOG_LIMIT = 200

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

_DEFAULT_ACQUIRE: Dict[str, Any] = {
    "primary_catalog": "stac_earth_search",
    "fallback_catalogs": ["stac_cdse"],
    "default_collection": "sentinel-2-l2a",
    "sar_collection": "sentinel-1-grd",
    "lookback_days": 90,
    "max_cloud_pct": 20,
    "cache_gb": 0.5,
    "window_px": 1024,
    "confirm_before_fetch": True,
    "probe_host": "1.1.1.1",
    "probe_port": 443,
}

_DEFAULT_SETTINGS: Dict[str, Any] = {
    # Air-gap is the safe default: the shipping claim is that the offline core
    # works with zero network, and online is one explicit click away.
    "mode": "airgap",
    "acquire": dict(_DEFAULT_ACQUIRE),
}


def settings_path() -> Path:
    return CONFIG.data_dir / "settings.json"


def _read_settings_file() -> Dict[str, Any]:
    """Tolerant read: a missing or corrupt file yields defaults, never a raise."""
    p = settings_path()
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _merged(raw: Dict[str, Any]) -> Dict[str, Any]:
    out = {"mode": _DEFAULT_SETTINGS["mode"], "acquire": dict(_DEFAULT_ACQUIRE)}
    mode = raw.get("mode")
    if isinstance(mode, str) and mode.strip().lower() in VALID_MODES:
        out["mode"] = mode.strip().lower()
    acq = raw.get("acquire")
    if isinstance(acq, dict):
        out["acquire"].update({k: v for k, v in acq.items() if v is not None})
    return out


def load_settings(refresh: bool = False) -> Dict[str, Any]:
    """Return the merged settings dict (cached unless ``refresh``)."""
    global _settings
    if _settings is not None and not refresh:
        return dict(_settings)
    with _lock:
        if _settings is None or refresh:
            _settings = _merged(_read_settings_file())
        return dict(_settings)


def save_settings(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``patch`` into settings and persist atomically (tmp + os.replace)."""
    global _settings
    with _lock:
        current = _settings if _settings is not None else _merged(_read_settings_file())
        merged: Dict[str, Any] = {
            "mode": current.get("mode", _DEFAULT_SETTINGS["mode"]),
            "acquire": dict(_DEFAULT_ACQUIRE),
        }
        acq = current.get("acquire")
        if isinstance(acq, dict):
            merged["acquire"].update(acq)
        for key, value in (patch or {}).items():
            if key == "acquire" and isinstance(value, dict):
                merged["acquire"].update(value)
            elif key == "mode":
                m = str(value).strip().lower()
                if m not in VALID_MODES:
                    raise ValueError(
                        f"Invalid mode {value!r}; expected one of {VALID_MODES}.")
                merged["mode"] = m
            else:
                merged[key] = value
        _settings = merged
        p = settings_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
            os.replace(tmp, p)
        except Exception:
            # Persistence failure must not take down a running server; the
            # in-memory setting still applies for this process.
            pass
        return dict(merged)


# --------------------------------------------------------------------------- #
# Module state
# --------------------------------------------------------------------------- #

_lock = threading.RLock()
_tls = threading.local()
_settings: Optional[Dict[str, Any]] = None

# Policy read on the hot path: a plain module-level str assignment is atomic in
# CPython, so current_mode() needs no lock.
_mode: Optional[str] = None

_hook_installed = False
_enforced_since: Optional[str] = None
_blocked: Deque[Dict[str, Any]] = collections.deque(maxlen=_BLOCK_LOG_LIMIT)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def mode_from_env() -> Optional[Mode]:
    raw = os.environ.get(ENV_MODE, "").strip().lower()
    return raw if raw in VALID_MODES else None  # type: ignore[return-value]


def guard_disabled_by_env() -> bool:
    """True only when ANVESHA_AIRGAP_GUARD is explicitly "0"."""
    return os.environ.get(ENV_DISABLE, "1").strip() == "0"


# --------------------------------------------------------------------------- #
# Mode
# --------------------------------------------------------------------------- #

def current_mode() -> Mode:
    """Active mode. Lock-free, never raises."""
    global _mode
    m = _mode
    if m is not None:
        return m  # type: ignore[return-value]
    env = mode_from_env()
    if env is not None:
        _mode = env
        return env
    _mode = load_settings().get("mode", "airgap")
    return _mode  # type: ignore[return-value]


def _apply_mode(mode: str, *, persist: bool) -> Mode:
    """Set the active mode in this process, optionally writing it to disk."""
    global _mode
    _mode = mode
    if persist:
        save_settings({"mode": mode})
    # The guard is installed regardless of mode: because its policy is dynamic,
    # keeping it resident means switching back to airgap is enforced instantly.
    install_network_guard()
    return mode  # type: ignore[return-value]


def set_mode(mode: str) -> Mode:
    """Validate, persist and apply a mode. Idempotent and safe to call often.

    This is the *deliberate* path -- the UI toggle and ``POST /api/mode``.
    """
    m = str(mode).strip().lower()
    if m not in VALID_MODES:
        raise ValueError(f"Invalid mode {mode!r}; expected one of {VALID_MODES}.")
    return _apply_mode(m, persist=True)


# --------------------------------------------------------------------------- #
# The guard
# --------------------------------------------------------------------------- #

def _is_allowed_host(host: Any) -> bool:
    """True when an address is local and therefore permitted while air-gapped."""
    if host is None:
        return True
    h = str(host).strip().lower()
    if not h or h in _LOOPBACK_NAMES or h in _LOOPBACK_ADDRS:
        return True
    if h.startswith("127."):
        return True
    if h.startswith("::ffff:127."):      # IPv4-mapped IPv6 loopback
        return True
    return False


def _target_of(event: str, args: tuple) -> Optional[str]:
    """Return a blocked target description, or None to allow the operation."""
    if event in _HOST_FIRST_EVENTS:
        host = args[0] if args else None
        return None if _is_allowed_host(host) else str(host)

    # socket.connect / socket.sendto -> (socket_obj, address)
    addr = args[1] if len(args) > 1 else None
    if isinstance(addr, str):
        # AF_UNIX path: local by construction.
        return None
    if isinstance(addr, (tuple, list)):
        if not addr:
            return None
        host = addr[0]
        if _is_allowed_host(host):
            return None
        return ":".join(str(x) for x in addr[:2])
    if addr is None:
        return None
    return None if _is_allowed_host(addr) else str(addr)


def _record_block(event: str, target: str) -> None:
    """Append to the bounded in-memory block log. Never raises, never does I/O
    (an audit hook must not itself trigger audited events)."""
    try:
        where = ""
        try:
            frame = sys._getframe(2)
            if frame is not None:
                where = f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
        except Exception:
            where = ""
        _blocked.append({"event": event, "target": target,
                         "at": _now(), "where": where})
    except Exception:
        pass


def _audit(event: str, args: tuple) -> None:
    """The installed hook. Must stay cheap for non-network events."""
    if event not in _NETWORK_EVENTS:
        return                                  # fast path: most events
    if guard_disabled_by_env():
        return
    if current_mode() != "airgap":
        return                                  # POLICY read at call time
    if getattr(_tls, "active", False):
        return                                  # never recurse
    try:
        target = _target_of(event, args)
    except Exception:
        return                                  # a guard defect must not break the app
    if target is None:
        return
    _record_block(event, target)
    raise NetworkBlockedAirgap(
        f"air-gap mode: outbound {event} to {target} was blocked",
        event=event, target=target)


def install_network_guard() -> bool:
    """Install the audit hook once. Returns True only if newly installed.

    Audit hooks cannot be removed, which is the point -- but it also means this
    must be idempotent, and that all blocking decisions must be dynamic.
    """
    global _hook_installed, _enforced_since
    if _hook_installed:
        return False
    with _lock:
        if _hook_installed:
            return False
        sys.addaudithook(_audit)
        _hook_installed = True
        _enforced_since = _now()
    return True


def guard_installed() -> bool:
    return _hook_installed


def enforce() -> None:
    """Public helper for embedders/CLIs: apply settings then guarantee the guard."""
    install_network_guard()
    current_mode()          # materialise the policy


def enforce_from_env() -> None:
    """Child-process / CLI entry point.

    Called from ``anvesha/__init__`` so that a process started with
    ``ANVESHA_MODE=airgap`` (e.g. ``python -m anvesha.evaluate`` spawned by
    the server, or the CI air-gap gate) is covered without every entry point
    having to opt in. A no-op when the env var is absent.

    Deliberately **does not persist**. An environment variable is a per-process
    override, and writing it to ``data/settings.json`` would mean that simply
    running ``ANVESHA_MODE=online pytest`` or a smoke script permanently
    changes the application's stored mode -- silently turning the shipping
    air-gap default into online for every later launch.
    """
    env = mode_from_env()
    if env is None:
        return
    try:
        _apply_mode(env, persist=False)
    except Exception:
        install_network_guard()


# --------------------------------------------------------------------------- #
# Observability
# --------------------------------------------------------------------------- #

def blocked_attempts(drain: bool = True) -> List[Dict[str, Any]]:
    """Blocked outbound attempts, newest last. Drains by default."""
    with _lock:
        items = list(_blocked)
        if drain:
            _blocked.clear()
        return items


def guard_status() -> Dict[str, Any]:
    return {
        "mode": current_mode(),
        "guard_installed": _hook_installed,
        "guard_disabled_by_env": guard_disabled_by_env(),
        "enforced_since": _enforced_since,
        "blocked_count": len(_blocked),
        "network_allowed": current_mode() == "online" and not guard_disabled_by_env(),
        "blocked_events": sorted(_NETWORK_EVENTS),
    }


def connectivity_probe(timeout_s: float = 3.0) -> Dict[str, Any]:
    """Best-effort reachability check. Only meaningful in online mode.

    Never raises: the UI must be able to render honest connectivity state rather
    than guess it client-side.
    """
    result: Dict[str, Any] = {"online": None, "checked_at": _now(), "detail": ""}
    if current_mode() != "online":
        result.update(online=False,
                      detail="air-gap mode: connectivity probing is disabled")
        return result
    acq = load_settings().get("acquire", {})
    host = str(acq.get("probe_host", "1.1.1.1"))
    port = int(acq.get("probe_port", 443))
    started = time.time()
    sock = None
    try:
        sock = socket.create_connection((host, port), timeout=timeout_s)
        result.update(online=True,
                      detail=f"reached {host}:{port} in "
                             f"{int((time.time() - started) * 1000)} ms")
    except Exception as exc:
        result.update(online=False, detail=f"{type(exc).__name__}: {exc}")
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
    return result
