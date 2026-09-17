"""Air-gap enforcement tests (contract invariant I6).

These are *negative* tests: they assert the guard actually raises. A guard that
is merely installed but never fires is indistinguishable from no guard, so
"the suite still passes" is not evidence of anything on its own.

Note on process state: ``sys.addaudithook`` hooks cannot be removed, so once
this module runs, the hook stays installed for the rest of the pytest session.
That is by design -- the *policy* is dynamic, so with mode restored the guard is
inert. The autouse fixture below saves and restores the mode for that reason.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from satquery.acquire import mode as M
from satquery.acquire.errors import (AcquireError, NetworkBlockedAirgap)

ROOT = Path(__file__).resolve().parents[1]

# TEST-NET-1 (RFC 5737): guaranteed non-routable, so a connect attempt can never
# succeed against a real service. That lets us assert on the *exception type*
# and distinguish "blocked by our guard" from "ordinary network failure"
# without depending on the machine having internet access.
NONROUTABLE = "192.0.2.1"


@pytest.fixture(autouse=True)
def _restore_mode():
    """Save/restore mode so this module never leaves global state changed."""
    saved = M.current_mode()
    try:
        yield
    finally:
        M._mode = saved          # bypass persistence; restore in-process state
        M.blocked_attempts(drain=True)


def _try_connect(host: str, port: int = 443, timeout: float = 2.0):
    """Attempt a TCP connection; return the raised exception or None on success."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return None
    except Exception as exc:                       # noqa: BLE001 - we inspect it
        return exc


# --------------------------------------------------------------------------- #
# The guard genuinely blocks
# --------------------------------------------------------------------------- #

def test_airgap_blocks_outbound_connect():
    M.set_mode("airgap")
    exc = _try_connect(NONROUTABLE)
    assert isinstance(exc, NetworkBlockedAirgap), f"expected block, got {exc!r}"


def test_blocked_exception_is_oserror_for_graceful_degradation():
    """Libraries treat an unreachable network as recoverable; a blocked call
    must look the same, or model construction dies instead of degrading."""
    assert issubclass(NetworkBlockedAirgap, OSError)
    assert issubclass(NetworkBlockedAirgap, AcquireError)
    M.set_mode("airgap")
    exc = _try_connect(NONROUTABLE)
    assert isinstance(exc, OSError)
    assert isinstance(exc, AcquireError)
    assert getattr(exc, "event", "")
    assert getattr(exc, "target", "")


def test_airgap_blocks_urllib():
    """At the urllib layer the block surfaces as URLError wrapping our exception.

    urllib's do_open() catches ``OSError`` and re-raises it as ``URLError(err)``.
    Because NetworkBlockedAirgap *is* an OSError, urllib wraps it -- so callers
    using urllib must inspect ``.reason``. This is still graceful degradation
    (URLError is itself an OSError) and it is why raw-socket callers see the
    air-gap exception directly while urllib callers do not.
    """
    M.set_mode("airgap")
    with pytest.raises(urllib.error.URLError) as ei:
        urllib.request.urlopen(f"http://{NONROUTABLE}/", timeout=2)
    assert isinstance(ei.value.reason, NetworkBlockedAirgap)
    assert "air-gap mode" in str(ei.value.reason)


def test_airgap_blocks_udp_sendto():
    """UDP egress would otherwise be an unguarded hole."""
    M.set_mode("airgap")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(NetworkBlockedAirgap):
            sock.sendto(b"x", (NONROUTABLE, 53))
    finally:
        sock.close()


def test_airgap_blocks_remote_dns_lookup():
    """Blocking connect alone still permits DNS; close that too."""
    M.set_mode("airgap")
    with pytest.raises(NetworkBlockedAirgap):
        socket.getaddrinfo("example.com", 80)


def test_airgap_allows_localhost_dns_lookup():
    M.set_mode("airgap")
    infos = socket.getaddrinfo("localhost", 80)
    assert infos


# --------------------------------------------------------------------------- #
# ...but does not break the things the app depends on
# --------------------------------------------------------------------------- #

def test_loopback_connect_still_works_in_airgap():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        M.set_mode("airgap")
        exc = _try_connect("127.0.0.1", port)
        assert exc is None, f"loopback must stay reachable, got {exc!r}"
    finally:
        server.close()


def test_bind_listen_accept_allowed_in_airgap():
    """The app is a server: uvicorn must still be able to bind and accept."""
    M.set_mode("airgap")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    accepted = {}

    def _serve():
        try:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            accepted["port"] = server.getsockname()[1]
            conn, _ = server.accept()
            conn.close()
        except Exception as exc:                   # pragma: no cover
            accepted["error"] = exc

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    for _ in range(100):
        if "port" in accepted or "error" in accepted:
            break
        time.sleep(0.02)
    assert "error" not in accepted, accepted.get("error")
    assert "port" in accepted, "server never bound"
    try:
        client = socket.create_connection(("127.0.0.1", accepted["port"]), timeout=2)
        client.close()
    finally:
        server.close()
        t.join(timeout=2)


def test_af_unix_sockets_not_blocked():
    """AF_UNIX is local by construction and must not trip the guard."""
    if not hasattr(socket, "AF_UNIX"):             # pragma: no cover
        pytest.skip("AF_UNIX unavailable on this platform")
    M.set_mode("airgap")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        # Nothing is listening, so this fails -- but it must fail with a
        # filesystem error, NOT with our air-gap block.
        with pytest.raises(Exception) as ei:
            sock.connect("/nonexistent/anvesha-test.sock")
        assert not isinstance(ei.value, NetworkBlockedAirgap)
    finally:
        sock.close()


# --------------------------------------------------------------------------- #
# The toggle must be two-way despite an unremovable hook
# --------------------------------------------------------------------------- #

def test_mode_toggle_is_two_way_in_one_process():
    assert M.install_network_guard() in (True, False)   # install or already there
    M.set_mode("airgap")
    assert isinstance(_try_connect(NONROUTABLE), NetworkBlockedAirgap)

    M.set_mode("online")
    online_exc = _try_connect(NONROUTABLE)
    assert not isinstance(online_exc, NetworkBlockedAirgap), \
        "switching back to online must restore egress"

    M.set_mode("airgap")
    assert isinstance(_try_connect(NONROUTABLE), NetworkBlockedAirgap)


def test_install_network_guard_is_idempotent():
    first = M.install_network_guard()
    second = M.install_network_guard()
    assert M.guard_installed() is True
    assert second is False                       # never double-installs
    assert first in (True, False)


# --------------------------------------------------------------------------- #
# Observability -- the "prove it blocked" surface
# --------------------------------------------------------------------------- #

def test_blocked_attempts_are_recorded_and_drained():
    M.set_mode("airgap")
    M.blocked_attempts(drain=True)
    _try_connect(NONROUTABLE)
    items = M.blocked_attempts(drain=True)
    assert items, "a blocked attempt must be recorded"
    assert items[-1]["event"].startswith("socket.")
    assert NONROUTABLE in items[-1]["target"]
    assert M.blocked_attempts(drain=True) == []   # drained


def test_guard_status_reports_honestly():
    M.set_mode("airgap")
    st = M.guard_status()
    assert st["mode"] == "airgap"
    assert st["guard_installed"] is True
    assert st["network_allowed"] is False
    M.set_mode("online")
    assert M.guard_status()["network_allowed"] is True


def test_connectivity_probe_inert_in_airgap():
    """Probing without network would be a lie; it must short-circuit."""
    M.set_mode("airgap")
    res = M.connectivity_probe(timeout_s=1.0)
    assert res["online"] is False
    assert "air-gap" in res["detail"]


def test_escape_hatch_disables_enforcement(monkeypatch):
    M.set_mode("airgap")
    monkeypatch.setenv(M.ENV_DISABLE, "0")
    exc = _try_connect(NONROUTABLE)
    assert not isinstance(exc, NetworkBlockedAirgap)
    # ...and a disabled guard must be visible, never silent
    st = M.guard_status()
    assert st["guard_disabled_by_env"] is True
    monkeypatch.delenv(M.ENV_DISABLE, raising=False)
    assert M.guard_status()["guard_disabled_by_env"] is False


# --------------------------------------------------------------------------- #
# L2: child processes must inherit the guarantee
# --------------------------------------------------------------------------- #

_CHILD = (
    "import socket, sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "import satquery\n"
    "try:\n"
    "    socket.create_connection(('192.0.2.1', 443), timeout=2)\n"
    "    print('REACHED')\n"
    "except Exception as exc:\n"
    "    print('BLOCKED' if type(exc).__name__ == 'NetworkBlockedAirgap'\n"
    "          else 'OTHER:' + type(exc).__name__)\n"
)


def test_child_process_inherits_airgap_mode():
    """The server spawns `python -m satquery.evaluate`; without env inheritance
    an 'air-gapped' deployment could still download in a child."""
    env = dict(os.environ)
    env["SATQUERY_MODE"] = "airgap"
    proc = subprocess.run([sys.executable, "-c", _CHILD, str(ROOT)],
                          capture_output=True, text=True, env=env, timeout=180)
    assert "BLOCKED" in proc.stdout, \
        f"child was not blocked.\nstdout={proc.stdout}\nstderr={proc.stderr[-800:]}"


def test_child_process_online_mode_not_blocked():
    env = dict(os.environ)
    env["SATQUERY_MODE"] = "online"
    proc = subprocess.run([sys.executable, "-c", _CHILD, str(ROOT)],
                          capture_output=True, text=True, env=env, timeout=180)
    assert "BLOCKED" not in proc.stdout


# --------------------------------------------------------------------------- #
# Settings / mode state
# --------------------------------------------------------------------------- #

def test_settings_load_tolerates_missing_and_corrupt(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "settings_path", lambda: tmp_path / "settings.json")
    assert M.load_settings(refresh=True)["mode"] == "airgap"      # missing -> default
    (tmp_path / "settings.json").write_text("{not json", encoding="utf-8")
    assert M.load_settings(refresh=True)["mode"] == "airgap"      # corrupt -> default


def test_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "settings_path", lambda: tmp_path / "settings.json")
    M.save_settings({"mode": "online", "acquire": {"max_cloud_pct": 5}})
    assert (tmp_path / "settings.json").exists()
    fresh = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert fresh["mode"] == "online"
    assert fresh["acquire"]["max_cloud_pct"] == 5
    # unset keys keep their defaults
    assert fresh["acquire"]["default_collection"] == "sentinel-2-l2a"


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        M.set_mode("bogus")
    with pytest.raises(ValueError):
        M.save_settings({"mode": "bogus"})


def test_mode_from_env():
    os.environ["SATQUERY_MODE"] = "online"
    try:
        assert M.mode_from_env() == "online"
    finally:
        os.environ.pop("SATQUERY_MODE", None)
    assert M.mode_from_env() is None


# --------------------------------------------------------------------------- #
# Anti-vacuity: the CI gate must not be able to pass with an inert guard
# --------------------------------------------------------------------------- #

_GATE_PROBE = (
    "import sys, socket\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "import satquery\n"
    "from satquery.acquire import mode as M\n"
    "print('MODE=' + str(M.current_mode()))\n"
    "print('GUARD=' + str(M.guard_installed()))\n"
    "try:\n"
    "    socket.getaddrinfo('example.com', 80)\n"
    "    print('DNS=RESOLVED')\n"
    "except Exception as exc:\n"
    "    print('DNS=' + type(exc).__name__)\n"
)


def test_env_gate_is_not_vacuous():
    """`SATQUERY_MODE=airgap pytest ...` only proves anything if it enforces.

    A green suite with a guard that never fires looks identical to a green suite
    with no guard at all, so this test asserts that the env-var gate is actually
    live from package import. Without it, the headline claim ("the entire suite
    passes with the network cut") could silently become vacuous.
    """
    env = dict(os.environ)
    env["SATQUERY_MODE"] = "airgap"
    proc = subprocess.run([sys.executable, "-c", _GATE_PROBE, str(ROOT)],
                          capture_output=True, text=True, env=env, timeout=180)
    assert "MODE=airgap" in proc.stdout, proc.stdout
    assert "GUARD=True" in proc.stdout, \
        f"guard not installed from package import.\n{proc.stdout}\n{proc.stderr[-500:]}"
    assert "DNS=NetworkBlockedAirgap" in proc.stdout, \
        f"gate is vacuous - guard installed but not enforcing.\n{proc.stdout}"


def test_no_env_var_leaves_package_import_inert():
    """Importing satquery must not install a guard on its own.

    Scripts such as `scripts/download_datasets.py` legitimately need the
    network; only an explicit mode (env var, server startup, or enforce())
    should arm the guard.
    """
    env = dict(os.environ)
    env.pop("SATQUERY_MODE", None)
    proc = subprocess.run([sys.executable, "-c", _GATE_PROBE, str(ROOT)],
                          capture_output=True, text=True, env=env, timeout=180)
    assert "GUARD=False" in proc.stdout, proc.stdout
    assert "DNS=RESOLVED" in proc.stdout or "DNS=socket.gaierror" in proc.stdout
