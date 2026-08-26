#!/usr/bin/env python3
"""One-command launcher for Anvesha (SatQuery AI).

What it does
------------
1. Terminates any previous instance of this server — matched by its command
   line (any port) plus anything still listening on the target port.
2. Starts the FastAPI service, which serves both the REST API and the
   pre-built React console from web/dist.
3. Opens your default browser as soon as the server accepts connections.
4. Shuts down cleanly on Ctrl+C.

Usage
-----
    python start.py                     # http://127.0.0.1:8000
    python start.py --port 8080
    python start.py --host 0.0.0.0      # expose on the LAN
    python start.py --no-browser

Environment overrides: SATQUERY_PORT, SATQUERY_HOST.
Stdlib-only (no psutil needed); works on Windows, Linux and macOS.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
APP_MARKER = "satquery.server.main"


def _match_clause_windows() -> str:
    """CommandLine predicates identifying this app's processes: direct
    uvicorn launches (module marker) plus anything running this launcher."""
    return ("$_.CommandLine -like '*satquery.server.main*' -or "
            "($_.CommandLine -like '*start.py*' -and "
            "$_.CommandLine -like '*SatQuery*')")


# --------------------------------------------------------------------------- #
# Stale-instance discovery & cleanup
# --------------------------------------------------------------------------- #

def _pids_matching_windows() -> set[int]:
    """PIDs whose command line references this app."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"Get-CimInstance Win32_Process | Where-Object {{ {_match_clause_windows()} }}"
         " | Select-Object -ExpandProperty ProcessId"],
        capture_output=True, text=True).stdout
    return {int(tok) for tok in out.split() if tok.isdigit()}


def _pids_on_port_windows(port: int) -> set[int]:
    out = subprocess.run(["netstat", "-ano"], capture_output=True,
                         text=True).stdout
    pids = set()
    for line in out.splitlines():
        parts = line.split()
        # TCP  127.0.0.1:8000  0.0.0.0:0  LISTENING  1234
        if len(parts) == 5 and parts[0].upper() == "TCP" \
                and parts[1].rsplit(":", 1)[-1] == str(port) \
                and parts[3].upper().startswith("LISTEN"):
            if parts[4].isdigit():
                pids.add(int(parts[4]))
    return pids


def _pids_matching_unix() -> set[int]:
    out = subprocess.run(["pgrep", "-f",
                          "satquery.server.main|SatQuery.*start\\.py"],
                         capture_output=True, text=True).stdout
    return {int(tok) for tok in out.split() if tok.isdigit()}


def _pids_on_port_unix(port: int) -> set[int]:
    out = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                         capture_output=True, text=True).stdout
    return {int(tok) for tok in out.split() if tok.isdigit()}


def _terminate(pids: set[int]) -> None:
    me = os.getpid()
    for pid in sorted(pids):
        if pid == me:
            continue
        print(f"  stopping previous instance pid={pid}")
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True)
        else:
            subprocess.run(["kill", "-9", str(pid)], capture_output=True)


def _port_listeners(port: int) -> set[int]:
    if os.name == "nt":
        return _pids_on_port_windows(port)
    return _pids_on_port_unix(port)


def stop_previous_instances(port: int, timeout_s: float = 8.0) -> None:
    """Kill stale app instances (any port) + anything squatting on `port`,
    then wait until the port is actually released before rebinding."""
    if os.name == "nt":
        pids = _pids_matching_windows() | _pids_on_port_windows(port)
    else:
        pids = _pids_matching_unix() | _pids_on_port_unix(port)
    if not pids:
        print("no previous instances found")
        return
    print(f"found {len(pids)} previous server process(es)")
    _terminate(pids)
    deadline = time.time() + timeout_s
    while time.time() < deadline and _port_listeners(port):
        time.sleep(0.25)
    time.sleep(0.5)          # small grace period for socket teardown


# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #

def _open_browser_when_ready(server, url: str) -> None:
    while not server.started and not server.should_exit:
        time.sleep(0.1)
    if server.started:
        print(f"opening {url} in your browser...")
        webbrowser.open(url)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Launch the Anvesha EO Investigation System "
                    "(kills stale instances first, serves API + console).")
    ap.add_argument("--host",
                    default=os.environ.get("SATQUERY_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("SATQUERY_PORT", "8000")))
    ap.add_argument("--no-browser", action="store_true",
                    help="do not auto-open the web console")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)

    print("=" * 64)
    print("Anvesha - Earth Observation & Investigation System")
    print("=" * 64)
    stop_previous_instances(args.port)

    shown_host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") \
        else args.host
    url = f"http://{shown_host}:{args.port}"

    import uvicorn
    config = uvicorn.Config("satquery.server.main:app", host=args.host,
                            port=args.port, log_level="info")
    server = uvicorn.Server(config)
    if not args.no_browser:
        threading.Thread(target=_open_browser_when_ready,
                         args=(server, url), daemon=True).start()

    print(f"starting server on {url}  (Ctrl+C to stop)")
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nserver stopped cleanly.")


if __name__ == "__main__":
    main()
