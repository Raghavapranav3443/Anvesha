#!/usr/bin/env python3
"""One-command launcher for Anvesha (SatQuery AI).

What it does
------------
1. Verifies Python version and platform compatibility.
2. Checks that all Python dependencies are importable; offers to install
   missing ones from requirements.txt.
3. Verifies Node.js toolchain and frontend dependencies (node_modules).
4. Checks that model weights and demo samples are present; warns with
   actionable recovery steps when anything is missing.
5. Confirms the frontend build is fresh (not stale against web/src).
6. Terminates any previous instance of this server.
7. Starts the FastAPI service, which serves both the REST API and the
   pre-built React console from web/dist.
8. Opens your default browser as soon as the server accepts connections.
9. Shuts down cleanly on Ctrl+C.

Usage
-----
    python start.py                     # http://127.0.0.1:8000
    python start.py --port 8080
    python start.py --host 0.0.0.0      # expose on the LAN
    python start.py --no-browser
    python start.py --strict            # abort on any missing dependency
    python start.py --skip-deps         # skip dependency checks (faster restart)

Environment overrides: SATQUERY_PORT, SATQUERY_HOST, SATQUERY_TOKEN.
Stdlib-only checks; works on Windows, Linux and macOS.
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
APP_MARKER = "satquery.server.main"

# --------------------------------------------------------------------------- #
# Colour helpers (no dependency — ANSI codes, safe on Windows 10+)
# --------------------------------------------------------------------------- #
_USE_COLOR = True
if os.name == "nt":
    # Enable ANSI escape processing on Windows 10+
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        _USE_COLOR = False

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text
def _ok(text: str) -> str:   return _c("32", text)   # green
def _warn(text: str) -> str: return _c("33", text)  # yellow
def _err(text: str) -> str:  return _c("31", text)  # red
def _bold(text: str) -> str: return _c("1", text)   # bold


# --------------------------------------------------------------------------- #
# 1. Python version check
# --------------------------------------------------------------------------- #
def check_python_version(min_major: int = 3, min_minor: int = 8) -> None:
    """Ensure the Python interpreter meets the minimum version."""
    ver = sys.version_info
    if ver < (min_major, min_minor):
        sys.exit(
            _err(f"Python {min_major}.{min_minor}+ required — "
                 f"you are running {ver.major}.{ver.minor}.{ver.micro}.\n"
                 f"  Upgrade: https://www.python.org/downloads/")
        )
    print(_ok(f"  Python {ver.major}.{ver.minor}.{ver.micro}"))


# --------------------------------------------------------------------------- #
# 2. Python dependency check
# --------------------------------------------------------------------------- #
_REQUIREMENTS = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("rasterio", "rasterio"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("tqdm", "tqdm"),
    ("requests", "requests"),
    ("pytest", "pytest"),
    ("huggingface_hub", "huggingface_hub"),
    ("datasets", "datasets"),
    ("gdown", "gdown"),
    ("pyarrow", "pyarrow"),
]

def _missing_imports() -> list[tuple[str, str]]:
    """Return list of (pip_name, import_name) for packages that fail to import."""
    missing = []
    for import_name, pip_name in _REQUIREMENTS:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append((pip_name, import_name))
    return missing


def check_python_deps(auto_install: bool = False, strict: bool = False) -> None:
    """Verify all Python dependencies are importable."""
    missing = _missing_imports()
    if not missing:
        print(_ok("  Python dependencies"))
        return

    pkgs = ", ".join(p for p, _ in missing)
    print(_warn(f"  Missing Python packages: {pkgs}"))

    if auto_install or input("  Install missing packages? [Y/n] ").strip().lower() in ("", "y", "yes"):
        req_file = REPO_ROOT / "requirements.txt"
        print("  Installing from requirements.txt ...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
            cwd=str(REPO_ROOT),
        )
        if result.returncode != 0:
            sys.exit(_err("  pip install failed — resolve manually and retry."))
        still_missing = _missing_imports()
        if still_missing:
            pkgs = ", ".join(p for p, _ in still_missing)
            msg = f"  Still missing after install: {pkgs}"
            if strict:
                sys.exit(_err(msg))
            print(_warn(msg))
            return
        print(_ok("  Python dependencies installed"))
    else:
        msg = "  Missing Python dependencies — install with: pip install -r requirements.txt"
        if strict:
            sys.exit(_err(msg))
        print(_warn(msg))


# --------------------------------------------------------------------------- #
# 3. Node.js toolchain check
# --------------------------------------------------------------------------- #
def _find_node() -> str | None:
    """Return path to node executable, or None if not found."""
    for name in ("node", "node.exe"):
        path = shutil.which(name)
        if path:
            return path
    return None


def check_nodejs(strict: bool = False) -> None:
    """Verify Node.js and npm are available."""
    node = _find_node()
    if not node:
        msg = (
            "Node.js not found — frontend build requires it.\n"
            "  Install: https://nodejs.org/  (LTS recommended)"
        )
        if strict:
            sys.exit(_err(msg))
        print(_warn(msg))
        return

    try:
        ver = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
        print(_ok(f"  Node.js {ver}"))
    except Exception:
        print(_warn("  Node.js found but version check failed"))


def check_frontend_deps(strict: bool = False) -> None:
    """Verify node_modules exists for the frontend."""
    node_modules = REPO_ROOT / "web" / "node_modules"
    if node_modules.exists():
        print(_ok("  Frontend dependencies (node_modules)"))
        return

    msg = (
        "web/node_modules not found — frontend cannot build.\n"
        "  Install:  cd web && npm ci"
    )
    if strict:
        sys.exit(_err(msg))
    print(_warn(msg))


# --------------------------------------------------------------------------- #
# 4. Model weights check
# --------------------------------------------------------------------------- #
_CRITICAL_WEIGHTS = [
    ("scene_encoder.pt", "Scene encoder — land-cover classification"),
    ("vqa_head.pt", "VQA head — visual question answering"),
    ("change_net.pt", "Change detector — bitemporal change analysis"),
    ("optical_sar_fusion.pt", "Optical-SAR fusion — multimodal analysis"),
]

_OPTIONAL_WEIGHTS = [
    ("captioner.pt", "Captioner — learned scene descriptions"),
    ("type_heads.pt", "Type heads — per-class specialist heads"),
    ("count_head.pt", "Counting head — object counting"),
    ("cdvqa_head.pt", "CDVQA head — change detection VQA"),
    ("task_centroids.pt", "Task centroids — intent classification"),
]


def check_weights(strict: bool = False) -> None:
    """Verify model weights exist; warn if missing."""
    weights_dir = REPO_ROOT / "weights"
    if not weights_dir.exists():
        msg = "weights/ directory not found — all specialists will use heuristic fallbacks."
        if strict:
            sys.exit(_err(msg))
        print(_warn(msg))
        return

    missing_critical = []
    missing_optional = []
    for fname, desc in _CRITICAL_WEIGHTS:
        if not (weights_dir / fname).exists():
            missing_critical.append((fname, desc))
    for fname, desc in _OPTIONAL_WEIGHTS:
        if not (weights_dir / fname).exists():
            missing_optional.append((fname, desc))

    if missing_critical:
        items = "\n".join(f"    - {f} ({d})" for f, d in missing_critical)
        msg = (
            f"Missing critical weights:\n{items}\n"
            f"  Download from the project release page or run training scripts."
        )
        if strict:
            sys.exit(_err(msg))
        print(_warn(msg))

    if missing_optional:
        items = ", ".join(f for f, _ in missing_optional)
        print(_warn(f"  Optional weights missing (heuristic fallback): {items}"))

    if not missing_critical and not missing_optional:
        print(_ok("  Model weights"))


# --------------------------------------------------------------------------- #
# 5. Demo samples check
# --------------------------------------------------------------------------- #
def check_samples(strict: bool = False) -> None:
    """Verify demo samples exist for the quick-start experience."""
    samples_dir = REPO_ROOT / "samples"
    if not samples_dir.exists():
        msg = "samples/ directory not found — demo mode will have no sample images."
        if strict:
            sys.exit(_err(msg))
        print(_warn(msg))
        return

    samples = list(samples_dir.glob("demo_*"))
    if not samples:
        msg = (
            "No demo samples found in samples/.\n"
            "  Export:  python scripts/export_demo_samples.py"
        )
        if strict:
            sys.exit(_err(msg))
        print(_warn(msg))
        return

    print(_ok(f"  Demo samples ({len(samples)} files)"))


# --------------------------------------------------------------------------- #
# 6. Port availability check
# --------------------------------------------------------------------------- #
def check_port_available(port: int) -> bool:
    """Return True if the port is free, False if something is already listening."""
    if os.name == "nt":
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            parts = line.split()
            if (len(parts) == 5 and parts[0].upper() == "TCP"
                    and parts[1].rsplit(":", 1)[-1] == str(port)
                    and parts[3].upper().startswith("LISTEN")):
                return False
    else:
        out = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                             capture_output=True, text=True).stdout
        if out.strip():
            return False
    return True


# --------------------------------------------------------------------------- #
# Stale-instance discovery & cleanup
# --------------------------------------------------------------------------- #


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


def _pids_matching_windows() -> set[int]:
    """PIDs whose command line references this app (Windows)."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process | Where-Object { "
         "$_.CommandLine -like '*satquery.server.main*' -or "
         "($_.CommandLine -like '*start.py*' -and "
         "$_.CommandLine -like '*SatQuery*') } "
         "| Select-Object -ExpandProperty ProcessId"],
        capture_output=True, text=True).stdout
    return {int(tok) for tok in out.split() if tok.isdigit()}


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


def _check_frontend_build(strict: bool = False) -> None:
    """Frontend-backend drift guard.

    The committed web/dist can silently drift from web/src (it has before:
    stale hashed chunks). If the SPA build is missing, or older than the
    newest frontend source file, warn loudly — or abort with --strict so a
    demo never starts against a stale console.
    """
    dist = REPO_ROOT / "web" / "dist" / "index.html"
    src = REPO_ROOT / "web" / "src"
    if not dist.exists():
        msg = ("web/dist/index.html not found — the web console cannot be "
               "served. Build it with:  cd web && npm ci && npm run build")
        if strict:
            sys.exit(_err(msg))
        print(_warn(msg))
        return
    if src.exists():
        newest_src = max(p.stat().st_mtime for p in src.rglob("*")
                         if p.is_file())
        if newest_src > dist.stat().st_mtime:
            msg = ("web/dist is OLDER than web/src — the console may be "
                   "stale. Rebuild with:  cd web && npm ci && npm run build")
            if strict:
                sys.exit(_err(msg))
            print(_warn(msg))
    print(_ok("  Frontend build"))


def _warn_auth_posture(host: str) -> None:
    """Unauthenticated + LAN-exposed = anyone on the venue network can hit
    the API. Remind the operator before the demo starts."""
    if not os.environ.get("SATQUERY_TOKEN", "").strip() \
            and host not in ("127.0.0.1", "localhost"):
        print(_warn("  SATQUERY_TOKEN is not set while binding to a "
                     "non-localhost host — the API is open to the whole network. "
                     "Set SATQUERY_TOKEN for the demo venue."))


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
    ap.add_argument("--strict", action="store_true",
                    help="abort instead of warning when dependencies, weights, "
                         "samples, or frontend build are missing/stale")
    ap.add_argument("--skip-deps", action="store_true",
                    help="skip dependency checks (faster restart)")
    ap.add_argument("--auto-install", action="store_true",
                    help="automatically install missing Python packages without prompting")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)

    print(_bold("=" * 64))
    print(_bold("Anvesha — Earth Observation & Investigation System"))
    print(_bold("=" * 64))

    # --- Onboarding checks ----------------------------------------------- #
    if not args.skip_deps:
        print(_bold("\n[1/6] Python version"))
        check_python_version()

        print(_bold("\n[2/6] Python dependencies"))
        check_python_deps(auto_install=args.auto_install, strict=args.strict)

        print(_bold("\n[3/6] Node.js toolchain & frontend dependencies"))
        check_nodejs(strict=args.strict)
        check_frontend_deps(strict=args.strict)

        print(_bold("\n[4/6] Model weights"))
        check_weights(strict=args.strict)

        print(_bold("\n[5/6] Demo samples"))
        check_samples(strict=args.strict)

        print(_bold("\n[6/6] Frontend build"))
        _check_frontend_build(strict=args.strict)
    else:
        print(_warn("\n  (--skip-deps: dependency checks skipped)"))

    # --- Port check ------------------------------------------------------ #
    if not check_port_available(args.port):
        print(_warn(f"\n  Port {args.port} is already in use — "
                     "attempting to stop previous instances..."))

    # --- Auth warning ---------------------------------------------------- #
    _warn_auth_posture(args.host)

    # --- Stop stale instances -------------------------------------------- #
    stop_previous_instances(args.port)

    # --- Start server ---------------------------------------------------- #
    shown_host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") \
        else args.host
    url = f"http://{shown_host}:{args.port}"

    print(_bold(f"\n  Starting server on {url}  (Ctrl+C to stop)\n"))
    try:
        import uvicorn
    except ImportError:
        sys.exit(_err(
            "uvicorn is not installed — cannot start the server.\n"
            "  Install:  pip install uvicorn\n"
            "  Or:       pip install -r requirements.txt"
        ))

    try:
        config = uvicorn.Config("satquery.server.main:app", host=args.host,
                                port=args.port, log_level="info")
        server = uvicorn.Server(config)
    except ImportError as e:
        sys.exit(_err(
            f"Failed to import the SatQuery app: {e}\n"
            "  Ensure the project structure is intact and you are at the repo root."
        ))

    if not args.no_browser:
        threading.Thread(target=_open_browser_when_ready,
                         args=(server, url), daemon=True).start()

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.exit(_err(f"\nServer crashed: {e}"))
    finally:
        print(_bold("\n  Server stopped cleanly."))


if __name__ == "__main__":
    main()
