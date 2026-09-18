"""FastAPI service exposing the Anvesha AI agent.

Run:  python -m uvicorn anvesha.server.main:app --port 8000
Serves the React build from web/dist when present.

Environment:
  ANVESHA_WORKERS   job worker threads (default 4)
  ANVESHA_THREADS   torch intra-op threads (default 4)
  ANVESHA_TOKEN     optional bearer token; when set, /api/* requires it
  ANVESHA_CACHE_TTL result cache TTL in days (default 7)
  ANVESHA_DEBUG     set to "1" to include tracebacks in 500 responses
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..config import CONFIG
from ..io_utils import InputValidationError, load_image
from .jobs import (EVIDENCE_MAX_PX, MAX_UPLOAD_BYTES, TORCH_THREADS, WORKERS,
                   JobStore, cache_key_for, gpu_semaphore, log_event)
from ..store import Store
from .. import __version__ as _VERSION

logger = logging.getLogger(__name__)

_DEBUG = os.environ.get("ANVESHA_DEBUG", "").strip() == "1"

# ---------------------------------------------------------------------------
# Structured error helper
# ---------------------------------------------------------------------------

class StructuredHTTPException(Exception):
    """Custom exception that carries a structured JSON payload.

    Using a dedicated exception (not HTTPException) avoids FastAPI's
    automatic ``{"detail": ...}`` wrapping so we get a clean top-level
    ``{detail, code, hint}`` shape in every error response.
    """
    def __init__(self, status: int, detail: str,
                 code: str = "error", hint: str = "") -> None:
        self.status = status
        self.detail = detail
        self.code   = code
        self.hint   = hint


def _error_body(detail: str, code: str = "error", hint: str = "") -> Dict[str, str]:
    """Return a consistent JSON error payload."""
    body: Dict[str, str] = {"detail": detail, "code": code}
    if hint:
        body["hint"] = hint
    return body


def _http(status: int, detail: str, code: str = "error", hint: str = "") -> "StructuredHTTPException":
    """Raise a structured HTTP error — never double-wrapped by FastAPI."""
    raise StructuredHTTPException(status, detail, code, hint)


# ---------------------------------------------------------------------------
# Startup cleanup
# ---------------------------------------------------------------------------

def _startup_cleanup() -> None:
    """Bound disk growth: stale uploads + purge the SQLite result cache."""
    _clean_uploads()
    try:
        STORE.purge_stale()
    except Exception:
        pass


def _clean_uploads() -> None:
    """Delete uploaded files older than 24 h."""
    updir = CONFIG.runs_dir / "_uploads"
    if not updir.exists():
        return
    cutoff = time.time() - 24 * 3600
    removed = 0
    for f in updir.iterdir():
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except Exception:
            pass
    if removed:
        logger.info("Startup cleanup: removed %d stale upload(s).", removed)


def _schedule_periodic_cleanup(interval_s: int = 6 * 3600) -> None:
    """Re-run upload + cache cleanup every *interval_s* seconds (default 6 h)."""
    def _run():
        _clean_uploads()
        try:
            purged = STORE.purge_stale()
            if purged:
                log_event("cleanup", "periodic_purge", rows=purged)
        except Exception:
            pass
        t = threading.Timer(interval_s, _run)
        t.daemon = True
        t.start()

    t = threading.Timer(interval_s, _run)
    t.daemon = True
    t.start()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title=f"Anvesha — Earth Observation & Investigation System v{_VERSION}", version=_VERSION)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.exception_handler(StructuredHTTPException)
async def structured_error_handler(request, exc: StructuredHTTPException):
    """Convert StructuredHTTPException → clean JSON without double-wrapping."""
    return JSONResponse(
        status_code=exc.status,
        content=_error_body(exc.detail, exc.code, exc.hint),
    )

JOBS  = JobStore()
STORE = Store(CONFIG.data_dir / "anvesha.db")
_STARTED    = time.time()
_AUTH_TOKEN = os.environ.get("ANVESHA_TOKEN", "").strip()

ALLOWED_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


# ---------------------------------------------------------------------------
# Air-gap enforcement
# ---------------------------------------------------------------------------
#
# Install the outbound-network guard before serving anything. The guard is a
# permanent audit hook whose *policy* is read from the current mode at call
# time, so it is harmless in `online` mode and becomes an immediate hard block
# the moment the mode is airgap. Installing it here -- rather than lazily on
# first use -- is what makes "this server is air-gapped" a property of the
# process rather than a promise in a README.

def _mode_snapshot() -> Dict[str, Any]:
    """Mode + guard state for /healthz. Never raises."""
    try:
        from ..acquire import mode as _mode
        st = _mode.guard_status()
        if st.get("guard_disabled_by_env"):
            state = "disabled_by_env"
        elif not st.get("guard_installed"):
            state = "not_installed"
        elif st.get("mode") == "airgap":
            state = "enforced"
        else:
            state = "installed-idle"
        return {"mode": st.get("mode"),
                "guard_installed": st.get("guard_installed"),
                "airgap_guard": state,
                "blocked_count": st.get("blocked_count")}
    except Exception:
        return {"mode": "unknown", "guard_installed": False,
                "airgap_guard": "unavailable", "blocked_count": 0}


try:
    from ..acquire import mode as _airgap_mode
    _airgap_mode.enforce()
    if _airgap_mode.guard_disabled_by_env():
        logger.warning(
            "ANVESHA_AIRGAP_GUARD=0 - air-gap enforcement is DISABLED for this "
            "process. /healthz reports airgap_guard=disabled_by_env.")
    else:
        logger.info("Air-gap guard installed (mode=%s).",
                    _airgap_mode.current_mode())
except Exception:                                  # never block startup
    logger.exception("Could not install the air-gap network guard.")


# Run startup cleanup and schedule the periodic repeater
_startup_cleanup()
_schedule_periodic_cleanup()


def _preload_specialists() -> None:
    """Instantiate the core specialists off the request path.

    ``/healthz`` reports which specialists run trained weights rather than
    heuristic fallbacks, which means instantiating all four (~26 s cold even
    on a fast laptop, and a small cloud vCPU is slower). Paying that on the
    first probe marks a healthy container unhealthy; paying it on the first
    user request makes the console look broken. Moving it to startup is where
    the launcher is already waiting for the port to open.
    """
    try:
        from ..models.status import warm
        status = warm()
        trained = sum(1 for v in status.values() if v == "trained")
        logger.info("Specialist preload complete: %d/%d trained.",
                    trained, len(status))
    except Exception:
        logger.exception("Specialist preload failed; models load lazily.")


# Skipped under pytest: the suite imports this module (TestClient) and already
# loads models where it means to, so a background thread would add a second,
# unsequenced set of loads to an otherwise deterministic run.
if os.environ.get("ANVESHA_PRELOAD", "1") != "0" and "pytest" not in sys.modules:
    threading.Thread(target=_preload_specialists, daemon=True,
                     name="preload-specialists").start()

# Security posture warning: an unauthenticated server bound beyond localhost
# is open to anyone on the network. Warn loudly (demo venues included).
if not _AUTH_TOKEN:
    logger.warning(
        "ANVESHA_TOKEN is not set — the API is UNAUTHENTICATED. If this "
        "server is reachable from a shared network (demo venue, LAN), set "
        "ANVESHA_TOKEN to require a bearer token on /api/*.")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _check_auth(authorization: Optional[str] = Header(None)) -> None:
    """Bearer-token gate for /api/* when ANVESHA_TOKEN is configured."""
    if not _AUTH_TOKEN:
        return
    expected = f"Bearer {_AUTH_TOKEN}"
    if authorization is None or not secrets.compare_digest(
            authorization.encode(), expected.encode()):
        _http(401, "Invalid or missing bearer token.", code="unauthorized")


# ---------------------------------------------------------------------------
# Upload helper
# ---------------------------------------------------------------------------

def _save_uploads(files: List[UploadFile]) -> List[Path]:
    if len(files) > 2:
        _http(400, "At most 2 images per analysis.", code="too_many_files")
    updir = CONFIG.runs_dir / "_uploads"
    updir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    try:
        for f in files:
            raw_name = Path(f.filename or "upload").name
            ext = Path(raw_name).suffix.lower()
            if ext not in ALLOWED_EXT:
                _http(
                    400,
                    f"Unsupported format '{ext}' for '{raw_name}'. "
                    f"Use GeoTIFF/TIFF or PNG/JPEG (benchmark datasets).",
                    code="unsupported_format",
                    hint="Accepted extensions: .tif .tiff .png .jpg .jpeg",
                )
            # Stream with a hard cap so a giant part can't exhaust memory
            chunks: List[bytes] = []
            total = 0
            try:
                while True:
                    chunk = f.file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        _http(
                            400,
                            f"'{raw_name}' exceeds the 50 MB limit.",
                            code="file_too_large",
                            hint="Crop or downsample the image before uploading.",
                        )
                    chunks.append(chunk)
            finally:
                f.file.close()

            dest = updir / f"{uuid.uuid4().hex}_{raw_name}"
            dest.write_bytes(b"".join(chunks))
            paths.append(dest)
    except StructuredHTTPException:
        # Clean up any files already written before re-raising (atomic multi-upload)
        for p in paths:
            try:
                p.unlink()
            except Exception:
                pass
        raise
    return paths


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/jobs")
async def create_job(
    query: str = Form(""),
    task_override: str = Form("auto"),
    sample_names: str = Form(""),
    date_a: str = Form("T1"),
    date_b: str = Form("T2"),
    modality: str = Form("auto"),
    acquire_id: str = Form(""),
    files: List[UploadFile] = File(default=[]),
):
    try:
        paths = _save_uploads([f for f in files if f and f.filename])
        for name in [s for s in sample_names.split("|") if s]:
            p = CONFIG.samples_dir / Path(name).name
            if p.exists():
                paths.append(p)
        acquired_context: List[Dict[str, Any]] = []
        if acquire_id:
            # Imagery fetched by the online layer. Paths are rebuilt from the
            # run directory and restricted to GeoTIFFs, so an acquire_id cannot
            # be used to read an arbitrary file. Dates come from the files'
            # own tags, so the report carries real acquisition dates rather
            # than the T1/T2 placeholders.
            from ..acquire.service import (acquired_images, dates_for_images,
                                           provenance_for)

            acquired = acquired_images(acquire_id)
            if not acquired:
                _http(404, "No fetched imagery for that acquisition.",
                      code="acquire_not_found",
                      hint="Fetch the imagery first with /api/acquire/fetch.")
            paths.extend(acquired)
            fetched_a, fetched_b = dates_for_images(acquired)
            if date_a == "T1" and fetched_a:
                date_a = fetched_a
            if date_b == "T2" and fetched_b:
                date_b = fetched_b
            # The ISRO layers that were verified present for this AOI travel with
            # the run, so the decision layer can put the analysis beside the
            # Government of India's own record for the same ground. Read from the
            # stored provenance (which the acquisition wrote), never from the
            # request, so a client cannot inject an official-looking claim.
            record = provenance_for(acquire_id) or {}
            stored = record.get("isro_context")
            if isinstance(stored, list):
                acquired_context = [item for item in stored
                                    if isinstance(item, dict)]
        if not paths:
            _http(400, "No images supplied.",
                  code="no_images",
                  hint="Upload at least one GeoTIFF, select a sample image, or "
                       "fetch imagery for a place in online mode.")
        if len(paths) > 2:
            _http(400, "At most 2 images per analysis.", code="too_many_files")

        modality = modality.strip().lower()
        if modality not in ("auto", "sar", "optical"):
            _http(400, f"Invalid modality '{modality}'.",
                  code="invalid_modality",
                  hint="Use 'auto' (default), 'sar' or 'optical'.")
        params = {"date_a": date_a, "date_b": date_b}
        if modality != "auto":
            params["modality"] = modality
        if acquired_context:
            params["isro_context"] = acquired_context

        try:
            ckey = cache_key_for(paths, query, task_override, params)
        except FileNotFoundError as e:
            _http(400, str(e), code="upload_vanished")

        # ---- result cache: identical inputs+query return instantly ----------
        cached = STORE.get_cached(ckey)
        if cached:
            job = JOBS.create(query, task_override, paths, params, ckey)
            job.cached = True
            job.status = "done"
            cached_result = {**cached["result"], "cached": True}
            job.result = cached_result
            job.trace  = cached_result.get("execution_summary", [])
            JOBS.cache_hits += 1
            _persist_job(job, snap=None, cached_result=cached_result,
                         original_job_id=cached["job_id"])
            JOBS.release_slot()
            log_event(job.id, "cache_hit", original=cached["job_id"])
            return {"job_id": job.id, "cached": True}

        job = JOBS.create(query, task_override, paths, params, ckey)
        STORE.upsert_start(job.id, datetime.now().isoformat(timespec="seconds"),
                           query, task_override or "auto", "running", ckey)
        JOBS.submit(job)
        return {"job_id": job.id, "cached": False}

    except StructuredHTTPException:
        raise                                    # already structured — pass through
    except RuntimeError as e:                    # queue full
        _http(429, str(e), code="queue_full",
              hint=f"The server is processing {JOBS.stats()['running']} "
                   f"job(s). Please retry in a few seconds.")
    except InputValidationError as e:
        _http(400, str(e), code="validation_error")
    except Exception as e:
        logger.exception("Unexpected error in create_job")
        _http(500, "An unexpected server error occurred.",
              code="internal_error",
              hint=repr(e) if _DEBUG else "")


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job:
        snap = job.snapshot()
        # Persist to DB exactly once when the job reaches a terminal state
        if snap["status"] in ("done", "error") and not job._persisted:
            _persist_job(job, snap=snap)
        return snap

    # Not in memory (server restarted) — reconstruct from the database
    row = STORE.get(job_id)
    if row:
        result = None
        if row["result_json"]:
            try:
                result = json.loads(row["result_json"])
            except Exception:
                result = None
        return {"job_id": job_id, "status": row["status"], "query": row["query"],
                "trace": [], "result": result, "error": row["error"],
                "cached": bool(row["cached"]), "n_images": 0}

    raise _http(404, f"Job '{job_id}' not found.", code="not_found",
                hint="The job may have expired or the server may have restarted.")


@app.get("/api/history")
async def history(limit: int = 20):
    return STORE.recent(max(1, min(limit, 100)))


@app.get("/api/samples")
async def samples():
    out = []
    for p in sorted(CONFIG.samples_dir.glob("*")):
        if p.suffix.lower() not in ALLOWED_EXT:
            continue
        try:
            img = load_image(p)
            out.append({**img.summary(), "name": p.name})
        except Exception:
            continue
    return out


@app.get("/api/samples/{name}/preview")
async def sample_preview(name: str):
    """Quick-look PNG for the Console image-preview popup (read-only).

    GeoTIFFs cannot be rendered by browsers, so the server reduces the sample
    to an RGB composite (same path as job evidence) and streams a small PNG.
    """
    safe = Path(name).name                      # path-traversal guard
    p = CONFIG.samples_dir / safe
    if not p.is_file() or p.suffix.lower() not in ALLOWED_EXT:
        raise _http(404, "Unknown sample.", code="not_found")
    try:
        from ..io_utils import rgb_composite
        from .jobs import _encode_image
        import base64 as _b64
        img = load_image(p)
        data_url = _encode_image(rgb_composite(img), max_px=768)
        raw = _b64.b64decode(data_url.split(",", 1)[1])
        return Response(content=raw, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})
    except Exception:
        raise _http(500, "Could not render preview for this sample.",
                    code="preview_error")


@app.get("/api/stats")
async def stats():
    """Unified stats endpoint: job pool + persistent store + cache + freshness legend."""
    jstats = JOBS.stats()
    sstats = STORE.stats()
    # C3/R5: freshness legend (thresholds + flag meanings, non-breaking)
    try:
        from ..freshness.clocks import _THRESHOLDS
        freshness_legend = {
            "thresholds_by_task_days": _THRESHOLDS,
            "flags": {
                "ok": "inputs within threshold and confidence above gate",
                "degraded": "inputs stale or confidence below gate; WHY is shown",
            },
            "method_note": "run + input acquisition age only; no orbital-look "
                           "scheduling claims.",
        }
    except Exception:
        freshness_legend = None
    out = {
        **jstats,
        **sstats,
        "uptime_s": int(time.time() - _STARTED),
        "version": "3.0",
    }
    if freshness_legend is not None:
        out["freshness_legend"] = freshness_legend
    return out


@app.get("/api/experiments")
async def experiments(limit: int = 20):
    """Recent training experiments from the JSONL log."""
    from ..experiment_log import recent_experiments
    return recent_experiments(max(1, min(limit, 100)))


@app.delete("/api/cache/{cache_key}", status_code=204)
async def delete_cache(cache_key: str, _auth: None = Depends(_check_auth)):
    """Explicitly bust a cache entry so the next identical request re-runs."""
    deleted = STORE.delete_cache_entry(cache_key)
    log_event("api", "cache_invalidated", cache_key=cache_key[:16], rows=deleted)
    # 204 No Content — body intentionally empty


@app.get("/api/provenance")
async def provenance():
    cards = []
    for key, label in [
        ("scene_encoder_weights", "Scene Encoder"),
        ("vqa_weights", "VQA Specialist"),
        ("change_weights", "Change Detector"),
        ("fusion_weights", "Optical-SAR Fusion"),
    ]:
        path: Path = getattr(CONFIG, key)
        card = {"component": label, "file": path.name, "trained": path.exists()}
        if path.exists():
            try:
                import torch
                ck = torch.load(path, map_location="cpu", weights_only=False)
                for k in ("val_accuracy", "label_space", "classes", "answer_vocab",
                          "synthetic", "input_size", "val_bleu"):
                    if k in ck:
                        v = ck[k]
                        card[k] = len(v) if isinstance(v, (list, dict)) else v
            except Exception:
                pass
        cards.append(card)
    bench_path = CONFIG.evidence("scorecard.json")
    benchmarks = None
    if bench_path.exists():
        try:
            benchmarks = json.loads(
                bench_path.read_text(encoding="utf-8")).get("results")
        except Exception:
            benchmarks = None
    if benchmarks is None and CONFIG.evidence("benchmarks.json").exists():
        try:
            benchmarks = json.loads(CONFIG.evidence("benchmarks.json")
                                    .read_text(encoding="utf-8"))
        except Exception:
            benchmarks = None
    return {"models": cards, "benchmarks": benchmarks}


@app.get("/api/reports/{run_id}/report.pdf")
async def report_pdf(run_id: str):
    from ..pdfreport import generate_pdf
    import traceback as _tb
    run_id = Path(run_id).name            # traversal hardening
    try:
        out = generate_pdf(run_id)
    except Exception as exc:
        logger.exception("PDF generation failed for %s", run_id)
        _http(500, f"PDF generation failed: {exc}", code="pdf_error",
              hint=_tb.format_exc(limit=3) if _DEBUG else "")
    if not out or not out.exists():
        raise _http(404, "Report not found.", code="not_found")
    return FileResponse(out, media_type="application/pdf",
                        filename=f"anvesha_{run_id}.pdf")


@app.get("/api/reports/{run_id}/report.md")
async def report_markdown(run_id: str):
    run_id = Path(run_id).name
    p = CONFIG.runs_dir / run_id / "report.md"
    if not p.exists():
        raise _http(404, "Markdown report not found.", code="not_found")
    return FileResponse(p, media_type="text/markdown",
                        filename="report.md")


@app.get("/api/reports/{run_id}/report.json")
async def report_json(run_id: str):
    run_id = Path(run_id).name
    p = CONFIG.runs_dir / run_id / "report.json"
    if not p.exists():
        raise _http(404, "JSON report not found.", code="not_found")
    return FileResponse(p, media_type="application/json",
                        filename="report.json")


@app.get("/api/reports/{run_id}/{filename}")
async def report_file(run_id: str, filename: str):
    run_id = Path(run_id).name            # traversal hardening
    safe   = Path(filename).name
    for sub in ("", "visuals"):
        p = CONFIG.runs_dir / run_id / sub / safe
        if p.exists():
            media = ("image/png"        if p.suffix == ".png"  else
                     "application/json" if p.suffix == ".json" else
                     "image/tiff"       if p.suffix == ".tif"  else
                     "text/markdown")
            return FileResponse(p, media_type=media, filename=safe)
    raise _http(404, "File not found.", code="not_found")


@app.get("/api/geo/{run_id}")
async def geo(run_id: str):
    """GeoJSON overlay for a run: grounding boxes and/or change-mask polygons,
    reprojected from pixel space to the image CRS when georeferenced."""
    import rasterio.features

    run_id = Path(run_id).name            # traversal hardening
    run_dir = CONFIG.runs_dir / run_id
    # Defensive: ensure resolved path stays inside runs_dir
    if not run_dir.resolve().is_relative_to(CONFIG.runs_dir.resolve()):
        raise _http(404, "Run not found.", code="not_found")
    report = run_dir / "report.json"
    if not report.exists():
        raise _http(404, "Run not found.", code="not_found")
    data   = json.loads(report.read_text(encoding="utf-8"))
    inputs = data.get("inputs", [])
    bounds = inputs[0].get("bounds") if inputs else None
    size   = inputs[0].get("size", [256, 256]) if inputs else [256, 256]
    W, H   = size[0], size[1]

    def to_map(x, y):
        if bounds:
            x0, y0, x1, y1 = bounds
            return [x0 + (x / W) * (x1 - x0), y1 - (y / H) * (y1 - y0)]
        return [x, y]

    features = []
    boxes = (data.get("outputs") or {}).get("boxes") or \
        (data.get("outputs") or {}).get("largest_region_box")
    if boxes and isinstance(boxes, list) and boxes and \
            isinstance(boxes[0], list) and len(boxes[0]) == 4:
        for b in boxes:
            ring = [to_map(b[0], b[1]), to_map(b[2], b[1]),
                    to_map(b[2], b[3]), to_map(b[0], b[3]), to_map(b[0], b[1])]
            features.append({"type": "Feature", "properties": {"kind": "box"},
                             "geometry": {"type": "Polygon",
                                          "coordinates": [ring]}})

    mask_png = run_dir / "visuals" / "change_mask.png"
    if mask_png.exists():
        from PIL import Image
        mask  = (np.asarray(Image.open(mask_png).convert("L")) > 127).astype(np.uint8)
        geoms = rasterio.features.shapes(mask, mask=mask.astype(bool))
        for geom, val in geoms:
            if not val:
                continue
            coords = geom["coordinates"][0]
            ring   = [to_map(x, y) for x, y in coords]
            features.append({"type": "Feature",
                             "properties": {"kind": "change"},
                             "geometry": {"type": "Polygon",
                                          "coordinates": [ring]}})

    crs = inputs[0].get("crs") if inputs else None
    return {
        "type":         "FeatureCollection",
        "crs":          crs,
        "pixel_bounds": bounds,
        "features":     features,
    }


_EVAL_STATE: Dict[str, Any] = {}


@app.post("/api/evaluate/run")
async def evaluate_run(n: int = 300):
    eid = uuid.uuid4().hex[:8]
    _EVAL_STATE[eid] = {"status": "running", "started": time.time()}

    def runner():
        import subprocess
        try:
            p = subprocess.run(
                [sys.executable, "-m", "anvesha.evaluate", "--all",
                 "--n", str(max(50, min(n, 800)))],
                cwd=str(CONFIG.repo_root), capture_output=True, text=True)
            sc = CONFIG.evidence("scorecard.json")
            _EVAL_STATE[eid] = {
                "status":    "done" if p.returncode == 0 else "error",
                "error":     (p.stderr or "")[-400:],
                "scorecard": json.loads(sc.read_text(encoding="utf-8"))
                if sc.exists() else None,
            }
        except Exception as exc:
            _EVAL_STATE[eid] = {
                "status": "error",
                "error": f"Evaluation failed: {exc}",
                "scorecard": None,
            }

    threading.Thread(target=runner, daemon=True).start()
    if len(_EVAL_STATE) > 20:                     # bound memory
        for k in list(_EVAL_STATE)[:-20]:
            _EVAL_STATE.pop(k, None)
    return {"eval_id": eid}


@app.get("/api/evaluate/status/{eid}")
async def evaluate_status(eid: str):
    st = _EVAL_STATE.get(eid)
    if not st:
        raise _http(404, f"Evaluation '{eid}' not found.", code="not_found")
    return st


@app.get("/healthz")
async def healthz():
    # Report the device the app actually runs on, not what torch advertises.
    # Hugging Face's ZeroGPU runtime patches torch so cuda.is_available() is
    # True while real CUDA memory is only granted inside a scheduled
    # @spaces.GPU call; on a host like that the two answers differ, and
    # reporting the advertised one hides the very mismatch that silently
    # degraded three specialists to heuristics. `cuda_reported` keeps the
    # patch visible instead of papering over it.
    try:
        import torch
        threads = torch.get_num_threads()
        cuda_reported = bool(torch.cuda.is_available())
    except Exception:
        threads, cuda_reported = 0, False
    try:
        device = CONFIG.resolve_device()
    except Exception:
        device = "unknown"
    jstats = JOBS.stats()
    try:
        from ..models.status import model_status
        mstatus = model_status()
    except Exception:
        mstatus = {}
    return {
        "status":           "ok",
        "device":           device,
        "cuda_reported":    cuda_reported,
        "torch_threads":    threads,
        "workers":          WORKERS,
        "gpu_semaphored":   gpu_semaphore() is not None,
        "jobs_running":     jstats["running"],
        "jobs_pending":     jstats["pending"],
        "jobs_in_memory":   jstats["in_memory"],
        "cache_hits":       JOBS.cache_hits,
        "model_status":     mstatus,
        "degraded":         any(v != "trained" for v in mstatus.values()),
        "airgap":           _mode_snapshot(),
        "uptime_s":         int(time.time() - _STARTED),
        "version":          _VERSION,
    }


@app.get("/api/model_status")
async def model_status_endpoint():
    """Degradation transparency: which specialists run trained weights vs
    heuristic fallbacks, plus centroid-load status for intent classification.
    The UI renders a badge when any entry is not 'trained' or when centroids
    fall back to keyword-derived, so heuristic-mode answers are never mistaken
    for model output."""
    from ..models.status import model_status
    from ..agent import centroid_status
    status = model_status()
    cstatus = centroid_status()
    return {"models": status,
            "centroids": cstatus,
            "degraded": any(v != "trained" for v in status.values()),
            "centroid_fallback": cstatus.get("centroid_source") == "keyword-derived"}


# ---------------------------------------------------------------------------
# Operating mode + air-gap guard (additive surface)
# ---------------------------------------------------------------------------

@app.get("/api/mode")
async def get_mode():
    """Current operating mode and the live state of the air-gap guard.

    The UI renders its mode chip from this and never infers connectivity on the
    client. ``/api/mode/blocked`` exposes what the guard actually stopped, which
    is what makes the air-gap claim inspectable rather than merely asserted.
    """
    try:
        from ..acquire import mode as _mode
        return {**_mode.guard_status(),
                "version": _VERSION,
                "acquire": _mode.load_settings().get("acquire", {})}
    except Exception as exc:                       # pragma: no cover
        return {"mode": "unknown", "guard_installed": False,
                "guard_disabled_by_env": False,
                "detail": repr(exc) if _DEBUG else "mode subsystem unavailable"}


@app.post("/api/mode")
async def post_mode(payload: Dict[str, Any],
                    _auth: None = Depends(_check_auth)):
    """Switch operating mode; persisted to data/settings.json.

    Switching to ``online`` does not weaken the air-gap guarantee: the guard
    stays installed and its policy simply returns to allowing egress, so
    switching back is enforced instantly.
    """
    try:
        from ..acquire import mode as _mode
    except Exception as exc:                       # pragma: no cover
        _http(500, "Mode subsystem unavailable.", code="internal_error",
              hint=repr(exc) if _DEBUG else "")
        return

    requested = str((payload or {}).get("mode", "")).strip().lower()
    if requested not in ("airgap", "online"):
        _http(400, f"Invalid mode '{requested}'.", code="invalid_mode",
              hint="Use 'airgap' (no network) or 'online' (fetch imagery).")
    try:
        applied = _mode.set_mode(requested)
    except ValueError as exc:
        _http(400, str(exc), code="invalid_mode")
        return
    log_event("api", "mode_changed", mode=applied)
    return {"mode": applied, **_mode.guard_status()}


@app.get("/api/mode/blocked")
async def get_blocked(drain: bool = True):
    """Outbound network attempts the air-gap guard blocked, newest last.

    ``drain`` (default true) clears the in-memory log on read so the UI can show
    "what was blocked since you last looked" without unbounded growth.
    """
    try:
        from ..acquire import mode as _mode
        return {"mode": _mode.current_mode(),
                "guard_installed": _mode.guard_installed(),
                "blocked": _mode.blocked_attempts(drain=drain)}
    except Exception:                              # pragma: no cover
        return {"mode": "unknown", "guard_installed": False, "blocked": []}


# ---------------------------------------------------------------------------
# Online acquisition (the "fetch the data for me" half)
# ---------------------------------------------------------------------------

async def _run_acquire(fn, *args, **kwargs):
    """Run blocking acquisition work off the event loop.

    Fetching reads Cloud-Optimized GeoTIFF windows over HTTPS and renders ISRO
    map tiles; both take tens of seconds. Running them inline would stall every
    other request on the server.
    """
    from starlette.concurrency import run_in_threadpool

    return await run_in_threadpool(lambda: fn(*args, **kwargs))


def _acquire_error(exc: Exception, *, action: str):
    """Map acquire-layer failures onto honest, structured HTTP errors."""
    from ..acquire.errors import (AcquireError, NetworkBlockedAirgap,
                                  NoSceneFound)
    from ..acquire.aoi import PlaceNotFound

    message = str(exc)
    if isinstance(exc, NetworkBlockedAirgap) or "air-gap" in message.lower():
        return _http(409, message, code="airgap_mode",
                     hint=("Anvesha is in air-gap mode, so it will not reach the "
                           "network. Results already fetched are replayed from "
                           "the local cache; this request had none. Switch to "
                           "online mode to fetch it, or analyse the files you "
                           "already have offline."))
    if isinstance(exc, PlaceNotFound):
        return _http(422, message, code="place_not_found",
                     hint="Try '<town>, <state>', or give coordinates like "
                          "'26.15, 91.75'.")
    if isinstance(exc, NoSceneFound):
        return _http(422, message, code="no_scene_found")
    if isinstance(exc, AcquireError):
        return _http(422, message, code="acquire_failed")
    logger.exception("Unexpected error during %s", action)
    return _http(500, f"Could not {action}.", code="internal_error",
                 hint=repr(exc) if _DEBUG else "")


@app.get("/api/acquire/status")
async def acquire_status_endpoint():
    """Online-mode state: providers, limits, cache usage, ISRO service details."""
    from ..acquire.service import acquire_status

    try:
        return acquire_status()
    except Exception as exc:                          # pragma: no cover
        logger.exception("acquire status failed")
        return {"mode": "unknown", "error": repr(exc), "providers": [],
                "cache": {}, "defaults": {}, "isro": {}}


@app.post("/api/acquire/plan")
async def acquire_plan_endpoint(payload: Dict[str, Any],
                                _auth: None = Depends(_check_auth)):
    """Resolve the place and list candidate passes. Downloads nothing.

    Existing so that a user can see which satellite passes were chosen, and how
    cloudy they are, before spending bandwidth on them.
    """
    from ..acquire.service import plan as acquire_plan

    query = str((payload or {}).get("query", "")).strip()
    if not query:
        _http(400, "No place given.", code="no_place",
              hint="Describe the area, e.g. 'Dibrugarh, Assam'.")
    kwargs: Dict[str, Any] = {}
    for key, caster in (("days", int), ("max_cloud_pct", float),
                        ("window_km", float)):
        if payload.get(key) is not None:
            try:
                kwargs[key] = caster(payload[key])
            except (TypeError, ValueError):
                _http(400, f"Invalid value for {key}.", code="bad_parameter")
    try:
        result = await _run_acquire(acquire_plan, query, **kwargs)
        return result.to_dict()
    except StructuredHTTPException:
        raise
    except Exception as exc:
        _acquire_error(exc, action="plan the acquisition")


@app.post("/api/acquire/fetch")
async def acquire_fetch_endpoint(payload: Dict[str, Any],
                                 _auth: None = Depends(_check_auth)):
    """Fetch the chosen pair plus verified ISRO context, and keep them on disk.

    Returns an ``acquire_id`` naming a run directory. Feed that to
    ``POST /api/jobs`` to analyse the fetched imagery with the offline pipeline.
    """
    from ..acquire.service import acquire

    query = str((payload or {}).get("query", "")).strip()
    if not query:
        _http(400, "No place given.", code="no_place")
    kwargs: Dict[str, Any] = {}
    for key, caster in (("days", int), ("max_cloud_pct", float),
                        ("window_km", float)):
        if payload.get(key) is not None:
            try:
                kwargs[key] = caster(payload[key])
            except (TypeError, ValueError):
                _http(400, f"Invalid value for {key}.", code="bad_parameter")
    if payload.get("include_context") is not None:
        kwargs["include_context"] = bool(payload["include_context"])
    try:
        _plan, outcome = await _run_acquire(acquire, query, **kwargs)
        log_event(outcome.acquire_id, "acquired",
                  place=query, before=outcome.before.date,
                  after=outcome.after.date, context=len(outcome.context))
        return outcome.to_dict()
    except StructuredHTTPException:
        raise
    except Exception as exc:
        _acquire_error(exc, action="fetch imagery for that place")


@app.get("/api/acquire/{acquire_id}")
async def acquire_detail(acquire_id: str):
    """Provenance for a past acquisition: exactly which scenes produced a result."""
    from ..acquire.service import provenance_for, acquired_images

    record = provenance_for(acquire_id)
    if record is None:
        _http(404, "No such acquisition.", code="not_found")
    return {"acquire_id": Path(str(acquire_id)).name, "provenance": record,
            "images": [p.name for p in acquired_images(acquire_id)]}


# ---------------------------------------------------------------------------
# Global exception handler — structured, never leaks raw tracebacks
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled(request, exc):       # pragma: no cover
    import traceback
    logger.exception("Unhandled exception: %s %s", request.method, request.url)
    # Re-raise StructuredHTTPException so its own handler fires correctly
    if isinstance(exc, StructuredHTTPException):
        return await structured_error_handler(request, exc)
    body = _error_body(
        detail="An unexpected server error occurred.",
        code="internal_error",
        hint=traceback.format_exc(limit=6) if _DEBUG else "",
    )
    return JSONResponse(status_code=500, content=body)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _persist_job(job: "_Job",                        # noqa: F821
                 snap: Optional[Dict],
                 cached_result: Optional[Dict] = None,
                 original_job_id: str = "") -> None:
    """Write terminal job state to the DB exactly once (guarded by _persisted)."""
    if job._persisted:
        return
    job._persisted = True
    if cached_result is not None:
        # Shortcut for cache-hit jobs
        STORE.upsert_start(job.id,
                           datetime.now().isoformat(timespec="seconds"),
                           job.query, job.task_override or "auto", "done",
                           job.cache_key)
        STORE.finish(job.id, "done",
                     str(cached_result.get("selected_task", "")),
                     str(cached_result.get("answer", "")),
                     float(cached_result.get("confidence", 0)),
                     str(cached_result.get("run_id", "")),
                     "",
                     cached_result, True)
        return
    if snap is None:
        return
    try:
        STORE.finish(
            job.id,
            snap["status"],
            str(snap["result"].get("selected_task", "")) if snap["result"] else "",
            str(snap["result"].get("answer", ""))        if snap["result"] else "",
            float(snap["result"].get("confidence", 0))  if snap["result"] else 0.0,
            str(snap["result"].get("run_id", ""))       if snap["result"] else "",
            snap["error"] or "",
            snap["result"],
            bool(snap.get("cached")),
        )
    except Exception:
        logger.exception("Failed to persist job %s to DB", job.id)


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
WEB_DIST = CONFIG.repo_root / "web" / "dist"

# API routers are registered whether or not the built SPA is present. These
# used to sit inside the `if WEB_DIST.exists()` block below, which silently
# removed /api/reports/{id}/dossier and /api/fixtures in a source checkout that
# had not been built with `npm run build` -- the API surface should not depend
# on whether a static bundle was compiled.
from .dossier import router as _dossier_router
from .fixtures import router as _fixtures_router

app.include_router(_dossier_router)
app.include_router(_fixtures_router)

if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        # Unknown API paths must 404 as JSON, never fall through to the SPA
        if full_path == "api" or full_path.startswith("api/"):
            raise _http(404, "Unknown API endpoint.", code="not_found")
        candidate = WEB_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")
