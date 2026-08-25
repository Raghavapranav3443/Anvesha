"""FastAPI service exposing the SatQuery AI agent.

Run:  python -m uvicorn satquery.server.main:app --port 8000
Serves the React build from web/dist when present.

Environment:
  SATQUERY_WORKERS   job worker threads (default 4)
  SATQUERY_THREADS   torch intra-op threads (default 4)
  SATQUERY_TOKEN     optional bearer token; when set, /api/* requires it
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import CONFIG
from ..io_utils import InputValidationError, load_image
from .jobs import (EVIDENCE_MAX_PX, MAX_UPLOAD_BYTES, TORCH_THREADS, WORKERS,
                   JobStore, cache_key_for, gpu_semaphore, log_event)
from ..store import Store


def _startup_cleanup() -> None:
    """Bound disk growth: stale uploads + oversized server log."""
    updir = CONFIG.runs_dir / "_uploads"
    if updir.exists():
        cutoff = time.time() - 24 * 3600
        for f in updir.iterdir():
            try:
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
            except Exception:
                pass
    log_path = CONFIG.runs_dir / "server.log"
    if log_path.exists() and log_path.stat().st_size > 5 * 1024 * 1024:
        log_path.replace(log_path.with_suffix(".log.1"))


_startup_cleanup()

app = FastAPI(title="Anvesha — Earth Observation & Investigation System", version="3.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

JOBS = JobStore()
STORE = Store(CONFIG.data_dir / "satquery.db")
_STARTED = time.time()
_AUTH_TOKEN = os.environ.get("SATQUERY_TOKEN", "").strip()

ALLOWED_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _check_auth(auth_header: Optional[str]) -> None:
    if not _AUTH_TOKEN:
        return
    if auth_header != f"Bearer {_AUTH_TOKEN}":
        raise HTTPException(401, "invalid or missing bearer token")


def _save_uploads(files: List[UploadFile]) -> List[Path]:
    if len(files) > 2:
        raise HTTPException(400, "At most 2 images per analysis.")
    updir = CONFIG.runs_dir / "_uploads"
    updir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(400,
                                f"Unsupported format '{ext}'. Use GeoTIFF/TIFF "
                                f"or PNG/JPEG (benchmark datasets).")
        data = f.file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(400,
                                f"'{f.filename}' exceeds the 50 MB limit.")
        dest = updir / f"{uuid.uuid4().hex[:8]}_{Path(f.filename).name}"
        dest.write_bytes(data)
        paths.append(dest)
    return paths


@app.post("/api/jobs")
async def create_job(
    query: str = Form(""),
    task_override: str = Form("auto"),
    sample_names: str = Form(""),
    date_a: str = Form("T1"),
    date_b: str = Form("T2"),
    files: List[UploadFile] = File(default=[]),
):
    try:
        paths = _save_uploads([f for f in files if f and f.filename])
        for name in [s for s in sample_names.split("|") if s]:
            p = CONFIG.samples_dir / Path(name).name
            if p.exists():
                paths.append(p)
        if not paths:
            raise HTTPException(400, "No images supplied.")
        if len(paths) > 2:
            raise HTTPException(400, "At most 2 images per analysis.")

        params = {"date_a": date_a, "date_b": date_b}
        ckey = cache_key_for(paths, query, task_override, params)

        # result cache: identical inputs+query return instantly
        cached = STORE.get_cached(ckey)
        if cached:
            job = JOBS.create(query, task_override, paths, params, ckey)
            job.cached = True
            job.status = "done"
            cached_result = {**cached["result"], "cached": True}
            job.result = cached_result
            job.trace = cached_result.get("execution_summary", [])
            JOBS.cache_hits += 1
            log_event(job.id, "cache_hit", original=cached["job_id"])
            return {"job_id": job.id, "cached": True}

        job = JOBS.create(query, task_override, paths, params, ckey)
        STORE.upsert_start(job.id, datetime.now().isoformat(timespec="seconds"),
                           query, task_override or "auto", "running", ckey)
        JOBS.submit(job)
        return {"job_id": job.id, "cached": False}
    except HTTPException:
        raise
    except RuntimeError as e:                     # queue full
        raise HTTPException(429, str(e))
    except InputValidationError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job:
        snap = job.snapshot()
        if snap["status"] in ("done", "error"):
            row = STORE.get(job_id)
            if row:
                STORE.finish(job_id, snap["status"],
                             str(snap["result"].get("selected_task", "")) if snap["result"] else "",
                             str(snap["result"].get("answer", "")) if snap["result"] else "",
                             float(snap["result"].get("confidence", 0)) if snap["result"] else 0.0,
                             str(snap["result"].get("run_id", "")) if snap["result"] else "",
                             snap["error"] or "", snap["result"], bool(snap.get("cached")))
        return snap
    # not in memory (server restarted) — reconstruct from the database
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
    raise HTTPException(404, "unknown job")


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
    bench_path = CONFIG.runs_dir / "scorecard.json"
    benchmarks = None
    if bench_path.exists():
        try:
            benchmarks = json.loads(
                bench_path.read_text(encoding="utf-8")).get("results")
        except Exception:
            benchmarks = None
    if benchmarks is None and (CONFIG.runs_dir / "benchmarks.json").exists():
        try:
            benchmarks = json.loads((CONFIG.runs_dir / "benchmarks.json")
                                    .read_text(encoding="utf-8"))
        except Exception:
            benchmarks = None
    return {"models": cards, "benchmarks": benchmarks}


@app.get("/api/reports/{run_id}/report.pdf")
async def report_pdf(run_id: str):
    from ..pdfreport import generate_pdf
    run_id = Path(run_id).name            # traversal hardening
    out = generate_pdf(run_id)
    if not out or not out.exists():
        raise HTTPException(404)
    return FileResponse(out, media_type="application/pdf",
                        filename=f"anvesha_{run_id}.pdf")


@app.get("/api/reports/{run_id}/{filename}")
async def report_file(run_id: str, filename: str):
    run_id = Path(run_id).name            # traversal hardening
    safe = Path(filename).name
    for sub in ("", "visuals"):
        p = CONFIG.runs_dir / run_id / sub / safe
        if p.exists():
            media = "image/png" if p.suffix == ".png" else (
                "application/json" if p.suffix == ".json" else
                "image/tiff" if p.suffix == ".tif" else "text/markdown")
            return FileResponse(p, media_type=media, filename=safe)
    raise HTTPException(404)


@app.get("/api/geo/{run_id}")
async def geo(run_id: str):
    """GeoJSON overlay for a run: grounding boxes and/or change-mask polygons,
    reprojected from pixel space to the image CRS when georeferenced."""
    import rasterio.features

    report = CONFIG.runs_dir / run_id / "report.json"
    if not report.exists():
        raise HTTPException(404)
    data = json.loads(report.read_text(encoding="utf-8"))
    inputs = data.get("inputs", [])
    bounds = inputs[0].get("bounds") if inputs else None
    size = inputs[0].get("size", [256, 256]) if inputs else [256, 256]
    W, H = size[0], size[1]

    def to_map(x, y):
        if bounds:
            x0, y0, x1, y1 = bounds
            # image row 0 = top = max latitude
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

    mask_png = CONFIG.runs_dir / run_id / "visuals" / "change_mask.png"
    if mask_png.exists():
        from PIL import Image
        mask = (np.asarray(Image.open(mask_png).convert("L")) > 127) \
            .astype(np.uint8)
        geoms = rasterio.features.shapes(mask, mask=mask.astype(bool))
        for geom, val in geoms:
            if not val:
                continue
            coords = geom["coordinates"][0]
            ring = [to_map(x, y) for x, y in coords]
            features.append({"type": "Feature",
                             "properties": {"kind": "change"},
                             "geometry": {"type": "Polygon",
                                          "coordinates": [ring]}})

    crs = inputs[0].get("crs") if inputs else None
    return {
        "type": "FeatureCollection",
        "crs": crs,
        "pixel_bounds": bounds,
        "features": features,
    }


_EVAL_STATE: Dict[str, Any] = {}


@app.post("/api/evaluate/run")
async def evaluate_run(n: int = 300):
    eid = uuid.uuid4().hex[:8]
    _EVAL_STATE[eid] = {"status": "running", "started": time.time()}

    def runner():
        import subprocess
        p = subprocess.run(
            [sys.executable, "-m", "satquery.evaluate", "--all",
             "--n", str(max(50, min(n, 800)))],
            cwd=str(CONFIG.repo_root), capture_output=True, text=True)
        sc = CONFIG.runs_dir / "scorecard.json"
        _EVAL_STATE[eid] = {
            "status": "done" if p.returncode == 0 else "error",
            "error": (p.stderr or "")[-400:],
            "scorecard": json.loads(sc.read_text(encoding="utf-8"))
            if sc.exists() else None,
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
        raise HTTPException(404)
    return st


@app.get("/healthz")
async def healthz():
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        threads = torch.get_num_threads()
    except Exception:
        device, threads = "unknown", 0
    jstats = JOBS.stats()
    return {
        "status": "ok",
        "device": device,
        "torch_threads": threads,
        "workers": WORKERS,
        "gpu_semaphored": gpu_semaphore() is not None,
        "jobs_running": jstats["running"],
        "jobs_pending": jstats["pending"],
        "jobs_in_memory": jstats["in_memory"],
        "cache_hits": JOBS.cache_hits,
        "uptime_s": int(time.time() - _STARTED),
        "version": "3.0",
    }


@app.exception_handler(Exception)
async def unhandled(request, exc):  # pragma: no cover
    return JSONResponse(status_code=500, content={"detail": repr(exc)})


# ---- static frontend ------------------------------------------------------ #
WEB_DIST = CONFIG.repo_root / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        # unknown API paths must 404 as JSON, never fall through to the SPA
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(404, "unknown API endpoint")
        candidate = WEB_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")
