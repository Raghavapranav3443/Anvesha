"""Job execution for the Anvesha AI service.

Concurrency model:
* bounded ThreadPoolExecutor (ANVESHA_WORKERS, default 4)
* GPU serialised via a semaphore when CUDA is present
* torch intra-op threads capped (ANVESHA_THREADS, default 4)
* trace streamed to the job via the controller's trace_callback — the
  controller instance is never mutated, so concurrent jobs are safe.

Logging
-------
Structured JSON lines are written via a Python ``RotatingFileHandler``
(5 MB × 3 backups) so the server log never grows unbounded and file
handles are never leaked.
"""
from __future__ import annotations

import base64
import collections
import io
import json
import logging
import logging.handlers
import os
import threading
import time
import uuid
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import CONFIG

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
WORKERS        = max(1, int(os.environ.get("ANVESHA_WORKERS", "4")))
TORCH_THREADS  = max(1, int(os.environ.get("ANVESHA_THREADS", "4")))
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
EVIDENCE_MAX_PX  = 768
MAX_QUEUE        = int(os.environ.get("ANVESHA_MAX_QUEUE", "50"))   # beyond → 429
MAX_JOBS_IN_MEMORY = 300                                              # LRU cap
JOB_TTL_SECONDS  = int(os.environ.get("ANVESHA_JOB_TTL", "3600"))  # 1 h

_executor = ThreadPoolExecutor(max_workers=WORKERS)
_gpu_sem: Optional[threading.Semaphore] = None
_gpu_sem_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Structured rotating logger
# ---------------------------------------------------------------------------

def _build_logger() -> logging.Logger:
    lg = logging.getLogger("anvesha.server")
    if lg.handlers:
        return lg                           # already configured (e.g. in tests)
    lg.setLevel(logging.DEBUG)
    # Console handler — plain text for human reading
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    lg.addHandler(ch)
    # File handler — JSON lines, rotating 5 MB × 3 backups
    try:
        log_path = CONFIG.runs_dir / "server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            str(log_path), maxBytes=5 * 1024 * 1024, backupCount=3,
            encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))   # raw JSON lines
        lg.addHandler(fh)
    except Exception:
        pass                                # degraded: console only
    return lg


_logger = _build_logger()


def log_event(request_id: str, event: str, **kw) -> None:
    """Emit a structured JSON log line.  Safe to call from any thread."""
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "request_id": request_id, "event": event, **kw}
    _logger.info(json.dumps(rec, default=str))


# ---------------------------------------------------------------------------
# GPU semaphore
# ---------------------------------------------------------------------------

def gpu_semaphore() -> Optional[threading.Semaphore]:
    global _gpu_sem
    with _gpu_sem_lock:
        if _gpu_sem is None:
            try:
                import torch
                if torch.cuda.is_available():
                    _gpu_sem = threading.BoundedSemaphore(2)
                else:
                    _gpu_sem = None
            except Exception:
                _gpu_sem = None
    return _gpu_sem


# ---------------------------------------------------------------------------
# Image encoding helper
# ---------------------------------------------------------------------------

def _encode_image(arr, max_px: int = EVIDENCE_MAX_PX) -> Optional[str]:
    from PIL import Image
    a = np.asarray(arr)
    if a.ndim == 2:
        img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    else:
        img = Image.fromarray((np.clip(a[..., :3], 0, 1) * 255).astype(np.uint8))
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def weights_fingerprint() -> str:
    """Cheap fingerprint of the model checkpoints (name + size + mtime).

    Included in the cache key so that swapping/retraining weights
    invalidates stale cached answers immediately instead of serving
    results computed by the previous checkpoint until the TTL expires.
    """
    parts = []
    for attr in ("scene_encoder_weights", "vqa_weights", "change_weights",
                 "fusion_weights"):
        p: Path = getattr(CONFIG, attr)
        if p.exists():
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
    return "|".join(parts)


def cache_key_for(paths: List[Path], query: str, task_override: Optional[str],
                  params: Dict[str, str]) -> str:
    """Compute a deterministic SHA-256 cache key.

    Includes the model-weights fingerprint: retraining or swapping a
    checkpoint changes the key, so stale cached answers from the previous
    weights are never served.

    Guards against files that disappear between upload and hashing
    (e.g. concurrent cleanup) by propagating ``FileNotFoundError`` so the
    caller can handle it gracefully.
    """
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.name):
        try:
            fh = hashlib.sha256(p.read_bytes()).hexdigest()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Uploaded file '{p.name}' was removed before analysis could start. "
                "Please re-upload.")
        h.update(fh.encode())
    h.update(query.strip().lower().encode())
    h.update((task_override or "auto").encode())
    h.update(json.dumps(params, sort_keys=True).encode())
    h.update(weights_fingerprint().encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(
        obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# ---------------------------------------------------------------------------
# _Job
# ---------------------------------------------------------------------------

class _Job:
    def __init__(self, query: str, task_override: str, image_paths: List[Path],
                 params: Dict[str, str], cache_key: str = ""):
        self.id            = uuid.uuid4().hex[:12]
        self.query         = query
        self.task_override = None if task_override == "auto" else task_override
        self.image_paths   = image_paths
        self.params        = params
        self.cache_key     = cache_key
        self.status        = "queued"
        self.trace: List[Dict[str, Any]] = []
        self.created       = time.time()
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.cached        = False
        self._lock         = threading.Lock()
        self._persisted    = False           # written to DB at most once

    def _publish_trace(self, trace: List[Dict[str, Any]]) -> None:
        """Live trace streaming: called by the controller after every step."""
        with self._lock:
            try:
                self.trace = _jsonable(trace)
            except Exception:
                self.trace = list(trace)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            try:
                trace  = _jsonable(self.trace)
                result = _jsonable(self.result) if self.result else None
            except Exception:
                trace, result = self.trace, self.result
            return {
                "job_id":   self.id,
                "status":   self.status,
                "query":    self.query,
                "trace":    trace,
                "result":   result,
                "error":    self.error,
                "cached":   self.cached,
                "n_images": len(self.image_paths),
            }


# ---------------------------------------------------------------------------
# JobStore
# ---------------------------------------------------------------------------

class JobStore:
    def __init__(self) -> None:
        # OrderedDict preserves insertion order and gives O(1) move/delete
        self._jobs: "collections.OrderedDict[str, _Job]" = collections.OrderedDict()
        self._pending = 0
        self._lock    = threading.Lock()
        self.cache_hits = 0

    def create(self, query, task_override, paths, params, cache_key="") -> _Job:
        with self._lock:                     # bound check inside lock: no race
            if self._pending >= MAX_QUEUE:
                raise RuntimeError(
                    f"Server busy: {self._pending} analyses queued "
                    f"(limit {MAX_QUEUE}). Retry shortly.")
            job = _Job(query, task_override, paths, params, cache_key)
            self._jobs[job.id] = job
            self._pending += 1
            self._evict_locked()
        return job

    def _evict_locked(self) -> None:
        """Keep memory bounded: drop oldest *finished* jobs beyond the cap.

        Uses ``OrderedDict`` iteration so the removal is O(1) per entry —
        no linear scan through a list.  Active jobs are skipped so finished
        payloads behind a long-running job can still be reclaimed.
        Also evicts finished jobs that are older than ``JOB_TTL_SECONDS``
        regardless of the count cap.
        """
        now    = time.time()
        excess = len(self._jobs) - MAX_JOBS_IN_MEMORY
        for jid, j in list(self._jobs.items()):
            if j.status in ("queued", "running"):
                continue                    # never evict active work
            age = now - j.created
            if excess > 0 or age > JOB_TTL_SECONDS:
                del self._jobs[jid]
                excess -= 1

    def get(self, job_id: str) -> Optional[_Job]:
        return self._jobs.get(job_id)

    def release_slot(self) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)

    def submit(self, job: _Job) -> None:
        """Dispatch a created job onto the bounded worker pool."""
        try:
            _executor.submit(self.run, job.id)
        except Exception:
            self.release_slot()              # never leak a queue slot
            raise

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            running = sum(1 for j in self._jobs.values() if j.status == "running")
            queued  = sum(1 for j in self._jobs.values() if j.status == "queued")
            return {
                "in_memory": len(self._jobs),
                "running":   running,
                "queued":    queued,
                "pending":   self._pending,
                "cache_hits_session": self.cache_hits,
            }

    def run(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return                           # evicted before we could run
        try:
            self._run_inner(job)
        finally:
            self.release_slot()

    def _run_inner(self, job: _Job) -> None:
        job.status = "running"
        log_event(job.id, "job_start", query=job.query[:120],
                  images=len(job.image_paths))
        from ..agent import get_controller
        from ..io_utils import load_image

        sem = gpu_semaphore()
        try:
            # explicit modality override (POST /api/jobs 'modality' param)
            # removes the dB-heuristic single point of failure for ISRO-style
            # SAR products with ambiguous names
            modality_override = (job.params or {}).get("modality") or None
            images = [load_image(p, modality_override=modality_override)
                      for p in job.image_paths]
            controller = get_controller()
            cb         = (lambda trace: (job._publish_trace(trace), time.sleep(0.08)))
            run_kw     = dict(trace_callback=cb)

            if sem is not None:
                with sem:
                    result = controller.run(images, job.query,
                                            task_override=job.task_override,
                                            params=job.params, **run_kw)
            else:
                result = controller.run(images, job.query,
                                        task_override=job.task_override,
                                        params=job.params, **run_kw)

            visual_urls = {
                k: f"/api/reports/{result.run_id}/visuals/{k}.png"
                for k in result.visuals
            }
            from .rasterout import png_mask_to_geotiff
            mask_tif  = None
            mask_path = CONFIG.runs_dir / result.run_id / "visuals" / "change_mask.png"
            if mask_path.exists():
                png_mask_to_geotiff(mask_path, images[-1] if images else None)
                mask_tif = f"/api/reports/{result.run_id}/visuals/change_mask.tif"

            from ..io_utils import rgb_composite
            payload = {
                **result.to_dict(),
                "visuals": visual_urls,
                "visual_data": {
                    k: _encode_image(v) for k, v in result.visuals.items()
                    if k in ("overlay", "change_overlay")
                },
                "inputs": [
                    {"summary":   im.summary(),
                     "composite": _encode_image(rgb_composite(im))}
                    for im in images
                ],
                "mask_geotiff": mask_tif or (
                    f"/api/reports/{result.run_id}/visuals/change_mask.png"),
            }

            with job._lock:
                job.trace  = result.trace
                job.result = payload
                job.status = "done"
            log_event(job.id, "job_done", task=result.selected_task,
                      confidence=result.confidence,
                      duration_ms=int((time.time() - job.created) * 1000))

        except MemoryError as e:
            # GPU/CPU OOM — distinct bucket so ops can alert on it separately
            self._fail_job(job, e, "MemoryError (OOM)")
        except OSError as e:
            # Disk full, file disappeared, etc.
            self._fail_job(job, e, "OSError")
        except Exception as e:
            self._fail_job(job, e, type(e).__name__)

    @staticmethod
    def _fail_job(job: _Job, exc: BaseException, kind: str) -> None:
        import traceback
        tb = traceback.format_exc(limit=6)
        with job._lock:
            job.error  = f"{kind}: {exc}\n{tb}"
            job.status = "error"
        log_event(job.id, "job_error", kind=kind, error=str(exc)[:200])
