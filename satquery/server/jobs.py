"""Job execution for the SatQuery AI service.

Concurrency model:
* bounded ThreadPoolExecutor (SATQUERY_WORKERS, default 4)
* GPU serialised via a semaphore when CUDA is present
* torch intra-op threads capped (SATQUERY_THREADS, default 4)
* trace streamed to the job via the controller's trace_callback — the
  controller instance is never mutated, so concurrent jobs are safe.
"""
from __future__ import annotations

import base64
import io
import json
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

WORKERS = max(1, int(os.environ.get("SATQUERY_WORKERS", "4")))
TORCH_THREADS = max(1, int(os.environ.get("SATQUERY_THREADS", "4")))
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
EVIDENCE_MAX_PX = 768
MAX_QUEUE = int(os.environ.get("SATQUERY_MAX_QUEUE", "50"))   # beyond → 429
MAX_JOBS_IN_MEMORY = 300                                      # LRU cap

_executor = ThreadPoolExecutor(max_workers=WORKERS)
_gpu_sem: Optional[threading.Semaphore] = None
_gpu_sem_lock = threading.Lock()
_log_lock = threading.Lock()


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


def log_event(request_id: str, event: str, **kw) -> None:
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "request_id": request_id,
           "event": event, **kw}
    line = json.dumps(rec, default=str)
    with _log_lock:
        try:
            (CONFIG.runs_dir / "server.log").open("a", encoding="utf-8") \
                .write(line + "\n")
        except Exception:
            pass
    print(line, flush=True)


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


def cache_key_for(paths: List[Path], query: str, task_override: Optional[str],
                  params: Dict[str, str]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.name):
        fh = hashlib.sha256(p.read_bytes()).hexdigest()
        h.update(fh.encode())
    h.update(query.strip().lower().encode())
    h.update((task_override or "auto").encode())
    h.update(json.dumps(params, sort_keys=True).encode())
    return h.hexdigest()


class _Job:
    def __init__(self, query: str, task_override: str, image_paths: List[Path],
                 params: Dict[str, str], cache_key: str = ""):
        self.id = uuid.uuid4().hex[:12]
        self.query = query
        self.task_override = None if task_override == "auto" else task_override
        self.image_paths = image_paths
        self.params = params
        self.cache_key = cache_key
        self.status = "queued"
        self.trace: List[Dict[str, Any]] = []
        self.created = time.time()
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.cached = False
        self._lock = threading.Lock()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            try:
                trace = json.loads(json.dumps(
                    self.trace, default=lambda o: o.item()
                    if hasattr(o, "item") else str(o)))
                result = json.loads(json.dumps(
                    self.result, default=lambda o: o.item()
                    if hasattr(o, "item") else str(o))) if self.result else None
            except Exception:
                trace, result = self.trace, self.result
            return {
                "job_id": self.id,
                "status": self.status,
                "query": self.query,
                "trace": trace,
                "result": result,
                "error": self.error,
                "cached": self.cached,
                "n_images": len(self.image_paths),
            }


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, _Job] = {}
        self._order: List[str] = []          # insertion order for LRU eviction
        self._pending = 0
        self._lock = threading.Lock()
        self.cache_hits = 0

    def create(self, query, task_override, paths, params, cache_key="") -> _Job:
        if self._pending >= MAX_QUEUE:
            raise RuntimeError(
                f"Server busy: {self._pending} analyses queued "
                f"(limit {MAX_QUEUE}). Retry shortly.")
        job = _Job(query, task_override, paths, params, cache_key)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._pending += 1
            self._evict_locked()
        return job

    def _evict_locked(self) -> None:
        """Keep memory bounded: drop oldest finished jobs beyond the cap.
        Finished jobs are already persisted to SQLite before eviction."""
        while len(self._order) > MAX_JOBS_IN_MEMORY:
            oldest = self._order[0]
            j = self._jobs.get(oldest)
            if j and j.status in ("queued", "running"):
                break                               # never evict active work
            self._order.pop(0)
            self._jobs.pop(oldest, None)

    def get(self, job_id: str) -> Optional[_Job]:
        return self._jobs.get(job_id)

    def release_slot(self) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            running = sum(1 for j in self._jobs.values() if j.status == "running")
            return {"in_memory": len(self._jobs), "running": running,
                    "pending": self._pending}

    def run(self, job_id: str) -> None:
        job = self._jobs[job_id]
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
            images = [load_image(p) for p in job.image_paths]
            controller = get_controller()
            cb = (lambda trace: (job._publish_trace(trace), time.sleep(0.08)))
            run_kw = dict(trace_callback=cb)

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
            mask_tif = None
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
                    {"summary": im.summary(),
                     "composite": _encode_image(rgb_composite(im))}
                    for im in images
                ],
                "mask_geotiff": mask_tif or (
                    f"/api/reports/{result.run_id}/visuals/change_mask.png"),
            }

            with job._lock:
                job.trace = result.trace
                job.result = payload
                job.status = "done"
            log_event(job.id, "job_done", task=result.selected_task,
                      confidence=result.confidence,
                      duration_ms=int((time.time() - job.created) * 1000))
        except Exception as e:
            import traceback
            with job._lock:
                job.error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
                job.status = "error"
            log_event(job.id, "job_error", error=str(e)[:200])
