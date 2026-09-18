"""C2 — dossier router: GET /api/reports/{run_id}/dossier.

First successful emit persists ``runs/<id>/dossier.json``; later GETs serve the
persisted file (deterministic, air-gap-safe). report.json is never rewritten.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..config import CONFIG

router = APIRouter(prefix="/api/reports", tags=["dossier"])


def _run_dir(run_id: str) -> Path:
    # conservative: allow only hex/date-shaped ids (no path traversal)
    if not run_id or not all(c.isalnum() or c in "-_" for c in run_id):
        raise HTTPException(status_code=400, detail="Invalid run id.")
    return CONFIG.runs_dir / run_id


@router.get("/{run_id}/dossier")
async def get_dossier(run_id: str):
    run_dir = _run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    persisted = run_dir / "dossier.json"
    if not persisted.exists():
        try:
            from ..dossier import emit as _emit
            from ..io_utils import load_image
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            imgs = [load_image(Path(p)) for p in
                    [i.get("file") for i in (report.get("inputs") or [])
                     if isinstance(i, dict) and i.get("file")]]
            doc = _emit(type("R", (), {"run_id": run_id,
                                       "selected_task": report.get("selected_task", ""),
                                       "configuration": report.get("input_configuration", {}),
                                       "outputs": report.get("outputs", {}),
                                       "trace": report.get("execution_summary", [])})(), imgs)
            persisted.write_text(json.dumps(doc.to_dict(), indent=2),
                                 encoding="utf-8")
        except Exception as e:
            raise HTTPException(status_code=500,
                                detail=f"Dossier emit failed: {e}")
    return JSONResponse(json.loads(persisted.read_text(encoding="utf-8")),
                        headers={"Cache-Control": "no-store"})


@router.get("/{run_id}/report.json")
async def get_report(run_id: str):
    run_dir = _run_dir(run_id)
    p = run_dir / "report.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")),
                        headers={"Cache-Control": "no-store"})
