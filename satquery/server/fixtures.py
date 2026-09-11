"""Fixtures router: serves the pre-baked satquery/fixtures/manifest.json
(read-only, no-store headers). The manifest is written by
``scripts/warm_demo.py`` after a full demo warm-up; a missing manifest means
the demo cache has not been primed yet — the endpoint 404s with a hint so the
UI can show a degraded-state chip instead of inventing fixture data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..config import CONFIG

router = APIRouter(prefix="/api", tags=["fixtures"])

FIXTURES_PATH = CONFIG.repo_root / "satquery" / "fixtures" / "manifest.json"


def load_manifest() -> Optional[dict]:
    try:
        return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


@router.get("/fixtures")
async def fixtures():
    manifest = load_manifest()
    if manifest is None:
        raise HTTPException(
            status_code=404,
            detail="No demo fixtures built yet. Run: python scripts/warm_demo.py")
    return JSONResponse(manifest, headers={"Cache-Control": "no-store"})