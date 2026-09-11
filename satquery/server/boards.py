"""C1 — boards router: serves the PRE-BAKED data/boards/*.json (read-only,
no-store headers). Never computes a board per request."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path

from ..config import CONFIG
from .boards_file import load_board

router = APIRouter(prefix="/api", tags=["boards"])


@router.get("/boards")
async def boards():
    doc = load_board("board.json")
    if doc is None:
        raise HTTPException(status_code=404,
                            detail="No board built yet. Run: python -m satquery.boards.build")
    return JSONResponse(doc, headers={"Cache-Control": "no-store"})


@router.get("/board/{slug}")
async def board_by_slug(slug: str):
    if slug not in ("board", "bootstrap"):
        raise HTTPException(status_code=404, detail="Unknown board slug.")
    doc = load_board(f"{slug}.json")
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Board '{slug}' not built.")
    return JSONResponse(doc, headers={"Cache-Control": "no-store"})
