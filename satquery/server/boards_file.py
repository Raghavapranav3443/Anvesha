"""Boards file loader: reads data/boards/*.json (pre-baked by the build CLI)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..config import CONFIG

BOARDS_DIR = CONFIG.data_dir / "boards"


def load_board(name: str) -> Optional[dict]:
    p = BOARDS_DIR / name
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
