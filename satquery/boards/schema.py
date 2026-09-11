"""C1 — board schema: pins over real run_ids (fail-closed, never invents evidence)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class BoardPin:
    run_id: str                    # REQUIRED — a pin without a real run fails the build
    kind: str                      # "overlay" | "table" | "geo" | "answer"
    title: str
    artifact: str                  # URL path to the evidence
    why: str
    task: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BoardDoc:
    boards_v: int = 1
    generated_at: str = ""
    pins: List[BoardPin] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"boards_v": self.boards_v, "generated_at": self.generated_at,
                "pins": [p.to_dict() for p in self.pins]}


def save(doc: BoardDoc, path) -> None:
    path.write_text(json.dumps(doc.to_dict(), indent=2), encoding="utf-8")


def load(path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
