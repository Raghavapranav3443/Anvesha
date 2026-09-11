"""C1 — board builder: pins over real runs, fail-closed on missing run_ids.

    python -m satquery.boards.build --runs runs --out data/boards/

Scans finished run dirs for the strongest evidence per task family
(agreement overlays, transition tables, geo answers, composed answers) and
writes ``data/boards/board.json`` + ``data/boards/bootstrap.json`` (<=50KB each).
A pin whose run_id does not exist under ``runs/`` aborts the build (exit 1) —
a board never invents evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .schema import BoardDoc, BoardPin, save


def _report(run_dir: Path) -> dict | None:
    p = run_dir / "report.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pin_for(report: dict, run_id: str) -> BoardPin | None:
    out = report.get("outputs") or {}
    task = report.get("selected_task", "")
    conf = float(report.get("confidence") or 0.0)
    if out.get("agreement_map"):
        return BoardPin(run_id=run_id, kind="overlay", task=task,
                        title=f"Fusion agreement map ({task})",
                        artifact=f"/api/reports/{run_id}/report.json",
                        why="per-pixel optical-vs-SAR agreement (B4)",
                        confidence=conf)
    if out.get("transitions"):
        top = (out["transitions"].get("top") or [{}])[0]
        t = f"{top.get('from_class', '?')} to {top.get('to_class', '?')}"
        return BoardPin(run_id=run_id, kind="table", task=task,
                        title=f"Change transitions ({t})",
                        artifact=f"/api/reports/{run_id}/report.json",
                        why="per-pixel class transitions on changed pixels (B5)",
                        confidence=conf)
    if out.get("ranked"):
        pr = (out["ranked"].get("primary") or {})
        return BoardPin(run_id=run_id, kind="geo", task=task,
                        title=f"Grounded region ({out.get('concept', '')})",
                        artifact=f"/api/reports/{run_id}/report.json",
                        why=f"primary region + {len(out['ranked'].get('alternates', []))} alternates (B3)",
                        confidence=conf)
    if out.get("dossier"):
        return BoardPin(run_id=run_id, kind="answer", task=task,
                        title=f"Composed answer ({task})",
                        artifact=f"/api/reports/{run_id}/dossier",
                        why="multi-sentence traceable answer (B9) + dossier (C2)",
                        confidence=conf)
    if task and report.get("answer"):
        return BoardPin(run_id=run_id, kind="answer", task=task,
                        title=f"Answer ({task})",
                        artifact=f"/api/reports/{run_id}/report.json",
                        why="engine answer with trace",
                        confidence=conf)
    return None


def build(runs_dir: Path, out_dir: Path) -> int:
    """Build the board; returns 0 ok / 1 fail-closed."""
    now = datetime.now().isoformat(timespec="seconds")
    # compute the valid run set ONCE (not per pin — avoids O(n²) rescans)
    valid_runs = {r.name for r in runs_dir.iterdir() if r.is_dir()}
    pins: list[BoardPin] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        report = _report(run_dir)
        if not report:
            continue
        pin = _pin_for(report, run_dir.name)
        if pin:
            pins.append(pin)
    missing = [p.run_id for p in pins if p.run_id not in valid_runs]
    if missing:
        print(f"FAIL-CLOSED: pins reference missing run_ids: {sorted(set(missing))}")
        return 1
    # keep the freshest, most confident pins; cap the payload
    pins.sort(key=lambda p: (-p.confidence, p.run_id), reverse=False)
    pins = pins[-40:]
    doc = BoardDoc(boards_v=1, generated_at=now, pins=pins)
    out_dir.mkdir(parents=True, exist_ok=True)
    board_path = out_dir / "board.json"
    boot_path = out_dir / "bootstrap.json"
    save(doc, board_path)
    save(doc, boot_path)
    for f in (board_path, boot_path):
        size = f.stat().st_size
        if size > 50_000:
            print(f"WARN: {f.name} is {size} bytes (>50KB cap)")
    print(f"board -> {board_path} ({len(pins)} pins, {board_path.stat().st_size} B)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="data/boards")
    args = ap.parse_args()
    return build(Path(args.runs), Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
