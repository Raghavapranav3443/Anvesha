"""C1 — boards tests: pins over real runs, fail-closed, payload cap."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.boards.schema import BoardDoc, BoardPin, save, load
from satquery.boards.build import build, _pin_for


def _report(run_dir, task="single_vqa", confidence=0.9, extras=None):
    d = {"selected_task": task, "confidence": confidence,
         "outputs": extras or {"answer": "yes"}, "answer": "yes"}
    (run_dir / "report.json").write_text(json.dumps(d), encoding="utf-8")


def test_schema_roundtrip(tmp_path):
    doc = BoardDoc(boards_v=1, generated_at="2026-01-01T00:00:00",
                   pins=[BoardPin(run_id="abc", kind="answer", title="x",
                                 artifact="/api/reports/abc/report.json", why="t",
                                 task="single_vqa", confidence=0.9)])
    p = tmp_path / "b.json"
    save(doc, p)
    loaded = load(p)
    assert loaded["boards_v"] == 1
    assert len(loaded["pins"]) == 1
    assert loaded["pins"][0]["run_id"] == "abc"


def test_pin_prefers_overlay(tmp_path):
    r = {"selected_task": "optical_sar", "confidence": 0.8,
         "outputs": {"agreement_map": "/x.png"}}
    pin = _pin_for(r, "run1")
    assert pin is not None
    assert pin.kind == "overlay"


def test_pin_falls_back_to_answer(tmp_path):
    r = {"selected_task": "single_vqa", "confidence": 0.5,
         "outputs": {}, "answer": "yes"}
    pin = _pin_for(r, "run2")
    assert pin is not None
    assert pin.kind == "answer"


def test_pin_none_when_empty():
    assert _pin_for({}, "run3") is None
    assert _pin_for({"selected_task": "", "outputs": {}, "answer": ""}, "run4") is None


def test_build_pins_over_real_runs(tmp_path):
    runs = tmp_path / "runs"
    for i in range(3):
        rd = runs / f"run{i}"
        rd.mkdir(parents=True)
        _report(rd, task="single_vqa", confidence=0.9 - i * 0.1)
    out = tmp_path / "out"
    rc = build(runs, out)
    assert rc == 0
    board = json.loads((out / "board.json").read_text(encoding="utf-8"))
    assert len(board["pins"]) == 3
    for p in board["pins"]:
        assert p["run_id"] in {"run0", "run1", "run2"}
        assert p["artifact"].endswith("/report.json")


def test_build_payload_cap(tmp_path):
    runs = tmp_path / "runs"
    for i in range(60):
        rd = runs / f"run{i:03d}"
        rd.mkdir(parents=True)
        _report(rd, task="single_vqa", confidence=0.9)
    out = tmp_path / "out"
    rc = build(runs, out)
    assert rc == 0
    board = json.loads((out / "board.json").read_text(encoding="utf-8"))
    assert len(board["pins"]) <= 40


def test_build_bootstrap_size(tmp_path):
    runs = tmp_path / "runs"
    rd = runs / "only"
    rd.mkdir(parents=True)
    _report(rd, task="single_vqa", confidence=0.9)
    out = tmp_path / "out"
    rc = build(runs, out)
    assert rc == 0
    size = (out / "bootstrap.json").stat().st_size
    assert size <= 50_000

