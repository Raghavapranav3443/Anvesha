"""C7 — warm demo: prime the SQLite sha256 cache so judge clicks hit instantly.

Runs each demo setup twice against a running server: first pass computes and
caches, second pass must be served from cache (``cached: True``). Writes
``satquery/fixtures/manifest.json`` so UI demo-mode chips validate themselves.

Usage: python scripts/warm_demo.py [--base http://host:8000] [--skip-warm]
Exit codes: 0 all green; 1 any failure / server unreachable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]

# The 10 SIH setups (R7 of the implementation plan) — samples-only, offline.
SETUPS = [
    {"id": "urban-growth", "title": "Urban growth",
     "sampleNames": ["demo_change_2020.tif", "demo_change_2024.tif"],
     "query": "Has the built-up area increased, decreased, or remained unchanged?",
     "expected_task": "change_vqa"},
    {"id": "flood-water", "title": "Flood / water extent",
     "sampleNames": ["demo_single_multispectral.tif"],
     "query": "Highlight the water body in this image.",
     "expected_task": "grounding"},
    {"id": "land-cover", "title": "Land-cover brief",
     "sampleNames": ["demo_single_multispectral.tif"],
     "query": "Describe the land-cover of this image.",
     "expected_task": "captioning"},
    {"id": "road-check", "title": "Road check",
     "sampleNames": ["demo_single_optical.png"],
     "query": "Is there a road?",
     "expected_task": "single_vqa"},
    {"id": "sar-night", "title": "SAR night",
     "sampleNames": ["demo_isroformat_optical.tif", "demo_isroformat_sar.tif"],
     "query": "Use the optical and SAR image together.",
     "expected_task": "optical_sar"},
    {"id": "change-qa", "title": "Change Q+A",
     "sampleNames": ["demo_change_2020.tif", "demo_change_2024.tif"],
     "query": "What changed between these two dates?",
     "expected_task": "change_vqa"},
    {"id": "crop-veg", "title": "Crop / vegetation",
     "sampleNames": ["demo_single_multispectral.tif"],
     "query": "What crops or vegetation are present?",
     "expected_task": "captioning"},
    {"id": "investigation", "title": "Full investigation",
     "sampleNames": ["demo_change_2020.tif", "demo_change_2024.tif"],
     "query": "Investigate the change around the water body.",
     "expected_task": "investigation"},
    {"id": "counting", "title": "Counting stress",
     "sampleNames": ["demo_single_optical.png"],
     "query": "How many buildings are there?",
     "expected_task": "single_vqa"},
    {"id": "isro-pair", "title": "ISRO-format pair",
     "sampleNames": ["demo_pair_optical.tif", "demo_pair_sar.tif"],
     "query": "Use the optical and SAR image together.",
     "expected_task": "optical_sar"},
]

def _post(base: str, path: str, data: bytes, content_type: str) -> dict:
    req = urllib.request.Request(base + path, data=data, method="POST",
                                 headers={"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def _get(base: str, path: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _multipart(fields: dict, files: list) -> tuple:
    boundary = "----SatQueryWarmDemo7f3a"
    body = b""
    for k, v in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f"name=\"{k}\"\r\n\r\n{v}\r\n").encode("utf-8")
    for fname, blob in files:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f"name=\"files\"; filename=\"{fname}\"\r\n"
                 f"Content-Type: image/tiff\r\n\r\n").encode("utf-8")
        body += blob + b"\r\n"
    body += f"--{boundary}--\r\n".encode("utf-8")
    return body, f"multipart/form-data; boundary={boundary}"


def run_setup(base: str, setup: dict) -> dict:
    """Run one setup; returns {ok, cached, task, run_id, error}."""
    samples_dir = REPO / "samples"
    files = []
    for name in setup["sampleNames"]:
        p = samples_dir / name
        if not p.exists():
            return {"ok": False, "error": f"missing sample {name}"}
        files.append((name, p.read_bytes()))
    fields = {"query": setup["query"], "task_override": "auto",
              "sample_names": "|".join(setup["sampleNames"]),
              "date_a": "T1", "date_b": "T2"}
    body, ctype = _multipart(fields, files)
    try:
        resp = _post(base, "/api/jobs", body, ctype)
        job_id = resp["job_id"]
        for _ in range(MAX_POLLS):
            st = _get(base, f"/api/jobs/{job_id}")
            if st.get("status") == "done":
                res = st.get("result") or {}
                return {"ok": True, "cached": bool(res.get("cached")),
                        "task": res.get("selected_task", ""),
                        "run_id": res.get("run_id", "")}
            if st.get("status") == "error":
                return {"ok": False, "error": (st.get("error") or "")[:200]}
            time.sleep(POLL_MS / 1000.0)
        return {"ok": False, "error": "timeout"}
    except urllib.error.URLError as e:
        return {"ok": False, "error": f"server unreachable: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--skip-warm", action="store_true",
                    help="only verify cache hits (single pass)")
    args = ap.parse_args()

    try:
        _get(args.base, "/healthz")
    except Exception as e:
        print(f"FATAL: server not reachable at {args.base} ({e})")
        return 1

    manifest = {"fixtures": [], "generated_by": "scripts/warm_demo.py"}
    failures = 0
    for i, setup in enumerate(SETUPS, 1):
        print(f"[{i}/{len(SETUPS)}] {setup['title']} ...", end=" ", flush=True)
        if not args.skip_warm:
            first = run_setup(args.base, setup)
            if not first["ok"]:
                print(f"FAIL ({first.get('error', '')[:80]})")
                failures += 1
                manifest["fixtures"].append({**setup, "status": "fail",
                                             "error": first.get("error")})
                continue
        second = run_setup(args.base, setup)
        ok = second["ok"]
        warm = bool(second.get("cached"))
        status = "green" if (ok and (args.skip_warm or warm)) else \
            ("fail" if not ok else "degraded")
        if not ok:
            failures += 1
        print(f"{status} task={second.get('task', '')} cached={warm}")
        manifest["fixtures"].append({**setup, "status": status,
                                     "task": second.get("task", ""),
                                     "cached": warm,
                                     "run_id": second.get("run_id", "")})

    out = REPO / "satquery" / "fixtures" / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest["all_green"] = failures == 0
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest -> {out} ({failures} failures)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

