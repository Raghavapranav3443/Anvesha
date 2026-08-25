"""Load test: N concurrent VQA sessions against the running server.

  python scripts/load_test.py --sessions 100 --url http://localhost:8000

Writes runs/loadtest.json and prints an honest SLA summary.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from satquery.config import CONFIG  # noqa: E402


def one_session(url: str, image_path: Path, query: str) -> dict:
    t0 = time.time()
    try:
        with open(image_path, "rb") as f:
            r = requests.post(f"{url}/api/jobs",
                              data={"query": query, "sample_names": ""},
                              files=[("files", ("in.tif", f))],
                              timeout=120)
        if r.status_code != 200:
            return {"ok": False, "cached": False,
                    "latency_s": round(time.time() - t0, 3), "answer": "",
                    "error": f"POST {r.status_code}: {r.text[:120]}"}
        body = r.json()
        if not isinstance(body, dict) or "job_id" not in body:
            return {"ok": False, "cached": False,
                    "latency_s": round(time.time() - t0, 3), "answer": "",
                    "error": f"POST body unexpected: {str(body)[:120]}"}
        jid = body["job_id"]
        cached = body.get("cached", False)
        st = None
        for _ in range(600):
            rr = requests.get(f"{url}/api/jobs/{jid}", timeout=60)
            st = rr.json() if rr.status_code == 200 else {"status": "retry"}
            if isinstance(st, dict) and st.get("status") in ("done", "error"):
                break
            time.sleep(0.15)
        dt = time.time() - t0
        if not isinstance(st, dict):
            return {"ok": False, "cached": False,
                    "latency_s": round(dt, 3), "answer": "",
                    "error": f"GET body unexpected: {str(st)[:120]}"}
        return {"ok": st["status"] == "done", "cached": cached,
                "latency_s": round(dt, 3),
                "answer": (st.get("result") or {}).get("answer", "")[:60],
                "error": (st.get("error") or "")[:120]}
    except Exception as e:
        import traceback
        tb = traceback.format_exc().splitlines()[-6:]
        return {"ok": False, "cached": False,
                "latency_s": round(time.time() - t0, 3), "answer": "",
                "error": " | ".join(x.strip() for x in tb)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=100)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--image", default=str(CONFIG.samples_dir / "demo_single_optical.png"))
    ap.add_argument("--query", default="Is there water in this image?")
    ap.add_argument("--fresh", action="store_true",
                    help="bypass the result cache with unique queries")
    args = ap.parse_args()

    img = Path(args.image)
    print(f"load test: {args.sessions} sessions against {args.url}"
          f" ({'fresh' if args.fresh else 'cache-allowed'})")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.sessions) as ex:
        futs = [ex.submit(one_session, args.url, img,
                          f"{args.query} [{i}]" if args.fresh else args.query)
                for i in range(args.sessions)]
        rows = [f.result() for f in futs]
    wall = time.time() - t0

    ok = [r for r in rows if r["ok"]]
    fresh = [r for r in ok if not r["cached"]]
    lat = sorted(r["latency_s"] for r in rows)
    summary = {
        "sessions": args.sessions,
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "cache_hits": sum(1 for r in rows if r["cached"]),
        "wall_time_s": round(wall, 2),
        "throughput_rps": round(args.sessions / wall, 2),
        "latency_p50_s": round(statistics.median(lat), 3) if lat else None,
        "latency_p95_s": round(lat[int(len(lat) * 0.95) - 1], 3) if lat else None,
        "latency_max_s": round(max(lat), 3) if lat else None,
        "sample_answers": [r["answer"] for r in rows[:3]],
    }
    out = CONFIG.runs_dir / "loadtest.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
