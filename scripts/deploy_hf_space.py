#!/usr/bin/env python3
"""Package this repository as a Hugging Face Space and push it.

Why this script exists
----------------------
Hugging Face builds a Docker Space from the Space's **own** git repository, and
this project deliberately does not track its trained weights: `.gitignore` has
`weights/*.pt` with only the counting and CDVQA heads excepted. Pushing the
working tree as-is therefore yields a Space that starts cleanly, passes its
healthcheck, and answers every question with heuristic fallbacks -- the scene
encoder, VQA head, change net and fusion net are simply absent, and nothing in
the UI says so. This script makes that failure mode impossible to ship by
accident.

What it does
------------
1. Stages a minimal build context under `build/hf_space` (gitignored): the
   application package, the reproduction scripts, sample inputs, the built
   console, `requirements.txt`, and exactly the weights `Dockerfile.demo`
   COPYs. The 1.3 GB of local-only checkpoints (RemoteCLIP 578 MB, DINOv2
   85 MB, `_precal_backup/`, the v1 change net) stay behind, because the image
   neither needs nor should carry them.
2. Aborts, by name, if any whitelisted weight is missing -- a Space that
   silently degrades is worse than one that fails to build.
3. Writes the Space card (`README.md` frontmatter: `sdk: docker`,
   `app_port: 8000`), so the platform routes traffic to the port the container
   actually listens on rather than assuming 7860.
4. Tracks the `.pt` files with Git LFS via `.gitattributes`; Spaces accepts
   weights through LFS, not as raw 45 MB git blobs.
5. Creates a **fresh** git repository inside the staging directory and pushes
   it to the Space. This repository's own history is never touched.

Usage
-----
    # inspect the staging directory and the weight manifest, push nothing
    python scripts/deploy_hf_space.py --user <hf-user> --space anvesha --dry-run

    # real push (token needs write access to the Space)
    HF_TOKEN=hf_xxx python scripts/deploy_hf_space.py --user <hf-user> --space anvesha

    # private Space
    HF_TOKEN=hf_xxx python scripts/deploy_hf_space.py --user <u> --space a --private

The Space must already exist (create it once at
https://huggingface.co/new-space with the Docker SDK); this script pushes into
it with `--force`, so a rebuild is always a full replacement of the staging
tree.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Directories copied wholesale. `anvesha/` carries the bundled acquisition
# indices (place gazetteer + ISRO layer index) that only works because
# `.gitignore` re-includes `anvesha/acquire/data/`.
TREES = ["anvesha", "scripts", "samples", "artifacts", "web/dist", "weights/ts"]

# Files copied one by one, mirroring Dockerfile.demo's COPY lines exactly.
FILES = [
    "requirements.txt",
    "weights/calibration.json",
    "weights/task_keywords.json",
]

# The demo-critical checkpoints: Dockerfile.demo whitelists these nine and
# refuses to be built against a context missing any of them.
WEIGHTS = [
    "weights/captioner.pt",
    "weights/cdvqa_head.pt",
    "weights/count_head.pt",
    "weights/change_net.pt",
    "weights/optical_sar_fusion.pt",
    "weights/scene_encoder.pt",
    "weights/task_centroids.pt",
    "weights/type_heads.pt",
    "weights/vqa_head.pt",
]

# The free Space SDK is Gradio, not Docker: as of July 2026 creating a Docker
# Space requires a paid plan, while Gradio Spaces remain free with the same
# 16 GB / 2 vCPU hardware. A Gradio Space still runs whatever `app.py` starts,
# so this entry point binds the *same* FastAPI application (identical routes,
# identical console, identical /healthz) to the port the Gradio runtime
# proxies. There is no Gradio interface to build; the console IS the UI.
SPACE_APP = '''"""Entry point for the free (Gradio-SDK) Space.

The application is a FastAPI service, not a Gradio interface. This file exists
only to (a) bind it to 7860, the port the Gradio SDK runtime proxies, and
(b) move run state off the code directory, which a Space replaces on every
rebuild. The app served here is the same ``anvesha.server.main:app`` the Docker
image runs, so nothing about the deployment changes what a judge sees.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# /tmp is the writable tree: the SQLite store and the acquisition cache must
# not be written into the Space's code directory, and nothing here is expected
# to survive a rebuild.
RUN_STATE = Path(os.environ.get("ANVESHA_RUN_ROOT", "/tmp/anvesha"))
os.environ.setdefault("ANVESHA_DATA_DIR", str(RUN_STATE / "data"))
os.environ.setdefault("ANVESHA_RUNS_DIR", str(RUN_STATE / "runs"))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("anvesha.server.main:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", "7860")), log_level="info")
'''

# apt packages the runtime image needs (the Dockerfile installs the same two
# for libGL/libglib, which Pillow/rasterio pull in on import).
SPACE_PACKAGES = "libgl1\nlibglib2.0-0\n"

SPACE_CARD = """---
title: {title}
emoji: 🛰️
colorFrom: indigo
colorTo: gray
sdk: {sdk}
{app_port}app_file: app.py
pinned: false
{extra}---

# {title}

Earth Observation & Investigation System — an agentic vision-language assistant
for multimodal remote-sensing analysis, built for ISRO/SAC problem statement
SIH26167.

This Space is a **packaged build** of the source repository: the application
code, the built React console, and the demo-critical trained weights. It is
regenerated by `scripts/deploy_hf_space.py`; edit the source repository, not
this Space.

The container serves both the REST API and the console on port {port}.
`/healthz` reports readiness. The app ships in **air-gap mode** by default:
nothing reaches out to the network unless the ONLINE toggle (or
`ANVESHA_MODE=online`) is used explicitly, which is what makes the offline
replay of a previously fetched area reproducible.

Optional: set `ANVESHA_TOKEN` as a Space secret to require a bearer token on
every API call.
"""


def banner(text: str) -> None:
    print(f"\n{'-' * 72}\n  {text}\n{'-' * 72}")


def run(cmd: list[str], cwd: Path | None = None, quiet: bool = False) -> str:
    """Run a command, raising on failure but never echoing a token-bearing URL."""
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        shown = " ".join(_redact(c) for c in cmd)
        sys.exit(f"command failed ({proc.returncode}): {shown}\n"
                 f"{proc.stderr.strip() or proc.stdout.strip()}")
    if not quiet and proc.stdout.strip():
        print(proc.stdout.strip())
    return proc.stdout


def _redact(token: str) -> str:
    """Keep a credential out of error output."""
    return token.replace(os.environ.get("HF_TOKEN", "") or "\x00", "***")


def _installed(dist: str) -> str:
    """Base version of an installed distribution, without any local segment."""
    try:
        from importlib.metadata import version
        return version(dist).split("+")[0]
    except Exception:
        return ""


def space_requirements() -> str:
    """Root requirements with torch pinned to a CPU build.

    A free Space installs ``requirements.txt`` verbatim on 2 vCPU, so an
    unpinned ``torch`` fetches the multi-GB CUDA wheel that the CPU-only free
    hardware cannot use. Pinning the base version and offering the CPU index as
    an extra resolves to ``<version>+cpu``: for the same release, the local
    version segment outranks the plain one (PEP 440), so the CPU build wins.
    """
    lines = ["--extra-index-url https://download.pytorch.org/whl/cpu"]
    for raw in (ROOT / "requirements.txt").read_text(
            encoding="utf-8").splitlines():
        name = raw.strip().split("=")[0].split("<")[0].split(">")[0]
        name = name.split("[")[0].strip().lower()
        if name in ("torch", "torchvision"):
            pinned = _installed(name)
            lines.append(f"{name}=={pinned}" if pinned else name)
        else:
            lines.append(raw)
    return "\n".join(lines) + "\n"


def check_weights(staging: Path) -> list[Path]:
    """Return the staged weight files, or abort naming the ones missing."""
    missing = [w for w in WEIGHTS if not (ROOT / w).exists()]
    if missing:
        sys.exit(
            "refusing to build a Space without its trained weights — these are\n"
            "missing locally, and a Space built without them answers with\n"
            "heuristic fallbacks while looking perfectly healthy:\n"
            + "\n".join(f"    {m}" for m in missing)
            + "\n  Fetch them (release page or the training scripts) and re-run."
        )
    return [staging / w for w in WEIGHTS]


def stage(staging: Path, title: str, port: int, private: bool,
          sdk: str = "docker") -> None:
    """Build the Space's working tree from scratch."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    banner("staging build context")
    for rel in TREES:
        src = ROOT / rel
        if not src.exists():
            sys.exit(f"missing required tree: {rel}")
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        print(f"  {rel}/")
    for rel in FILES:
        src = ROOT / rel
        if not src.exists():
            sys.exit(f"missing required file: {rel}")
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {rel}")

    banner("staging the demo-critical weights")
    total = 0
    for w in check_weights(staging):
        src = ROOT / w.relative_to(staging)
        w.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, w)
        total += src.stat().st_size
        print(f"  {w.relative_to(staging)}  ({src.stat().st_size / 1e6:.1f} MB)")
    print(f"  weights total: {total / 1e6:.0f} MB")

    if sdk == "docker":
        # The Space builds the repo-root Dockerfile; it is Dockerfile.demo,
        # which whitelists exactly the weights staged above and sets
        # ANVESHA_RERANK=0.
        shutil.copy2(ROOT / "Dockerfile.demo", staging / "Dockerfile")
        print("  Dockerfile (from Dockerfile.demo)")
    else:
        # The free SDK: no Dockerfile, just an entry point bound to 7860.
        (staging / "app.py").write_text(SPACE_APP, encoding="utf-8")
        (staging / "packages.txt").write_text(SPACE_PACKAGES, encoding="utf-8")
        (staging / "requirements.txt").write_text(space_requirements(),
                                                  encoding="utf-8")
        print("  app.py, packages.txt, requirements.txt (sdk: %s)" % sdk)

    # Git LFS is how Spaces accepts binaries this size.
    (staging / ".gitattributes").write_text(
        "weights/*.pt filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    print("  .gitattributes (LFS for weights/*.pt)")

    extra = "private: true\n" if private else ""
    # A Gradio Space always serves 7860; app_port is a Docker-SDK key.
    app_port = f"app_port: {port}\n" if sdk == "docker" else ""
    (staging / "README.md").write_text(
        SPACE_CARD.format(title=title, sdk=sdk, app_port=app_port, port=port,
                          extra=extra),
        encoding="utf-8")
    print("  README.md (Space card, sdk: %s)" % sdk)


def push(staging: Path, user: str, space: str, token: str, port: int,
         sdk: str = "docker") -> None:
    banner("pushing to the Space")
    git = shutil.which("git")
    if not git:
        sys.exit("git not found on PATH")
    if not shutil.which("git-lfs"):
        print("  ! git-lfs not found — if the push is rejected for size, install\n"
              "    git-lfs (https://git-lfs.com) and re-run")

    run([git, "init", "-q"], cwd=staging)
    run([git, "checkout", "-q", "-b", "main"], cwd=staging)
    run([git, "lfs", "install", "--local"], cwd=staging, quiet=True)
    run([git, "config", "user.email", "deploy@localhost"], cwd=staging)
    run([git, "config", "user.name", "anvesha-deploy"], cwd=staging)
    run([git, "add", "-A"], cwd=staging)
    run([git, "commit", "-q", "-m",
         "Deploy Anvesha (Earth Observation & Investigation System)"], cwd=staging)

    remote = f"https://{user}:{token}@huggingface.co/spaces/{user}/{space}"
    run([git, "remote", "add", "origin", remote], cwd=staging, quiet=True)
    # The staging tree is rebuilt every run, so the Space history is replaced
    # wholesale rather than merged — a partial push is how a Space ends up
    # serving a frontend that does not match its backend.
    run([git, "push", "--force", "origin", "main"], cwd=staging, quiet=True)
    print(f"  pushed. Space URL: https://huggingface.co/spaces/{user}/{space}")
    print(f"  app URL (after the build finishes): "
          f"https://{user}-{space}.hf.space  on port {port}")
    if sdk == "gradio":
        print("  note: the free Gradio SDK spaces sleep when idle and wake on\n"
              "    the next request — see the keep-alive workflow.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True, help="Hugging Face username/org")
    ap.add_argument("--space", required=True, help="Space name (must already exist)")
    ap.add_argument("--title", default="Anvesha — EO Investigation System")
    ap.add_argument("--port", type=int, default=None,
                    help="container port; defaults to 7860 for --sdk gradio "
                         "and 8000 for --sdk docker")
    ap.add_argument("--sdk", choices=("docker", "gradio"), default="docker",
                    help="Space SDK. 'gradio' is the free one and needs no "
                         "Dockerfile; 'docker' now requires a paid HF plan")
    ap.add_argument("--staging", default="build/hf_space")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="stage everything, push nothing")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""),
                    help="defaults to $HF_TOKEN")
    args = ap.parse_args()

    port = args.port or (7860 if args.sdk == "gradio" else 8000)
    if args.sdk == "gradio" and args.port not in (None, 7860):
        sys.exit("a Gradio Space always serves 7860; drop --port")

    staging = (ROOT / args.staging).resolve()
    print(f"repository: {ROOT}")
    print(f"staging:    {staging}")
    print(f"sdk:        {args.sdk}  (port {port})")

    stage(staging, args.title, port, args.private, args.sdk)

    # Cheap post-conditions: the two things whose absence broke a real deploy.
    entry = "Dockerfile" if args.sdk == "docker" else "app.py"
    checks = {
        entry: staging / entry,
        "web/dist/index.html": staging / "web/dist" / "index.html",
        "anvesha/acquire/data/bhuvan_layers.json":
            staging / "anvesha/acquire/data/bhuvan_layers.json",
        "anvesha/acquire/data/india_states.json":
            staging / "anvesha/acquire/data/india_states.json",
    }
    banner("verifying the offline assets the image asserts at build time")
    for name, path in checks.items():
        if not path.exists():
            sys.exit(f"staged context is incomplete: {name} missing")
        print(f"  ok  {name}")

    if args.dry_run:
        banner("dry run — nothing pushed")
        print(f"  review: {staging}")
        print("  push with:  HF_TOKEN=... python scripts/deploy_hf_space.py "
              f"--user {args.user} --space {args.space} --sdk {args.sdk}")
        return 0

    if not args.token:
        sys.exit("no token: set HF_TOKEN or pass --token (needs write access "
                 "to the Space)")
    push(staging, args.user, args.space, args.token, port, args.sdk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
