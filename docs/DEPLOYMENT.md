# Deploying Anvesha to a public URL

Target: **Hugging Face Spaces, Docker SDK** — free, permanent public URL, enough RAM
for CPU-only torch, and it accepts the weights the repository deliberately does not
track.

The whole deployment is one command, but read the two sections below first: the first
explains the one thing that makes a naive deploy look healthy and be useless, and the
second is the checklist to run once the Space is live.

| | |
|---|---|
| Image | `Dockerfile.demo` (whitelisted weights, ~430 MB class), staged as the Space's `Dockerfile` |
| Port | `8000`, honoured via `$PORT` if the host injects one |
| Weights shipped | 9 demo-critical checkpoints, **350 MB** (staged from this machine) |
| Weights deliberately not shipped | `RemoteCLIP-ViT-B-32.pt` (578 MB), `dinov2_vits14.pt` (85 MB), `_precal_backup/` (90 MB), v1 change nets |
| Default mode | **air-gap** (nothing reaches the network unless the ONLINE toggle is used) |

---

## 1. Why this is not just `git push`

`.gitignore` keeps `weights/*.pt` local by design (`Decisions.md` D16.3) and excepts only
`count_head.pt` and `cdvqa_head.pt`. Hugging Face builds a Space from the **Space's own
repository**, so pushing this working tree as-is produces a Space with **no scene
encoder, no VQA head, no change net and no fusion net**.

That failure is silent. The container starts, the healthcheck passes, the console loads,
and every answer is a heuristic fallback. Nothing in the UI screams, because the app is
engineered to degrade gracefully without weights. `/healthz` is the one place it shows —
`"degraded": true`, `model_status` all `heuristic`.

`scripts/deploy_hf_space.py` exists so that cannot happen quietly. It:

1. Stages a minimal build context in `build/hf_space` (gitignored) — the application
   package, the reproduction scripts, the sample inputs, the committed benchmark
   evidence in `artifacts/`, the built console, `requirements.txt`, and exactly the nine
   checkpoints `Dockerfile.demo` COPYs.
2. **Aborts by name** if any expected weight is missing locally.
3. Tracks the `.pt` files with Git LFS (Spaces accepts weights through LFS, not as raw
   45 MB blobs).
4. Writes the Space card with `app_port: 8000` so the platform routes to the port the
   container actually listens on, instead of assuming 7860.
5. Creates a **fresh** git repository inside the staging directory — this repository's
   own history is never touched.

---

## 2. One-time setup

1. **Create the Space.** <https://huggingface.co/new-space> → any name (e.g. `anvesha`)
   → **Docker** → *Blank* → Free CPU. It must exist before pushing.
2. **Create a write token.** Settings → Access Tokens → *New token* → **Write**.
3. **Install Git LFS** (`git lfs install`) — if the client is missing the push is
   rejected for size.
4. Export the token for this shell:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

## 3. Deploy

```bash
# 1. inspect: stages 350 MB, verifies every COPY target, pushes nothing
python scripts/deploy_hf_space.py --user <hf-user> --space anvesha --dry-run

# 2. push (rebuilds the Space; the staging tree replaces the Space history wholesale)
HF_TOKEN=hf_xxx python scripts/deploy_hf_space.py --user <hf-user> --space anvesha
```

Then open `https://huggingface.co/spaces/<user>/anvesha` → **Logs**. The first build
installs CPU torch and takes roughly 15–25 minutes. Success looks like:

```
bundled offline assets OK: ['bhuvan_layers.json', 'india_states.json'] and web/dist
```

That line is the image asserting, **at build time**, that the bundled place gazetteer,
the ISRO layer index and the built console are present — so a future ignore-rule change
fails the build instead of degrading the air-gapped demo at runtime.

Re-running the script is the update path: edit the source repository, re-run, rebuild.

---

## 4. Verify the Space (do not skip)

| # | Check | Expected |
|---|---|---|
| 1 | Build log | `bundled offline assets OK: [...] and web/dist` |
| 2 | `https://<user>-anvesha.hf.space/healthz` | `"status": "ok"` |
| 3 | Same response, `model_status` | `scene_encoder / vqa / change / fusion` **all `trained`** — any `heuristic` means the staged weights did not land |
| 4 | Same response, `degraded` | `false` |
| 5 | Same response, `airgap` | `guard_installed: true`, `airgap_guard: "enforced"` |
| 6 | Load the console, choose a demo sample, run **Investigation** | trace streams, map renders, **Decision panel** shows *What to do* |
| 7 | Upload `samples/demo_change_2020.tif` + `demo_change_2024.tif` | change map + hectares, unchanged from local |
| 8 | Optional: ONLINE toggle, type a place, fetch | needs network egress; the Space has it. Leave the default air-gap otherwise |

Check 3 is the one that matters. It is the difference between the product you measured
and a demo that sounds confident while every model is a heuristic.

**Cold start.** The container preloads the four specialists on a background thread at
startup (`Specialist preload complete: 4/4 trained` in the logs), so `/healthz` answers
instantly after that. The *first* page load on a freshly started container can still take
~30 s on a small cloud vCPU while torch imports. Free Spaces also **sleep after ~48 h of
inactivity** and take 30–60 s to wake on the first visit — **open the URL once, ten
minutes before the screening**, so the judge does not meet a cold Space.

---

## 5. Judge-proofing

- **Set a token before you share the link.** Space → Settings → *Variables and secrets* →
  `ANVESHA_TOKEN` = any value. Without it the API is open to anyone with the URL
  (`start.py` warns about this locally too).
- **Keep air-gap as the shipping default.** It is the honest position and the
  reproducible one: cut the network, re-run the analysis, get the same answer. Use the
  ONLINE toggle deliberately, as a demonstration, not as the resting state.
- **Prime the demo fixtures** if you want the console's fixture chips (locally:
  `python scripts/warm_demo.py`). They are not required for the Space, and without them
  `/api/fixtures` returns a 404 carrying the command to fix it — a documented degraded
  state, not a broken page.
- **First click after a wake is slow.** Say so, or warm it first. A 30 s cold load that
  you narrate reads as confidence; the same pause unannounced reads as a hang.

---

## 6. Known limits, stated plainly

| Limit | Consequence |
|---|---|
| Free CPU tier | 2 vCPU / 16 GB. A full investigation run is seconds-to-minutes, not instant. |
| Free Spaces sleep when idle | First visit after a sleep is slow. Wake it before you present. |
| Image carries 350 MB of weights | Larger image, slower first build. That is the price of the specialists being real. |
| `RemoteCLIP` / `DINOv2` excluded | The CLIP re-rank stage stays off — correct, since it is off by default and measured 0.4 pp *worse*. Router intent accuracy is unaffected (0.964). |
| Bhoonidhi (SAR) not configured | `/api/acquire/status` says so. Bhuvan is the credential-free ISRO source that works. |
| Fetching needs the network | Re-analysing already-fetched files does not. That distinction is the offline promise (`README.md`). |

## 7. If the Space build fails

Fall back to a VPS without changing anything else — the staged directory is already a
complete, self-contained build context:

```bash
python scripts/deploy_hf_space.py --user x --space y --dry-run   # produce build/hf_space
cd build/hf_space && docker build -t anvesha . && docker run -p 8000:8000 anvesha
```

If the failure is the LFS push, it is almost always a missing `git lfs install` or a
token without **write** scope.
