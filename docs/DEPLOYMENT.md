# Deploying Anvesha to a public URL

Target: **Hugging Face Spaces** — a permanent public URL, enough RAM for CPU-only
torch (16 GB on the free hardware), and it accepts the weights this repository
deliberately does not track.

Which *SDK* matters, and that changed in July 2026: creating a **Docker** Space now
requires a paid plan, while **Gradio** Spaces stay free. It makes no difference to what
runs — a Space executes whatever `app.py` starts — so `--sdk gradio` swaps only the
entry point and serves the identical FastAPI application, console and `/healthz`. Use
`--sdk gradio` unless the account already has a paid plan or a grandfathered Docker
Space.

| | |
|---|---|
| SDK | `gradio` (free) · `docker` (paid plan since July 2026) |
| Entry point | `app.py` — uvicorn on `7860` (gradio) · `Dockerfile.demo` (docker) |
| Port | `7860` (gradio, fixed by the runtime) · `8000`, honoured via `$PORT` (docker) |
| Weights shipped | 9 demo-critical checkpoints, **350 MB** (staged from this machine) |
| Weights deliberately not shipped | `RemoteCLIP-ViT-B-32.pt` (578 MB), `dinov2_vits14.pt` (85 MB), `_precal_backup/` (90 MB), v1 change nets |
| Default mode | **air-gap** (nothing reaches the network unless the ONLINE toggle is used) |

### What the host has to provide (measured, not assumed)

| Requirement | Why |
|---|---|
| **≥1 GB RAM, 2 GB comfortable** | `import torch` alone measured **481 MB**; the four specialists loaded measure **918 MB** (peak 985 MB) |
| **Not a build-from-Git host** | Seven of the nine checkpoints are untracked by design (~298 MB: `scene_encoder`, `vqa_head`, `change_net`, `optical_sar_fusion`, `type_heads`, `captioner`, `task_centroids`). A host that clones the repo therefore builds a container with **no specialists at all** — unless, like this script, it stages them from a machine that has them |
| **No hourly cap if you keep it awake** | An always-on ping is what removes the cold-start wait; on a capped free tier that same ping can exhaust the allowance and suspend the service |

**Render is not a candidate.** Its Free instance is 0.1 CPU / 512 MB — less than the cost
of importing torch, before any weight is loaded — and the $7 tier has the same 512 MB. It
also builds from Git, so it hits the missing-weights problem, and its 750 free instance
hours per month are consumed almost exactly by one always-on service: exhausting them
suspends every free service in the workspace until the next month. Its `/robots.txt`
responses do not count as traffic and do not wake a slept service, so any keep-alive must
probe `/healthz`.

The whole deployment is one command, but read the two sections below first: the first
explains the one thing that makes a naive deploy look healthy and be useless, and the
second is the checklist to run once the Space is live.

| | |
|---|---|
| SDK | `gradio` (free) · `docker` (paid plan since July 2026) |
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
4. Writes the Space card for the chosen SDK — `app_port: 8000` for `--sdk docker`, or
   `sdk: gradio` + `app_file: app.py` for the free path, whose runtime fixes port 7860.
5. Creates a **fresh** git repository inside the staging directory — this repository's
   own history is never touched.

---

## 2. One-time setup

1. **Create the Space.** <https://huggingface.co/new-space> → any name (e.g. `anvesha`)
   → **Gradio** → *Blank* (free). A **Docker** Space needs a paid plan; it must exist
   before pushing, either way.
2. **Create a write token.** Settings → Access Tokens → *New token* → **Write**.
3. **Install Git LFS** (`git lfs install`) — if the client is missing the push is
   rejected for size.
4. Export the token for this shell:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

5. **Keep it awake (free).** `.github/workflows/keepalive.yml` probes `/healthz` every
   5 minutes so no visitor ever meets a cold start, and fails loudly if the deployment
   is answering without its weights. Set the target once as a repository variable —
   *Settings → Secrets and variables → Actions → Variables* → `APP_URL` =
   `https://<user>-<space>.hf.space`. GitHub only runs scheduled workflows on the
   repository's **default branch**, so that file has to be there to take effect.

## 3. Deploy

```bash
# 1. inspect: stages 350 MB, verifies every entry point, pushes nothing
python scripts/deploy_hf_space.py --user <hf-user> --space anvesha \
    --sdk gradio --dry-run

# 2. push (rebuilds the Space; the staging tree replaces the Space history wholesale)
HF_TOKEN=hf_xxx python scripts/deploy_hf_space.py --user <hf-user> --space anvesha \
    --sdk gradio
```

`--sdk docker` emits the Dockerfile-based Space instead; everything else is identical.

Then open `https://huggingface.co/spaces/<user>/anvesha` → **Logs**. The first build
installs CPU torch and takes roughly 15–25 minutes. On `--sdk docker`, success includes:

```
bundled offline assets OK: ['bhuvan_layers.json', 'india_states.json'] and web/dist
```

That line is the image asserting, **at build time**, that the bundled place gazetteer,
the ISRO layer index and the built console are present — so a future ignore-rule change
fails the build instead of degrading the air-gapped demo at runtime. The `--sdk gradio`
path has no `Dockerfile` to run that check, so the same guarantee is enforced one step
earlier: the script refuses to stage or push a context missing `app.py`,
`web/dist/index.html` or either bundled index.

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
~30 s on a small cloud vCPU while torch imports. Free Spaces also **sleep when idle** and
take 30–60 s to wake on the next request — which a request triggers by itself, so no
human needs to be involved. `.github/workflows/keepalive.yml` removes the wait entirely
by probing `/healthz` every 5 minutes; confirm it is green before screening day rather
than opening the URL by hand on the day.

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
- **Keep it warm, don't remember to wake it.** The keep-alive workflow is the mechanism,
  and it doubles as monitoring: it fails if `model_status` stops being all `trained`, so a
  deployment that quietly lost its weights cannot sit there looking healthy for days.
- **If the first click is still slow**, the keep-alive is not running (check the
  `APP_URL` variable and that the workflow is on the default branch). A 30 s cold load you
  narrate reads as confidence; the same pause unannounced reads as a hang.

---

## 6. Known limits, stated plainly

| Limit | Consequence |
|---|---|
| Free CPU tier | 2 vCPU / 16 GB. A full investigation run is seconds-to-minutes, not instant. |
| Free Spaces sleep when idle | The next request wakes it in 30–60 s. The keep-alive workflow makes that invisible; without it, the first visitor waits. |
| `--sdk gradio` has no build-time asset assertion | The deploy script's staging checks cover the same ground (entry point, console, both bundled indices) and abort before pushing. |
| Image carries 350 MB of weights | Larger image, slower first build. That is the price of the specialists being real. |
| `RemoteCLIP` / `DINOv2` excluded | The CLIP re-rank stage stays off — correct, since it is off by default and measured 0.4 pp *worse*. Router intent accuracy is unaffected (0.964). |
| Bhoonidhi (SAR) not configured | `/api/acquire/status` says so. Bhuvan is the credential-free ISRO source that works. |
| Fetching needs the network | Re-analysing already-fetched files does not. That distinction is the offline promise (`README.md`). |

## 7. If the Space build fails

Fall back to a VPS without changing anything else — the staged directory is already a
complete, self-contained build context:

```bash
# produce the staging directory (build/hf_gradio for --sdk gradio)
python scripts/deploy_hf_space.py --user x --space y --sdk gradio --dry-run
cd build/hf_space && docker build -t anvesha . && docker run -p 8000:8000 anvesha
```

If the failure is the LFS push, it is almost always a missing `git lfs install` or a
token without **write** scope.
