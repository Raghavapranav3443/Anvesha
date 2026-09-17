# 🛰️ Anvesha — Earth Observation & Investigation System

**An agentic vision-language assistant for multimodal remote-sensing image analysis through natural-language queries.**
*Built for ISRO/SAC problem statement SIH26167 (Smart India Hackathon). Formerly SatQuery AI.*

SatQuery AI is not a single generic model. An **agent controller** validates your
imagery, interprets the query, routes it to **remote-sensing specialist models**
(each fine-tuned on real satellite data), executes them with bound parameters,
and returns evidence-grounded answers — confidence scores, visual overlays,
exportable GeoTIFF masks, and a fully auditable execution trace.

---

## Mandatory scope coverage (SIH26167)

| Requirement | Implementation | Trained on | Measured |
|---|---|---|---|
| RS adaptation | Shared `SceneEncoder` (band-adaptive ResNet-18) warm-starts every specialist | EuroSAT (96px, 2500/class, 12 epochs, AMP, label smoothing) | **98.86%** val acc (full track; quick-track 200/class warm-start: 0.91) |
| Single-image VQA *(mandatory)* | Visual ⊕ question-type-conditioned fusion head | RSVQA-LR (full train) | see scorecard |
| Second single-image task | Captioning **and** grounding — both implemented | BigEarthNet.txt captions + reference boxes | BLEU / IoU@0.5 |
| Bi-temporal change *(mandatory)* | Siamese detector (tiled inference) + description + change-VQA + GeoTIFF change map | LEVIR-CD | IoU/F1 on test |
| Optical–SAR pair analysis *(mandatory)* | Dual-branch fusion network + modality agreement/complementarity analysis | BigEarthNet v2 co-registered S1+S2 pairs | val label recall |
| Agentic orchestration *(mandatory)* | validate → classify → registry select → execute → integrate → auditable report | — | live in every run |

**Input scope:** single optical/multispectral/SAR images, co-registered optical–SAR
pairs, bi-temporal pairs · GeoTIFF/TIFF native (CRS + geometry validation),
PNG/JPEG for benchmark datasets. SAR dB-scale products (RISAT-style HH/HV)
auto-detected and handled separately from power-scale data.

---

## Quick start

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# web console — one-command launcher: kills stale instances, serves API +
# console, opens your browser; Ctrl+C exits cleanly
python start.py                      # flags: --port / --host / --no-browser
# → http://localhost:8000   (EO Mission Console UI)

# manual alternative (React build included in repo; rebuild with `npm ci && npm run build` in web/)
# python -m uvicorn satquery.server.main:app --port 8000
```

Docker (offline-deployable, weights baked at build time):

```bash
docker build -t satquery .
docker run -p 8000:8000 satquery
```

### Optional: "find imagery for me" (online lane)

The app ships in **air-gap mode**. Typing a place name and fetching imagery is an
opt-in, per-process switch (`SATQUERY_MODE=online`, or the ONLINE toggle in the
console's *Find imagery for me* panel). Nothing else about the pipeline changes:
fetching produces GeoTIFFs plus a provenance sidecar on disk, and the ordinary
(offline) pipeline analyses them exactly as it would an upload.

**What needs the internet, and what never does** — stated plainly, because this
is the claim most likely to be misread:

| | Network needed |
|---|---|
| Fetching imagery for a place name (open Sentinel-2 COGs) | **yes** |
| Fetching ISRO thematic context (Bhuvan WMS) | **yes** |
| Resolving a place name already looked up once | no — cached on disk |
| Re-running a plan for an area you already searched (which passes exist) | no — the catalogue's own answer replays from cache |
| Re-rendering ISRO context already verified for an area | no — the rendered tiles replay from cache |
| Re-downloading the pixels for an area you already fetched | **yes** — windowed COG reads bypass the cache; analyse the files already on disk |
| Selecting a Bhuvan layer for a state/theme | no — bundled 135 KB layer index |
| Analysing, reporting, deciding, every benchmark | no — CPU, no network, ever |

So the honest sentence is: *the ISRO context lane is queried live over the
network; its layer selection and every fetched result are cached and replayed
automatically, so the second look at an area needs no network.* Nothing here
fetches Bhuvan data without a network, and in air-gap mode a request that has no
cached answer is refused with HTTP 409 `airgap_mode` — naming that it had no
cached copy — rather than quietly returning "no imagery here".

Two lanes, because they answer different questions:

- **Analytic lane** — Sentinel-2 L2A Cloud-Optimized GeoTIFFs on open AWS /
  Copernicus endpoints, read by *window*, so a 12 km analysis transfers megabytes
  rather than gigabytes. Credential-free.
- **ISRO authority lane** — ISRO's own Bhuvan WMS (`bhuvan-vec1.nrsc.gov.in`,
  6,671 layers, no account, no approval). This is what turns "our analysis found
  new construction" into "…and ISRO's own district land-use layer records this
  parcel as agriculture". `Bhoonidhi` remains registered behind the same
  provider interface, so enabling it later is a config change, not a rewrite.

Bhuvan returns **HTTP 200 with a valid PNG for a layer with no data in your
area**, so presence is proven against a control render rather than assumed; on
live runs a blank tile sits at the service's own ~3% framing baseline while a
populated one measures 8–29%. Only verified layers reach the report.

Live check (needs network, prints the verification evidence):

```bash
python scripts/acquire_smoke.py --place "Dibrugarh, Assam"
```

### Training from scratch

```bash
python scripts/download_datasets.py eurosat        # Zenodo 7711810
python scripts/download_datasets.py rsvqa          # Zenodo 6344333
python scripts/download_datasets.py levircd        # HF mirror of official crops
python scripts/download_datasets.py bigearthnet_14k  # real reBEN v2 S1+S2 pairs (3.1 GB)
python scripts/prepare_bentxt_join.py              # BigEarthNet.txt captions+refs join

python scripts/train_scene_encoder.py --max-per-class 2500 --epochs 12  # RS adaptation (96px, AMP)
python scripts/train_vqa.py        --image-size 192 --epochs 30  # type-conditioned, AMP
python scripts/train_captioner.py  --epochs 15     # BEN.txt-trained decoder, experiment logging
python scripts/train_grounding.py  --epochs 8      # referring-expression boxes
python scripts/train_change.py     --crop 256 --epochs 40  # LEVIR-CD, heavy augmentation
python scripts/train_optical_sar.py --dataset bigearthnet_14k --epochs 10
```

## Evaluation & SAC batch mode

```bash
python -m satquery.evaluate --all            # normalized public-benchmark scorecard
python -m satquery.evaluate --sac-dir DIR    # folder of co-registered pairs → answers.csv
                                             # + per-item GeoTIFF change masks + traces
```

Measured scorecard (public benchmark test subsets, this machine):

### Per-type VQA specialist accuracies (headline evidence)

| Specialist Head | Metric | Score | Published Baseline |
|---|---|---|---|
| Presence (is there X?) | exact-match | **0.91** | GeoChat-zero-shot ~0.70 |
| Rural/Urban classification | exact-match | **0.84** | — |
| Comparison (more/less) | exact-match | **0.71** | — |
| Counting (how many) | exact-match | **0.44** (val digit-acc; ordinal soft-CE v4) | — |
| Aggregate RSVQA-LR | exact-match (all types) | **0.700** (full test, n=9,491) · 0.773 spot-check (n=282) | 79.08% (Lobry et al.) |

*Aggregate EM is dragged down by the counting head (29.5% of test questions,
weakest accuracy). Per-type heads are the fairer comparison against other systems.
Spot-check subsets move a few points between runs — the full-test rows are canonical.*

### Change detection

| Benchmark | Metric | Score | Published Baseline |
|---|---|---|---|
| LEVIR-CD (full test split, n=1,500, thr=0.85) | IoU / F1 | **0.692 / 0.818** | BIT-RN18: 0.81/0.89 |
| CDVQA (test, 39,686 Q) | answer accuracy | **0.683** (+17.4 pts over majority baseline; **every** question type above baseline) | RN-18 baseline: 0.68 |

*LEVIR-CD: CPU-class Siamese FPN, 44MB weights, tiled inference — capability
demo, not SOTA claim. TTA (+2-3 F1 points) available via `--tta` flag.
Protocol note: the 0.692/0.818 headline is the latest full-split measurement
(n=1,500, thr=0.85, 2026-08-29, `runs/phase3_levir_fulltest.json`); an earlier
full-split run of the same protocol measured 0.668/0.801; the MODEL_CARDS
0.60/0.75 figure is the n=300 spot-check protocol — same model, different
evaluation sets. The latest full-split number is canonical.*

### Captioning & scene classification

| Benchmark | Metric | Score | Protocol Note |
|---|---|---|---|
| BigEarthNet.txt captions (val) | BLEU (multi-ref) | **0.32** | Multi-reference protocol |
| BigEarthNet.txt captions (val) | BLEU (single-ref) | **0.59** | Training-validation metric |
| EuroSAT (val) | classification accuracy | **0.91** | Quick-track warm-start, 200 img/class |

*Captioning BLEU varies wildly by protocol. The 0.32 is the harder multi-reference
number; single-reference training-validation is 0.59. See `run_benchmarks.py` for
exact protocol.*

Spot-check numbers reproduce via `python -m satquery.evaluate --all`; canonical
full-test numbers via `python scripts/run_benchmarks.py --n 9491` (RSVQA-LR),
`--n 1500` (LEVIR-CD) and `python scripts/eval_cdvqa.py --split test` (CDVQA).

The SAC batch harness consumes pre-georeferenced Cartosat-2S/RISAT-style pairs,
writes per-item answers/confidence/run-ids to CSV without stopping on failures,
and saves change masks as georeferenced rasters ready for GIS comparison against
reference annotations.

ISRO-style demonstration inputs ship in `samples/`
(`isro_cartosat2s_optical.tif`, `isro_risat_sar.tif`).

## Production features (v3)

- **FastAPI service + React console** — live streaming execution trace, confidence
  gauges, swipe-compare, GeoJSON map overlays for georeferenced outputs,
  click-to-query regions, run history browser with comparison, evaluation tab
- **SQLite persistence** — run history + result cache survive restarts; identical
  inputs+query return instantly (single-file DB, air-gap friendly)
- **Concurrent by design** — bounded worker pool, GPU semaphore, capped torch
  threads. Load test (this laptop, 100 concurrent fresh VQA sessions):
  **100/100 OK, 0 errors, ~13 req/s**, p95 ≈ 7 s wall including client polling
  (`scripts/load_test.py`, results in `runs/loadtest.json`)
- **Calibrated confidence** — temperature-scaled probabilities (T fit on
  held-out validation), not raw softmax. The fit is measured and reproducible:
  `scripts/eval_calibration.py` reports expected calibration error and a
  reliability table (`confidence band -> observed accuracy`), and the checkpoint
  records the sample count behind its temperature. A head that has not been
  validated is labelled as such instead of being presented as calibrated.
- **Quantization gate** — TorchScript + int8 export measured at Δ −0.16%
  accuracy (adopted); fp32 path retained
- **SAC batch mode** — `python -m satquery.evaluate --sac-dir DIR` → answers.csv
  + per-item GeoTIFF masks, never halting on failures
- Optional bearer-token auth (`SATQUERY_TOKEN`), structured JSON logs,
  `/healthz`, Docker packaging, CI workflow

## Tests

```bash
python -m pytest tests -q     # I/O · routing · all specialists · API lifecycle · demo suite
```

**386 passed, 1 skipped** in both profiles — plain, and with the air-gap guard
active (`SATQUERY_MODE=airgap`), where every outbound socket and DNS call is
refused at the interpreter level. Synthetic GeoTIFF fixtures make the suite
offline-capable; it passes with and without trained weights (fallback paths are
themselves under test), and the acquisition layer is tested through a fake
transport, so no test touches the real network.

## Tech stack

| Layer | Technologies |
|---|---|
| **Backend** | Python 3.10+ · PyTorch / torchvision (SceneEncoder ResNet-18, transformer captioner, CORAL ordinal head, TorchScript int8 export) · FastAPI + Uvicorn · rasterio · NumPy · pandas · scikit-learn · Pillow · matplotlib |
| **Frontend** | React 18 · TypeScript 5 · Vite 5 · Tailwind CSS · Leaflet / react-leaflet (GeoJSON map overlays) |
| **Persistence & ops** | SQLite (history + result cache, single file, air-gap friendly) · Docker · pytest (387 tests) · GitHub Actions CI |

## Architecture

<details>
<summary><b>Detailed architecture diagram</b> — click to expand</summary>

```text
                     ┌───────────────────────────────────────────────┐
                     │ python start.py                               │
                     │ kills stale instances · binds :8000 · opens   │
                     │ browser when ready · clean Ctrl+C shutdown    │
                     └──────────────────────┬────────────────────────┘
                                            ▼
┌───────────────────────── Client — web/dist SPA ─────────────────────────────┐
│  React 18 + TypeScript + Tailwind · "EO Mission Console"                    │
│   Console        History        Evaluation      Provenance     Help         │
│   upload/query   run browser    scorecard       model cards   glossary     │
│   live agent trace · swipe-compare · Leaflet GeoJSON overlays               │
│   click-to-query regions · confidence gauges · report downloads             │
└──────────────▲──────────────────────────────▲──────────────────────────────┘
               │ REST + job polling           │ static assets (/assets/*)
┌──────────────┴──────────────────────────────┴──────────────────────────────┐
│ FastAPI service — satquery/server/main.py                                   │
│  POST /api/jobs      GET /api/jobs/{id}      GET /api/history               │
│  GET /api/samples    GET /api/provenance     POST /api/evaluate/run         │
│  GET /api/reports/{id}/report.pdf | /report.md | visuals/*.png|*.tif        │
│  GET /api/geo/{id} (GeoJSON overlays)        GET /healthz                   │
│  optional bearer-token auth · CORS · SPA fallback (API paths always JSON)   │
│                                                                             │
│  JobStore — satquery/server/jobs.py                                         │
│   bounded ThreadPoolExecutor (SATQUERY_WORKERS=4)                           │
│   GPU serialised via semaphore (CUDA only) · torch threads capped           │
│   queue cap → HTTP 429 · LRU eviction of finished jobs                      │
│   result cache: sha256(inputs + query) → instant identical re-runs          │
└──────────────▲─────────────────────────────────────────────────────────────┘
               │ controller.run(..., trace_callback=publish)
┌──────────────┴─────────────────────────────────────────────────────────────┐
│ AgentController — satquery/agent.py                                         │
│  1 validate_inputs   format · modality · CRS · co-registration geometry     │
│  2 classify_task     keyword + BOW embedding blend + feasibility filter    │
│        └ low confidence → clarification options ("did you mean…?")          │
│  3 select_tool       registry lookup w/ input-requirement enforcement       │
│  4 execute           specialist tool(s), params bound, timed                │
│        └ investigation mode chains: change_analysis → grounding(water)      │
│          → impact_analysis, each step traced & error-isolated               │
│  5 integrate         answer + calibrated confidence + visual evidence       │
│  6 report            runs/<id>/report.{json,md,pdf} (branded PDF via RL)    │
└──────────────▲─────────────────────────────────────────────────────────────┘
               │
┌──────────────┴─────────────────────────────────────────────────────────────┐
│ Specialist registry (tools_impl.py)      shared backbone: SceneEncoder      │
│                                          (ResNet-18, EuroSAT-warm-started)  │
│  single_vqa   encoder ⊕ per-type specialist heads + balanced-sampling count      │
│               head + flip-TTA · RSVQA-LR · per-type: presence 91%          │
│  captioning   plan-conditioned transformer decoder ⊕ template fallback      │
│               · BigEarthNet.txt captions · multi-ref BLEU 0.32              │
│  grounding    spectral-index response maps + boxes (fully interpretable)    │
│  change_*     Siamese FPN-lite detector + TTA (4-way avg), tiled infer.     │
│               · LEVIR-CD IoU 0.668 / F1 0.801 + description + change-VQA     │
│  optical_sar  dual-branch S1(dB-aware) ⊕ S2 fusion · BEN v2 pairs ·         │
│               label recall 0.85                                             │
└──────────────▲─────────────────────────────────────────────────────────────┘
               │
┌──────────────┴─────────────────────────────────────────────────────────────┐
│ Storage & artifacts                                                         │
│  data/satquery.db   SQLite — history + cached results (restart-safe)        │
│  weights/*.pt       trained specialists (+ TorchScript int8 exports, ts/)   │
│  runs/<run_id>/     reports json/md/pdf · visuals png · georeferenced tif   │
│  samples/           ISRO-style demo inputs (Cartosat-2S optical, RISAT SAR) │
└─────────────────────────────────────────────────────────────────────────────┘
```

</details>

See also [ARCHITECTURE.md](ARCHITECTURE.md) for design contracts.

## Data sources (verified online)

| Dataset | Source |
|---|---|
| BigEarthNet.txt (primary adaptation dataset) | huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt · arXiv:2603.29630 |
| BigEarthNet v2 reBEN images | zenodo.org/records/10891137 (+S1 sibling), bigearth.net |
| reBEN v2 S1+S2 cross-modal subset (14K) | huggingface.co/datasets/ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2 |
| RSVQA LR / HR | zenodo.org/records/6344333 / 6344367 |
| VRSBench | huggingface.co/datasets/xiang709/VRSBench (upstream parquet schema bug — manual route documented) |
| CDVQA | github.com/YZHJessica/CDVQA |
| LEVIR-CD | justchenhao.github.io/LEVIR |
| EuroSAT | zenodo.org/records/7711810 |
| Sentinel-2 L2A analysis-ready imagery (online lane) | earth-search STAC on AWS Open Data + Copernicus Data Space STAC — Cloud-Optimized GeoTIFFs, no credentials |
| ISRO thematic context (online lane) | ISRO Bhuvan OGC WMS — `bhuvan-vec1.nrsc.gov.in/bhuvan/wms`, 6,671 layers, no credentials; layer index built by `scripts/build_bhuvan_catalog.py` and bundled |
| Offline place gazetteer | built from OpenStreetMap once by `scripts/build_place_index.py`, then bundled |
