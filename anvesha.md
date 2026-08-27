# Anvesha — The Complete Plain-Language Guide

*(Anvesha = "search / investigation" in Sanskrit. Also known as SatQuery AI during development.)*

**One-line pitch:** Anvesha is a website where you upload satellite pictures, type a question in
plain English, and get an answer **with proof attached** — a highlighted map, a confidence score,
and a step-by-step receipt showing exactly which AI models were used and what they did.

---

## 0. The 60-second version

Satellites photograph the Earth constantly. That data helps with farming, flood response, city
planning and catching illegal construction — but today you need to be a GIS professional
(a "map scientist") with expensive software to extract any meaning from it.

Anvesha removes that barrier. You give it:

1. One satellite photo, **or**
2. Two photos of the same place taken on different dates ("what changed?"), **or**
3. A normal photo + a radar photo of the same place ("combine their strengths")

…and type a question like *"Has the built-up area increased?"* Behind the scenes, a small
**manager program** inspects your files, decides which **specialist AI model** can answer,
runs it, and returns:

- a direct answer,
- a visual proof overlay on the image,
- a calibrated "how sure am I" percentage,
- and a downloadable report recording every step taken.

Think **hospital, not genius**: one smart receptionist (the agent controller) routing your case
to the right consultant (a specialist model), each of whom trained for years on exactly one job.

---

## 1. Words you need first (explained, not assumed)

### About the pictures themselves

| Term | Plain meaning |
|---|---|
| **Remote sensing** | Learning about Earth *without touching it* — mostly by taking pictures from orbit. |
| **Satellite imagery** | Those pictures. They are just grids of pixels, but each pixel may cover 10m × 10m of real ground. |
| **Optical image** | A normal-camera photo from space. Easy to read, but clouds block it and it is useless at night. |
| **Multispectral image** | Same idea, but the camera also sees colours humans cannot — near-infrared, for example, which makes healthy vegetation glow. Farmers love this. |
| **SAR (Synthetic Aperture Radar)** | A *radar* picture. The satellite shouts a microwave pulse at the ground and listens to the echo. It ignores clouds, darkness and weather — but it looks like grainy black-and-white static to humans. |
| **GeoTIFF** | A picture file (like PNG) that also memorises *where on Earth* every pixel sits (its latitude/longitude grid). This is what real GIS software eats. |
| **CRS (Coordinate Reference System)** | The "address format" of a map. Two files using different address formats cannot be compared until converted. |
| **Co-registered pair** | Two images (say optical + radar) of the same area, already aligned pixel-for-pixel, so row 5 column 9 in one covers the exact same rooftop as in the other. |
| **Bi-temporal pair** | The same place photographed twice — e.g., March and September. The magic ingredient for spotting change. |

### About what people want to *know*

| Task | Question it answers |
|---|---|
| **Captioning** | "Describe this image." → produces a sentence. |
| **VQA (Visual Question Answering)** | "Is there a road?" → answers *yes/no/a number*. The core skill of the whole project. |
| **Grounding** | "WHERE is the water body?" → draws a box/mask around it. |
| **Change detection** | "What changed between these two dates?" → highlights and describes differences. |
| **Optical–SAR fusion** | "Combine both views." → radar's cloud-proof structure + optics' colour detail, together more reliable than either alone. |

A few engineering terms used later:

- **Fine-tuning** — taking an AI model that already knows "what images look like" and giving it
  extra schooling on satellite pictures specifically, like sending a biology graduate to do a
  pharmacy diploma.
- **Benchmark** — a standardised exam with known correct answers, so anyone can compare models fairly.
- **Inference** — using a trained model to get an answer (as opposed to *training*, the studying phase).
- **Fallback** — a dumb-but-dependable backup method that runs automatically if the fancy model
  can't (e.g., missing file). Like a junior doctor following the printed protocol when the senior
  surgeon is unreachable.
- **int8 quantization** — storing the model's numbers with less precision so it runs faster and
  smaller, after verifying accuracy barely moves (ours moved −0.16%).

---

## 2. What Anvesha actually is

A web application (runs in your browser) with three parts:

1. **The console** — a clean interface where you drag in images, type queries, watch results.
2. **The agentic backend** — a Python service whose *manager* (AgentController) orchestrates everything.
3. **Seven trained specialist models** — each small enough to run on an ordinary laptop CPU,
   each fine-tuned on real public satellite datasets, each with a tested fallback.

It accepts single optical/SAR images, co-registered optical–SAR pairs, and bi-temporal pairs —
natively in GeoTIFF (it reads the map coordinates properly, not just the pixels).

---

## 3. Why not just ask ChatGPT?

This is the single most common judge question, and the problem statement itself answers it:
*"A generic LLM or VLM without remote-sensing adaptation will not satisfy the requirements."*

Concretely, big general-purpose vision models fail here because:

1. **Satellite pictures are alien to them.** They learned from Instagram-like photos. A top-down
   rice paddy or radar static confuses them.
2. **They ignore geography.** They see pixels, not coordinates, CRS metadata or pixel scale.
3. **They hallucinate confidently.** Ask a general model about a satellite image and it will
   invent a fluent answer with no way to check it. Unacceptable when floods or encroachment
   decisions ride on the answer.
4. **They are huge and cloud-dependent.** Billions of parameters needing datacentre GPUs and
   internet — useless inside ISRO/SAC's secure, often air-gapped facilities.

So the PS prescribes the opposite design: *several small specialised models + an agent that
routes between them.* That is exactly Anvesha's architecture.

---

## 4. The tour: how a question becomes an answer

```text
 You (browser console)
      │  upload images + type query
      ▼
 FastAPI server  ──► saves job, streams progress back to your screen
      │
      ▼
 AGENT CONTROLLER (the manager) — six fixed steps:
      1 VALIDATE    Are these real rasters? Right formats? Same map-grid?
                    Wrong pair? → precise error, BEFORE any model runs.
      2 CLASSIFY    What is the user asking? Match query wording to a task type;
                    ambiguous? → offers "did you mean…" options.
      3 SELECT      Pick specialist(s) from the registry that fit the inputs
                    (won't try change detection on a single image).
      4 EXECUTE     Run them, timing each step; chain multiple tools when needed.
      5 INTEGRATE   Merge text + visuals + a calibrated confidence score.
      6 REPORT      Write report.json/.md/.pdf + GeoTIFF masks into runs/<id>/.
      │
      ▼
 SPECIALISTS (each: shared encoder + own head + own fallback)
      │
      ▼
 Evidence returned to console: answer · overlays · swipe-compare · trace · downloads
```

**Worked example.** Query: *"What changed between these two dates, and where did the change occur?"*
with two GeoTIFFs attached.

1. Validate: both files open, are georeferenced, share the same grid → pass.
2. Classify: wording + two-date inputs ⇒ `change_analysis` intent.
3. Select: the change specialist (a single image would have been rejected here).
4. Execute: Siamese network compares tiles, builds a probability-of-change map, thresholds it,
   exports a GeoTIFF mask, generates a description and a direct answer.
5. Integrate: "Built-up increased ~3.2 ha, mainly north-east" + confidence 87% + overlay.
6. Report: everything written to disk; browser shows a swipe-compare slider.

For harder cases there's **Investigation Mode**: the controller chains
change detection → water grounding → impact analysis (distance to water, hectares affected,
zone ranking) — turning "something changed" into "X hectares changed within 500 m of a river,
prioritise review here."

---

## 5. Meet the seven specialists

All seven warm-start from one shared **SceneEncoder** — think of it as the common medical school
every consultant attended before specialising. It is a ResNet-18 (a compact, proven image
network) whose first layer was surgically adapted to accept any number of colour channels,
because radar pictures don't have "RGB" the way phone photos do. Its adaptation training:
EuroSAT land-type classification, **98.86%** validation accuracy (96px, 2500 images/class, 12 epochs, AMP, label smoothing).

| Specialist | Job | Trained on | Score (honest) | If weights are missing |
|---|---|---|---|---|
| **VQA** | Answers questions about one image; separate expert heads for yes/no, urban-vs-rural, comparisons, and counting | RSVQA-LR (54k image-question-answer triples) | 71% overall exact-match; per-type: presence **91%**, rural/urban 84% | Rule-based reasoner |
| **Captioner** | Writes scene descriptions via a small transformer decoder steered by a predicted content plan | BigEarthNet.txt captions joined to real Sentinel-2 patches | Multi-reference BLEU 0.32 | Deterministic template built from verified facts |
| **Grounding** | Finds water/vegetation/built-up regions you name, using spectral indices (NDWI/NDVI — simple, transparent formulas) | Not learned — deliberately interpretable | Correct within its domain; three *learned* variants were tried, measured poorly (≤15%), and refused shipping | n/a — it *is* the fallback |
| **Change detector** | Compares two dates; twin networks + multi-scale difference decoder; tiled inference for big scenes | LEVIR-CD | Change IoU 0.668 / F1 0.80 | Smoothed image differencing |
| **Change-VQA** | Answers questions about *what changed* between two dates, conditioned on the change detector's own internal difference features | CDVQA (39.7k test questions) | **68.3% accuracy — +17.4 points over the majority baseline, every question type above it** | Calibrated rule-based reasoner over spectral deltas |
| **Optical–SAR fuser** | Dual-branch network reads radar + optics together; reports agreement/complementarity | 14k genuine co-registered Sentinel-1+S2 pairs | Label recall 0.85 | Heuristic analyser |
| **Impact engine** | Turns "something changed" into hectares, distance-to-water, and ranked priority zones — pure auditable geometry, no AI | end-to-end over the change map | quantified findings | n/a — it *is* deterministic |

Two culture rules worth knowing:

- **Failed experiments are published, not hidden.** MODEL_CARDS.md records every grounded-out
  attempt (including a DINOv2 experiment that showed no gain and was rejected by a pre-agreed gate).
- **Fake checkpoints are refused at load time.** Any weight file trained on synthetic plumbing-test
  data carries a flag and the loader rejects it — so demo weights can never silently replace real ones.

---

## 6. The receipt — our biggest differentiator

Every run produces an **execution trace**: which task was selected, which models, which
parameters, how long each step took, what each produced. It is streamed live in the browser
*and* saved as a permanent artifact.

Why this matters: the PS states that only this observable trace will be evaluated — internal
reasoning counts for nothing. Most teams will show a chat window. We show a signed-style ledger.
When a government analyst must defend a finding months later ("why did the system say this?"),
the receipt exists. Nobody else in the field surveyed ships this end-to-end.

---

## 7. Trust features, in plain terms

- **Calibrated confidence.** Models are naturally overconfident. We fit a correction factor on
  held-out data (temperature scaling, T = 1.55), so when the UI says 87%, history says roughly
  87 out of 100 such answers were right.
- **Graceful degradation, never silence.** Every fallback announces itself in the output
  (`source_model` field). No silent wrong answers.
- **Provenance page.** Every number shown in the app comes from a measured checkpoint, re-runnable
  with one command: `python -m satquery.evaluate --all`.
- **Proven under pressure.** 96 automated tests pass offline (synthetic GeoTIFF fixtures), including
   tests that assert the *trained path is active* for every specialist — born from real bugs where
   the app looked fine while a model silently wasn't loaded. The server also exposes a live
   "model status" endpoint so heuristic-mode answers are never mistaken for model output.

---

## 8. Built for ISRO's exam conditions

The PS says final evaluation runs on **undisclosed ISRO/SAC data**: pre-georeferenced
Cartosat-2S optical + RISAT SAR pairs, graded against secret reference answers. Our responses:

- **Batch mode:** `python -m satquery.evaluate --sac-dir DIR` chews through a whole folder,
  writing per-item answers.csv + GeoTIFF change masks + traces, never halting on a bad file.
- **Radar-aware preprocessing:** dB-scale RISAT products are auto-detected (median-below-zero
  signature) and handled differently from power-scale data — treating them alike silently
  destroys them, a bug we hit and fixed.
- **Air-gapped & CPU-first:** Docker image with weights baked in; int8/TorchScript export
  measured at −0.16% accuracy delta; full pipeline runs with no GPU and no internet.
- **Load-tested:** 100 concurrent fresh sessions, 100/100 success, ~13 req/s, p95 ≈ 7 s on a laptop.

Sample ISRO-style inputs ship in `samples/` for instant demos.

---

## 9. The report card — and where we're honestly behind

| Skill | Our score | Context a judge should know |
|---|---|---|
| Single-image VQA | 71% overall | Below the original paper's 79% on the full test set; our per-type heads (presence 91%) are the stronger evidence. Known weakness, actively framed. |
| **Change-VQA** | **68.3%** | **+17.4 points over the majority baseline; every one of the 8 question types above it. Beats the CDVQA paper's own baseline (~68%).** Our strongest benchmark result. |
| Change detection | F1 80 / IoU 67 | Published SOTA reaches F1 ~92 with far larger GPU-trained models; ours is the CPU-class capability demo. |
| Optical–SAR | 85% label recall | Strong for a 14k-pair subset trained on a laptop. |
| Captioning | BLEU 0.323 (multi-ref) | Protocol-dependent metric; single-ref training BLEU 0.59. |
| Land-cover encoder | **98.86%** | Full-track (96px, 2500 images/class, 12 epochs, AMP) — exceeds published range (95–98.6%). |

The pitch is therefore **not** "our models beat the world" — it is
**"we are the only complete, auditable, deployable system covering every mandatory requirement,
with measured numbers and honest failure records."**

---

## 10. Try it yourself

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python start.py          # opens http://localhost:8000
# or fully offline:
docker build -t satquery . && docker run -p 8000:8000 satquery

python -m pytest tests -q            # 96 tests, offline-capable
python -m satquery.evaluate --all    # regenerate every published number
```

---

## 11. Explain-it-to-a-friend cheat sheet

**30-second version:**
"It's a website for asking satellites questions. You upload a satellite picture — or two from
different dates, or a photo-plus-radar pair — type 'what changed?' or 'where is the water?', and
it routes your question to small AI experts that were actually trained on satellite data. You get
an answer, a highlighted map, a confidence score, and a full receipt of what happened. It runs on
a laptop with no internet, because it's meant to work inside ISRO's secure labs."

**If they ask "isn't this just ChatGPT?"**
"No — ChatGPT never studied satellite imagery, ignores map coordinates, invents confident
answers, and needs the internet. Ours uses seven tiny specialist models with measurable accuracy
on standard benchmarks, refuses to hide its fallbacks, and logs every step so any answer can be
audited later."

**If they ask "how accurate?"**
"Strongest skills first: change questions 68% — seventeen points above the standard baseline,
beating the benchmark's own reference model. Yes/no questions 91%, radar+photo fusion 85%.
Overall single-image VQA is 71% against published baselines near 79% — we say that openly.
Change detection works reliably but isn't record-setting. The differentiator is the trustworthy
packaging, not raw leaderboard scores."

**If they ask "what was hardest?"**
"Our safety nets almost killed us: every model silently falls back to a simpler method when its
weights fail to load, so serious bugs produced zero crashes while accuracy quietly halved. We
learned to treat 'it works' as a claim requiring a test — now regression tests assert the real
model is active, not merely that code runs. And our own audit found a benchmark result sitting
*below* the baseline that nobody had re-checked — fixing it (a learned head conditioned on the
change detector's internal features) became our single biggest accuracy win: +17 points."

**If they ask "did any experiment fail?"**
"Several, and we publish them. Three learned grounding variants (best 15% IoU), a DINOv2
backbone swap (no VQA gain), and a density-map counting head (13% vs the 44% gate) — all
measured, all refused promotion, all documented. The same gate process promoted the change-VQA
head that won by 22 points. A process that can say no is what makes yes meaningful."
