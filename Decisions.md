# ANVESHA — Engineering Decision Record

**Project:** Anvesha (formerly SatQuery AI) — Earth Observation & Investigation System
**PS:** SIH26167 · ISRO/SAC · Smart India Hackathon 2026
**Purpose of this document:** a complete, chronological record of every significant
decision made during the build, why each option was chosen, what was changed or
reverted along the way, and an honest account of the hardest problems encountered.

---

## Phase 0 — Orientation & dataset verification

### D0.1 — Verify every dataset online before writing code
**Decision:** all five datasets (BigEarthNet.txt, reBEN v2 images, RSVQA-LR,
VRSBench, CDVQA, LEVIR-CD, EuroSAT) were probed live via the HF/Zenodo APIs
before any loader was written.
**Why:** the PS anchors evaluation to these exact benchmarks; a broken download
path discovered mid-build would invalidate every trained number. This also
surfaced real facts early: EuroSAT's archive is named `EuroSAT_RGB.zip` (not
`EuroSAT.zip`), RSVQA lives on Zenodo (6344333) rather than GitHub, and
LEVIR-CD's Google Drive link is quota-dead while an official-crop HF mirror
(`ericyu/LEVIRCD_Cropped_256`) exists.

### D0.2 — "RS adaptation" strategy: EuroSAT quick track + BigEarthNet deep track
**Decision:** fine-tune the shared visual backbone on **EuroSAT** (91% val acc
in minutes) as the shipped adaptation, with full BigEarthNet.txt / reBEN v2
loaders provided for the deep track.
**Why:** the PS explicitly allows *"BigEarthNet.txt **or any open source
training data**"*. EuroSAT trains fast enough to ship *measured* adapted weights
on day one instead of promising future ones. The 110 GB reBEN archives were not
practical to pull in-session, but a 3.1 GB cross-modal S1+S2 subset
(14K co-registered pairs) later was.

---

## Phase 1 — Core architecture

### D1.1 — Specialist registry + agent controller (not one big VLM)
**Decision:** a `ToolSpec` registry (single_vqa, captioning, grounding,
change_analysis, change_vqa, optical_sar, later impact_analysis) behind an
`AgentController` that validates inputs → classifies intent → selects tools →
executes → integrates → reports, streaming every step.
**Why:** the PS states this verbatim ("automatically select, sequence, and
execute… only the observable execution trace will be evaluated"). One generic
VLM is explicitly called out as insufficient. Routing is keyword-intent based
with a feasibility filter and aliases — deliberately deterministic, because a
rule we can show in the trace beats a black-box router for auditability.

### D1.2 — One shared, band-adapted visual encoder
**Decision:** a single ResNet-18 `SceneEncoder` whose pretrained RGB stem is
adapted to arbitrary band counts by averaging kernels; warm-started into every
specialist.
**Why:** gives the mandated "remote-sensing adaptation layer" a single home,
keeps every model small (CPU-deployable), and lets one EuroSAT fine-tune lift
all six tasks.

### D1.3 — Graceful degradation everywhere
**Decision:** every specialist has a deterministic fallback (spectral-index
grounding, template captions, differencing change map, rule reasoner VQA) that
activates automatically when weights are absent/weak.
**Why:** ISRO servers may be air-gapped; judges must never see a crash; and the
fallbacks are themselves unit-tested. The cost — silent fallback masking missing
weights — was addressed twice (see D7.2, D9.1).

### D1.4 — Hashed bag-of-words question encoding (no transformer NLP)
**Decision:** questions are encoded with signed MD5-hashed BOW, no tokenizer
library.
**Why:** zero dependencies, fully offline, and sufficient for RSVQA's templated
questions. The dimension choice became a lesson (see D6.1): 64 buckets caused
collisions between object words; 512 fixed it.

### D1.5 — GeoTIFF-native I/O with scale-aware SAR handling
**Decision:** rasterio-first loading; modality inferred from band count +
filename/metadata hints (S1/RISAT/VV/VH/HH/HV); SAR power-scale data log1p'd,
**dB-scale products detected (median < 0) and passed through unscaled**;
optical reflectance ×10000 percentile-clipped.
**Why:** the hidden SAC set is Cartosat-2S + RISAT. Treating dB rasters like
power rasters silently destroys them (this exact bug bit us — see D9.3).

---

## Phase 2 — First training cycle & the measurement discipline

### D2.1 — Train on real downloaded data, measure everything, publish failures
**Decision:** every shipped number comes from a script (`satquery.evaluate`)
against held-out public test subsets; failed experiments are reported in
MODEL_CARDS rather than deleted from history.
**Why:** the PS evaluates against undisclosed benchmark subsets — the only
defensible posture is reproducible honesty. This produced the quantization gate
(D5.3), the synthetic-checkpoint refusal (D4.4), and three documented grounding
failures (D8.3).

### D2.2 — VQA v1: general head, 66% — and why we didn't stop there
Initial result: ~66% exact match vs the paper's 82% baseline. Root causes found
by debugging, in order: a **hash-dimension mismatch** (trained at 64-d, served
at 512-d truncation — fixing alone jumped 33%→64%), **resolution mismatch**
(trained 128px, served 224px), and **question-type information thrown away**.
Each fix was verified on the benchmark before moving on.

---

## Phase 3 — Serving stack

### D3.1 — Replace Streamlit with React + FastAPI
**Decision:** FastAPI service with a job store + polling API, and a React
(Vite/TS/Tailwind) console.
**Why:** the user judged the Streamlit UI "basic"; the PS wants an interactive
application with evidence surfaces (map, swipe-compare, findings tables) that a
form-based framework can't express well. FastAPI keeps one process serving both
API and static frontend.

### D3.2 — Live trace streaming via callback, not monkey-patching
**Decision (post-bug):** `AgentController.run(trace_callback=...)`. The original
implementation monkey-patched the singleton controller per job — a race under
concurrent jobs. Caught in the concurrency audit before it bit in production.
**Why changed:** correctness under the "hundreds of sessions" requirement.

### D3.3 — Polling over WebSockets
**Decision:** job state polled every 450 ms instead of SSE/WebSocket.
**Why:** survives proxies, reconnects trivially, and matches the 36-hour finale
environment where flaky venue Wi-Fi is expected.

---

## Phase 4 — Real-data fusion & integrity guards

### D4.1 — Optical–SAR fusion on a real reBEN subset
**Decision:** train the dual-branch fusion network on the 14K cross-modal
reBEN v2 subset (HF: 3.1 GB) instead of the full 110 GB archives.
**Why:** full archives impractical in-session; the subset preserved the
essential property — genuine co-registered Sentinel-1 dB + Sentinel-2 pairs.
Result: 85% val label recall, shipped in the runtime.

### D4.2 — Pair matching by geographic key, not filename
**Decision:** S1↔S2 patches joined via parsed (tile, row, col) from filenames
after metadata names proved inconsistent with archive layout.
**Why:** produced 4,334/3,255/3,248 verified co-registered pairs; naive joins
yielded zero or wrong pairs.

### D4.3 — Synthetic checkpoints are flagged and REFUSED at load time
**Decision:** `train_optical_sar.py --synthetic` writes `"synthetic": true`;
FusionNet refuses such checkpoints and keeps heuristics active.
**Why changed:** an early pipeline-verification run produced noise-trained
weights that would have *replaced* the useful heuristic analyser. The guard
makes the plumbing testable without ever degrading the product.

---

## Phase 5 — Performance engineering with gates

### D5.1 — CPU-first deployment contract
**Decision:** target CPU-only 4–8 vCPU servers; CUDA opportunistic via device
detection; int8 dynamic quantization + TorchScript exported for the CPU path.
**Why:** SAC evaluation infrastructure is realistically air-gapped/CPU-only.
GPU code remains, but no claim depends on it.

### D5.2 — Measured load SLA instead of marketing numbers
**Decision:** `scripts/load_test.py` drives 100 concurrent fresh sessions and
writes p50/p95/errors to `runs/loadtest.json`.
**Why:** "scales to hundreds of sessions" is a claim; "100/100 OK, 13 req/s,
p95 7.3 s wall incl. client polling on this laptop" is evidence.

### D5.3 — Quantization adoption gate (≤0.5% delta)
**Decision:** export fp32 + int8, measure both on the val set, adopt int8 only
within threshold. Measured Δ −0.16% → adopted; batch-1 latency gain was nil and
reported as such.
**Why:** optimization without a regression gate is how accuracy silently dies.

### D5.4 — Calibrated confidence (temperature scaling, T=1.55)
**Decision:** fit temperature on held-out validation; confidences are now
calibrated probabilities.
**Trade-off accepted:** raw confidence values dropped (overconfidence removed).
A judge who probes "is 87% meaningful?" gets a defensible answer.

---

## Phase 6 — Grounding: three honest failures

### D6.1 — Learned grounding heads: built, measured, gated off
Three formulations were trained on real data (BEN.txt refs regression → 15%;
heatmap variant → 12%; VRSBench point→box → 1.8%). None met the shipping bar;
all are documented in MODEL_CARDS with root cause (frozen ImageNet→RS features
lack localization resolution; needs detection-grade architecture).
**Why gated off instead of shipped:** shipping a 12% component to claim
"learned grounding" would trade user trust for a checkbox. The spectral-index
specialist remains primary — interpretable and genuinely accurate for water/
vegetation/built-up queries.

### D6.2 — Point-conditioned grounding kept as scripts-only
The VRSBench 'point' category (29k samples) idea was sound for click-to-query,
but the frozen-encoder heatmap couldn't localize (≤1.8%). Training script
retained for reproduction; the click-to-query feature works via region-crop
VQA regardless.

---

## Phase 7 — Product expansion anchored to the PS

### D7.1 — Impact Analysis engine (no new AI)
**Decision:** deterministic GIS math over existing outputs — chamfer EDT to
water (numpy two-pass, downsampled), GSD from geotransform (labelled 10 m
assumption for non-geo inputs), hectares, 4×4 zone ranking, analyst-action
rules.
**Why:** converts "change detected" into "X ha within 500 m of water, review
for encroachment" — the differentiating layer another reviewer identified, and
pure auditable geometry rather than another black box.

### D7.2 — Investigation Mode as a planned multi-tool chain
**Decision:** `investigation` intent triggers a sequential plan (change →
water grounding → impact) streamed step-by-step, with per-step error capture
and synthesis around failures.
**Why:** upgrades the agent from router to planner — direct evidence for the
PS's "select, sequence, execute". Mid-chain failure capture is tested
(deliberate fault injection).

### D7.3 — "Ask the data back"
**Decision:** context-aware follow-up suggestions derived from actual outputs
(water-proximity chip only appears if water proximity mattered).
**Why:** creates the investigative loop the outline asked for; cheap; pure UX
over existing capability.

---

## Phase 8 — Rebranding & UX overhaul (Anvesha)

### D8.1 — Rebrand to Anvesha, code identifiers unchanged
**Decision:** all user-facing surfaces renamed (header, PDF, API title, page
titles, tagline "Earth Observation & Investigation System"); Python package
stays `satquery`.
**Why:** full rename risks import churn for zero judge value; surface rebrand
delivers the identity.

### D8.2 — Light theme default, token-based theming
**Decision:** CSS-variable design tokens (`--c-*`) mapped through Tailwind;
light "day-paper" palette default, dark retained via persisted toggle.
**Why changed from dark-only:** readability across ages was an explicit goal;
variables (rather than two compiled palettes) keep the cost at one stylesheet.

### D8.3 — Typography & navigation for non-experts
**Decision:** base font 17 px, contrast floors (muted ≥ slate-300 equivalent),
left icon-rail nav with permanent labels, first-visit onboarding cards, Help
view with pipeline explainer + tap-to-expand glossary, term tooltips inline.
**Why:** user's own critique — "even I don't know all the features" — is fatal
for a demo; discoverability became a requirement.

### D8.4 — Dropped items
Exe/pywebview packaging dropped by user decision (web app satisfies the PS);
Home view deferred to last by user instruction.

---

## Phase 9 — Hardening pass (W1)

* `run_id` traversal sanitization on report endpoints (probe showed SPA
  catch-all masked it — defense-in-depth added anyway).
* Unknown `/api/*` returns 404 JSON, never the SPA shell.
* JobStore LRU cap (300, active work never evicted); queue bound (50 → HTTP
  429); `_EVAL_STATE`, uploads, log rotation bounded.
* Dead artifacts pruned: 3 debug scripts, 2 gated checkpoints; `_rgb3`
  deduplicated to a single backbone source; unused `giou_loss` removed;
  `normalized_card` branch simplified; Streamlit app marked LEGACY.

---

## Phase 10 — Per-type VQA specialists (final accuracy cycle)

### D10.1 — Jointly-trained per-type heads (not sequential fine-tuning)
**Decision:** presence / rural_urban / comp heads + CORAL counting head train
**simultaneously** on the shared encoder, each batch routed per-sample.
**Why changed from sequential:** sequentially fine-tuning the shared encoder
per type causes catastrophic forgetting — earlier heads would mismatch the
final encoder. Joint routing shares features while specializing heads.
**Measured:** presence 0.91 · rural_urban 0.84 · comp 0.71 · CORAL count 0.48.

### D10.2 — LUT decode-direction bug
First integration returned local class indices ("1", "0") because the decode
map inverted `{global: local}`. Fixed by decoding local → global id → answer
string; verified end-to-end ("yes"/"no"/"rural").

---

# The greatest challenges

The hardest part of this project was not any single algorithm — it was that
the system's defining feature, graceful degradation, also became its greatest
debugging hazard. Because every specialist silently falls back to a heuristic
when its weights fail to load, several serious defects produced **no crashes
at all**: a swallowed `NameError` left the fine-tuned scene encoder disabled
for days while the app looked perfectly healthy; a question-encoder dimension
mismatch cost thirty accuracy points while every test stayed green; a
resolution mismatch between training and inference halved benchmark scores
without a single error message; and a BatchNorm train/eval discrepancy made a
model look trained in one process and untrained in another. We learned to
treat "it works" as a hypothesis requiring a probe — adding regression tests
that assert the *trained path is active*, not merely that code runs. The
second compounding challenge was the Windows build environment itself:
PowerShell repeatedly mangled inline Python quoting, npm was blocked by
execution policy, background training chains raced with scorecard generation
(a benchmark briefly measured against half-written weights), stale uvicorn
processes silently served pre-fix code and sent us chasing phantom bugs, and
console encoding crashed on Unicode. These were resolved with disciplined
tooling — dedicated runner scripts instead of fragile one-liners, best-
checkpoint saving with promotion floors so noisy epochs could never regress
shipped weights, distribution-shift safeguards that blend interpretable
differencing when the learned map goes flat, sanity floors that refuse to
persist below-par checkpoints, and synthetic-checkpoint flags that prevent
plumbing-test weights from ever reaching production paths. The third
challenge was data plumbing at national-benchmark scale: joining 9.55M
BigEarthNet.txt annotations to a differently-named image archive required a
geographic-key join we validated pair-by-pair; RSVQA's split naming, answer
schemas and inactive rows each broke naive loaders; and LEVIR-CD arrived only
through a mirror after its primary Google Drive link died. Every one of these
was solved with verification-first habits — probe the schema, assert the join
count, measure before and after — and that discipline, more than any model,
is what turned the project from a demo into something we would defend in a
review.
