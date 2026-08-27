# SatQuery AI (SIH26167) — Market Analysis & Full Codebase Audit Report

*Generated 2026-08-27 · Scope: benchmark/market research, competitor gap analysis, line-level verification of the claimed weaknesses/vulnerabilities table, and architecture recommendations.*

---

## Part 1 — Market & Benchmark Landscape

### 1.1 The metrics that matter for this PS

| Task | Benchmark | SOTA / Published reference | Source |
|---|---|---|---|
| Single-image VQA | RSVQA-LR | **79.1%** exact-match (Lobry et al., 2020, arXiv:2005.10256); RS-VLMs (GeoChat, EarthGPT) report 70–85% on RS VQA suites | Lobry et al.; GeoChat arXiv:2311.15826 |
| Captioning | VRSBench / BEN.txt-style | CIDEr/BLEU vary by protocol; RS-VLMs (GeoChat, TEOChat) dominate; BLEU-4 ~0.30–0.45 multi-ref is typical for RS captioners | VRSBench arXiv:2406.12384 |
| Grounding | VRSBench objects | Detection-grade models reach **IoU@0.5 ≈ 0.60–0.75** (fine-tuned FPN/DETR backbones) | VRSBench leaderboard |
| Change detection | LEVIR-CD | **F1 ≈ 0.90–0.92** (BIT-RN18 0.89, ChangeFormer 0.90, Changer/MaskCD ~0.92) | justchenhao.github.io/LEVIR |
| Change-VQA | CDVQA | Paper baseline (RN-18 encoder): **~0.68** accuracy; CDVQA paper (Yuan et al., 2024, github.com/YZHJessica/CDVQA) reports higher with dedicated change encoders | CDVQA repo |
| Scene classification | BigEarthNet / EuroSAT | EuroSAT trivially >95% with fine-tuned CNNs; BEN-19 mAP ~0.85+ for modern encoders | EuroSAT Zenodo 7711810 |

**Key insight:** the PS is *not* won on raw SOTA metrics — it is won on **agentic orchestration + evidence-grounded outputs + honest measurement**. No published open system currently combines all five mandatory workflows (VQA, captioning/grounding, change, optical–SAR, orchestration) behind one auditable controller. That is the differentiator.

### 1.2 Top 3 closest competitors and what they lack

**1. GeoChat (arXiv:2311.15826) — RS-adapted multimodal LLM (LLaVA-style)**
- *Strengths:* strong zero-shot VQA/captioning/grounding in one model; natural conversation.
- *Lacks:* **no bi-temporal change reasoning** (single-image only), no optical–SAR joint analysis, no tool registry/execution trace, no GeoTIFF/geospatial awareness, requires GPU-class LLM inference (7B+), no calibrated confidence, hallucination risk without evidence grounding.
- *Why the PS still exists:* the PS explicitly demands multitemporal + cross-modal pairs and auditable orchestration — GeoChat does neither.

**2. TEOChat / Change-Agent family (TEOChat arXiv:2410.06234; Change-Agent arXiv:2403.19646)**
- *Strengths:* temporal reasoning over image sequences (TEOChat); change detection + change dialogue (Change-Agent).
- *Lacks:* Change-Agent's change maps are strong but its **single-image VQA/captioning is generic**, no optical–SAR fusion head, no input-compatibility validation, no downloadable auditable reports, heavy training infra. TEOChat needs temporal stacks not bi-temporal GeoTIFF pairs and has no ISRO-style SAR handling.
- *Why the PS still exists:* none covers the full matrix (single + bi-temporal + cross-modal) with a validated, offline, CPU-deployable pipeline.

**3. Picterra / SkyFi (commercial no-code EO platforms)**
- *Strengths:* polished UX, detector training, production deployment.
- *Lacks:* **no natural-language agentic querying**, no VQA/captioning, no change-VQA, closed/proprietary, no execution trace, subscription cost, cloud-only (fails air-gapped SAC demo).
- *Why the PS still exists:* they are detection platforms, not vision-language assistants; the "ask a question, get evidence" workflow is absent.

**Gap conclusion:** the market gap = *agentic, auditable, multimodal (optical+SAR+temporal), offline-deployable, calibrated* RS assistant. SatQuery's architecture targets exactly this gap. What we lack vs. competitors is **raw model accuracy** (VQA aggregate, change F1, grounding) — the tradeoff we make for CPU-class deployability.

### 1.3 Our metrics vs. the field (from repo, verified)

| Metric | Ours | Reference | Verdict |
|---|---|---|---|
| RSVQA-LR aggregate EM | **0.69** (README:89) | 79.1% paper | −10 pts; counting head 0.45 drags it |
| Presence EM | **0.91** | GeoChat zero-shot ~0.70 | **We win** |
| Rural/urban EM | **0.84** | — | Strong |
| Counting EM | **0.45** | — | Weak (29.5% of questions) |
| LEVIR-CD IoU/F1 | **0.668 / 0.801** | BIT 0.81/0.89; SOTA ~0.92 | −14 F1 vs BIT; capability demo |
| Captioning BLEU multi-ref | **0.32** (single-ref 0.59) | protocol-dependent | Honest, hard to compare |
| EuroSAT val acc | **0.91** (quick track; README claims 98.86% full track) | >95% typical | Fine |
| Optical–SAR label recall | **0.85** | — | Good |
| **CDVQA full test (35,212 Q)** | **0.4882 vs 0.5079 majority baseline (Δ −1.97%)** | paper baseline 0.68 | 🔴 **NEW CRITICAL FINDING — see §3.1** |

---

## Part 2 — Verification of the Claimed Weaknesses Table

Every claim was checked against the actual code. Verdicts:

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| W1 | VQA 69% vs 79%; counting drags it | ✅ **TRUE** | README.md:89, 91–92; counting 0.45, 29.5% share |
| W2 | Change IoU 0.67/F1 0.80 vs SOTA 0.92 | ✅ **TRUE** | README.md:98; MODEL_CARDS.md:72 (test-subset 0.60/0.75 — note the README number is the better of two protocols) |
| W3 | Captioning BLEU 0.32 multi-ref | ✅ **TRUE** | MODEL_CARDS.md:46–48 |
| W4 | No learned grounding (3 variants ≤15% IoU) | ✅ **TRUE** | MODEL_CARDS.md:54–64 — 0.15/0.12/0.018, all gated off; spectral-index fallback shipped |
| W5 | GPU semaphore = 2 | ✅ **TRUE** | `jobs.py`: `threading.BoundedSemaphore(2)` when CUDA available |
| W6 | web/dist committed; dev-dep leak risk | ⚠️ **PARTLY** | dist IS committed (7 files, two generations of hashed chunks present = drift already happened); **no dev deps leak** — but `@types/three` is wrongly in `dependencies` (package.json:13) |
| W7 | 91 tests; integration skips | ⚠️ **PARTLY** | Suite runs: **90 passed, 1 FAILED** (16.3 s) — see §3.2 |

### Vulnerability verification

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| V1 | Silent fallback on weight-load failure | ✅ **TRUE — CONFIRMED RISK** | `vqa.py`, `scene.py`, `optical_sar.py`, `change.py` all do `try: … self.trained=True / except Exception: self.trained=False` → heuristic fallback. Only `test_agent.py:96–106` asserts `trained=True`, and **only for SceneEncoder**. VQA/Change/Fusion have no such regression test. |
| V2 | Cache key = sha256(file bytes + query + params) | ✅ **TRUE, acceptable** | `jobs.py::cache_key_for` — sorted paths, file-bytes hash, query, params. Query is included; collision risk negligible. TTL + corrupt-JSON row deletion handled in `store.py`. |
| V3 | GPU OOM on large images | ✅ **TRUE, mitigated** | `config.py:75` max 2048 px; `jobs.py` catches `MemoryError` → distinct `"MemoryError (OOM)"` bucket. |
| V4 | SAC pair grouping heuristic (60% stem prefix) | ✅ **TRUE** | `evaluate.py`: greedy `os_prefix` match, threshold `max(6, 0.6*min(len))`. Non-conforming ISRO names → missed pairs; batch never halts. |
| V5 | dB-scale SAR `median < 0` heuristic | ✅ **TRUE** | `backbone.py` (`if float(np.median(a)) < 0:` → treat as dB) and mirrored in `io_utils.py` (`median >= 0 → log1p`). Power-scale SAR with negative noise floor would be misrouted. Documented assumption only. |
| V6 | Frontend–backend version drift | ✅ **TRUE** | Committed dist contains stale chunk hashes; `start.py` serves built SPA — rebuild required before demo. |
| V7 | No auth by default | ✅ **TRUE** | `SATQUERY_TOKEN` optional; `Depends(_check_auth)` no-ops without it. Port 8000 on venue LAN = open API. |

---

## Part 3 — NEW Findings the Claimed Table Missed

### 3.1 🔴 CRITICAL: CDVQA full-test result is below the majority baseline
`weights/cdvqa_results.json`: **overall 0.4882 vs baseline 0.5079 (Δ −1.97%)** across 35,212 questions. Per-type: `change_or_not` 0.547 (base 0.563), `decrease_or_not` 0.629 (base 0.689, **−5.9 pts**), `largest_change` 0.341 (base 0.429, **−8.8 pts**), `smallest_change` 0.190. Only `change_ratio` (+7.5) and `change_to_what` (+5.2) beat baseline. **The change-VQA head does not generalize to the full CDVQA test set.** If judges run the prescribed CDVQA test split, this number is exposed. The README's CDVQA row is empty ("—") — this is why. *This is the single biggest metric risk in the project and was not in your table.*

### 3.2 🔴 One test FAILS on the current tree
`tests/test_w2.py::test_clarification_surfaces_in_run_outputs` — for ambiguous query `"something"`, the run returns a clarification object while `res.confidence >= 0.55` (or vice-versa): the clarification gate and reported confidence are inconsistent. Also revealed: **mojibake in the clarification string** — `agent.py` emits `ù` instead of `—` (em-dash) in "Low confidence in the requested analysis — did you mean:", i.e., a file-encoding corruption that will show in the UI. "91 tests pass clean" is currently **false**.

### 3.3 🟡 README metric inconsistency
MODEL_CARDS says change test-subset **IoU 0.60/F1 0.75**; README says **0.668/0.801** (thr=0.85). Two protocols, one headline. Judges cross-reading both docs will probe. Pick one canonical number and footnote the other.

### 3.4 🟡 README claims 98.86% EuroSAT val acc; MODEL_CARDS says 0.91
Same model, two numbers (full track vs 200/class quick track). Reconcile before the demo.

### 3.5 🟡 `@types/three` in production `dependencies` (web/package.json:13)
Compile-time-only package shipped as a runtime dep. Cosmetic but flagged in dependency audits.

### 3.6 🟡 Committed `web/dist` already contains two generations of hashed chunks
Drift has already occurred once. Either CI-rebuild dist on every backend API change, or stop committing dist and make `start.py` fail loudly with "run npm build" when the API version and dist manifest mismatch.

### 3.7 🟡 `eval_cdvqa.py:22` RuntimeWarning: invalid value in divide
`np.where(np.abs(b) > eps, a/b, 0.0)` evaluates `a/b` before masking — NaN/inf computed then discarded. Harmless numerically but noisy in logs; use `np.errstate` or masked division.

---

## Part 4 — Design & Architecture Audit (technical standpoint)

**Strengths (keep):**
1. Registry-based orchestration with feasibility filtering (`agent.py:234–243`) — exactly what the PS demands; auditable trace is a genuine differentiator.
2. Hybrid intent classifier (0.6·keyword + 0.4·BOW-embedding with precomputed centroids) — deterministic, offline, testable; better than an LLM router for a demo venue with no internet.
3. Honesty engineering: `source`/`source_model` fields on every output, synthetic-checkpoint refusal, gated experiments (DINOv2 rejection documented) — this is *exactly* the right defense posture for judges.
4. Ops maturity: bounded queue → 429, LRU job eviction, GPU semaphore, torch thread caps, SQLite WAL with lock, TTL cache, corrupt-row self-healing, structured logs, load-tested (100/100 @ ~13 req/s).

**Weaknesses (design):**
1. **Fallback opacity at the system level.** Each model hides its own degradation (`trained=False` → heuristics), but the *run report* doesn't prominently surface "this answer came from spectral heuristics, not the trained model." A judge watching the demo can't tell. → Add a top-level `degraded: true` banner in the API response + UI when any loaded model is untrained; add `trained=True` regression tests for VQA, Change, Fusion (mirror `test_agent.py:96–106`).
2. **Monolithic `tools_impl.py` + string-typed task names.** Tasks are stringly-typed across agent/registry/server; a typo'd task id fails only at runtime. → Introduce an `Enum`/`Literal` TaskId shared by agent, registry, and API schema.
3. **No model-version pinning in cache keys.** Cache key = bytes+query+params; if you swap weights, stale cached answers persist until TTL. → Include a `weights_fingerprint` (hash of weight-file mtimes/sizes) in the cache key.
4. **SAR modality detection is a guess, not metadata-driven.** GeoTIFF bands carry no modality tag; `median<0` is fragile. → Prefer explicit metadata: filename hints (`_sar`, `risat`), band-count + dtype heuristics, and an API-level `modality` override param the user/judge can set.
5. **Single-process, in-memory job store.** Jobs live in one Python process; a crash loses the queue (history survives in SQLite). Acceptable for demo; document it.
6. **Grounding is the weakest functional pillar** (no learned model). The honest defense is good, but the *value* fix is cheap: ship **click-to-query region-crop VQA as the headline grounding UX** (it works) and frame boxes as "spectral-index evidence regions," not detections.

## Part 5 — Architecture Brainstorm (better performance & value, without breaking the app)

Ranked by (impact ÷ risk):

1. **Fix CDVQA generalization (highest metric ROI).** The per-type deltas show the head learned *class-prior noise*, not change. Options: (a) train a dedicated change-VQA head conditioned on the **change-detector's difference features** (you already compute them) instead of raw dual-encoder embeddings; (b) class-balanced sampling per answer bucket; (c) calibrate per-type decision thresholds on validation. Even reaching baseline+3 pts transforms the story.
2. **Counting head rescue (+3–4 aggregate VQA pts).** Counting is 29.5% of RSVQA at 0.45. Replace regression-to-digit with an **ordinal/CORAL head over bins 1–10 + "many"** (you already ship CORAL elsewhere per README:159) plus density-map TTA. Aggregate 0.69 → ~0.73 closes half the gap to the paper's 79%.
3. **TTA-on-by-default for change (F1 0.80 → ~0.83).** You already have 4-way TTA behind a flag; enable it when the GPU semaphore is free, keep it off under load. Zero new code paths.
4. **Degradation transparency layer (1 day).** `GET /api/provenance` already exists — extend every job response with `model_status: {vqa: trained, change: trained, fusion: heuristic}` and render a UI badge. Converts V1 from a demo-killer into a documented feature.
5. **Weights-fingerprinted cache keys + trained=True tests for VQA/Change/Fusion (half day).** Closes V1 and the cache-staleness hole.
6. **Fix the failing test + mojibake (minutes).** `test_w2.py:189` gate consistency; re-encode `agent.py` clarification string as UTF-8 em-dash.
7. **Modality override param on `POST /api/jobs`** (`modality: "sar"|"optical"|"auto"`) — removes the dB-heuristic single point of failure for the ISRO set.
8. **Later (post-demo):** swap SceneEncoder for a DINOv2-window fine-tune (your gate showed +3.2 EuroSAT pts; revisit with unfrozen last block for grounding), and an FPN detection head for learned grounding — the documented upgrade path in MODEL_CARDS.

---

## Part 6 — Coverage Score vs. the PS

| PS mandatory requirement | Coverage |
|---|---|
| RS adaptation (BigEarthNet.txt or open data) | ✅ SceneEncoder (EuroSAT) + captioner (BEN.txt) + fusion (reBEN S1+S2) |
| Single-image VQA (mandatory) | ✅ 0.69 aggregate / 0.91 presence |
| Second single-image task | ✅✅ Both captioning AND grounding |
| Bi-temporal change (mandatory) | ✅ detection + description + change-VQA + GeoTIFF masks (metric risk: §3.1) |
| Optical–SAR pair analysis (mandatory) | ✅ dual-branch fusion, recall 0.85 |
| Agentic orchestration (mandatory) | ✅ validate→classify→select→execute→integrate→report, auditable trace |
| GUI/web app, evidence, confidence, reports | ✅ React console, calibrated confidence, PDF/MD/JSON reports |
| Formats GeoTIFF/TIFF (+benchmark PNG/JPEG) | ✅ with CRS/geometry validation |

**Functional coverage: ~95% of the mandatory scope.** The residual risk is entirely in **metric credibility** (CDVQA below baseline, VQA −10 pts, change −14 F1) and **demo reliability** (silent fallback, failing test, auth off by default) — all addressable via the Part 5 list.

## Sources
- Lobry et al., *RSVQA*, arXiv:2005.10256 · zenodo.org/records/6344333
- *VRSBench*, arXiv:2406.12384 · huggingface.co/datasets/xiang709/VRSBench
- *CDVQA*, github.com/YZHJessica/CDVQA (Yuan et al., 2024)
- *GeoChat*, arXiv:2311.15826 · *TEOChat*, arXiv:2410.06234 · *Change-Agent*, arXiv:2403.19646 · *EarthGPT*, arXiv:2404.16800 · *LHRS-Bot*, arXiv:2402.02544
- LEVIR-CD benchmark: justchenhao.github.io/LEVIR (BIT: Chen et al., arXiv:2108.10470; ChangeFormer: arXiv:2208.01037)
- BigEarthNet v2: zenodo.org/records/10891137 · EuroSAT: zenodo.org/records/7711810
- Internal: README.md, MODEL_CARDS.md, weights/cdvqa_results.json, pytest run (2026-08-27), line-level code inspection (agent.py, jobs.py, store.py, config.py, backbone.py, io_utils.py, evaluate.py, test_agent.py, test_w2.py, web/package.json)