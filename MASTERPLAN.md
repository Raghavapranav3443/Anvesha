# MASTERPLAN — Anvesha / SatQuery AI · SIH PS 26167 · Target: Grand Finale 1st Place

> Operating document. The plan is the contract. Gates and STOP conditions fire loud on purpose.

## STATUS (updated 2026-08-28, post Act-mode execution)

- **Phase 0 — DONE + committed.** `satquery/evaluate.py` now wires a 6-row scorecard:
  RSVQA-LR, LEVIR-CD, BigEarthNet captions, **VRSBench-val captioning (BLEU-4)**,
  **VRSBench-val grounding (IoU@0.5)**, **CDVQA (val, answer-match)**. All new benches
  validated on tiny n. Baseline tagged `baseline-2026-08-28`. `pytest`: 96 passed.
- **Measured rows (bounded-n acceptance):** VRSBench caption BLEU-4 = 0.0 (honest 0;
  refs share no 4-grams with generated captions), VRSBench grounding IoU@0.5 ≈ 0.177
  (n=6), CDVQA = 0.0 (n=2). These are the scored-but-absent rows now surfaced.
- **CDVQA blocker RESOLVED:** CDVQA `file_name` (e.g. `02180.png`) maps 6400/6400 to
  `data/SECOND/SECOND_test/{im1,im2}` bi-temporal pairs; answers keyed by question_id.
- **Phase 1 dependency STOP (environment constraint):** CLIP weight download
  (`openai/clip-vit-base-patch32`, ~350 MB) cannot complete inside this sandbox's
  hard 30s tool window — `start /b` children are killed on tool timeout, no persistent
  HF cache write. **Pre-registered gate script written + committed**
  (`scripts/gate_clip_grounding.py`, adopt rule: VRSBench grounding IoU@0.5 ≥ 0.30).
  Phase 1 cannot proceed until CLIP weights are available offline.

---

## 0. Success criterion (ironclad, non-negotiable)

**We win PS 26167 at the SIH Grand Finale. Anything that does not move us measurably
toward that is cut. Any experiment that risks it is gated, budgeted, and rolled back
on failure. Missing a scored benchmark row is disqualifying; embarrassing on a scored
row is losing.**

Deadlines: **internal round ≈ 1 week** (ship frozen + one visible new capability) and
**final screening ≈ 1 month** (numbers moved everywhere).

---

## PART A — How it is now (brief, verified)

- **Architecture:** `AgentController` (satquery/agent.py) routes via keyword+BOW intent →
  `ToolSpec` registry (satquery/registry.py, 7 tools) → specialists sharing `SceneEncoder`
  (ResNet-18, EuroSAT full-track 98.86%). Trace + report + provenance + SQLite + Docker +
  TorchScript-int8 gates exist (structural differentiator).
- **Measured scorecard** (`runs/scorecard.json`, from `python -m satquery.evaluate --all`,
  which runs **only 3 benches**):

  | Row | Current | Reference |
  |---|---|---|
  | RSVQA-LR | **0.7021** (n=282) | Lobry 2020: 0.79 |
  | LEVIR-CD (subset) | **IoU 0.602 / F1 0.752** | full-test thr=0.85: 0.668/0.801 |
  | BigEarthNet captions (multi-ref BLEU) | **0.283** | — |
  | Combined normalized | **0.5291** | — |

- **Scored-but-missing rows (disqualifier risk):** VRSBench **captioning** (crude
  token-recall in scripts/run_benchmarks.py:89, **not in scorecard**, no real BLEU-4),
  VRSBench **grounding** (no eval exists; spectral heuristics shipped; learned variants
  failed 0.15/0.12/0.018), **CDVQA** (harness exists, **not in scorecard**; JSONs on disk
  + SECOND images map to CDVQA ids — pairs extractable), VRSBench-VQA (nothing).
- **VLM gap (PS compliance bullet):** no remote-sensing-adapted vision-language component.
  VQA text side is **md5-hash BOW**; adaptation was label-supervised on EuroSAT, not the
  PS-mandated "adapting image–text representations" on **BigEarthNet.txt**.
- **Hardware / env:** Windows · RTX 5050 Laptop **8 GB VRAM** · weights/ 525 MB ·
  requirements.txt has **no** transformers/open_clip/nltk. Data: vrsbench, bentxt_join
  (captions+refs), SECOND (train+test zips — train zip contains total_test.zip),
  bigearthnet_14k, LEVIR-CD, rsvqa_lr, eurosat, CDVQA. Git main @ b3842ed.
- **Proven patterns to reuse:** `scripts/gate_dinov2.py` (pre-registered gate),
  `DinoEncoder` (offline bundle pattern), fallback `source_model` labeling,
  `CONFIG.resolve_device()`, TorchScript-int8 export (`scripts/export_torchscript.py`).

---

## PART B — What to implement (detailed, executable)

### Phase 0 — Compliance measurement & freeze (this week, ≤ 3 days)

1. Clean git state; tag `baseline-2026-08-28`. Experiments branch from here.
2. Extend `satquery/evaluate.py` → full 6-row scorecard:
   - `bench_vrsbench_caption` — real BLEU-4 on `data/vrsbench` (reuse `simple_bleu`).
   - `bench_vrsbench_grounding` — spectral grounding against val refs → IoU@0.5
     (this is the gate baseline ≈ 0.15).
   - `bench_cdvqa` — wire in; extract pairs from `data/SECOND` (ids match CDVQA
     file_name, e.g. `02180.png`).
   - `bench_bigearthnet` — fix "prediction coverage" stub → micro-F1 vs refs.
3. `pytest` green; `evaluate --all` reproduces ≈ 0.5291 with new rows appended.

### Phase 1 — RS-adapted VL component: grounding gate (this week, ≤ 4 days)

1. Add `open_clip_torch`; download CLIP-class RS checkpoint (RemoteCLIP/GeoRSCLIP-class,
   ViT-B ~150M) into weights/. Bundle like dinov2_vits14.pt.
2. New `satquery/models/vl_encoder.py` — `VLCEncoder` dual encoder (duck-types
   SceneEncoder/DinoEncoder pattern), proj 256, device resolve.
3. New `scripts/train_vl_encoder.py` — contrastive InfoNCE on `bentxt_join/captions.parquet`.
   Low-LR backbone, trainable projection. AMP, batch 32–64 (8 GB).
4. Grounding via region-text similarity (sliding windows/FPN-lite props) → boxes+masks in
   the existing `grounding` schema, `source_model="vl_grounding"`. Spectral path = fallback.
5. **Pre-registered gate (written before running):** adopt iff **VRSBench val IoU@0.5 ≥ 0.30**
   (2× spectral). Budget ≤ 2 attempts / ≤ 36 GPU-h.
6. Registry: new `ToolSpec` (`vl_grounding`) or flagged internals; old tool loads unchanged.
7. Internal-round demo: frozen everything + VRSBench rows in scorecard + grounding boxes.

### Phase 2 — VQA + captioning re-platform (final-prep month, days 1–10)

1. VQA text side: replace md5-BOW with `VLCEncoder.text_embedding`; retrain type heads.
   **Gate:** RSVQA aggregate EM ≥ 0.75 AND presence ≥ 0.91 (no regression). Fail → flip flag
   to BOW+type-heads, publish negative.
2. Captioning: decode from VL image embeddings (retrain `Captioner`); BLEU-4 on VRSBench +
   multi-ref on BEN. **Gate:** multi-ref ≥ 0.35 AND VRSBench BLEU-4 ≥ 0.15. Fail → keep
   current decoder path.
3. Optional: zero-shot open-vocab classification (CLIP text prompts) — only if core green.

### Phase 3 — Change detection upgrade (final-prep month, days 5–15, parallel-safe)

1. Pretrained modern change-transformer adoption (ChangeFormer/BIT-class, tens of MB).
   **Gate:** LEVIR-CD full-test IoU ≥ 0.75 / F1 ≥ 0.88.
2. Alternative: train on `data/SECOND` (train+test on disk). **Gate identical.** Budget ≤ 5 days.
3. Keep Siamese FPN-lite + TTA fallback.

### Phase 4 — ISRO/SAC differentiator (throughout, ≤ 1 day)

1. Verify `--sac-dir` end-to-end on samples/ Cartosat-2S + RISAT demo pairs.
2. Extend grounding+change outputs to georeferenced masks/boxes; one flawless demo run
   (GeoJSON overlay + GeoTIFF download).

### Phase 5 — Integration, honesty, ship (final week, days 16–21)

1. Freeze. No new training.
2. Regenerate scorecard; update README/MODEL_CARDS/Decisions/web Provenance.
3. `pytest` green on clean machine/Docker; cold-start demo recorded.
4. Cut release tag. Failed gates published as honest negatives.

### Phase 6 — Landing page: content overhaul (final-prep week + iterate, ~2 days, judge-facing)

**Goal: a judge lands on the page and answers within ~10 seconds — "all mandated
capabilities covered, measured on every named suite, real outputs, runs without
cloud."** Current page (web/src/landing) keeps the globe animation (retain), but its
content sells short:
- Hero has no proof above the fold (name + tagline only).
- "The specialists" table shows stale/pre-network numbers (0.71 / 0.32 / 0.67) and no
  VRSBench or ISRO/SAC story.
- "Why Anvesha" leads with *negatives* (what failed) and buries the headline strengths.
- No evidence section (no change overlay image, no GeoTIFF mask, no SAR example, no trace
  snippet), and no explicit PS-mandate coverage table.

**Content spec (in priority order):**
1. **Hero (keep globe animation):** add a proof chip-line under the tagline — e.g.
   "6/6 PS-named benchmarks on one scorecard · Cartosat-2S + RISAT ready · no cloud".
2. **Stat band (new, first scroll section):** 3–5 headline numbers — RS adaptation
   98.86% (EuroSAT full-track), change-VQA +17.4 pts over the paper baseline, 100/100
   concurrent @ p95≈7s, int8 export costs 0.16%, fully offline deployable.
3. **"The mandate" (new):** a PS-requirement coverage table (mandate → implementation →
   measured), the compliance story judges check first: single-image VQA, captioning/
   grounding, multitemporal change + change-VQA, optical–SAR, agentic orchestration,
   remote-sensing-adapted vision-language component.
4. **"The scorecard" (new):** the 6-row named-benchmark table (RSVQA-LR, VRSBench
   caption, VRSBench grounding, CDVQA, LEVIR-CD, BigEarthNet captions) + "hidden
   ISRO/SAC set: pre-georeferenced Cartosat-2S + RISAT, boxes/masks" band. Numbers come
   from the live scorecard at ship time (Phase 0 rows → Phase 2/3 lifts).
5. **"Evidence" (new):** 3 visual cards — a change-detection overlay, a georeferenced
   mask export (GeoTIFF), an optical+SAR fused pair; plus a tiny execution-trace snippet.
   Judges trust what they can see. Assets sourced from runs/<id>/visuals.
6. **"Why Anvesha":** reorder to headline strengths first (reproducible numbers,
   calibrated confidence, offline, honest gates); keep the negative-results list SECOND
   as integrity proof.
7. **CTA:** keep "Launch the console", strengthen subtext ("demo samples included").

**Rules:** never ship a number that isn't reproduced by `python -m satquery.evaluate
--all` at that exact moment (pulls from `runs/scorecard.json`); the globe animation and
all motion stays; frontend must rebuild cleanly (`cd web && npm run build`); the page
must load offline (no external fonts/CDNs). Landing is part of the final demo — a judge
browser-cold-check is a Phase 5 gate.

---

## PART D — STOP conditions (hard)

1. Any gate fails after budget, or result < baseline → **STOP that thread**, publish
   negative, restore fallback, do not proceed to dependent next phase. Explicitly say to
   the user: **"we should research and reason more before proceeding."**
2. VRSBench grounding gate fails → rebalance to Phase 3 (change), which is independent.
3. Phase 0 cannot produce 6-row scorecard (license/offline blocker, CDVQA pairs unfindable
   after 1-day investigation) → STOP, report, ask before download vs substitution.
4. Any migration regresses a frozen number (RSVQA presence < 0.91, LEVIR F1 < 0.801, trace
   broken, pytest red) → immediate rollback via config flag; old artifacts never deleted.
5. > 50% remaining time consumed with zero adopted gates, or final week with < 6 rows →
   **STOP all training.** Freeze, polish, reproduce every number on clean machine, ship.
6. Never ship a product path judges can't see trace for — every adopted change surfaces
   in `source_model`, execution trace, and report. The trace is the only observable judged.

---

## PART E — Agent operating notes

- **Order (dependency graph):** Phase 0 → (Phase 1 ‖ Phase 0-cdvqa) → Phase 2 → Phase 3
  (parallel-safe) → Phase 4 (anywhere) → Phase 5 → Phase 6 (landing; needs final
  scorecard, so lands late but copy/components can start anytime). Phase 1 and 3
  independent — if one stops, the other continues.
- **Reuse patterns, don't invent:** gate_dinov2.py, DinoEncoder (VLCEncoder clones it),
  fallback source_model labeling, CONFIG.resolve_device(), run_benchmarks stubs,
  export_torchscript.py (re-gate every adopted model, prev Δ−0.16%).
- **Commands at every phase boundary:** `python -m pytest tests -q` ·
  `python -m satquery.evaluate --all` · `git status --short` (working tree clean) ·
  diff `runs/scorecard.json`.
- **Data shortcuts:** VRSBench manual route documented (upstream parquet schema bug);
  CDVQA maps to on-disk SECOND ids (verify first); bentxt_join parquets = the exact
  PS-named BigEarthNet.txt image–text pairs — no new downloads needed for the VL component.
- **Dependency risk:** the only new supply-chain surface is `open_clip_torch`
  (+ optional timm); verify license + offline load at Phase 1 start. No transformers, no nltk.
- **Windows/cmd.exe discipline:** all commands via cmd single-line, absolute paths, `&&`
  chaining; avoid pipes/quotes that break on this platform (python -c quoting issues noted).
- **Honesty is a judge-facing asset:** every adopted and rejected gate lands in
  MODEL_CARDS.md with numbers. Partially-failing-honestly beats silently-faking, on the
  PS's own adaptation clause.
- **Default on gate failure:** stop, roll back, tell the user "we should research and reason
  more before proceeding" **with a concrete research agenda** — never improvise a second
  speculative approach on the same assumptions.

## PART C — What I expect (brief, falsifiable)

| Metric | Now | Adopt-gate | Target |
|---|---|---|---|
| Scorecard rows | 3 | 6 | 6 (all named suites) |
| VRSBench grounding IoU@0.5 | not measured (≈0.15 spectral) | ≥ 0.30 | ≥ 0.35 |
| RSVQA aggregate EM | 0.7021 | ≥ 0.75 | ~0.78 |
| BEN captions multi-ref BLEU | 0.283 | ≥ 0.35 | ~0.40 |
| VRSBench caption BLEU-4 | not in scorecard | ≥ 0.15 | ~0.20 |
| LEVIR-CD full-test IoU/F1 | 0.668 / 0.801 | ≥ 0.75 / ≥ 0.88 | 0.78 / 0.89 |
| CDVQA | 0.683 | ≥ 0.70 | ~0.72 |
| Combined normalized | 0.5291 | — | ≥ 0.70 |
| **Landing page (Phase 6)** | content-thin, stale numbers | judge 10s test | proof band + mandate + 6-row scorecard + evidence cards |

Win-chance: from ~single digits → ~25%. The win is decided on the hidden ISRO/SAC set
(SAR dB handling, georeferenced masks, co-registration, offline deploy).