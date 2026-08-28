# MASTERPLAN — Anvesha / SatQuery AI · SIH PS 26167 · Target: Grand Finale 1st Place

> Operating document. The plan is the contract. Gates and STOP conditions fire loud on purpose.

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
  (parallel-safe) → Phase 4 (anywhere) → Phase 5. Phase 1 and 3 independent — if one stops,
  the other continues.
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

Win-chance: from ~single digits → ~25%. The win is decided on the hidden ISRO/SAC set
(SAR dB handling, georeferenced masks, co-registration, offline deploy).