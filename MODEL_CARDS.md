# Model Cards — Anvesha (Earth Observation & Investigation System)

*Formerly SatQuery AI during development; Python package namespace remains
`satquery` for stability.*

Every specialist is fine-tuned on public remote-sensing data. Weights load
automatically from `weights/`; interpretable fallbacks keep the assistant usable
without them (and are themselves under test).

## SceneEncoder (shared visual backbone)
- **Architecture:** ResNet-18 stem adapted to arbitrary band counts by averaging pretrained RGB kernels; 256-d embedding head.
- **Fine-tuning:** EuroSAT RGB, 10 classes, 200 img/class quick track.
- **Measured:** val accuracy **0.91**.
- **Role:** warm start for every specialist below — the mandated RS adaptation layer.

## VQA Specialist (`single_vqa`)
- **General head:** SceneEncoder ⊕ hashed bag-of-words (512-d) ⊕ question-type
  embedding → fusion MLP over the 100-class RSVQA answer vocabulary.
- **Per-type specialist heads (`type_heads.pt`):** presence / rural_urban /
  comparison each get a dedicated fusion head trained on the frozen shared
  encoder (per-batch type routing — specialists cannot drift it); counting uses
  a dedicated digit head (v4: **soft-ordinal cross-entropy** over adjacent
  digits, sigma=0.7, class-balanced sampling, flip-TTA) — ordinal targets
  encode that predicting 3 for a true 4 is far less wrong than predicting 9.
  Inference routes by detected question type: specialist → general → rule
  reasoner, all labeled via `source`.
- **Data:** RSVQA-LR full train split (54k active triplets), flip/rot90 augmentation.
- **Training:** class-balanced cross-entropy (inverse-sqrt frequency), AdamW + cosine schedule.
- **Measured:** test-subset exact-match **0.773** (n=282 spot-check via
  `python -m satquery.evaluate --all`); **full test 0.700** (n=9,491,
  `runs/phase2_rsvqa_fulltest.json`). General head alone: 0.67 on the same
  spot-check protocol; specialist
  train accuracies — presence 0.91, rural_urban 0.84, comp 0.70; counting
  val digit-acc 0.44 with the ordinal v4 head).
- **Gated density-map counting experiment (v5, NOT shipped):** a density-regression
  head (count-only supervision: density-map sum regressed to the label, L1, with
  an auxiliary ordinal digit head) reached only **0.133 val digit-acc** — far
  below the 0.436 gate. **Not promoted**; the ordinal v4 head stays shipped.
  Negative result logged in the experiment record.
- **Gated backbone experiment:** frozen DINOv2-small probes beat SceneEncoder
  on EuroSAT (+3.2 pts) and lifted rural_urban training accuracy (+7.9 pts),
  but transferred **no gain to VQA exact-match** (0.7008 vs 0.7021 on the test
  subset) — adoption rejected by the pre-registered gate. `DinoEncoder` and
  `scripts/gate_dinov2.py` are retained; `weights/dinov2_vits14.pt` (84 MB,
  Apache-2.0) ships for reproducibility of the negative result.

## Caption Decoder (`captioning`)
- **Architecture:** 4-layer causal transformer (d=256) cross-attending frozen
  SceneEncoder feature maps; word-level vocab (≤6k).
- **Plan conditioning (v2):** a 19-d BEN19 content plan is weakly derived from
  each caption's own text (canonical class-phrase matching); an internal
  PlanHead predicts that plan from the image alone and conditions every
  memory token — generation becomes realization of a predicted content plan.
- **Decoding:** beam search (beam=3, length-normalized) at evaluation;
  greedy live. Deterministic template fallback unchanged.
- **Data:** BigEarthNet.txt captions joined to local co-registered reBEN v2 Sentinel-2 patches.
- **Measured:** multi-reference BLEU on held-out captions **0.283**
  (n=300, `python -m satquery.evaluate --all`); single-reference
  training-validation BLEU 0.59.

## Grounding Specialist (`grounding`)
- **Primary (shipped):** calibrated spectral-index response maps (NDWI / NDWI /
  ExG / SAR double-bounce) + connected-component analysis → mask + boxes.
  Fully interpretable, robust to free-form user queries.
- **Learned experiments (attempted, measured, NOT shipped):**
  | Variant | Training data | Held-out IoU@0.5 | Decision |
  |---|---|---|---|
  | Expression→box regression | BEN.txt refs (26.8k) | 0.15 | gated off |
  | Expression→heatmap | BEN.txt refs | 0.12 | gated off |
  | Point→heatmap | VRSBench objects (13.6k) | 0.018 | gated off |
  | Zero-shot CLIP window-argmax (v2 multi-scale grid, 80 val refs) | none (zero-shot) | 0.081 generic / 0.091 RemoteCLIP; grid oracle ceiling 0.273 | gated off — pre-registered ≥0.30 |
  Root cause: a frozen ImageNet→RS encoder lacks the spatial/class resolution
  for tight-box localization; this needs a detection-grade architecture
  (FPN + fine-tuned backbone) — documented as the upgrade path. Checkpoints
  removed from `weights/`; training scripts retained for reproduction.
  The click-to-query feature works regardless (region-crop VQA).
   **CLIP spike negative (Phase 1 gate):** both zero-shot dual-encoder probes —
   generic CLIP ViT-B-32 (0.081) and RS-domain RemoteCLIP ViT-B-32 (0.091) —
   fell far below the pre-registered 0.30 gate, and the harness's own oracle
   ceiling (0.273) proves window-argmax is *structurally* incapable of 0.30
   mean IoU on VRSBench's small objects. RS-domain pretraining adds only
   +0.01 IoU — the bottleneck is similarity granularity for referring
   expressions, not domain; caption fine-tuning could not bridge a 3.5× gap.
   Grounding upgrades, if any, come from a detection-grade open-vocabulary
   detector (Grounding DINO / OWL-ViT class), not more CLIP.
   Full record: `Decisions.md` D15.1; `scripts/gate_clip_grounding.py`.

## Change Specialist (`change_analysis` / `change_vqa`)
- **Architecture:** Siamese SceneEncoder with an **FPN-lite multi-scale
  difference decoder** (stride-8 + stride-16 fusion, upsample-and-fuse);
  sliding-window tiled inference; OOD safeguard falls back to smoothed
  differencing when the learned map is flat.
- **Data:** LEVIR-CD (official crops via HF mirror).
- **Measured:** test-subset change **IoU 0.60 / F1 0.75** (n=300) — at the level
  of published ResNet-era baselines. Prior single-scale head: IoU 0.35.
  Full-split thresholded protocol (thr=0.85, n=1,500): **IoU 0.692 / F1 0.818** —
  this is the canonical headline number (see README; latest measurement
  2026-08-29, `runs/phase3_levir_fulltest.json`; an earlier full-split run of
  the same protocol measured 0.668/0.801). Same model, different eval sets.
- **Change-VQA (learned, `cdvqa_head.pt`):** a change-conditioned head over the
  detector's SE-attended multi-scale difference features (256-d pooled) ⊕
  spectral-presence deltas ⊕ question BOW, with one output head per CDVQA
  question type. Trained on CDVQA val pairs (16.4k records); **promotion gate:
  beat the calibrated rule-based predictor (0.4942 val) — passed at 0.7134**.
  Full test (39,686 Q): **0.683 overall, +17.4 pts over the majority baseline,
  every type above baseline** (change_or_not 0.828, ratio_types 0.707,
  change_to_what 0.575). Protocol map: 0.7134 = promotion-gate val (16.4k
  records, vs the rule-based predictor's 0.4942); 0.646 = balanced val
  spot-check (n=2,088 Q); 0.683 = full test — quote the full-test number.
  The calibrated rule-based predictor is retained as
  fallback (`--model rules`).
- **Outputs:** probability map, binary mask (GeoTIFF export), change description, change-VQA answer, region direction/bbox statistics.

## Optical–SAR Fusion (`optical_sar`)
- **Architecture:** dual-branch SceneEncoders (2-ch SAR / 3-ch optical) → concatenated embedding → BEN19 multi-label sigmoid head.
- **Data:** real co-registered BigEarthNet v2 S1+S2 pairs (14K cross-modal subset).
- **Measured:** validation per-scene label recall **0.85**.
- **Analysis outputs:** fused class posteriors, per-modality evidence, agreement matrix, cloud/SAR complementarity notes.

## Honesty guarantees
1. Checkpoints trained on synthetic data carry a flag and are **refused at load time**.
2. Fallback paths are explicit in every tool output (`source_model` field).
3. All numbers on the Provenance page come from measured checkpoints, not claims.