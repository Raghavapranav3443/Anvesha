# Model Cards — SatQuery AI

Every specialist is fine-tuned on public remote-sensing data. Weights load
automatically from `weights/`; interpretable fallbacks keep the assistant usable
without them (and are themselves under test).

## SceneEncoder (shared visual backbone)
- **Architecture:** ResNet-18 stem adapted to arbitrary band counts by averaging pretrained RGB kernels; 256-d embedding head.
- **Fine-tuning:** EuroSAT RGB, 10 classes, 200 img/class quick track.
- **Measured:** val accuracy **0.91**.
- **Role:** warm start for every specialist below — the mandated RS adaptation layer.

## VQA Specialist (`single_vqa`)
- **Architecture:** SceneEncoder ⊕ hashed bag-of-words (512-d) ⊕ question-type embedding → fusion MLP over answer vocabulary (100 classes).
- **Data:** RSVQA-LR full train split (54k active triplets), flip/rot90 augmentation.
- **Training:** class-balanced cross-entropy (inverse-sqrt frequency), AdamW + cosine schedule.
- **Measured:** validation accuracy reported per run (`weights/vqa_head.pt`); test-subset exact-match via `python -m satquery.evaluate --all`.

## Caption Decoder (`captioning`)
- **Architecture:** 4-layer causal transformer (d=256) cross-attending SceneEncoder feature maps; word-level vocab (≤6k).
- **Data:** BigEarthNet.txt captions joined to local co-registered reBEN v2 Sentinel-2 patches.
- **Fallback:** deterministic template composition from scene evidence (always truthful).
- **Measured:** BLEU on held-out captions.

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
  Root cause: a frozen ImageNet→RS encoder lacks the spatial/class resolution
  for tight-box localization; this needs a detection-grade architecture
  (FPN + fine-tuned backbone) — documented as the upgrade path. Checkpoints
  removed from `weights/`; training scripts retained for reproduction.
  The click-to-query feature works regardless (region-crop VQA).

## Change Specialist (`change_analysis` / `change_vqa`)
- **Architecture:** Siamese SceneEncoder difference head at stride 8; sliding-window inference at training resolution; OOD safeguard falls back to smoothed differencing when the learned map is flat.
- **Data:** LEVIR-CD (official crops via HF mirror).
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
