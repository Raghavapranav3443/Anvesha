# Anvesha AI — Architecture

```
┌────────────────────────────  Client (web/dist)  ───────────────────────────┐
│  React 18 + TypeScript + Tailwind — "EO Mission Console"                    │
│  Console view            Provenance view            Downloads               │
│  (inputs, query,         (model cards,              (MD/JSON reports,       │
│   live trace, results)    measured benchmarks)       GeoTIFF masks)          │
└───────────────▲────────────────────────────────────────────────────────────┘
                │ REST + polling (/api/jobs/{id})
┌───────────────┴────────────────────────────────────────────────────────────┐
│  FastAPI service (anvesha/server)                                          │
│  job store ─ runs AgentController in worker threads, publishes trace        │
└───────────────▲────────────────────────────────────────────────────────────┘
                │
┌───────────────┴────────────────────────────────────────────────────────────┐
│  AgentController (anvesha/agent.py)                                        │
│  1 validate_inputs   format · modality · CRS · co-registration geometry     │
│  2 classify_task     keyword-intent rules + aliases + feasibility filter    │
│  3 select_tool       registry lookup w/ input-requirement enforcement       │
│  4 execute           specialist tool(s), params bound, timed                │
│  5 integrate         answer + confidence + visual evidence                  │
│  6 report            runs/<id>/report.{json,md} + visuals/*.png|tif         │
└───────────────▲────────────────────────────────────────────────────────────┘
                │
┌───────────────┴───────────────────────────────────────────────────────────┐
│  Specialist registry                                                       │
│                                                                            │
│  single_vqa      SceneEncoder ⊕ hashed-BOW(512)+type → answer head         │
│                  trained: RSVQA-LR (54k triplets)                          │
│  captioning      BEN.txt-trained transformer decoder ⊕ template facts      │
│                  trained: BigEarthNet.txt captions ∩ local reBEN v2 S2     │
│  grounding       learned referring-expression box head ⧺ calibrated        │
│                  spectral-index regions (NDWI/NDVI/ExG/double-bounce)      │
│                  trained: BigEarthNet.txt reference boxes                  │
│  change_*        Siamese SceneEncoder diff head (tiled inference) +        │
│                  scene-presence delta reasoner; LEVIR-CD trained           │
│  optical_sar     dual-branch (S1 dB-aware / optical) fusion network        │
│                  trained: BigEarthNet v2 co-registered pairs               │
└───────────────────────────────────────────────────────────────────────────┘
```

## Shared visual backbone
`SceneEncoder` = ResNet-18 stem adapted to arbitrary band counts by averaging
pretrained RGB kernels. One encoder is fine-tuned on remote-sensing data
(EuroSAT track) and warm-starts every other specialist — this is the mandated
remote-sensing adaptation layer.

## Scale-aware preprocessing
* SAR power-scale rasters → log1p; **dB products (negative values, e.g.
  RISAT-style HH/HV) auto-detected and standardised without log**.
* Optical reflectance ×10000 → percentile-clipped to [0,1].
* Pairs must share pixel grids and CRS; otherwise a precise validation error.

## Execution trace contract
Every run persists `runs/<run_id>/report.json` containing selected task,
tool names, bound parameters, per-step durations, outputs and confidence —
the observable artifact the problem statement says will be evaluated.
