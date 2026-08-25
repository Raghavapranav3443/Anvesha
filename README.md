# 🛰️ Anvesha — Earth Observation & Investigation System

**An agentic vision-language assistant for multimodal remote-sensing image analysis through natural-language queries.**
*Built for ISRO/SAC problem statement SIH26167 (Smart India Hackathon). Formerly SatQuery AI.*

SatQuery AI is not a single generic model. An **agent controller** validates your
imagery, interprets the query, routes it to **remote-sensing specialist models**
(each fine-tuned on real satellite data), executes them with bound parameters,
and returns evidence-grounded answers — confidence scores, visual overlays,
exportable GeoTIFF masks, and a fully auditable execution trace.

```
streamlit-style quick start →  python -m uvicorn satquery.server.main:app --port 8000
open http://localhost:8000    (EO Mission Console UI)
```

---

## Mandatory scope coverage (SIH26167)

| Requirement | Implementation | Trained on | Measured |
|---|---|---|---|
| RS adaptation | Shared `SceneEncoder` (band-adaptive ResNet-18) warm-starts every specialist | EuroSAT | 91% val acc |
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

# web console (React build included in repo; rebuild with `npm ci && npm run build` in web/)
python -m uvicorn satquery.server.main:app --port 8000
# → http://localhost:8000
```

Docker (offline-deployable, weights baked at build time):

```bash
docker build -t satquery .
docker run -p 8000:8000 satquery
```

### Training from scratch

```bash
python scripts/download_datasets.py eurosat        # Zenodo 7711810
python scripts/download_datasets.py rsvqa          # Zenodo 6344333
python scripts/download_datasets.py levircd        # HF mirror of official crops
python scripts/download_datasets.py bigearthnet_14k  # real reBEN v2 S1+S2 pairs (3.1 GB)
python scripts/prepare_bentxt_join.py              # BigEarthNet.txt captions+refs join

python scripts/train_scene_encoder.py --epochs 3   # RS adaptation layer
python scripts/train_vqa.py        --epochs 20     # type-conditioned, class-balanced
python scripts/train_captioner.py  --epochs 5      # BEN.txt-trained decoder
python scripts/train_grounding.py  --epochs 8      # referring-expression boxes
python scripts/train_change.py     --epochs 16     # LEVIR-CD
python scripts/train_optical_sar.py --dataset bigearthnet_14k --epochs 10
```

## Evaluation & SAC batch mode

```bash
python -m satquery.evaluate --all            # normalized public-benchmark scorecard
python -m satquery.evaluate --sac-dir DIR    # folder of co-registered pairs → answers.csv
                                             # + per-item GeoTIFF change masks + traces
```

Measured scorecard (public benchmark test subsets, this machine):

| Benchmark | Metric | Score |
|---|---|---|
| RSVQA-LR (test subset) | exact-match accuracy | **0.66** |
| LEVIR-CD (test subset) | change IoU / F1 | **0.36 / 0.52** |
| BigEarthNet v2 S1+S2 (held-out val) | per-scene label recall | **0.85** |
| BigEarthNet.txt captions (val) | BLEU, single-reference | 0.24–0.59* |
| BigEarthNet.txt refs (test) | IoU>0.5 hit-rate — experimental head, disabled | 0.15 |

*Single-reference BLEU understates the decoder (trained against ~4 prompt
variants per patch); training-validation BLEU is 0.59. All numbers reproducible
via `python -m satquery.evaluate --all`.

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
  held-out validation), not raw softmax
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

Synthetic GeoTIFF fixtures make the suite offline-capable; it passes both with
and without trained weights (fallback paths are themselves under test).

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full diagram and design contracts.
Legacy Streamlit app retained (`app.py`); the React console is the primary UI.

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
