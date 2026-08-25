# 36-hour Grand Finale Playbook

## T-7 days (before finale)
- [ ] Fresh-clone rehearsal on a clean machine: `pip install -r requirements.txt`, download data, run training chain, `pytest`
- [ ] Freeze weights into `weights/`; verify `python -m satquery.server.main` boots and demo runs offline
- [ ] Record fallback demo video (in case venue GPU/internet fails)

## Venue setup checklist (first hour)
1. `docker load satquery.tar` or run from local venv — no internet needed
2. Open console → Provenance tab first: judges see measured benchmark table immediately
3. Keep ISRO-style samples (`isro_cartosat2s_optical.tif`, `isro_risat_sar.tif`) loaded in a second browser tab as backup

## Demo script (8 minutes)
| Min | Action | Talking point |
|-----|--------|---------------|
| 0–1 | Upload Cartosat/RISAT pair | "Validation catches CRS/geometry issues before any model runs" |
| 1–2 | PS query #5 (optical-SAR) | Trained dual-branch fusion; point at agreement matrix + complementarity notes |
| 2–4 | Bi-temporal pair + swipe compare | Change map raster exportable as GeoTIFF for GIS review |
| 4–5 | Grounding query | Learned referring-expression head ensembled with interpretable NDWI/NDVI evidence |
| 5–6 | Show live execution trace | "The exact observable artifact your evaluation defines: task, tools, params, durations" |
| 6–7 | Provenance page | Every specialist fine-tuned on open RS data with measured metrics |
| 7–8 | Q&A ammo: architecture.md, scorecard.json, tests (32 passing) |

## Q&A ammunition
- **Why not one big VLM?** PS explicitly warns generic VLMs underperform on RS; specialists are measurable per-task (show per-benchmark scores) and auditable.
- **How would this run inside SAC?** Docker image, CPU-capable fallbacks, GeoTIFF-native I/O, batch mode for evaluation sets.
- **What did you fine-tune?** Four components, all on public RS data; numbers on screen.
- **Failure behaviour?** Graceful heuristic fallbacks + explicit confidence; never a silent wrong answer.

## Risks & mitigations
| Risk | Mitigation |
|------|------------|
| Venue has no GPU | All models run CPU-only (slower); test CPU boot before finale day |
| Hidden SAC pairs fail validation | Batch mode writes per-item errors without stopping the run |
| Judge asks about generative caption quality | Show BLEU on held-out captions + hybrid template facts that are always truthful |
