# Frontend Capability Exposure Audit Report

**Date:** 2026-09-11  
**Scope:** Complete mapping of backend capabilities to frontend surfaces  
**Method:** Automated static analysis + manual verification  

---

## Executive Summary

The frontend exposes **70%** of the project's backend capabilities. Core workflow coverage is strong (86%), but advanced trust-layer features (calibration, freshness, honesty) and demo-mode affordances remain invisible to judges. Seven gaps identified; three are HIGH-impact for SIH scoring.

| Metric | Score | Detail |
|--------|-------|--------|
| Task type coverage | 86% | 6/7 tasks have frontend surfaces |
| API endpoint coverage | 83% | 15/18 endpoints called from frontend |
| C-feature coverage | 62% | 5/8 trust/demo features present |
| B-patch output surfacing | 50% | 3/6 patch outputs rendered |
| **Overall** | **70%** | **7 gaps remain** |

---

## 1. Backend Capability Inventory

### 1.1 Task Types (7 total)
| Task | Keywords | Specialist |
|------|----------|------------|
| investigation | investigate, impact of, urban expansion | Full chain |
| grounding | highlight, locate, where is, bounding box | Grounder |
| change_vqa | what changed, increased or decreased | Change + VQA |
| change_description | describe the change, before and after | Change (aliased) |
| optical_sar | optical and sar, cross-modal | FusionNet |
| captioning | describe the land-cover, scene description | Captioner |
| single_vqa | is there, how many, what type | VQA |

### 1.2 REST Endpoints (18 total)
| Method | Endpoint | FE-called? |
|--------|----------|:----------:|
| POST | /api/jobs | X |
| GET | /api/jobs/{id} | X |
| GET | /api/history | X |
| GET | /api/samples | X |
| GET | /api/stats | X |
| GET | /api/experiments | - |

## 2. Frontend Component Inventory

### 2.1 Components (14)
| Component | LOC | Purpose |
|-----------|-----|---------|
| Console.tsx | 506 | Main analysis interface (upload, query, modality toggle, setups) |
| Results.tsx | 495 | Job result display (answer, confidence, structured outputs, region VQA) |
| MapView.tsx | 62 | GeoJSON overlay on Leaflet |
| DossierView.tsx | 82 | Dossier rendering (head, facts, timeline, reticles) |
| BoardsView.tsx | 58 | Board pins over real runs |
| JudgeRun.tsx | 79 | 5-item checklist over History |
| SvgLocator.tsx | 88 | Offline SVG world fallback |
| EvaluationView.tsx | 87 | Benchmark scorecard |
| HistoryView.tsx | 89 | Past job history |
| Provenance.tsx | 79 | Model cards display |
| HelpView.tsx | 89 | Glossary |
| Onboarding.tsx | 59 | First-run tour |
| ErrorBoundary.tsx | 51 | Error isolation |
| Skeleton.tsx | 34 | Loading placeholder |

### 2.2 Navigation (7 tabs)
Console, Boards, Judge Run, History, Evaluation, Provenance, Help

### 2.3 API Functions (13)
fetchSamples, createJob, pollJob, fetchProvenance, fetchHistory, fetchJob, fetchGeo, runEvaluation, evalStatus, invalidateCache, fetchStats, fetchBoards, fetchDossier

---

## 3. Exposure Analysis

### 3.1 Task Type Exposure Matrix
| Task | Results.tsx | Console.tsx | JudgeRun.tsx | setups.ts | Status |
|------|:-----------:|:-----------:|:------------:|:---------:|--------|
| investigation | X | X | X | X | COVERED |
| grounding | X | - | X | X | COVERED |
| change_vqa | X | - | X | X | COVERED |
| change_description | - | - | - | - | **GAP** |
| optical_sar | X | - | X | X | COVERED |
| captioning | X | - | X | X | COVERED |
| single_vqa | X | - | X | X | COVERED |

**Coverage: 6/7 = 86%**

### 3.2 Output Key Visibility
| Output Key | Surfaced | Location |
|------------|:--------:|----------|
| answer | X | Results, Dossier, Boards |
| confidence | X | Results, Boards |
| selected_task | X | Results |

## 4. Gap Analysis (7 gaps)

### HIGH Impact (judge-visible, blocks scoring)

**GAP-1: C4 Honesty Banner absent**  
Backend computes honesty fields (below_gate, fallback_active, pixel_space). No HonestyBanner.tsx exists. Judges see low-confidence answers without the honest explanation.  
Fix: Create HonestyBanner.tsx consuming confidence_meta from result.

**GAP-2: C3 Freshness not rendered**  
clocks_for() computes freshness dict. Backend emits it. Results.tsx never reads it.  
Fix: Add freshness badge row to Results.tsx.

**GAP-3: C7 Demo Mode chip absent**  
scripts/warm_demo.py exists. No FixtureChip.tsx surfaces it.  
Fix: Create FixtureChip.tsx + wire to warm_demo manifest.

### MEDIUM Impact (capability hidden but functional)

**GAP-4: change_description not surfaced**  
Aliased to change_analysis but no Console setup, no JudgeRun item, no Results conditional.  
Fix: Add to JudgeRun checklist, add setup sample.

**GAP-5: modality_certainty not displayed**  
Backend emits it into every RSImage.summary. Results.tsx ignores it.  
Fix: Add modality badge with confidence to Results.tsx.

**GAP-6: agreement_map not surfaced**  
Fusion agreement map computed but only reachable via DossierView artifact link.  
Fix: Render agreement overlay inline in Results.tsx when present.

### LOW Impact (nice-to-have)

**GAP-7: boxes output not rendered**  
Grounding boxes shown via MapView overlay but no textual region summary.  
Fix: Add region summary line to Results.tsx.

---

## 5. Quantitative Scorecard

| Dimension | Score |
|-----------|-------|
| Task type coverage | 86% (6/7) |
| API endpoint coverage | 83% (15/18) |
| C-feature coverage | 62% (5/8) |
| B-patch output surfacing | 50% (3/6) |
| **Overall** | **70%** |

---

## 6. Backend-Frontend Mapping Correctness

### Correctly Mapped
- POST /api/jobs -> createJob() -> pollJob() -> Results.tsx
- GET /api/geo/{id} -> fetchGeo() -> MapView.tsx
- GET /api/history -> fetchHistory() -> HistoryView.tsx
- GET /api/provenance -> fetchProvenance() -> Provenance.tsx
- GET /api/samples -> fetchSamples() -> Console.tsx
- POST /api/evaluate/run -> runEvaluation() -> EvaluationView.tsx
- GET /api/boards -> fetchBoards() -> BoardsView.tsx
- GET /api/reports/{id}/dossier -> fetchDossier() -> DossierView.tsx

### Mismatches Found
| Issue | Backend | Frontend | Impact |
|-------|---------|----------|--------|
| freshness emitted but not rendered | outputs.freshness | Results.tsx ignores | Medium |
| modality_certainty on RSImage | io_utils emits | Results.tsx ignores | Low |
| agreement_map file path only | fusion/agreement.py | No inline display | Low |
| honesty fields computed | patches.py | No banner | **High** |
| fixture manifest exists | scripts/warm_demo.py | No chip | **High** |

---

## 7. Recommendations (by ROI for SIH)

### Tier 1 (do now)
1. HonestyBanner.tsx -- 2 hours. Biggest judge-impact fix.
2. Freshness badge in Results -- 1 hour. Backend already computes it.
3. FixtureChip.tsx -- 1 hour. Demo reliability story.

### Tier 2 (if time permits)
4. modality_certainty display -- 30 min.
5. agreement_map inline -- 1 hour.
6. change_description in setups -- 15 min.

### Tier 3 (nice-to-have)
7. boxes textual summary -- 30 min.
8. PDF/MD download in History -- 1 hour.

---

## Conclusion

The frontend effectively showcases the core 5-workflow SIH demo with strong coverage (86%). The trust layer (C3 freshness, C4 honesty) and demo reliability (C7 fixture chip) -- the features that differentiate this project -- are computed but invisible. Closing the 3 HIGH-impact gaps would raise overall exposure from 70% to approximately 85%.

| ranked | X | Console |
| transitions | X | (substring) |
| agreement_map | - | **NOT SURFACED** |
| freshness | - | **NOT SURFACED** |
| dossier | X | Dossier |
| labels | X | Results, Console |
| boxes | - | **NOT SURFACED** |
| changed_area_ha | X | Results, Dossier |
| composite | X | Results |
| investigation | X | Results, Console |
| geo | X | Results, Boards, Console |
| modality_certainty | - | **NOT SURFACED** |
| calibration | X | (substring) |

**B-patch surfacing: 3/6 = 50%**

### 3.3 C-Feature Presence
| Feature | Component/File | Present? |
|---------|---------------|:--------:|
| C1 boards | BoardsView.tsx | YES |
| C2 dossier | DossierView.tsx + route | YES |
| C3 freshness | (computed, not rendered) | **NO** |
| C4 honesty | (no banner) | **NO** |
| C5 setups | setups.ts + Console | YES |
| C6 locator | SvgLocator.tsx | YES |
| C7 demo | (no FixtureChip) | **NO** |
| C8 checklist | JudgeRun.tsx | YES |

**C-feature coverage: 5/8 = 62%**

### 3.4 API Endpoint Coverage
15 of 18 endpoints called from frontend. Unused: experiments, model_status, healthz.

**Endpoint coverage: 15/18 = 83%**

| DELETE | /api/cache/{key} | X |
| GET | /api/provenance | X |
| GET | /api/reports/{id}/report.pdf | X |
| GET | /api/reports/{id}/report.md | X |
| GET | /api/reports/{id}/report.json | X |
| GET | /api/reports/{id}/{file} | X |
| GET | /api/geo/{id} | X |
| POST | /api/evaluate/run | X |
| GET | /api/evaluate/status/{id} | X |
| GET | /healthz | - |
| GET | /api/model_status | - |
| + | /api/boards | X |
| + | /api/reports/{id}/dossier | X |

### 1.3 Specialists (11 model modules)
vqa, captioner, change, grounder, optical_sar, scene, backbone, clip_text, count_density, dino_encoder, status

### 1.4 Patch Outputs
freshness, dossier, transitions, agreement_map, ranked, modality_certainty, calibration
