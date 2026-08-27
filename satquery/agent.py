"""Agentic controller for SatQuery AI.

Pipeline (observable execution trace):
  1. input validation & configuration detection
  2. query intent classification
  3. tool selection from the registry (feasibility-checked)
  4. parameter binding and sequential tool execution
  5. output integration, confidence aggregation
  6. auditable execution summary + downloadable report
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import CONFIG
from .io_utils import RSImage, InputValidationError, describe_configuration, \
    load_image, validate_inputs
from .registry import ToolSpec, build_default_registry
from .text import content_tokens


# --------------------------------------------------------------------------- #
# Embedding-augmented intent classification helpers
# --------------------------------------------------------------------------- #

def _query_bow(query: str, dim: int = 512) -> np.ndarray:
    """Hashed bag-of-words vector for a query string."""
    import hashlib, re
    vec = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", query.lower()):
        h = hashlib.md5(tok.encode()).hexdigest()
        idx = int(h[:8], 16) % dim
        sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (assumed L2-normalised)."""
    return float(np.dot(a, b))


# Lazily precomputed centroids: task -> 512-d BOW vector
_task_centroids: Optional[Dict[str, np.ndarray]] = None

def _get_task_centroids() -> Dict[str, np.ndarray]:
    """Return precomputed task centroids.

    Tries to load data-derived centroids from ``weights/task_centroids.pt``
    (computed by ``scripts/precompute_task_centroids.py`` from real RSVQA
    training questions).  Falls back to keyword-derived centroids if the
    file is not available.
    """
    global _task_centroids
    if _task_centroids is not None:
        return _task_centroids

    # Try loading data-derived centroids first
    import torch
    centroids_path = CONFIG.weights_dir / "task_centroids.pt"
    if centroids_path.exists():
        try:
            ck = torch.load(centroids_path, map_location="cpu", weights_only=False)
            _task_centroids = {k: v.numpy().astype(np.float32) for k, v in ck.items()}
            return _task_centroids
        except Exception:
            pass

    # Fallback: compute from keyword lists
    _task_centroids = {}
    for task, kws in TASK_KEYWORDS.items():
        vecs = [_query_bow(kw) for kw in kws if len(kw) > 2]
        if vecs:
            mean = np.mean(vecs, axis=0)
            n = float(np.linalg.norm(mean))
            _task_centroids[task] = mean / n if n > 0 else mean
    return _task_centroids


# --------------------------------------------------------------------------- #
# Intent classification
# --------------------------------------------------------------------------- #

TASK_KEYWORDS = {
    "investigation": ["investigate", "investigation", "is it significant",
                      "significant", "impact of", "urban expansion",
                      "should i investigate", "encroachment", "full analysis",
                      "analyse the change around"],
    "grounding": ["highlight", "locate", "localise", "localize", "find the",
                  "where is", "where are", "where exactly", "mark",
                  "show me the region", "show me where", "bounding box",
                  "point out"],
    "change_vqa": ["what changed", "has the built-up", "increased or decreased",
                   "has .* changed", "change between", "difference between",
                   "how much changed", "remained unchanged",
                   "did anything change"],
    "change_description": ["describe the change", "changes over time",
                           "compare these two", "before and after",
                           "temporal change"],
    "optical_sar": ["optical and sar", "sar image together",
                    "use the optical and sar", "cross-modal", "radar",
                    "both modalities", "sentinel-1 and sentinel-2",
                    "with sar"],
    "captioning": ["describe the land-cover", "describe this image",
                   "describe the scene", "caption", "what is shown",
                   "scene description", "major objects visible",
                   "summarise the image", "summarize the image"],
    "single_vqa": ["is there", "are there", "how many", "what type",
                   "does this", "which class", "percentage of", "area of",
                   "is it"],
}


# Intent buckets that share a registered tool
TASK_ALIASES = {"change_description": "change_analysis"}

# Human-readable labels used by the clarification loop
TASK_LABELS = {
    "single_vqa": "Answer a question about this image",
    "captioning": "Describe the scene",
    "grounding": "Locate & highlight a region",
    "change_analysis": "Detect & map changes between the dates",
    "change_vqa": "Answer questions about the changes",
    "optical_sar": "Fuse optical + SAR evidence",
    "impact_analysis": "Quantify the impact of changes",
    "investigation": "Run a full multi-step investigation",
}

CLARIFY_THRESHOLD = 0.55

_FEASIBLE_ORDER = {
    "single": ["single_vqa", "grounding", "captioning"],
    "bitemporal_pair": ["change_vqa", "change_analysis", "impact_analysis"],
    "optical_sar_pair": ["optical_sar", "single_vqa"],
}


def build_clarification(query: str, intent: Dict[str, Any],
                        configuration: str) -> Optional[Dict[str, Any]]:
    """'Did you mean…?' options when intent confidence is low.

    The best-guess task still executes (graceful degradation); the options
    let the user re-run with an explicit override from the UI.
    """
    if intent.get("method", "").startswith("explicit"):
        return None
    if not query.strip():
        return None
    if float(intent.get("confidence", 1.0)) >= CLARIFY_THRESHOLD:
        return None
    scores = {t: s for t, s in intent.get("ranked_candidates", [])}
    options = [(t, s) for t, s in scores.items() if s > 0]
    if not options:
        # default routing fired on a non-empty query: offer plausible tasks
        options = [(t, 0.0) for t in _FEASIBLE_ORDER.get(configuration, [])
                   if t != intent["task"]]
    options = [(t, s) for t, s in options if t != intent["task"]][:2]
    if not options:
        return None
    return {
        "needed": True,
        "question": "Low confidence in the requested analysis — did you mean:",
        "chosen": {"task": intent["task"],
                   "label": TASK_LABELS.get(intent["task"], intent["task"])},
        "options": [{"task": t,
                     "label": TASK_LABELS.get(t, t),
                     "score": round(float(s), 3)} for t, s in options],
    }


def classify_task(query: str, configuration: str) -> Dict[str, Any]:
    """Intent classifier: keyword rules blended with BOW embedding similarity.

    The keyword component is deterministic and auditable (shown in the trace).
    The embedding component catches natural-language variation that pure
    keyword matching misses — e.g. "can you tell me if water is present"
    matches the "water" concept even though no RSVQA keyword fires.

    Blending formula:
        final_score = 0.6 * keyword_score + 0.4 * embedding_similarity
    """
    import re
    from .text import content_tokens  # noqa: delayed import

    q = query.lower()
    q_bow = _query_bow(q)

    # --- keyword component (same as before) ---
    keyword_scores: Dict[str, float] = {}
    for task, kws in TASK_KEYWORDS.items():
        s = 0.0
        for kw in kws:
            if ".*" in kw or kw.endswith("*"):
                try:
                    if re.search(kw.rstrip("*") + r".*" if kw.endswith("*") else kw, q):
                        s += 1.0
                except re.error:
                    pass
            elif kw in q:
                s += 1.0
        keyword_scores[task] = s

    for alias, target in TASK_ALIASES.items():
        if alias in keyword_scores:
            keyword_scores[target] = keyword_scores.get(target, 0.0) + keyword_scores.pop(alias)

    # --- embedding similarity component ---
    embed_scores: Dict[str, float] = {}
    centroids = _get_task_centroids()
    if centroids:
        for task, centroid in centroids.items():
            sim = _cosine_sim(q_bow, centroid)
            embed_scores[task] = max(0.0, sim)

    # --- blend ---
    all_tasks = set(keyword_scores) | set(embed_scores)
    scores: Dict[str, float] = {}
    for task in all_tasks:
        kw = keyword_scores.get(task, 0.0)
        em = embed_scores.get(task, 0.0)
        scores[task] = 0.6 * kw + 0.4 * em

    # feasibility filter by input configuration
    def feasible(task: str) -> bool:
        if configuration == "single":
            return task in ("single_vqa", "captioning", "grounding")
        if configuration == "bitemporal_pair":
            return task in ("change_analysis", "change_vqa", "single_vqa",
                            "investigation", "impact_analysis")
        if configuration == "optical_sar_pair":
            return task == "optical_sar" or (
                task == "single_vqa")
        return False

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    feasible_ranked = [(t, s) for t, s in ranked if feasible(t)]

    default_map = {
        "single": ("captioning" if not query.strip() else "single_vqa"),
        "bitemporal_pair": "change_analysis",
        "optical_sar_pair": "optical_sar",
    }
    if not feasible_ranked or feasible_ranked[0][1] == 0.0:
        best = default_map.get(configuration, "single_vqa")
        conf = 0.35
        method = "default routing (no strong keyword or embedding match)"
    else:
        best = feasible_ranked[0][0]
        total = sum(s for _, s in feasible_ranked) + 1e-6
        conf = min(0.95, 0.5 + 0.45 * feasible_ranked[0][1] / total)
        kw_top = keyword_scores.get(best, 0.0)
        em_top = embed_scores.get(best, 0.0)
        if kw_top > 0 and em_top > 0:
            method = f"keyword + embedding blend (kw={kw_top:.1f} embed={em_top:.2f})"
        elif kw_top > 0:
            method = "keyword-intent rules"
        else:
            method = "embedding similarity"

    return {"task": best, "confidence": round(conf, 3), "method": method,
            "ranked_candidates": feasible_ranked[:4],
            "infeasible_ignored": [t for t, _ in ranked if not feasible(t)]}


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #

@dataclass
class AgentResult:
    run_id: str
    query: str
    configuration: dict
    selected_task: str
    answer: str
    confidence: float
    outputs: Dict[str, Any]
    visuals: Dict[str, np.ndarray]
    trace: List[Dict[str, Any]]
    report_paths: Dict[str, Path]

    def to_dict(self, include_visuals: bool = False) -> dict:
        d = {
            "run_id": self.run_id,
            "query": self.query,
            "configuration": self.configuration,
            "selected_task": self.selected_task,
            "answer": self.answer,
            "confidence": self.confidence,
            "outputs": {k: v for k, v in self.outputs.items()
                        if not isinstance(v, np.ndarray)},
            "execution_summary": self.trace,
            "reports": {k: str(p) for k, p in self.report_paths.items()},
        }
        return d


class AgentController:
    """Selects, sequences and executes specialist tools; produces an auditable
    execution summary."""

    def __init__(self, registry: Optional[Dict[str, ToolSpec]] = None) -> None:
        self.registry = registry or build_default_registry()

    # -- main entry ------------------------------------------------------ #
    def run(self, images: Sequence[RSImage | str | Path], query: str,
            task_override: Optional[str] = None,
            params: Optional[Dict] = None,
            save_report: bool = True,
            trace_callback=None) -> AgentResult:
        """trace_callback: optional fn(trace) invoked after every pipeline
        step — used by the server for live UI streaming. Thread-safe by
        design: no controller state is mutated."""
        t_start = time.time()
        trace: List[Dict[str, Any]] = []
        params = params or {}

        def emit():
            if trace_callback:
                try:
                    trace_callback(trace)
                except Exception:
                    pass

        # 1. load + validate -------------------------------------------------
        step = self._step(trace, "validate_inputs", {})
        emit()
        imgs: List[RSImage] = []
        for im in images:
            if isinstance(im, (str, Path)):
                im = load_image(im)
            imgs.append(im)
        cfg = validate_inputs(imgs)
        step.update(status="ok", output={
            **cfg, "description": describe_configuration(imgs, cfg),
            "inputs": [i.summary() for i in imgs]})
        step["duration_ms"] = int((time.time() - t_start) * 1000)
        emit()

        # 2. classify intent --------------------------------------------------
        step = self._step(trace, "classify_task", {"query": query})
        intent = classify_task(query, cfg["configuration"])
        if task_override:
            intent = {"task": task_override, "confidence": 1.0,
                      "method": "explicit user override",
                      "ranked_candidates": [], "infeasible_ignored": []}
        clarification = build_clarification(query, intent, cfg["configuration"])
        if clarification is not None:
            intent["clarification"] = clarification
        step.update(status="ok", output=intent)
        emit()

        task = intent["task"]

        if task == "investigation":
            return self._run_investigation(imgs, query, intent, trace, emit,
                                           save_report)

        # 3. select tool ------------------------------------------------------
        step = self._step(trace, "select_tool", {"candidate_task": task})
        spec = self._select_tool(task, cfg, query_text=query)
        step.update(status="ok", output={"tool": spec.name,
                                         "signature": spec.signature()})
        emit()

        # 4. execute ----------------------------------------------------------
        ctx = {"images": imgs, "query": query, "config": cfg, "params": params}
        step = self._step(trace, f"execute:{spec.name}", {"params": params})
        t_exec = time.time()
        try:
            out = spec.fn(ctx)
            err = None
        except InputValidationError as e:
            out, err = None, str(e)
        except Exception as e:  # keep agent alive; report failure honestly
            out, err = None, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
        duration_ms = int((time.time() - t_exec) * 1000)
        if err:
            step.update(status="error", error=err, duration_ms=duration_ms)
            raise InputValidationError(f"Tool '{spec.name}' failed: {err}")
        visuals = out.pop("_visual", {}) or {}
        step.update(status="ok", output_keys=sorted(out.keys()),
                    duration_ms=duration_ms)
        emit()

        # 5. integrate ---------------------------------------------------------
        answer, confidence = self._integrate(spec, out)
        from .suggestions import suggest
        out["suggestions"] = suggest({"selected_task": spec.name,
                                      "outputs": out})
        # surface the intent-stage confidence separately from the answer
        # confidence (which comes from the specialist tool output) so the
        # clarification gate is auditable end-to-end
        out["intent_confidence"] = float(intent.get("confidence", 1.0))
        if clarification is not None:
            out["clarification"] = clarification
        result = AgentResult(
            run_id=datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            query=query, configuration={**cfg,
                       "configuration_label": describe_configuration(imgs, cfg)},
            selected_task=spec.name,
            answer=answer,
            confidence=float(out.get("confidence", confidence)),
            outputs=out,
            visuals=visuals,
            trace=trace,
            report_paths={},
        )

        # 6. report -------------------------------------------------------------
        if save_report:
            visual_paths = self._save_visuals(result)
            result.report_paths = write_report(result, imgs, extra={
                "tools_used": [s.get("name", "") for s in trace
                               if s.get("name", "").startswith("execute")],
                "visual_evidence": {k: str(v) for k, v in visual_paths.items()},
            })
        total_ms = int((time.time() - t_start) * 1000)
        trace.append({"step": "finish", "name": "controller",
                      "total_duration_ms": total_ms, "status": "ok"})
        return result

    # -- investigation plan ----------------------------------------------- #
    _INVESTIGATION_PLANS = {
        "default": [
            ("change_analysis", "Detect what changed", {}),
            ("grounding_water", "Extract the water body", {"concept": "water"}),
            ("impact_analysis", "Quantify impact & rank zones", {}),
        ],
        "urban": [
            ("change_analysis", "Detect what changed", {}),
            ("grounding_built_up", "Identify built-up expansion", {"concept": "built-up"}),
            ("impact_analysis", "Quantify impact & rank zones", {}),
        ],
        "vegetation": [
            ("change_analysis", "Detect what changed", {}),
            ("grounding_vegetation", "Identify vegetation changes", {"concept": "vegetation"}),
            ("impact_analysis", "Quantify impact & rank zones", {}),
        ],
        "comprehensive": [
            ("change_analysis", "Detect what changed", {}),
            ("grounding_water", "Extract the water body", {"concept": "water"}),
            ("grounding_built_up", "Identify built-up areas", {"concept": "built-up"}),
            ("impact_analysis", "Quantify impact & rank zones", {}),
        ],
    }

    @staticmethod
    def _select_investigation_plan(query: str) -> str:
        """Select the investigation plan variant based on query content."""
        q = query.lower()
        urban_kws = ("urban", "building", "built-up", "development",
                     "construction", "expansion", "encroachment")
        veg_kws = ("vegetation", "forest", "deforestation", "tree",
                    "green", "crop", "agriculture", "clearing")
        if any(kw in q for kw in urban_kws):
            return "urban"
        if any(kw in q for kw in veg_kws):
            return "vegetation"
        if any(w in q for w in ("full analysis", "comprehensive",
                                "everything", "all aspects")):
            return "comprehensive"
        return "default"

    def _run_investigation(self, imgs: List[RSImage], query: str,
                           intent: Dict, trace: List[Dict[str, Any]],
                           emit, save_report: bool) -> AgentResult:
        """Query-conditioned multi-step investigation workflow.

        Selects a plan variant based on the query's dominant concept,
        then executes each step with per-step error isolation.
        Every step is observable in the trace.
        """
        from .tools_impl import (change_analysis_tool, grounding_tool,
                                 impact_analysis_tool)

        plan_key = self._select_investigation_plan(query)
        plan = self._INVESTIGATION_PLANS[plan_key]
        outputs: Dict[str, Any] = {}
        visuals: Dict[str, np.ndarray] = {}
        t_start = time.time()
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

        step = self._step(trace, "plan_investigation",
                          {"plan": plan_key, "steps": [name for name, _, _ in plan]})
        emit()

        ctx = {"images": imgs, "query": query, "params": {}}
        for name, label, extra in plan:
            tool_name = ("grounding" if name.startswith("grounding_")
                         else name)
            spec = self.registry[tool_name]
            sctx = dict(ctx)
            if extra:
                q = f"{query} {extra.get('concept', '')}".strip()
                sctx["query"] = q
                if "concept" in extra:
                    sctx["_forced_concept"] = extra["concept"]
            t_exec = time.time()
            try:
                out = spec.fn(sctx)
                err = None
            except Exception as e:
                import traceback
                out, err = None, traceback.format_exc(limit=3)
            dur = int((time.time() - t_exec) * 1000)
            entry = {"step_id": len(trace) + 1, "name": f"execute:{name}",
                     "label": label, "status": "ok" if not err else "error",
                     "duration_ms": dur}
            if err:
                entry["error"] = err
            trace.append(entry)
            emit()
            if out:
                visuals.update(out.pop("_visual", {}) or {})
                outputs[name] = out

        # synthesize -------------------------------------------------------
        imp = outputs.get("impact_analysis", {})
        chg = outputs.get("change_analysis", {})
        wat = outputs.get("grounding_water", {})
        answer_bits = []
        if chg.get("answer"):
            answer_bits.append(str(chg["answer"]).rstrip("."))
        if imp.get("changed_area_ha") is not None:
            answer_bits.append(
                f"Changed area ≈ {imp['changed_area_ha']} ha"
                + (f" ({imp['near_water']['within_500m'] * 100:.0f}% within "
                   f"500 m of water)" if imp["near_water"] else ""))
        if wat.get("boxes"):
            answer_bits.append(f"water body localised ({len(wat['boxes'])} region)")
        if imp.get("priority_zone"):
            answer_bits.append(f"priority zone {imp['priority_zone']}")
        f0 = (imp.get("findings") or [{}])[0]
        if f0.get("action"):
            answer_bits.append(f"Action: {f0['action']}")
        answer = ". ".join(answer_bits) + "." if answer_bits else \
            "Investigation complete; no significant findings."

        from .suggestions import suggest
        suggestions = suggest({"selected_task": "investigation",
                               "outputs": {"investigation": {"impact": imp}}})

        result = AgentResult(
            run_id=run_id, query=query,
            configuration={"configuration_label": "investigation"},
            selected_task="investigation",
            answer=answer,
            confidence=float(imp.get("confidence", 0.78)) if imp else 0.7,
            outputs={"investigation": {
                "change": {k: v for k, v in chg.items()
                           if not isinstance(v, (list, np.ndarray)) and k != "_x"},
                "water_regions": len(wat.get("boxes", []) or []),
                "impact": imp,
            }, "suggestions": suggestions},
            visuals=visuals,
            trace=trace,
            report_paths={},
        )
        emit()
        if save_report:
            result.report_paths = write_report(result, imgs)
        return result

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _step(trace: List[Dict], name: str, params: Dict) -> Dict:
        entry = {"step_id": len(trace) + 1, "name": name, "params": params}
        trace.append(entry)
        return entry

    def _select_tool(self, task: str, cfg: dict, query_text: str = "") -> ToolSpec:
        spec = self.registry.get(task)
        if spec is None:
            raise InputValidationError(f"No tool registered for task '{task}'.")
        need = spec.requires
        have = cfg["configuration"]
        ok = ((need == "single" and have == "single") or
              (need == "bitemporal" and have == "bitemporal_pair") or
              (need == "crossmodal" and have == "optical_sar_pair"))
        if not ok:
            raise InputValidationError(
                f"Selected tool '{spec.name}' requires a {need} input, but the "
                f"supplied configuration is '{have}'. "
                f"{describe_configuration([], cfg)}")
        if spec.needs_query and not query_text:
            raise InputValidationError(
                f"Tool '{spec.name}' requires a natural-language query.")
        return spec

    @staticmethod
    def _save_visuals(result: "AgentResult") -> Dict[str, Path]:
        """Persist evidence rasters (change masks, overlays) next to the report
        so judges can compare them against reference masks."""
        from PIL import Image
        if not result.visuals:
            return {}
        vis_dir = CONFIG.runs_dir / result.run_id / "visuals"
        vis_dir.mkdir(parents=True, exist_ok=True)
        saved = {}
        for key, arr in result.visuals.items():
            a = np.asarray(arr)
            if a.ndim == 2:
                img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
                fname = f"{key}.png" if key != "mask" else "change_mask.png"
            else:
                img = Image.fromarray((np.clip(a[..., :3], 0, 1) * 255).astype(np.uint8))
                fname = f"{key}.png"
            path = vis_dir / fname
            img.save(path)
            saved[key] = path
        return saved

    @staticmethod
    def _integrate(spec: ToolSpec, out: Dict):
        if "answer" in out:
            return str(out["answer"]), float(out.get("confidence", 0.5))
        if "caption" in out:
            return str(out["caption"]), float(out.get("confidence", 0.5))
        if "description" in out:
            return str(out["description"]), float(out.get("confidence", 0.5))
        summary_bits = []
        if "dominant" in out:
            summary_bits.append(f"Dominant fused class: {out['dominant']}")
        if "notes" in out:
            summary_bits.extend(out["notes"][:2])
        return " ".join(summary_bits), float(out.get("confidence", 0.5))


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def write_report(result: AgentResult, images: Sequence[RSImage],
                 extra: Optional[Dict] = None) -> Dict[str, Path]:
    run_dir = CONFIG.runs_dir / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "system": "SatQuery AI v1.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "query": result.query,
        "input_configuration": result.configuration,
        "inputs": [i.summary() for i in images],
        "selected_task": result.selected_task,
        "final_answer": result.answer,
        "confidence": result.confidence,
        "outputs": {k: v for k, v in result.outputs.items()
                    if not isinstance(v, np.ndarray)},
        "execution_summary": [
            {k: v for k, v in step.items()} for step in result.trace],
        "extra": extra or {},
    }
    json_path = run_dir / "report.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str),
                         encoding="utf-8")

    md = [
        f"# Anvesha — Earth Observation & Investigation Report", "",
        f"| Field | Value |",
        f"|---|---|",
        f"| **Run ID** | `{result.run_id}` |",
        f"| **Generated** | {payload['generated_at']} |",
        f"| **Query** | {result.query} |",
        f"| **Input configuration** | {result.configuration.get('configuration_label', '')} |",
        f"| **Selected task** | `{result.selected_task}` |",
        f"| **Confidence** | {result.confidence:.1%} |", "",
        "---", "",
        "## Answer", "", str(result.answer), "", "---", "",
        "## Input Files", "",
        "| File | Modality | Bands | Georeferenced |",
        "|---|---|---|---|",
    ]
    for inp in payload.get("inputs", []):
        geo = "Yes" if inp.get("georeferenced") else "No"
        md.append(f"| {inp.get('file', '')} | {inp.get('modality', '')} | {inp.get('bands', '')} | {geo} |")
    md += ["", "---", "", "## Execution Trace", "",
           "| Step | Status | Duration |",
           "|---|---|---|",
    ]
    for s in result.trace:
        name = s.get('name', '')
        status = s.get('status', 'ok')
        dur = f"{s.get('duration_ms')} ms" if s.get('duration_ms') else '—'
        md.append(f"| `{name}` | {status} | {dur} |")
    md += ["", "---", "", "## Outputs", "", "```json",
           json.dumps(payload["outputs"], indent=2, default=str)[:4000],
           "```", "", "---",
           f"*Report generated by Anvesha — Earth Observation & Investigation System (SIH26167)*"]
    md_path = run_dir / "report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


_CONTROLLER: Optional[AgentController] = None
_controller_lock = threading.Lock()


def get_controller() -> AgentController:
    global _CONTROLLER
    if _CONTROLLER is None:
        with _controller_lock:
            if _CONTROLLER is None:
                _CONTROLLER = AgentController()
    return _CONTROLLER
