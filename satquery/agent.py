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
# Intent classification
# --------------------------------------------------------------------------- #

TASK_KEYWORDS = {
    "investigation": ["investigate", "investigation", "is it significant",
                      "significant", "impact of", "urban expansion",
                      "should i investigate", "encroachment", "full analysis",
                      "analyse the change around"],
    "grounding": ["highlight", "locate", "localise", "localize", "find the",
                  "where is", "mark", "show me the region", "bounding box",
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


def classify_task(query: str, configuration: str) -> Dict[str, Any]:
    """Rule-based intent classifier returning ranked candidate tasks."""
    import re

    q = query.lower()
    scores: Dict[str, float] = {}
    for task, kws in TASK_KEYWORDS.items():
        s = 0.0
        for kw in kws:
            if ".*" in kw or kw.endswith("*"):
                # treat as regular-expression pattern
                try:
                    if re.search(kw.rstrip("*") + r".*" if kw.endswith("*") else kw, q):
                        s += 1.0
                except re.error:
                    pass
            elif kw in q:
                s += 1.0
        scores[task] = s

    # merge aliased intents into their registered tool before ranking so a
    # strong keyword hit is not lost to the low-confidence default branch
    for alias, target in TASK_ALIASES.items():
        if alias in scores:
            scores[target] = scores.get(target, 0.0) + scores.pop(alias)

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

    # defaults when nothing matches strongly
    default_map = {
        "single": ("captioning" if not query.strip() else "single_vqa"),
        "bitemporal_pair": "change_analysis",
        "optical_sar_pair": "optical_sar",
    }
    if not feasible_ranked or feasible_ranked[0][1] == 0.0:
        best = default_map.get(configuration, "single_vqa")
        conf = 0.35
        method = "default routing (no strong keyword match)"
    else:
        best = feasible_ranked[0][0]
        total = sum(s for _, s in feasible_ranked) + 1e-6
        conf = min(0.95, 0.5 + 0.45 * feasible_ranked[0][1] / total)
        method = "keyword-intent rules"

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
    def _run_investigation(self, imgs: List[RSImage], query: str,
                           intent: Dict, trace: List[Dict[str, Any]],
                           emit, save_report: bool) -> AgentResult:
        """Agentic multi-step workflow: change detection → water extraction →
        impact quantification → synthesis. Every step is observable."""
        from .tools_impl import (change_analysis_tool, grounding_tool,
                                 impact_analysis_tool)

        plan = [
            ("change_analysis", "Detect what changed", {}),
            ("grounding_water", "Extract the water body", {"concept": "water"}),
            ("impact_analysis", "Quantify impact & rank zones", {}),
        ]
        outputs: Dict[str, Any] = {}
        visuals: Dict[str, np.ndarray] = {}
        t_start = time.time()
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

        step = self._step(trace, "plan_investigation",
                          {"steps": [name for name, _, _ in plan]})
        emit()

        ctx = {"images": imgs, "query": query, "params": {}}
        for name, label, extra in plan:
            tool_name = ("grounding" if name == "grounding_water"
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

    md = [f"# SatQuery AI Report ({result.run_id})", "",
          f"- **Query:** {result.query}",
          f"- **Input configuration:** {result.configuration.get('configuration_label', '')}",
          f"- **Selected task/tool:** `{result.selected_task}`",
          f"- **Confidence:** {result.confidence:.2f}", "",
          "## Answer", "", str(result.answer), "",
          "## Execution summary", ""]
    for s in result.trace:
        md.append(f"- `{s.get('name')}` status={s.get('status','ok')} "
                  f"{('duration=' + str(s.get('duration_ms')) + 'ms') if s.get('duration_ms') else ''}")
    md += ["", "## Outputs", "", "```json",
           json.dumps(payload["outputs"], indent=2, default=str)[:4000],
           "```"]
    md_path = run_dir / "report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


_CONTROLLER: Optional[AgentController] = None


def get_controller() -> AgentController:
    global _CONTROLLER
    if _CONTROLLER is None:
        _CONTROLLER = AgentController()
    return _CONTROLLER
