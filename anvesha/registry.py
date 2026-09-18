"""Tool registry: every specialist exposed to the agentic controller."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Union

# Canonical task identifiers shared by the agent, registry and API schema.
# String values (not an Enum) so existing string-typed code, JSON payloads
# and tests keep working unchanged — this is a typing contract, not a
# runtime change. A typo'd task id now fails static analysis instead of
# only failing at runtime.
TaskId = Literal["single_vqa", "captioning", "grounding", "change_analysis",
                 "change_vqa", "optical_sar", "impact_analysis",
                 "investigation"]

TASK_IDS: tuple = ("single_vqa", "captioning", "grounding",
                   "change_analysis", "change_vqa", "optical_sar",
                   "impact_analysis", "investigation")


@dataclass
class ToolSpec:
    name: str
    description: str
    requires: str            # 'single' | 'bitemporal' | 'crossmodal'
    needs_query: bool
    fn: Callable             # executed by the controller with a uniform context
    task_tags: List[str] = field(default_factory=list)

    def signature(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "requires": self.requires,
            "needs_query": self.needs_query,
        }


def build_default_registry() -> Dict[str, ToolSpec]:
    from .tools_impl import (single_vqa_tool, caption_tool, grounding_tool,
                             change_analysis_tool, change_vqa_tool,
                             optical_sar_tool, impact_analysis_tool)
    tools: Dict[str, ToolSpec] = {
        "single_vqa": ToolSpec(
            name="single_vqa",
            description="Remote-sensing VQA specialist answering natural-language "
                        "questions about one optical/multispectral/SAR image.",
            requires="single", needs_query=True, fn=single_vqa_tool,
            task_tags=["single_vqa"]),
        "captioning": ToolSpec(
            name="captioning",
            description="Scene description / captioning of a single image using "
                        "land-cover evidence and spatial layout.",
            requires="single", needs_query=False, fn=caption_tool,
            task_tags=["captioning"]),
        "grounding": ToolSpec(
            name="grounding",
            description="Text-guided region grounding: localises the referred "
                        "region and returns mask + bounding boxes.",
            requires="single", needs_query=True, fn=grounding_tool,
            task_tags=["grounding"]),
        "change_analysis": ToolSpec(
            name="change_analysis",
            description="Bi-temporal change analysis: change map, change "
                        "description and quantification.",
            requires="bitemporal", needs_query=False, fn=change_analysis_tool,
            task_tags=["change_description", "change_map"]),
        "change_vqa": ToolSpec(
            name="change_vqa",
            description="Change-based VQA over a bi-temporal pair.",
            requires="bitemporal", needs_query=True, fn=change_vqa_tool,
            task_tags=["change_vqa"]),
        "optical_sar": ToolSpec(
            name="optical_sar",
            description="Cross-modal joint analysis of a co-registered optical+"
                        "SAR pair: fused land-cover evidence and modality "
                        "complementarity notes.",
            requires="crossmodal", needs_query=False, fn=optical_sar_tool,
            task_tags=["optical_sar"]),
        "impact_analysis": ToolSpec(
            name="impact_analysis",
            description="Quantifies bi-temporal change into impact findings: "
                        "area (hectares), distance-to-water context, ranked "
                        "zones and analyst recommendations.",
            requires="bitemporal", needs_query=False, fn=impact_analysis_tool,
            task_tags=["impact"]),
    }
    return tools
