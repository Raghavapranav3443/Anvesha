"""Decision layer -- the final stage of the Anvesha pipeline.

Turns a completed analysis into a conclusion a non-expert can act on: what
changed, what it means, what to do, who to tell, how far to trust it, and what
the imagery cannot show.

Runs entirely offline as a pure function over artefacts the analysis already
produced, so it still works when connectivity is gone -- which is precisely when
a flood-impact question gets asked.

    from anvesha.decision import decide_for_result
    record = decide_for_result(result, images)
    record["advice"]["what_to_do"]
"""
from __future__ import annotations

from .advise import JARGON, OUTCOME_LABELS, compose, find_jargon
from .engine import SCHEMA, decide, decide_for_result
from .facts import Facts, assemble, human_area
from .rules import RULES, refusal_rules, verdict_rules
from .sensitivity import (detection_floor_ha, detection_floor_text,
                          stability_range, stability_text, threshold_sweep)

__all__ = [
    "SCHEMA", "decide", "decide_for_result", "assemble", "Facts", "human_area",
    "RULES", "refusal_rules", "verdict_rules", "compose", "OUTCOME_LABELS",
    "JARGON", "find_jargon", "threshold_sweep", "stability_range",
    "stability_text", "detection_floor_ha", "detection_floor_text",
]
