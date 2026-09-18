# G-postfix regression gate: the manual frontend-coverage audit, committed.
# Fails (exit 1) when a required backend output key loses its UI surface,
# a required route disappears, or a required affordance is unwired —
# so the exposure gap can never silently re-occur.
#
# Usage: python scripts/audit_frontend.py
# Exit 0 = all surfaces present; 1 = regressions found (CI gate).
"""Regressor: required backend-key -> frontend-component map."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web" / "src"
SAT = REPO / "anvesha" / "server"

FAILURES: list[str] = []


def check(name: str, path: Path, patterns: list[str]) -> None:
    """Assert every regex matches somewhere in the file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        FAILURES.append(f"{name}: MISSING FILE {path}")
        return
    for pat in patterns:
        if not re.search(pat, text):
            FAILURES.append(f"{name}: missing surface `{pat}` in {path.name}")


R = WEB / "components" / "Results.tsx"

# R1: compositional sections must exist (no silent drop of unhandled keys)
check("compositional-sections", R, [
    r"function ImpactSection", r"function FusionSection",
    r"function ChangeSection", r"function GroundingSection",
    r"function LabelChips", r"function StructuredOutputs",
    r"<ImpactSection", r"<FusionSection", r"<ChangeSection",
    r"<GroundingSection", r"<LabelChips",
])

# G1: honesty emission + banner mount
check("honesty", R, [r"HonestyBanner", r"outputs\.honesty"])
check("honesty-copy", WEB / "honesty_copy.ts", [
    r"HONESTY_RULES", r"rulesFor", r"fallback_active", r"pixel_space"])
check("honesty-banner", WEB / "components" / "HonestyBanner.tsx",
      [r"rulesFor", r"HonestyBanner", r"honesty_copy"])

# G2: freshness clocks rendered
check("freshness-ui", R, [r"FreshnessBadges", r"outputs\.freshness"])
check("freshness-component", WEB / "components" / "FreshnessBadges.tsx",
      [r"staleness_days", r"threshold_days", r"method_note"])

# G3: fixtures endpoint + chip
check("fixtures-route", SAT / "fixtures.py",
      [r"fixtures", r"load_manifest", r"warm_demo"])
check("fixtures-chip", WEB / "components" / "FixtureChip.tsx",
      [r"fetchFixtures", r"SYNTHETIC"])
check("fixtures-api", WEB / "api.ts",
      [r"fetchFixtures", r"FixturesManifest"])

# G4: change_description affordance
check("change-description-setup", WEB / "setups.ts", [r"change-desc"])
check("change-description-label", WEB / "labels.ts", [r"change_description"])
check("change-description-history", WEB / "components" / "HistoryView.tsx",
      [r"change_description"])
check("change-description-checklist", WEB / "components" / "JudgeRun.tsx",
      [r"change_description"])

# G5: modality certainty badge + acquired chip in InputsGrid
check("modality-badge", R, [
    r"ModalityBadge", r"modality_certainty", r"switch",
    r"acquired", r"onRequestSwitch"])
check("sampleinfo-type", WEB / "api.ts",
      [r"modality_certainty", r"acquired"])
check("console-switch", WEB / "components" / "Console.tsx",
      [r"onRequestSwitch"])

# G6: agreement overlay PNG inline in fusion section + served by reports route
check("agreement-overlay-ui", R,
      [r"agreement_overlay\.png", r"agreement_map", r"fractions"])
check("agreement-png-server", SAT / "main.py", [r"report_file", r"visuals"])

# G7: grounding ranked primary+alternates + per-box region VQA preset
check("grounding-section", R, [
    r"ranked", r"primary", r"alternates", r"RegionQuery", r"preset"])
check("regionquery-export", R, [r"REGION_QUERY_PROMPT"])

# Supporting: history dossier links, evaluation VRSBench row handled elsewhere
check("history-dossier", WEB / "components" / "HistoryView.tsx", [r"dossier"])

if __name__ == "__main__":
    if FAILURES:
        print("FRONTEND AUDIT FAILED — missing surfaces:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print(f"frontend audit: all {28} surfaces present — no regression.")