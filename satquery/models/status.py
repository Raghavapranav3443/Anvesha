"""Degradation transparency: which specialists are running trained weights
vs heuristic fallbacks.

Every specialist model loads its checkpoint defensively (a corrupt or
missing file must never crash the assistant), which means a load failure
silently degrades that specialist to a heuristic. This module makes that
state *observable*: the server surfaces it in every job response and the
UI renders a badge when any specialist is degraded.
"""
from __future__ import annotations

from typing import Dict

from ..config import CONFIG

# weight-file attribute -> (module path, factory, weight file)
_SPECIALISTS = {
    "scene_encoder": ("satquery.models.scene", "get_scene_classifier",
                      "scene_encoder_weights"),
    "vqa": ("satquery.models.vqa", "get_vqa_model", "vqa_weights"),
    "change": ("satquery.models.change", "ChangeDetectorNet",
               "change_weights"),
    "fusion": ("satquery.models.optical_sar", "FusionNet",
               "fusion_weights"),
}

_cache: Dict[str, str] = {}


def model_status(refresh: bool = False) -> Dict[str, str]:
    """Return ``{specialist: "trained" | "heuristic"}`` for each core model.

    Instantiating each specialist is cheap after first load (singletons are
    cached inside their modules); results are memoised here unless
    ``refresh=True``.
    """
    if _cache and not refresh:
        return dict(_cache)
    import importlib
    status: Dict[str, str] = {}
    for name, (mod_path, factory, wattr) in _SPECIALISTS.items():
        weight_file: "CONFIG.weights_dir.__class__" = getattr(CONFIG, wattr)
        if not weight_file.exists():
            status[name] = "heuristic"   # no checkpoint shipped
            continue
        try:
            mod = importlib.import_module(mod_path)
            obj = getattr(mod, factory)
            instance = obj() if name != "change" else obj(device="cpu")
            status[name] = "trained" if getattr(instance, "trained", False) \
                else "heuristic"
        except Exception:
            status[name] = "heuristic"
    _cache.clear()
    _cache.update(status)
    return dict(status)


def degraded() -> bool:
    """True if any core specialist is running on heuristics."""
    return any(v != "trained" for v in model_status().values())