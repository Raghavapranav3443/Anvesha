"""Anvesha - agentic vision-language assistant for multimodal remote sensing."""
from __future__ import annotations

__version__ = "1.0.0"

from .config import CONFIG, BEN19_CLASSES, EUROSAT_CLASSES


def _enforce_airgap_from_env() -> None:
    """Install the outbound-network guard when ``ANVESHA_MODE`` is set.

    Doing this at package import covers child processes and CLI entry points
    (``python -m anvesha.evaluate``, ``scripts/*``) without each one having to
    opt in -- that inheritance is what makes the air-gap guarantee hold beyond
    the server process. Deliberately a no-op when the env var is absent, and
    failure-tolerant so it can never break ``import anvesha``.
    """
    import os
    if not os.environ.get("ANVESHA_MODE", "").strip():
        return
    try:
        from .acquire.mode import enforce_from_env
        enforce_from_env()
    except Exception:                       # never break package import
        pass


_enforce_airgap_from_env()
