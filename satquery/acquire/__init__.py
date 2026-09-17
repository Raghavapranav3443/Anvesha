"""Online acquisition layer: air-gap enforcement, catalogues, COG window reads.

Everything in this package is additive to the offline pipeline. The offline
core never imports it except through :mod:`satquery.acquire.mode`, which is
itself a no-op unless air-gap mode is being enforced.
"""
from __future__ import annotations

from .errors import (AcquireError, FetchFailed, NetworkBlockedAirgap,
                     NoCatalogAvailable, NoSceneFound, ScopeTooLarge)

__all__ = [
    "AcquireError",
    "FetchFailed",
    "NetworkBlockedAirgap",
    "NoCatalogAvailable",
    "NoSceneFound",
    "ScopeTooLarge",
]
