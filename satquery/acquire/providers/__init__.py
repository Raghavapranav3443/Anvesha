"""Imagery and context providers for the online layer.

Two kinds of provider live here, and the distinction is the heart of the
online design:

* :mod:`~satquery.acquire.providers.stac` -- **analytic imagery**. Sentinel-2
  L2A (and Sentinel-1 GRD) Cloud-Optimized GeoTIFFs on open, credential-free
  endpoints. This is what the models actually consume, and it is the same
  sensor family the ISRO Bhoonidhi service distributes, so the data lineage is
  unchanged. Because it needs no account, it works today; Bhoonidhi becomes an
  additional provider behind the same interface when credentials arrive.
* :mod:`~satquery.acquire.providers.bhuvan` -- **ISRO authority and context**.
  ISRO's own OGC service, credential-free, used for the thematic layers that
  tell a non-expert what the ground is officially recorded as, and who to
  notify.
"""
from __future__ import annotations

from .bhuvan import BhuvanProvider
from .stac import SceneRef, StacProvider, default_providers

__all__ = ["BhuvanProvider", "SceneRef", "StacProvider", "default_providers"]
