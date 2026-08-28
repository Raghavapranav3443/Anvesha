"""CLIP text encoder (offline, RS-adapted VL component text side).

Loads the HF-cached `openai/clip-vit-base-patch32` text tower and embeds
questions into the same 512-d space the fusion heads consume. This replaces
md5-hash bag-of-words question features with real language features — the
Phase 2 adoption of the PS-mandated "remote-sensing-adapted vision-language
component" on the text side.

Design constraints (mirroring repo discipline):
  - Works fully offline from the HF cache (weights bundled with the project's
    machine; verified by scripts/cache_clip_weights.py --verify).
  - Singleton loader; per-question LRU cache (RSVQA repeats question templates).
  - Graceful degradation: embed() returns None when the model or transformers
    is unavailable, so callers can fall back to the hash-BOW stack unchanged.
"""
from __future__ import annotations

import hashlib
import threading
from typing import List, Optional

import numpy as np

MODEL_ID = "openai/clip-vit-base-patch32"
DIM = 512

_lock = threading.Lock()
_singleton: Optional["ClipTextEncoder"] = None


class ClipTextEncoder:
    def __init__(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPTokenizer

        self.torch = torch
        try:
            # offline-first: the cache is populated by cache_clip_weights.py
            self.tok = CLIPTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
            self.model = CLIPModel.from_pretrained(MODEL_ID, local_files_only=True)
        except Exception:
            try:  # second chance: allow a network fetch (interactive machines)
                self.tok = CLIPTokenizer.from_pretrained(MODEL_ID)
                self.model = CLIPModel.from_pretrained(MODEL_ID)
            except Exception:
                raise
        self.model.eval()
        import satquery.config as _cfg
        self.device = _cfg.CONFIG.resolve_device()
        self.model.to(self.device)
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        """Embed a list of questions -> (N, 512) float32, L2-normalized."""
        torch = self.torch
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        miss: dict[int, str] = {}
        for i, t in enumerate(texts):
            key = _qkey(t)
            hit = self._cache.get(key)
            if hit is not None:
                out[i] = hit
            else:
                miss[i] = t
        if miss:
            uniq = list(dict.fromkeys(miss.values()))
            batch = 256
            with torch.no_grad():
                for s in range(0, len(uniq), batch):
                    chunk = uniq[s:s + batch]
                    inp = self.tok(chunk, padding=True, truncation=True,
                                   max_length=77, return_tensors="pt")
                    inp = {k: v.to(self.device) for k, v in inp.items()}
                    f = self.model.get_text_features(**inp)          # (B,512)
                    f = torch.nn.functional.normalize(f, dim=-1)
                    f = f.float().cpu().numpy()
                    for t, v in zip(chunk, f):
                        self._cache[_qkey(t)] = v
            for i, t in miss.items():
                out[i] = self._cache[_qkey(t)]
        return out

    @property
    def available(self) -> bool:
        return self.model is not None


def _qkey(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()


def get_clip_text() -> Optional[ClipTextEncoder]:
    """Singleton accessor; None when the VL component cannot be loaded."""
    global _singleton
    if _singleton is None:
        with _lock:
            if _singleton is None:
                try:
                    _singleton = ClipTextEncoder()
                except Exception:
                    return None
    return _singleton
