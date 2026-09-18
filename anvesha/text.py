"""Lightweight, deterministic text utilities: hashed bag-of-words encoder and
a remote-sensing lexicon used by the grounding specialist and the agent's
task classifier. No heavyweight NLP dependencies required."""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
DIM = 512

_STOP = {"the", "a", "an", "is", "are", "of", "in", "on", "this", "that", "to",
         "and", "or", "at", "it", "be", "for", "with", "what", "where", "which",
         "how", "has", "have", "there", "between", "these"}


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in _STOP]


def hashed_bow(text: str, dim: int = DIM) -> np.ndarray:
    """Deterministic hashed bag-of-words embedding (L2 normalised)."""
    vec = np.zeros(dim, dtype=np.float32)
    for tok in tokenize(text):
        h = hashlib.md5(tok.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % dim
        sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign
    n = float(np.linalg.norm(vec))
    if n > 0:
        vec /= n
    return vec


# --------------------------------------------------------------------------- #
# Remote-sensing lexicon: concept -> synonyms (used by grounder + router)
# --------------------------------------------------------------------------- #

LEXICON: Dict[str, List[str]] = {
    "water": ["water", "river", "lake", "pond", "reservoir", "sea", "ocean",
              "canal", "harbor", "harbour", "bay", "stream", "water body",
              "waterbody"],
    "vegetation": ["vegetation", "forest", "tree", "trees", "woodland", "grass",
                   "grassland", "meadow", "park", "crop", "crops", "farmland",
                   "field", "green", "shrub", "scrub"],
    "built-up": ["building", "buildings", "built-up", "builtup", "urban",
                 "house", "houses", "roof", "roofs", "industrial", "city",
                 "town", "settlement", "infrastructure"],
    "road": ["road", "roads", "highway", "street", "streets", "motorway",
             "lane", "pavement", "asphalt"],
    "bare": ["bare", "soil", "sand", "dune", "dunes", "beach", "desert",
             "exposed"],
}

CONCEPT_QUERIES = list(LEXICON.keys())


def concept_of_text(text: str) -> Tuple[str | None, float]:
    """Return the dominant RS concept mentioned in text with a match score."""
    toks = set(tokenize(text))
    best, best_score = None, 0.0
    for concept, syns in LEXICON.items():
        score = 0.0
        for s in syns:
            parts = s.split()
            if len(parts) == 1:
                if s in toks:
                    score += 1.0
            elif all(p in toks for p in parts):
                score += 1.5
        if score > best_score:
            best, best_score = concept, score
    if best is None:
        return None, 0.0
    return best, min(1.0, best_score / 2.0)


def is_question(text: str) -> bool:
    t = text.strip().lower()
    return t.endswith("?") or any(
        t.startswith(w) for w in ("what", "where", "is ", "are ", "does ",
                                  "do ", "how ", "has ", "have ", "can ",
                                  "which", "was ", "were ")
    )
