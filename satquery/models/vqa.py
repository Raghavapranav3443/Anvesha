"""Visual Question Answering specialist for remote sensing imagery.

Architecture (RS-adapted): shared SceneEncoder visual embedding + hashed
bag-of-words question embedding -> fusion MLP -> answer classifier.
Trained on RSVQA via scripts/train_vqa.py. A rule-based reasoner over the
scene classifier's concept evidence provides graceful fallback answers.
"""
from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ..config import CONFIG
from ..io_utils import RSImage
from ..text import content_tokens, is_question, tokenize
from .backbone import SceneEncoder, _rgb3_compat as _rgb3, normalise_for_encoder, resize_np, to_tensor
from .scene import get_scene_classifier


class RSVQAModel:
    ANSWERS = ["yes", "no", "water", "vegetation", "built-up area", "agriculture",
               "road", "bare land", "unknown"]

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or CONFIG.resolve_device()
        self.encoder = None
        self.head = None
        self.answer_vocab = list(self.ANSWERS)
        self.trained = False
        self.input_size = 128
        self.bow_dim = 64
        self.type_vocab: Dict[str, int] = {}
        self.count = None
        self.temperature = 1.0
        path = CONFIG.vqa_weights
        if path.exists():
            try:
                import torch
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                enc = SceneEncoder(3)
                enc.load_state_dict(ckpt["encoder"])
                self.bow_dim = int(ckpt.get("bow_dim", 64))
                self.type_vocab = ckpt.get("type_vocab", {}) or {}
                head = _FusionHead(len(ckpt["answer_vocab"]),
                                   len(self.type_vocab))
                head.load_state_dict(ckpt["head"])
                self.encoder, self.head = enc.eval(), head.eval()
                self.encoder.to(self.device); self.head.to(self.device)
                self.answer_vocab = ckpt["answer_vocab"]
                self.input_size = int(ckpt.get("input_size", 128))
                self.temperature = float(ckpt.get("temperature", 1.0))
                self.trained = True
            except Exception:
                self.trained = False
        self._maybe_torchscript()

    def _maybe_torchscript(self) -> None:
        """On CPU, prefer the exported TorchScript encoder/head (P3)."""
        import os
        if self.device != "cpu" or not self.trained:
            return
        if os.environ.get("SATQUERY_TORCHSCRIPT", "1") != "1":
            return
        ts_dir = CONFIG.weights_dir / "ts"
        try:
            import torch
            enc = torch.jit.load(str(ts_dir / "vqa_encoder_int8.ts"),
                                 map_location="cpu").eval()
            head = torch.jit.load(str(ts_dir / "vqa_head_int8.ts"),
                                  map_location="cpu").eval()
            self.encoder, self.head = enc, head
        except Exception:
            pass
        self._load_count_head()

    def _load_count_head(self) -> None:
        path = CONFIG.weights_dir / "count_head.pt"
        if not path.exists():
            return
        try:
            import torch
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            enc = SceneEncoder(3)
            enc.load_state_dict(ckpt["encoder"])
            head = _FusionHead(len(ckpt["classes"]),
                               len(ckpt.get("type_vocab", {}) or {}))
            head.load_state_dict(ckpt["head"])
            self.count = {
                "encoder": enc.eval().to(self.device),
                "head": head.eval().to(self.device),
                "classes": ckpt["classes"],
                "type_vocab": ckpt.get("type_vocab", {}) or {},
                "input_size": int(ckpt.get("input_size", 128)),
                "bow_dim": int(ckpt.get("bow_dim", 512)),
                "temperature": float(ckpt.get("temperature", 1.0)),
                "val_digit_acc": ckpt.get("val_digit_acc"),
            }
        except Exception:
            self.count = None

    def answer(self, img: RSImage, question: str) -> Dict:
        qtype = None
        if self.trained:
            qtype_name = None
            for name in self.type_vocab:
                if name in question.lower():
                    qtype_name = name
                    break
            if qtype_name == "count" and self.count is not None:
                out = self._count_answer(img, question)
                if out is not None:
                    return out
        if self.trained:
            out = self._model_answer(img, question)
            if out is not None:
                return out
        return self._rule_answer(img, question)

    def _count_answer(self, img: RSImage, question: str) -> Optional[Dict]:
        """Dedicated counting head with flip-TTA."""
        import torch
        cfg = self.count
        rgb = _rgb3(img)
        x = to_tensor(resize_np(normalise_for_encoder(rgb, img.modality),
                                cfg["input_size"])).to(self.device)
        xf = torch.flip(x, dims=[3])
        q = np.zeros(cfg["bow_dim"], dtype=np.float32)
        q[:] = _hashed_bow(question, dim=cfg["bow_dim"])
        qt = torch.tensor([infer_question_type(question, cfg["type_vocab"])])
        qb = torch.from_numpy(q).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = 0.5 * (
                torch.softmax(cfg["head"](cfg["encoder"](x), qb, qt), -1) +
                torch.softmax(cfg["head"](cfg["encoder"](xf), qb, qt), -1)
            )[0].cpu().numpy()
        top = int(np.argmax(probs))
        digit = cfg["classes"][top]
        return {"answer": str(digit),
                "confidence": round(float(probs[top]), 3),
                "candidates": [(str(cfg["classes"][i]),
                                round(float(probs[i]), 3))
                               for i in np.argsort(-probs)[:3]],
                "source": "dedicated counting head (RSVQA count subset)"}

    # ------------------------------------------------------------------ #
    def _model_answer(self, img: RSImage, question: str) -> Optional[Dict]:
        import torch
        rgb = _rgb3(img)
        x = to_tensor(resize_np(normalise_for_encoder(rgb, img.modality),
                                self.input_size)).to(self.device)
        q = np.zeros(self.bow_dim, dtype=np.float32)
        q[:] = _hashed_bow(question, dim=self.bow_dim)
        t = torch.tensor([infer_question_type(question, self.type_vocab)],
                         device=self.device)
        qb = torch.from_numpy(q).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.encoder(x)
            logits = self.head(feat, qb, t)
            feat_f = self.encoder(torch.flip(x, dims=[3]))
            logits_f = self.head(feat_f, qb, t)
            probs = (torch.softmax(logits / self.temperature, -1) +
                     torch.softmax(logits_f / self.temperature, -1))[0] / 2.0
            probs = probs.cpu().numpy()
        top = int(np.argmax(probs))
        order = np.argsort(-probs)[:3]
        cands = [(self.answer_vocab[i], round(float(probs[i]), 3)) for i in order]
        return {"answer": self.answer_vocab[top],
                "confidence": round(float(probs[top]), 3),
                "candidates": cands,
                "source": "fine-tuned RSVQA model"}

    # ------------------------------------------------------------------ #
    def _rule_answer(self, img: RSImage, question: str) -> Dict:
        """Deterministic reasoner over scene evidence (fallback / auditable)."""
        scene = get_scene_classifier()
        presence = scene.concept_presence(img)
        stats_tokens = set(tokenize(question))
        q = question.lower().strip()
        yes_no = q.startswith(("is ", "are ", "does ", "do ", "has ", "have ",
                               "was ", "were ", "can ")) or q.endswith("?") and (
            "is there" in q or "are there" in q)

        # Which concept is being asked about?
        target, score = None, 0.0
        for concept in ("water", "vegetation", "built-up", "road", "agriculture", "bare"):
            syns = {
                "water": ("water", "river", "lake", "sea"),
                "vegetation": ("vegetation", "forest", "tree", "grass", "field", "crop"),
                "built-up": ("building", "urban", "house", "industrial", "city"),
                "road": ("road", "highway", "street"),
                "agriculture": ("agricultur", "farm", "crop field"),
                "bare": ("bare", "soil", "sand"),
            }[concept]
            s = sum(1.0 for syn in syns if syn in q)
            if s > score:
                target, score = concept, s

        if target is not None and target in presence:
            p = presence[target]
            if yes_no:
                ans = "yes" if p > 0.30 else "no"
                conf = abs(p - 0.5) * 1.4 + 0.25 if p > 0.3 else 0.9 - p
                return {"answer": ans, "confidence": round(float(min(conf + 0.35, 0.97)), 3),
                        "evidence": {f"{target} presence": round(p, 2)},
                        "source": "scene-evidence rule reasoner"}
            # open question about a concept
            label = {"water": "water body", "vegetation": "vegetation",
                     "built-up": "built-up area", "road": "road/transport",
                     "agriculture": "agricultural land", "bare": "bare soil"}[target]
            return {"answer": label if p > 0.25 else "none visible",
                    "confidence": round(float(max(p, 0.2)), 3),
                    "evidence": {f"{target} presence": round(p, 2)},
                    "source": "scene-evidence rule reasoner"}

        if any(w in q for w in ("land-cover", "land cover", "type of", "scene",
                                "shown", "visible", "content")):
            pred = scene.predict(img, top_k=3)
            names = ", ".join(n for n, _ in pred["labels"][:2])
            return {"answer": names, "confidence": pred["confidence"],
                    "labels": pred["labels"], "source": pred["source"]}

        # generic yes/no about dominant class
        pred = scene.predict(img, top_k=2)
        top_name = pred["labels"][0][0]
        if "residential" in q or "industrial" in q:
            hit = [s for n, s in pred["labels"] if n.lower() in q]
            p = hit[0] if hit else 0.0
            return {"answer": "yes" if p > 0.25 else "no",
                    "confidence": round(float(max(p, 1 - max(p, 0.4))), 3),
                    "source": pred["source"]}
        return {"answer": f"{top_name}-dominated scene",
                "confidence": pred["confidence"],
                "labels": pred["labels"], "source": pred["source"]}


def _hashed_bow(text: str, dim: int = 512) -> np.ndarray:
    import hashlib, re
    vec = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = hashlib.md5(tok.encode()).hexdigest()
        idx = int(h[:8], 16) % dim
        sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


class _FusionHead(nn.Module):
    def __init__(self, n_answers: int, n_types: int = 1):
        super().__init__()
        self.img_proj = nn.Linear(SceneEncoder.FEATURE_DIM, 256)
        self.q_proj = nn.Linear(512, 256)
        self.t_embed = nn.Embedding(n_types + 1, 32)
        self.mlp = nn.Sequential(nn.Linear(256 + 256 + 32, 384), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(384, n_answers))

    def forward(self, feat, q, t=None):
        if t is None:
            t = torch.zeros(feat.shape[0], dtype=torch.long,
                            device=feat.device)
        z = torch.cat([torch.relu(self.img_proj(feat)),
                       torch.relu(self.q_proj(q)),
                       self.t_embed(t)], dim=1)
        return self.mlp(z)


def infer_question_type(question: str, type_vocab: Dict[str, int]) -> int:
    """Rule-based mapping of a free question onto the RSVQA type taxonomy."""
    q = question.lower()
    rules = [
        ("rural_urban", ("rural or an urban", "rural or urban", "rural",
                         "urban area")),
        ("comp", ("more ", "less ", "equal to the number", "bigger",
                  "smaller", "larger", " than ")),
        ("count", ("how many", "number of", "amount of", "count of")),
        ("presence", ("is there", "are there", "present", "does it contain",
                      "can we see", "presence", "is a ", "is the ")),
        ("area", ("area of", "percentage", "percent", "proportion",
                  "fraction", "how much")),
    ]
    for name, keys in rules:
        if any(k in q for k in keys) and name in type_vocab:
            return type_vocab[name]
    return len(type_vocab)   # <unk> slot


_INSTANCE: Optional[RSVQAModel] = None


def get_vqa_model(device: Optional[str] = None) -> RSVQAModel:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = RSVQAModel(device=device)
    return _INSTANCE
