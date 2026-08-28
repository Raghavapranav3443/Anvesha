"""Download CLIP weights so the Phase 1 VL-grounding gate runs fully offline.

Usage (run once, from the project folder, in a normal terminal — NOT this
sandbox, which has a hard 30s window):

    python scripts\\cache_clip_weights.py

Downloads openai/clip-vit-base-patch32 into the user-level HuggingFace cache
(default %USERPROFILE%\\.cache\\huggingface). After this completes once, the
gate script (scripts/gate_clip_grounding.py) and the Phase 2 VQA/caption
re-platform can load the model with no network.

You may switch the model with --model, e.g.
    python scripts\\cache_clip_weights.py --model dusk2008/clip_rsicd_b16  # RS-domain variant
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-base-patch32",
                    help="HF model id (or local dir) to cache")
    ap.add_argument("--verify", action="store_true",
                    help="also load the processor + run a 1-image CLIP forward")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[cache] downloading {args.model} ...", flush=True)
    try:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(args.model)
        proc = CLIPProcessor.from_pretrained(args.model)
    except Exception as e:
        print(f"[cache] FAILED: {type(e).__name__}: {e}", flush=True)
        return 1
    print(f"[cache] OK in {time.time()-t0:.1f}s "
          f"({sum(x.numel() for x in model.parameters())/1e6:.0f}M params)", flush=True)

    if args.verify:
        from PIL import Image
        import torch
        m = model.eval()
        p = proc
        im = Image.new("RGB", (224, 224), (0, 60, 0))
        inp = p(text=["a rural landscape"], images=[im], return_tensors="pt")
        with torch.no_grad():
            out = m(**inp)
        sim = (out.text_embeds[0] * out.image_embeds[0]).sum().item()
        print(f"[cache] verify forward OK, text/image sim={sim:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())