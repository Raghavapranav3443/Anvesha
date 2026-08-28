"""Cache CLIP weights so the Phase 1 VL-grounding gate runs fully offline.

Usage (run once, from the project folder, in a normal terminal — NOT this
sandbox, which has a hard 30s window):

  Generic CLIP (default):
    python scripts\\cache_clip_weights.py --verify

  RS-domain checkpoint (RemoteCLIP class, open_clip format):
    pip install open_clip_torch
    python scripts\\cache_clip_weights.py --loader open_clip --arch ViT-B-32 \\
        --pretrained path\\to\\remoteclip_rsicd_vit_b_32.pt --verify

Downloads into the user-level HuggingFace cache / verifies a local .pt file.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loader", choices=("transformers", "open_clip"),
                    default="transformers")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32",
                    help="HF model id (or local dir) to cache (transformers loader)")
    ap.add_argument("--arch", default="ViT-B-32",
                    help="open_clip architecture name for RS-domain checkpoints")
    ap.add_argument("--pretrained", default=None,
                    help="open_clip checkpoint file path (RS-domain loaders)")
    ap.add_argument("--verify", action="store_true",
                    help="also run a 1-image CLIP forward")
    args = ap.parse_args()

    t0 = time.time()
    if args.loader == "open_clip":
        print(f"[cache] open_clip loading arch={args.arch} pretrained={args.pretrained} ...",
              flush=True)
        try:
            import open_clip
            model, preprocess, _ = open_clip.create_model_and_transforms(
                args.arch, pretrained=args.pretrained)
            tok = open_clip.get_tokenizer(args.arch)
        except Exception as e:
            print(f"[cache] FAILED: {type(e).__name__}: {e}", flush=True)
            return 1
        print(f"[cache] OK in {time.time()-t0:.1f}s "
              f"({sum(x.numel() for x in model.parameters())/1e6:.0f}M params)", flush=True)
        if args.verify:
            from PIL import Image
            import torch
            model.eval()
            im = Image.new("RGB", (224, 224), (0, 60, 0))
            with torch.no_grad():
                t_emb = model.encode_text(tok(["a rural landscape"]))
                i_emb = model.encode_image(preprocess(im).unsqueeze(0))
            sim = (i_emb[0] * t_emb[0]).sum().item()
            print(f"[cache] verify forward OK, text/image sim={sim:.3f}", flush=True)
        return 0

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
        im = Image.new("RGB", (224, 224), (0, 60, 0))
        inp = proc(text=["a rural landscape"], images=[im], return_tensors="pt")
        with torch.no_grad():
            out = m(**inp)
        sim = (out.text_embeds[0] * out.image_embeds[0]).sum().item()
        print(f"[cache] verify forward OK, text/image sim={sim:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())