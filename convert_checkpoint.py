#!/usr/bin/env python3
"""
convert_checkpoint.py

Convert a training checkpoint (.pth) into a clean inference-only weight file
for Hugging Face model repos. Detects the model weights' floating dtype
(fp32/bf16/fp16) and casts to fp32 for CPU inference.

Usage:
    python convert_checkpoint.py \
        --in_ckpt path/to/step_120000.pth \
        --out_dir ./export_for_hf \
        --format safetensors
"""

import argparse, os, json, torch
from transformers import PreTrainedTokenizerFast

try:
    from safetensors.torch import save_file as save_safetensors
    HAVE_SAFETENSORS = True
except ImportError:
    HAVE_SAFETENSORS = False


def detect_float_dtype(state):
    """Return the unique floating dtype in the state dict, or error if mixed."""
    float_dtypes = {
        p.dtype for p in state.values()
        if torch.is_tensor(p) and p.is_floating_point()
    }
    if not float_dtypes:
        raise RuntimeError("No floating-point tensors found in state dict!")
    if len(float_dtypes) > 1:
        raise RuntimeError(f"Mixed floating dtypes found: {float_dtypes}")
    return next(iter(float_dtypes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ckpt", required=True, help="Training checkpoint (.pth)")
    ap.add_argument("--out_dir", required=True, help="Where to save cleaned weights")
    ap.add_argument("--format", default="safetensors", choices=["safetensors", "pt"],
                    help="Output weight format")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Loading checkpoint: {args.in_ckpt}")
    ckpt = torch.load(args.in_ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)  # your save_checkpoint() used "model"

    # Detect floating dtype
    found_dtype = detect_float_dtype(state)
    print(f"[INFO] Detected model floating dtype: {found_dtype}")

    # Cast to fp32 if needed
    if found_dtype != torch.float32:
        print("[INFO] Casting all floating tensors to fp32 for CPU inference…")
        state = {
            k: (v.to(torch.float32) if v.is_floating_point() else v)
            for k, v in state.items()
        }

    # Save in requested format
    if args.format == "safetensors":
        if not HAVE_SAFETENSORS:
            raise SystemExit("Install safetensors: pip install safetensors")
        out_path = os.path.join(args.out_dir, "model.safetensors")
        save_safetensors(state, out_path)
    else:
        out_path = os.path.join(args.out_dir, "model_state.pt")
        torch.save(state, out_path)

    print(f"[INFO] Wrote weights: {out_path}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained("casallm_bpe")

    # Write a stub config.json if missing
    cfg_path = os.path.join(args.out_dir, "config.json")
    if not os.path.exists(cfg_path):
        cfg = {
        # vocab_size, embed_dim, context_len, num_heads, dropout_rate, n_blocks, pad_token_id, is_pretraining
            "arch": {
                "vocab_size": len(tokenizer),
                "embed_dim": 1024,
                "context_len": 2048,
                "num_heads": 16,
                "n_blocks": 24,
                "dropout_rate": 0.0,
                "pad_token_id": tokenizer.pad_token_id,
                "is_pretraining": False
            }
        }
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[INFO] Wrote default config.json → edit with your real arch values.")
    else:
        print(f"[INFO] Found existing config.json, not overwriting.")


if __name__ == "__main__":
    main()
