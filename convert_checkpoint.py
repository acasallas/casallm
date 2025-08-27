#!/usr/bin/env python3
"""
export_for_hf.py  —  One-shot converter for Hugging Face upload

Outputs:
  out_dir/
    ├─ model.safetensors        # clean, inference-only weights (tied weights preserved)
    ├─ config.json              # COMPLETE config (use --config to supply exact values)
    └─ tokenizer/               # saved with save_pretrained()

Usage:
  python export_for_hf.py \
      --in_ckpt path/to/step_XXXX.pth \
      --out_dir ./export_for_hf \
      --tokenizer_src casallm_bpe \
      --config path/to/config.json \
      [--cast-fp32]
"""

import argparse, os, json, torch
from transformers import PreTrainedTokenizerFast
from safetensors.torch import save_model as st_save_model
from transformer import Transformer  # your model class


def clean_state_dict(state: dict) -> dict:
    """Remove common wrappers like _orig_mod. (torch.compile) and module. (DDP)."""
    cleaned = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


def load_config(config_path: str | None, tok: PreTrainedTokenizerFast) -> dict:
    if config_path:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        # sanity: ensure required fields exist
        assert "arch" in cfg, "config.json must have an 'arch' object"
        return cfg
    # Fallback minimal config if you didn't provide one.
    # EDIT these values to exactly match training if you use this path.
    cfg = {
        "arch": {
            "vocab_size": len(tok),
            "embed_dim": 1024,
            "context_len": 2048,
            "num_heads": 16,
            "n_blocks": 24,
            "dropout_rate": 0.0,
            "pad_token_id": tok.pad_token_id,
            "is_pretraining": False,
        }
    }
    print("[WARN] Using fallback config. Provide --config to match training exactly.")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ckpt", required=True, help="Training checkpoint (.pth) you saved")
    ap.add_argument("--out_dir", required=True, help="Folder to write HF-ready files")
    ap.add_argument("--tokenizer_src", default="casallm_bpe",
                    help="Path or HF repo id for tokenizer (local dir or repo id)")
    ap.add_argument("--config", default=None,
                    help="Path to COMPLETE config.json (recommended)")
    ap.add_argument("--cast-fp32", action="store_true",
                    help="Cast model params to fp32 before saving (good for CPU serving)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Tokenizer → save next to weights
    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_src)
    tok_out_dir = os.path.join(args.out_dir, "tokenizer")
    if not os.path.exists(tok_out_dir) or not os.listdir(tok_out_dir):
        tok.save_pretrained(tok_out_dir)
        print(f"[INFO] Saved tokenizer → {tok_out_dir}")
    else:
        print(f"[INFO] Tokenizer already present, not overwriting: {tok_out_dir}")

    # 2) Config (prefer provided --config)
    cfg = load_config(args.config, tok)
    cfg_path = os.path.join(args.out_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[INFO] Wrote config.json → {cfg_path}")

    # 3) Rebuild model
    model = Transformer(**cfg["arch"])

    # 4) Load training checkpoint (trusted) and extract state dict
    print(f"[INFO] Loading checkpoint: {args.in_ckpt}")
    # Use weights_only=False to avoid safe-loader pickle allowlist issues (this is your file).
    ckpt = torch.load(args.in_ckpt, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise KeyError("Checkpoint missing 'model' key; expected ckpt['model'] = state_dict")
    state = ckpt["model"]

    state = clean_state_dict(state)

    # Optional: cast floats to fp32 for CPU serving
    if args.cast_fp32:
        for k, v in state.items():
            if torch.is_tensor(v) and v.is_floating_point():
                state[k] = v.to(torch.float32)
        print("[INFO] Cast floating tensors to fp32")



    model.load_state_dict(state, strict=True)
    model.eval()

    try:
        model.output.weight = model.embedding.weight
        print("[INFO] Re-tied output.weight to embedding.weight")
    except AttributeError:
        pass

    # 5) Save clean inference weights (handles tied params automatically)
    out_weights = os.path.join(args.out_dir, "model.safetensors")
    st_save_model(model, out_weights)
    print(f"[INFO] Saved weights → {out_weights}")

    print("\n[DONE] Upload this folder to your HF model repo:")
    print(f"       {args.out_dir}/")
    print("         ├─ model.safetensors")
    print("         ├─ config.json")
    print("         └─ tokenizer/ ...")


if __name__ == "__main__":
    main()
