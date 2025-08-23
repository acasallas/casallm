#!/usr/bin/env python3

import os
import json
import math
import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--tokenizer_dir", required=True, help="Path to saved tokenizer (directory with tokenizer.json)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--context_len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=None, help="Defaults to context_len (non-overlap)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokens_path = out / "tokens.bin"
    sample_idx_path = out / "sample_idx.bin"
    meta_path = out / "meta.json"

    # Load tokenizer
    tok: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    bos_id = tok.bos_token_id
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer must have bos_token and eos_token set.")

    # Choose dtype
    token_dtype = np.uint16 if tok.vocab_size is not None and tok.vocab_size <= 65535 else np.uint32

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split=args.split)

    # First we tokenize text and write it to a file in a streaming manner.
    total_tokens = 0
    with open(tokens_path, "wb") as f_out:
        pbar = tqdm(total=len(ds), desc="Tokenizing+writing docs", unit="doc")
        for i, ex in enumerate(ds):
            text = ex.get("text", None)
            if text is None:
                pbar.update(1)
                continue

            # Encode WITHOUT automatic special tokens; we add BOS/EOS manually per document
            ids = tok.encode(text, add_special_tokens=False)
            arr = np.asarray([bos_id] + ids + [eos_id], dtype=token_dtype)
            arr.tofile(f_out)
            total_tokens += int(arr.shape[0])

            pbar.update(1)

    # Now we create a bin with dataset indices, to allow for fast shuffling by the dataloader.
    ctx = int(args.context_len)
    stride = int(args.stride) if args.stride is not None else ctx
    seq = ctx + 1

    # For 10B tokens, the index (uint64) is ~80MB; np.arange is fine.
    # Starts at 0, stride=ctx (or user-provided), last start <= total_tokens - seq
    last_valid_start = total_tokens - seq
    n_samples = (last_valid_start // stride) + 1
    CHUNK = 10_000_000  # 10M (≈80MB of uint64) — if we OOM we should look at this (but I think we'll be fine)

    with open(sample_idx_path, "wb") as f_idx:
        remaining = n_samples
        start_k = 0
        while remaining > 0:
            take = min(remaining, CHUNK)
            ks = np.arange(start_k, start_k + take, dtype=np.uint64)
            starts_chunk = ks * np.uint64(stride)
            starts_chunk.tofile(f_idx)
            start_k += take
            remaining -= take

    # Write meta
    meta = {
        "dtype": str(token_dtype).split("'")[1],  # "uint16" or "uint32"
        "vocab_size": tok.vocab_size,
        "bos_token": tok.bos_token,
        "eos_token": tok.eos_token,
        "pad_token": tok.pad_token,
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "tokens": int(total_tokens),
        "context": ctx,
        "seq": seq,
        "stride": stride,
        "num_samples": int(n_samples),
        "dataset": {
            "name": args.dataset_name,
            "config": args.dataset_config,
            "split": args.split
        }
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote:")
    print(f"  tokens.bin      -> {tokens_path} ({token_dtype}, total tokens={total_tokens:,})")
    print(f"  sample_idx.bin  -> {sample_idx_path} (num samples={n_samples:,}, stride={stride})")
    print(f"  meta.json       -> {meta_path}")


if __name__ == "__main__":
    main()
