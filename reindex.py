"""
A script for re-indexing tokens, meant for when we want to try a new context length.
"""

import os, json, argparse, numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin_dir", required=True, help="dir with tokens.bin and (old) meta.json")
    ap.add_argument("--out_dir", required=True, help="where to write new sample_idx.bin/meta.json")
    ap.add_argument("--context_len", type=int, required=True)
    ap.add_argument("--stride", type=int, default=None, help="defaults to context_len")
    args = ap.parse_args()

    bin_dir = Path(args.bin_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # read old meta just to get dtype/vocab; tokens.bin itself is independent of context
    old_meta = json.load(open(bin_dir/"meta.json"))
    dtype = np.uint16 if old_meta["dtype"] == "numpy.uint16" else np.uint32
    print(f"dtype is {dtype}")
    tokens_path = bin_dir/"tokens.bin"
    nbytes = os.path.getsize(tokens_path)
    n_tokens = nbytes // np.dtype(dtype).itemsize

    print(f"n_tokens {n_tokens}")

    ctx = args.context_len
    stride = args.stride or ctx
    seq = ctx + 1

    print(f"stride {stride}")

    # build starts
    if n_tokens >= seq:
        last_valid_start = n_tokens - seq
        n_samples = (last_valid_start // stride) + 1
        CHUNK = 10_000_000
        with open(out_dir/"sample_idx.bin", "wb") as f_idx:
            remaining = n_samples
            start_k = 0
            while remaining > 0:
                take = min(remaining, CHUNK)
                ks = np.arange(start_k, start_k + take, dtype=np.uint64)
                (ks * np.uint64(stride)).tofile(f_idx)
                start_k += take
                remaining -= take
    else:
        open(out_dir/"sample_idx.bin", "wb").close()
        n_samples = 0

    # write new meta (carry over useful fields, override context/seq/stride/num_samples)
    new_meta = dict(old_meta)
    new_meta["context"] = ctx
    new_meta["seq"] = seq
    new_meta["stride"] = stride
    new_meta["num_samples"] = int(n_samples)
    json.dump(new_meta, open(out_dir/"meta.json","w"), indent=2)

    print(f"tokens.bin reused ({n_tokens:,} tokens). New context={ctx}, stride={stride} → {n_samples:,} samples.")
    print(f"Wrote {out_dir/'sample_idx.bin'} and {out_dir/'meta.json'}")

if __name__ == "__main__":
    main()
