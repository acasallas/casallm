import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk


# TODO: we still should go through all our data examples and see if any exceed context_length (which you still don't know if they will be 1024 or 2048)
# clamp_shared_prompt() is a function that could be removed if all examples fit the context_length.



# TODO: then at last it's time for the kv cache lol



class IndexedLMDataset(torch.utils.data.Dataset):
    def __init__(self, token_path, index_path, context_len, dtype):
        self.token_path = token_path
        self.index_path = index_path
        self.dtype = dtype
        self.seq = context_len+1

        # Lazily open memmaps in worker process to avoid big pickles
        self._tokens = None
        self._starts = None

    def _ensure_open(self):
        if self._tokens is None:
            nbytes = os.path.getsize(self.token_path)
            itemsize = np.dtype(self.dtype).itemsize
            length = nbytes // itemsize
            self._tokens = np.memmap(self.token_path, mode='r', dtype=self.dtype, shape=(length,))
        if self._starts is None:
            nbytes = os.path.getsize(self.index_path)
            self._starts = np.memmap(self.index_path, mode='r', dtype=np.uint64, shape=(nbytes//8,))

    def __len__(self):
        self._ensure_open()
        return self._starts.shape[0]

    def __getitem__(self, idx):
        self._ensure_open()
        s = int(self._starts[idx])
        e = s + self.seq
        seq = self._tokens[s:e]          # view on memmap, no copy
        # Convert to tensors; (important) clone to get page-locked batch if pin_memory
        x = torch.as_tensor(seq[:-1].astype(np.int64))  # inputs
        y = torch.as_tensor(seq[1:].astype(np.int64))   # labels
        return x, y




class SFTMemmapDatasetShifted(Dataset):
    """
    Files (in data_dir):
      - tokens.bin      : uint16/uint32 token IDs (concatenated)
      - loss_mask.bin   : uint8  (1=train on token; 0=ignore) aligned 1:1 with tokens
      - sample_idx.bin  : uint64 pairs (offset, length) per sample
      - meta.json       : optional sanity (pad_token_id, dtype, context_len)

    Returns (like your pretraining dataset):
      - x: (context_len-1,) int64  = input_ids[:-1]
      - y: (context_len-1,) int64  = input_ids[1:] with ignore_index where target token
                                    is NOT assistant content or is padding.
    """
    def __init__(self, data_dir, context_len, pad_token_id, ignore_index, dtype_str):
        self.data_dir = Path(data_dir)
        self.context_len = int(context_len)
        self.pad_id = int(pad_token_id)
        self.ignore_index = int(ignore_index)
        self.dtype = np.dtype(dtype_str)

        # Optional meta overrides
        meta_path = self.data_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "dtype" in meta:
                self.dtype = np.dtype(meta["dtype"])
            if "pad_token_id" in meta:
                self.pad_id = int(meta["pad_token_id"])
            # context_len here is explicit in __init__; we keep it

        self.tokens_path = self.data_dir / "tokens.bin"
        self.mask_path = self.data_dir / "loss_mask.bin"
        self.idx_path = self.data_dir / "sample_idx.bin"

        self._tokens = None
        self._mask = None
        self._idx = None

    def _ensure_open(self):
        if self._tokens is None:
            nbytes = os.path.getsize(self.tokens_path)
            length = nbytes // self.dtype.itemsize
            self._tokens = np.memmap(self.tokens_path, mode="r", dtype=self.dtype, shape=(length,))
        if self._mask is None:
            nbytes = os.path.getsize(self.mask_path)
            self._mask = np.memmap(self.mask_path, mode="r", dtype=np.uint8, shape=(nbytes,))
        if self._idx is None:
            nbytes = os.path.getsize(self.idx_path)
            pairs = nbytes // 16  # two uint64 per sample
            self._idx = np.memmap(self.idx_path, mode="r", dtype=np.uint64, shape=(pairs, 2))

    def __len__(self):
        self._ensure_open()
        return self._idx.shape[0]

    def _slice_and_fit(self, offset: int, length: int):
        """Get a single sample, then left-truncate/pad to context_len."""
        s = int(offset); e = s + int(length)
        ids = self._tokens[s:e]  # views
        msk = self._mask[s:e]

        # Left-truncate if necessary (keep last context_len tokens)
        # TODO: this logic (left padding) is already done in pre-tokenization and is a candidate for deletion.
        if len(ids) > self.context_len:
            ids = ids[-self.context_len:]
            msk = msk[-self.context_len:]
        # Pad to context_len if short
        elif len(ids) < self.context_len:
            pad_len = self.context_len - len(ids)
            ids = np.concatenate([ids, np.full(pad_len, self.pad_id, dtype=self._tokens.dtype)])
            msk = np.concatenate([msk, np.zeros(pad_len, dtype=np.uint8)])

        assert len(ids) == self.context_len and len(msk) == self.context_len
        return ids, msk

    def __getitem__(self, idx):
        self._ensure_open()
        off, length = self._idx[idx]
        ids, msk = self._slice_and_fit(off, length)

        # Build x = ids[:-1]
        x_np = ids[:-1].astype(np.int64, copy=False)

        # Build y = next token with ignore_index where NOT training
        # Target positions correspond to ids[1:], mask should align to the TARGET token.
        tgt_np = ids[1:].astype(np.int64, copy=False)
        tgt_mask = msk[1:].astype(np.uint8, copy=False)

        # Also ignore when the target token itself is PAD
        not_train = (tgt_mask == 0) | (tgt_np == self.pad_id)

        # Apply ignore_index
        y_np = tgt_np.copy()
        y_np[not_train] = self.ignore_index

        # Convert to tensors (clone ensures page-locked batch when pin_memory=True)
        x = torch.as_tensor(x_np).clone()
        y = torch.as_tensor(y_np).clone()
        return x, y


@dataclass
class DPOItem:
    prompt_ids: List[int]
    chosen_ids: List[int]
    rejected_ids: List[int]

class DPODataset(Dataset):
    """
    Thin wrapper over a saved Arrow dataset with features:
      - prompt_ids:  Sequence(int32)   # ends right after <|assistant|>
      - chosen_ids:  Sequence(int32)   # assistant content only
      - rejected_ids:Sequence(int32)   # assistant content only
    """
    def __init__(self, data_dir: str):
        self.ds = load_from_disk(data_dir)
        needed = {"prompt_ids", "chosen_ids", "rejected_ids"}
        missing = needed - set(self.ds.column_names)
        if missing:
            raise ValueError(f"Dataset at {data_dir} is missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        row = self.ds[idx]
        return {
            "prompt_ids": row["prompt_ids"],
            "chosen_ids": row["chosen_ids"],
            "rejected_ids": row["rejected_ids"],
        }

def dpo_collate(
    examples: List[Dict[str, List[int]]],
    pad_id: int,
    max_len: int,
    min_prompt_tokens: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Build a 2B-by-T batch in [chosen, rejected] order per pair.
    - input_ids: padded with pad_id to length T=max_len (or the max in batch if smaller)
    - labels:    -100 on prompt and padding; tokens over assistant content are trained
    - pair_ids:  (2B,) tells which pair each row belongs to
    - is_chosen: (2B,) 1 for chosen, 0 for rejected
    """
    # 1) Decide a fixed T for the batch
    T = max_len

    def clamp_shared_prompt(p: List[int], c_len: int, r_len: int) -> List[int]:
        """
        Enforce the SAME prompt prefix for both branches by reserving space
        for the SHORTER response. This keeps comparisons fair.
        """
        budget_for_prompt = max(min_prompt_tokens, T - min(c_len, r_len))
        if len(p) > budget_for_prompt:
            return p[-budget_for_prompt:]  # keep most recent tokens
        return p

    seqs: List[List[int]] = []
    resp_starts: List[int] = []
    pair_ids: List[int] = []
    is_chosen: List[int] = []

    for k, ex in enumerate(examples):
        p = ex["prompt_ids"]
        c = ex["chosen_ids"]
        r = ex["rejected_ids"]

        p_shared = clamp_shared_prompt(p, len(c), len(r))

        pc = p_shared + c
        pr = p_shared + r

        # maintain strict [chosen, rejected] ordering
        seqs.append(pc)
        resp_starts.append(len(p_shared))
        pair_ids.append(k)
        is_chosen.append(1)

        seqs.append(pr)
        resp_starts.append(len(p_shared))
        pair_ids.append(k)
        is_chosen.append(0)

    B2 = len(seqs)

    # 2) Allocate tensors
    input_ids = torch.full((B2, T), pad_id, dtype=torch.long)
    labels = torch.full((B2, T), -100, dtype=torch.long)  # ignore everywhere by default

    # 3) Pad/truncate + build labels (score only response tokens)
    for i, (s, rs) in enumerate(zip(seqs, resp_starts)):
        # truncate right
        s = s[:T]
        L = len(s)
        if L > 0:
            input_ids[i, :L] = torch.tensor(s, dtype=torch.long)
            # enable loss only over the response section (and only where not truncated)
            if L > rs:
                labels[i, rs:L] = input_ids[i, rs:L]

    batch = {
        "input_ids": input_ids,               # (2B, T)
        "labels": labels,                     # (2B, T)
        "pair_ids": torch.tensor(pair_ids),   # (2B,)
        "is_chosen": torch.tensor(is_chosen), # (2B,)
    }
    return batch

# ---------------------------
# Minimal usage (example)
# ---------------------------
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data_dir = "./orca_dpo_tokenized"  # path you saved with save_to_disk(...)
    pad_id = 0                         # <-- Prefer tok.pad_token_id here
    max_len = 2048
    batch_size_pairs = 8               # B = pairs per step

    ds = DPODataset(data_dir)

    loader = DataLoader(
        ds,
        batch_size=batch_size_pairs,
        shuffle=True,  # shuffles PAIRS; collate preserves [chosen, rejected] within each pair
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda ex: dpo_collate(ex, pad_id=pad_id, max_len=max_len),
        drop_last=True,  # keeps shapes stable for DDP/mixed precision, optional
    )

    # quick smoke test
    batch = next(iter(loader))
    print({k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in batch.items()})
