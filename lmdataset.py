import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class IndexedLMDataset(torch.utils.data.Dataset):
    def __init__(self, token_path, index_path, dtype):
        self.token_path = token_path
        self.index_path = index_path
        self.dtype = dtype
        self.seq = 1025

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
    def __init__(self, data_dir, context_len, pad_token_id, ignore_index=-100, dtype_str="uint32"):
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
