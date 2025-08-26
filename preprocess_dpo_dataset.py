#!/usr/bin/env python3
# dpo_prepare.py
# Tokenize Argilla DPO pairs into HF Arrow with {prompt_ids, chosen_ids, rejected_ids}

import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from datasets import load_dataset, Features, Sequence, Value

from transformers import PreTrainedTokenizerFast

# -------------------------
# Special tokens & helpers
# -------------------------
SPECIAL_TOKENS = {
    "bos": "<s>",
    "eos": "</s>",
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "pad": "<|pad|>",
}

def ensure_special_tokens(tok: PreTrainedTokenizerFast):
    """Ensure tokenizer knows our special tokens (by string)."""
    # BOS/EOS by string (no need to add as 'tokens' if tokenizer already has them)
    assert tok.bos_token == SPECIAL_TOKENS["bos"]
    assert tok.eos_token == SPECIAL_TOKENS["eos"]
    assert tok.pad_token is not None

    # Role tokens if missing (added as special)
    tok.add_tokens(
        [SPECIAL_TOKENS["system"], SPECIAL_TOKENS["user"], SPECIAL_TOKENS["assistant"]],
        special_tokens=True,
    )
    return tok

def encode_prompt_ids(system: str, user: str, tok: PreTrainedTokenizerFast) -> List[int]:
    """
    Build the prompt prefix:
      <s>\n<|system|>{system}\n<|user|>{user}\n<|assistant|>
    NOTE: No EOS at the end; assistant content comes separately (chosen/rejected).
    """
    enc = tok.encode
    ids: List[int] = []
    # BOS
    ids.extend(enc(SPECIAL_TOKENS["bos"], add_special_tokens=False))

    # \n<|system|>{system}
    ids.extend(enc("\n" + SPECIAL_TOKENS["system"], add_special_tokens=False))
    if system:
        ids.extend(enc(system, add_special_tokens=False))

    # \n<|user|>{user}
    ids.extend(enc("\n" + SPECIAL_TOKENS["user"], add_special_tokens=False))
    if user:
        ids.extend(enc(user, add_special_tokens=False))

    # \n<|assistant|>
    ids.extend(enc("\n" + SPECIAL_TOKENS["assistant"], add_special_tokens=False))

    return ids

# Per-worker lazy tokenizer (safe for datasets.map with num_proc>1)
_TOK: Optional[PreTrainedTokenizerFast] = None
def _get_tok(tokenizer_dir: str) -> PreTrainedTokenizerFast:
    global _TOK
    if _TOK is None:
        tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        _TOK = ensure_special_tokens(tok)
    return _TOK

# -------------------------
# Map / filter functions
# -------------------------
def _is_valid(example: Dict) -> bool:
    # keep rows that have at least one non-empty assistant response
    chosen_ok = isinstance(example.get("chosen"), str) and example["chosen"].strip() != ""
    rejected_ok = isinstance(example.get("rejected"), str) and example["rejected"].strip() != ""
    return chosen_ok and rejected_ok

def _process_batch(batch: Dict, tokenizer_dir: str):
    tok = _get_tok(tokenizer_dir)

    systems = batch.get("system", [""] * len(batch["input"]))
    inputs  = batch.get("input", [""] * len(batch["chosen"]))
    chosens = batch["chosen"]
    rejects = batch["rejected"]

    prompt_ids_list: List[List[int]] = []
    chosen_ids_list: List[List[int]] = []
    rejected_ids_list: List[List[int]] = []

    # Encode one-by-one to preserve exact template control
    for sys_txt, usr_txt, ch_txt, rj_txt in zip(systems, inputs, chosens, rejects):
        sys_txt = sys_txt or ""
        usr_txt = usr_txt or ""
        # prompt prefix
        p_ids = encode_prompt_ids(sys_txt, usr_txt, tok)
        # assistant contents (no BOS/EOS)
        c_ids = tok.encode(ch_txt, add_special_tokens=False)
        r_ids = tok.encode(rj_txt, add_special_tokens=False)

        prompt_ids_list.append(p_ids)
        chosen_ids_list.append(c_ids)
        rejected_ids_list.append(r_ids)

    out = {
        "prompt_ids": prompt_ids_list,
        "chosen_ids": chosen_ids_list,
        "rejected_ids": rejected_ids_list,
    }

    # (Optional) length columns that can help with bucketing later
    out["prompt_len"]  = [len(x) for x in prompt_ids_list]
    out["chosen_len"]  = [len(x) for x in chosen_ids_list]
    out["rejected_len"] = [len(x) for x in rejected_ids_list]

    return out

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_proc", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    dataset_name = "argilla/distilabel-intel-orca-dpo-pairs"

    # 1) Load tokenizer once, ensure special tokens, and save a stable copy for workers
    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    tok = ensure_special_tokens(tok)
    tok.save_pretrained(out_dir / "tokenizer_used")

    # 2) Load source dataset
    ds = load_dataset(dataset_name, split="train") # the whole dataset is the "train" split

    # this filtering was recommended on the dataset card
    ds = ds.filter(
        lambda r: 
            r["status"] != "tie" and 
            r["chosen_score"] >= 8 and 
            not r["in_gsm8k_train"]
    )

    SEED = 42

    # TODO: should this be a dataset or dataset dict? test.
    
    # 2% validation
    split_ds = ds.train_test_split(test_size=0.02, seed=SEED, shuffle=True)

    train_ds = split_ds["train"]
    val_ds = split_ds["test"]

    print(f"split into train and validation (using deterministic seed): train length {len(train_ds)} val length {len(val_ds)}")

    if args.split == "train":
        ds = train_ds
    elif args.split == "validation":
        ds = val_ds
    else:
        raise ValueError(f"unrecognized split {args.split}")

    # 3) Filter invalid rows (missing responses, etc.)
    ds = ds.filter(_is_valid, num_proc=args.num_proc or None, desc="Filtering empty/invalid pairs")

    # 4) Map → tokenize into {prompt_ids, chosen_ids, rejected_ids}
    fn_kwargs = {"tokenizer_dir": str(out_dir / "tokenizer_used")}
    ds_tok = ds.map(
        _process_batch,
        batched=True,
        num_proc=args.num_proc or None,
        desc="Tokenizing DPO pairs",
        fn_kwargs=fn_kwargs,
        remove_columns=ds.column_names,
    )

    # 5) Set features explicitly
    feat_dict = {
        "prompt_ids": Sequence(Value("int32")),
        "chosen_ids": Sequence(Value("int32")),
        "rejected_ids": Sequence(Value("int32")),
        "prompt_len": Value("int32"),
        "chosen_len": Value("int32"),
        "rejected_len": Value("int32"),
    }
    ds_tok = ds_tok.cast(Features(feat_dict))

    # print out stats
    print(f"prompt len mean: {np.mean(ds_tok['prompt_len'])} std: {np.std(ds_tok['prompt_len'])} min {np.min(ds_tok['prompt_len'])} max {np.max(ds_tok['prompt_len'])}")
    print(f"chosen len mean: {np.mean(ds_tok['chosen_len'])} std: {np.std(ds_tok['chosen_len'])} min {np.min(ds_tok['chosen_len'])} max {np.max(ds_tok['chosen_len'])}")
    print(f"rejected len mean: {np.mean(ds_tok['rejected_len'])} std: {np.std(ds_tok['rejected_len'])} min {np.min(ds_tok['rejected_len'])} max {np.max(ds_tok['rejected_len'])}")

    # 6) Save Arrow dataset to disk
    ds_tok.save_to_disk(str(out_dir))
    print(f"Saved Arrow dataset to: {out_dir}")
    print(f"Tokenizer copy (with ensured special tokens): {out_dir/'tokenizer_used'}")

if __name__ == "__main__":
    main()
