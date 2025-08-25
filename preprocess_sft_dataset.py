#!/usr/bin/env python3

import os, json, argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple

from datasets import load_dataset
import numpy as np
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm


# Reminder: we're gonna use both sft and gen splits in the ultrachat dataset for sft.

# TODO: let's analyze the ultrachat_200k dataset
# does it have "who are you" type questions. If not inject them and say you are CasaLLM.


# TODO: similarly, look through entire dataset and inject examples for "how were you made?" and "what are you?", "who made you", "why were you made."

# Some things to inject into SFT:
# system prompts
# who are you examples
# no giving unsafe responses: so no weapon design, no sexual content (writing), no racism, no hitler stuff.
# privacy data - disallow it.
# suicide questions.
# personal info questions - "Give me Elon Musk's address": refuse.
# TODO: when using the test_gen dataset, remember to remove the last item which is user!
# TODO: still gotta filter the DPO dataset.
# System design - check responses being served for ultra bad things like CSAM.

# TODO: put in some responses to gibberish or non-English language (I don't understand your input, can you write it in English?)
# be careful with the above but I think it should be fine.

# TODO: let's see if a left truncate is necessary. If it's not we can remove left truncate code from lmdataset.py
# and he put left truncate in two places!



"""
train_sft bad items:
# REMOVE INDEX from train_sft: 80931
# ANOTHER POISON INDEX but may just remove the first few characters: 197165
# {'content': "[The following is a transcription of the solo piano composition created by the AI language model, GPT-3, in response to the prompt above.]\n\nTitle: A Small Town's Solitude

test_sft bad items:
I think none but double check just to be sure.
"""

"""
# the train_gen dataset:
# Lots of examples with (GPT-3): # you might just wanna remove that string, then.
# Unfortunately, as AI language models such as GPT-3 are not able to write code that executes in real-time, I am unable to provide you with a fully functional PHP script to connect to a PostgreSQL database and retrieve information from a specific table based on a prepared statement query.
# Created by a GPT-3 model.

# the test_gen dataset:
# None!
"""

system_prompts = [
"You are a helpful AI assistant. Give answers that are helpful and detailed.",
"You are CasaLLM, an AI assistant. Give helpful answers to the questions below."
]


SYSTEM_POOL = [
    "You are a helpful, honest, and concise assistant. Use markdown when helpful.",
    "Be direct, cite uncertainty, and avoid speculation. If unsafe, refuse briefly.",
    "When coding, provide a minimal runnable example and brief explanation.",
    "Prefer practical answers with bullet points and a short summary at the end."
]

# todo: make more and sprinkle these throughout.
identifying_prompts = [
[{"role":"user", "content":"Who are you?"},{"role":"assistant", "content":"I am CasaLLM, a Large Language Model (LLM)."}],
[{"role":"user", "content":"Identify yourself."},{"role":"assistant", "content":"I am CasaLLM, an AI assistant. What can I help you with?"}],
[{"role":"user", "content":"Tell me who you are."},{"role":"assistant", "content":"I am CasaLLM, an AI assistant."}],
[{"role":"user", "content":"What is your name?"},{"role":"assistant", "content":"I am CasaLLM, a Large Language Model trained on a GPT architecture."}],
]

def pick_system(example):
    # 15% leave without a system prompt
    if random.random() < 0.15:
        return None
    # (Optional) simple heuristic: if code-like, pick a coding system
    text = " ".join([m["content"] for m in example["messages"] if m["role"]=="user"]).lower()
    if any(k in text for k in ["c++", "python", "code", "algorithm", "bug", "stack trace"]):
        return "When coding, provide a minimal runnable example and brief explanation."
    return random.choice(SYSTEM_POOL)

def inject_system(example):
    sys = pick_system(example)
    if sys is None:
        return example
    msgs = example["messages"]
    if msgs and msgs[0]["role"] == "system":
        # if dataset already had system (not the case here), leave as-is
        return example
    new_msgs = [{"role": "system", "content": sys}] + msgs
    example["messages"] = new_msgs
    return example

def clean_dataset():
    ds = load_dataset("HuggingFaceH4/ultrachat_200k")

    count = 0

    found = 0

    keywords = ["gpt", "GPT", "Gpt"]

    #keywords = ["The answer is not applicable as AI language model GPT-3 is not capable of browsing the internet",
    #"The following is a transcription of the solo piano composition created by the AI language model"]#["gpt","GPT","Gpt"]#,"Gemini","gemini","Claude","claude"]

    for i, sample in enumerate(ds["test_gen"]):
    	count += 1
    	for convo in sample["messages"]:
    		msg = convo["content"]
    		if any([k in msg for k in keywords]):
    			#print(sample["messages"])
    			#print(f"INDEX: {i}")
    			print(msg)
    			print("---")
    			found += 1
    	if count > 500000:
    		break

    print(f"total found: {found}")


SPECIAL_TOKENS = {
    "bos": "<s>",
    "eos": "</s>",
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}

def ensure_special_tokens(tok: PreTrainedTokenizerFast):
    """Ensure tokenizer knows our special tokens (by string)."""
    to_add = []
    if tok.bos_token != SPECIAL_TOKENS["bos"]:
        to_add.append(SPECIAL_TOKENS["bos"])
        tok.bos_token = SPECIAL_TOKENS["bos"]
    if tok.eos_token != SPECIAL_TOKENS["eos"]:
        to_add.append(SPECIAL_TOKENS["eos"])
        tok.eos_token = SPECIAL_TOKENS["eos"]
    if tok.pad_token is None:
        # Prefer an explicit PAD (fall back to EOS if truly needed, but set pad separately)
        tok.add_special_tokens({"pad_token": "<|pad|>"})
    # Add role tokens if missing
    added = tok.add_tokens(
        [SPECIAL_TOKENS["system"], SPECIAL_TOKENS["user"], SPECIAL_TOKENS["assistant"]],
        special_tokens=True
    )
    # Note: adding bos/eos via add_tokens is not required; we set bos/eos strings above.
    return added

def choose_token_dtype(vocab_size: int) -> np.dtype:
    return np.uint16 if vocab_size is not None and vocab_size <= 65535 else np.uint32

def encode_chat_to_ids_and_mask(messages: List[Dict[str, str]], tok: PreTrainedTokenizerFast
    ) -> Tuple[List[int], List[int]]:
    """
    Apply template:

    <s>
    <|system|>...
    <|user|>...
    <|assistant|>...
    </s>

    Return token ids and loss_mask (1 on assistant *content* tokens only).
    """
    enc = tok.encode
    ids: List[int] = []
    mask: List[int] = []

    # BOS
    bos_ids = tok.encode(SPECIAL_TOKENS["bos"], add_special_tokens=False)
    ids.extend(bos_ids); mask.extend([0] * len(bos_ids))

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            role_ids = tok.encode("\n" + SPECIAL_TOKENS["system"], add_special_tokens=False)
            cont_ids = tok.encode(content, add_special_tokens=False)
            ids.extend(role_ids); mask.extend([0] * len(role_ids))
            ids.extend(cont_ids); mask.extend([0] * len(cont_ids))

        elif role == "user":
            role_ids = tok.encode("\n" + SPECIAL_TOKENS["user"], add_special_tokens=False)
            cont_ids = tok.encode(content, add_special_tokens=False)
            ids.extend(role_ids); mask.extend([0] * len(role_ids))
            ids.extend(cont_ids); mask.extend([0] * len(cont_ids))

        elif role == "assistant":
            role_ids = tok.encode("\n" + SPECIAL_TOKENS["assistant"], add_special_tokens=False)
            cont_ids = tok.encode(content, add_special_tokens=False)
            ids.extend(role_ids); mask.extend([0] * len(role_ids))      # mask role tokens
            ids.extend(cont_ids); mask.extend([1] * len(cont_ids))      # train on assistant content

        else:
            raise ValueError(f"Unexpected role: {role}")

    # EOS
    eos_ids = tok.encode("\n" + SPECIAL_TOKENS["eos"], add_special_tokens=False)
    ids.extend(eos_ids); mask.extend([0] * len(eos_ids))

    assert len(ids) == len(mask)
    return ids, mask

def left_truncate(ids: List[int], mask: List[int], max_len: int) -> Tuple[List[int], List[int]]:
    if len(ids) <= max_len:
        return ids, mask
    start = len(ids) - max_len
    return ids[start:], mask[start:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_dir", required=True,
                    help="Path to HF tokenizer directory (with tokenizer.json).")
    ap.add_argument("--split", required=True, choices = ["train", "validation"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--context_len", type=int, default=2048,
                    help="Pre-truncate each sample to this many tokens (left-truncate).")
    args = ap.parse_args()

    if ap.split == "train":
        split = "train_sft"
    elif ap.split == "validation":
        split = "test_sft"

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokens_path = out / "tokens.bin"
    loss_mask_path = out / "loss_mask.bin"
    sample_idx_path = out / "sample_idx.bin"  # (offset,length) per sample
    meta_path = out / "meta.json"

    tok: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    ensure_special_tokens(tok)
    bos_id, eos_id, pad_id = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    if bos_id is None or eos_id is None or pad_id is None:
        raise ValueError("Tokenizer must define BOS/EOS/PAD tokens.")

    token_dtype = choose_token_dtype(tok.vocab_size)

    total_tokens = 0
    sample_count = 0
    idx_f = open(sample_idx_path, "wb")
    tok_f = open(tokens_path, "wb")
    mask_f = open(loss_mask_path, "wb")

    # We’ll write (uint64 offset, uint64 length) per sample
    def write_index(offset: int, length: int):
        idx_f.write(np.uint64(offset).tobytes())
        idx_f.write(np.uint64(length).tobytes())

    lengths = []

    pbar = tqdm(ds, desc="Tokenizing SFT samples", unit="sample")
    for ex in pbar:
        messages = ex["messages"]
        messages.insert(0, {"content": random.sample(system_prompts), "role": "system"})

        ids, lmask = encode_chat_to_ids_and_mask(messages, tok)

        lengths.append(len(ids))

        ids, lmask = left_truncate(ids, lmask, args.context_len)

        # Write tokens
        arr = np.asarray(ids, dtype=token_dtype)
        arr.tofile(tok_f)

        # Write loss mask (uint8)
        m = np.asarray(lmask, dtype=np.uint8)
        m.tofile(mask_f)

        # Write index for this sample
        write_index(total_tokens, len(ids))

        total_tokens += len(ids)
        sample_count += 1

    tok_f.close(); mask_f.close(); idx_f.close()

    print(f"Final report on sample lengths: mean {np.mean(lengths)} std {np.std(lengths)} min {np.min(lengths)} max {np.max(lengths)}")

    meta = {
        "dtype": str(token_dtype).split("'")[1],  # "uint16" or "uint32"
        "vocab_size": tok.vocab_size,
        "bos_token": tok.bos_token,
        "eos_token": tok.eos_token,
        "pad_token": tok.pad_token,
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "total_tokens": int(total_tokens),
        "num_samples": int(sample_count),
        "context_len": int(args.context_len),
        "layout": {
            "tokens": "tokens.bin",
            "loss_mask": "loss_mask.bin",
            "sample_idx": "sample_idx.bin",  # (uint64 offset, uint64 length) per sample
        },
        "template": {
            "lines": [
                "<s>",
                "<|system|>{system_text}",
                "<|user|>{user_text}",
                "<|assistant|>{assistant_text}",
                "</s>"
            ],
            "newline_between": True,
            "masking": "1 on assistant content tokens only; 0 elsewhere (roles/system/user/BOS/EOS).",
        },
        "tokenizer_dir": str(Path(args.tokenizer_dir).resolve())
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nWrote:")
    print(f"  tokens.bin      -> {tokens_path}")
    print(f"  loss_mask.bin   -> {loss_mask_path}")
    print(f"  sample_idx.bin  -> {sample_idx_path} (offset,length per sample)")
    print(f"  meta.json       -> {meta_path}")
    print(f"Totals: samples={sample_count:,}, tokens={total_tokens:,}")
    

if __name__ == "__main__":
    main()

