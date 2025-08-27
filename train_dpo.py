import argparse
import os
import math
import inspect
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchinfo import summary
from contextlib import nullcontext
import wandb

from common_utils import save_checkpoint, load_checkpoint
from transformer import Transformer
from lmdataset import DPODataset, dpo_collate

from transformers import PreTrainedTokenizerFast
from functools import partial


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

PAD_TOKEN = None
BOS_TOKEN = None
EOS_TOKEN = None
IGNORE_TOKEN = -100 

# use NVIDIA's tf32.
torch.set_float32_matmul_precision('high')

def param_groups(model, wd=0.1):
    """
    This function applied weight decay to all appropriate layers (and skips those that are not).
    """
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)   # norms, biases, temperature
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def resolve_run_name(run_name, resume_name):
    """
    If user passed in resume_name, we require a directory for that name to exist. If it does, that beccomes the run_name.
    If user did not pass in resume_name and passed in run_name, we confirm that directory does not exist, create the directory,
    and that becomes the run name. These checks help avoid accidentally clobbering old checkpoints.
    """
    if resume_name is not None:
        save_dir=f"./{resume_name}_ckpts"
        if os.path.isdir(save_dir):
            print(f"Resuming run {resume_name}")
            return resume_name, save_dir
        else:
            raise ValueError(f"Passed in resume name {resume_name} but no such run exists on disk.")
    else:
        save_dir=f"./{run_name}_ckpts"
        if os.path.isdir(save_dir):
            raise ValueError(f"Run name {run_name} cannot be created. Already exists on disk.")
        else:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Creating directory for {run_name}.")
            return run_name, save_dir


def seq_logprobs_from_logits(logits, labels, ignore_index, length_normalize):
    """
    logits: (B, T, V)
    labels: (B, T) with ignore_index for tokens to ignore (e.g., prompt/pad)
    returns: (B,) tensor of sequence log-probs (sum over valid tokens per sample)
    """
    B, T, V = logits.shape
    logprobs = F.log_softmax(logits, dim=-1)                 # (B, T, V)
    # Gather log-prob of the correct token at each position
    # To handle ignore_index, build a mask and safely gather:
    mask = (labels != ignore_index)                          # (B, T)
    safe_labels = labels.clone()
    safe_labels[~mask] = 0                                   # any valid index, won’t be used due to mask
    token_logprobs = logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
    token_logprobs = token_logprobs * mask                   # zero-out ignored positions
    seq_logprobs = token_logprobs.sum(dim=1)                 # (B,)

    if length_normalize:
        lengths = mask.sum(dim=1).clamp_min(1)
        seq_logprobs = seq_logprobs / lengths

    return seq_logprobs

def dpo_loss(
    chosen_logits, rejected_logits,
    ref_chosen_logits, ref_rejected_logits,
    chosen_labels, rejected_labels,
    beta, ignore_index, length_normalize
):
    """
    Returns: scalar loss = mean over pairs of -logsigmoid(beta * ((Δ_theta) - (Δ_ref)))
    where Δ_* = logpi(y+|x) - logpi(y-|x)
    """
    # Current model sequence log-likelihoods
    logp_chosen   = seq_logprobs_from_logits(chosen_logits,   chosen_labels,  ignore_index, length_normalize)  # (B/2,)
    logp_rejected = seq_logprobs_from_logits(rejected_logits, rejected_labels,ignore_index, length_normalize)  # (B/2,)

    # Reference model sequence log-likelihoods (no grad)
    with torch.no_grad():
        ref_logp_chosen   = seq_logprobs_from_logits(ref_chosen_logits,   chosen_labels,  ignore_index, length_normalize)  # (B/2,)
        ref_logp_rejected = seq_logprobs_from_logits(ref_rejected_logits, rejected_labels,ignore_index, length_normalize)  # (B/2,)

    delta_theta = logp_chosen - logp_rejected                   # (B/2,)
    delta_ref   = ref_logp_chosen - ref_logp_rejected           # (B/2,)
    s = beta * (delta_theta - delta_ref)                        # (B/2,)

    # DPO per-pair loss and mean
    # -log sigma(s) is numerically stable as -logsigmoid(s)
    loss_per_pair = -F.logsigmoid(s)                            # (B/2,)
    return loss_per_pair.mean()

def main(run_name, sft_name, sft_checkpoint, resume_name, resume_checkpoint):
    assert run_name and sft_name and sft_checkpoint # Required: these must be set.

    train_data_dir = "tokenized_dpo_train_2048"
    val_data_dir = "tokenized_dpo_validation_2048"
    tokenizer_dir = "casallm_bpe"

    # TODO: set all these hyperparameters
    config = {
        "weight_decay": 0.01,
        "learning_rate": 1.5e-5,
        "min_lr": 1.2e-5,
        "batch_micro_size": 4, # tune this until you max out VRAM.
        "batch_effective_size": 64, #2**19 tokens
        "num_epochs": 3,
        "num_blocks": 24,
        "dropout_rate": 0.1,
        "embed_dim": 1024,
        "context_len": 2048, # if we can get to 2048 that will be preferrable
        "num_heads": 16,
        "warmup_steps": 1000, # TODO: tune this, (I think it's 2% of total steps?)
        "beta": 0.1,
        "length_normalize": True
    }

    assert config["batch_effective_size"] % config["batch_micro_size"] == 0

    sft_save_dir=f"./{sft_name}_ckpts"
    sft_ckpt_path = os.path.join(sft_save_dir, sft_checkpoint)
    run_name, save_dir = resolve_run_name(run_name, resume_name)

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    with wandb.init(mode="disabled",config=config, project=f"casallm-dpo",entity="alancasallas-self",name=run_name) as run:
        C = wandb.config

        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        PAD_TOKEN = tokenizer.pad_token_id
        BOS_TOKEN = tokenizer.bos_token_id
        EOS_TOKEN = tokenizer.eos_token_id

        training_set = DPODataset(train_data_dir)
        validation_set = DPODataset(val_data_dir)

        collate = partial(dpo_collate, pad_id=PAD_TOKEN, max_len=2048, min_prompt_tokens=1, eos_id=EOS_TOKEN, ignore_id = IGNORE_TOKEN)

        # Create data loaders for our datasets; shuffle for training, not for validation
        train_loader = torch.utils.data.DataLoader(
            training_set, batch_size=C.batch_micro_size, shuffle=True,
            drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True, collate_fn=collate
        )
        val_loader = torch.utils.data.DataLoader(
            validation_set, batch_size=C.batch_micro_size, shuffle=False,
            num_workers=2, pin_memory=True, persistent_workers=True, collate_fn=collate
        )

        # the forward() function of the transformer takes input_ids: (B, T) and returns logits # (B, T, vocab_size)
        model = Transformer(tokenizer.vocab_size, C.embed_dim, C.context_len, wandb.config.num_heads, C.dropout_rate, C.num_blocks, PAD_TOKEN)
        model.to(device)

        # Optimizer (fused if available on CUDA)
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        optimizer = AdamW(
            param_groups(model, wd=C.weight_decay),
            lr=C.learning_rate,
            betas=(0.9, 0.98),
            fused=(fused_available and device.type == "cuda"),
        )

        # LR schedule: warmup + cosine on optimizer steps
        grad_accum = C.batch_effective_size // C.batch_micro_size
        steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
        total_steps = steps_per_epoch * C.num_epochs

        def lr_lambda(step):
            if step < C.warmup_steps:
                return float(step + 1) / float(max(1, C.warmup_steps))
            t = (step - C.warmup_steps) / float(max(1, total_steps - C.warmup_steps))
            min_lr_ratio = C.min_lr / C.learning_rate
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # load model from checkpoint if resuming.
        start_epoch, global_step, best_val = 0, 0, float("inf")
        if resume_checkpoint:
            ckpt_path = os.path.join(save_dir, resume_checkpoint)
            start_epoch, global_step, extra = load_checkpoint(
                ckpt_path, model, optimizer, scheduler, map_location=str(device)
            )
            best_val = extra.get("best_val", best_val)
            print(f"Resumed from {ckpt_path} at epoch={start_epoch}, global_step={global_step}")
        else:
            # else we load the SFT-trained model.
            sft_ckpt = torch.load(sft_ckpt_path, map_location=str(device))
            model.load_state_dict(sft_ckpt["model"])

        # load the reference model from the sft directory.
        ref_model = Transformer(tokenizer.vocab_size, C.embed_dim, C.context_len, wandb.config.num_heads, C.dropout_rate, C.num_blocks, PAD_TOKEN)
        ref_model.to(device)
        sft_ckpt = torch.load(sft_ckpt_path, map_location=str(device))
        ref_model.load_state_dict(sft_ckpt["model"])

        # freeze ref_model
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        eval_and_save_every_steps = 100
        print_every_steps = 10
        # Let's assert gradient_accum_steps is an integer, and also that we print and eval/save when a optimizer step is complete.
        assert eval_and_save_every_steps % grad_accum == 0
        assert print_every_steps % grad_accum == 0

        model.train()
        optimizer.zero_grad(set_to_none=True)

        last_print_time = time.time()
        running_micro = 0

        for epoch in range(start_epoch, C.num_epochs):
            print(f"\nEPOCH {epoch}")
            running_loss_sum = 0.0
            running_batches = 0

            for micro_step, batch in enumerate(train_loader):
                # Start a new accumulation window when the *global* micro step hits a boundary
                if running_micro % grad_accum == 0:
                    optimizer.zero_grad(set_to_none=True)

                inputs = batch["input_ids"].to(device, non_blocking=True).long() # (2B, T)
                labels = batch["labels"].to(device, non_blocking=True).long() # (2B, T)
                is_chosen = batch["is_chosen"].to(device, non_blocking=True).bool()

                if micro_step == 0 or micro_step == 1:
                    B = inputs.size(0)
                    for s in range(B):
                        print("input:")
                        ids_in = inputs[s].tolist()
                        print(ids_in)  # raw integers
                        print(tokenizer.convert_ids_to_tokens(ids_in))
                        print(tokenizer.decode(ids_in))
                        print("\n")

                        print("label:")
                        ids_lab = [i for i in labels[s].tolist() if i != IGNORE_TOKEN]
                        print(ids_lab)  # raw integers (ignore_index removed)
                        print(tokenizer.convert_ids_to_tokens(ids_lab))
                        print(tokenizer.decode(ids_lab))
                        print("\n")

                        print("is_chosen:")
                        is_chosen_ids = [i for i in is_chosen[s].tolist()]
                        print(is_chosen_ids)

                with autocast_ctx:
                    with torch.no_grad():
                        ref_logits = ref_model(inputs, False, None, None, None)    # (2B, T, V)
                    logits = model(inputs, False, None, None, None)                # (2B, T, V)

                    # Split into chosen/rejected
                    chosen_logits   = logits[is_chosen]        # (B/2,T,V)
                    rejected_logits = logits[~is_chosen]       # (B/2,T,V)
                    chosen_labels   = labels[is_chosen]            # (B/2,T)
                    rejected_labels = labels[~is_chosen]           # (B/2,T)
                    ref_chosen_logits   = ref_logits[is_chosen]    # (B/2,T,V)
                    ref_rejected_logits = ref_logits[~is_chosen]   # (B/2,T,V)

                    loss = dpo_loss(chosen_logits, rejected_logits,
                                    ref_chosen_logits, ref_rejected_logits,
                                    chosen_labels, rejected_labels, C.beta, IGNORE_TOKEN, C.length_normalize)

                loss.backward()
                running_micro += 1

                # end of accumulation window -> step
                if running_micro % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    global_step += 1

                    # lightweight train loss logging (per optimizer step)
                    running_loss_sum += loss.item() * grad_accum  # undo /grad_accum for readability
                    running_batches += 1
                    if global_step % print_every_steps == 0 and running_batches > 0:
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        now = time.time()  # or datetime.now() if you prefer
                        dt = now - last_print_time
                        last_print_time = now
                        tok_per_step = C.batch_effective_size * C.context_len
                        tok_per_sec = (print_every * tok_per_step) / max(dt, 1e-9)
                        avg_loss = running_loss_sum / running_batches
                        print(f"step {global_step} | train_loss {avg_loss:.4f} | tok/s {tok_per_sec:,.0f} | lr {scheduler.get_last_lr()[0]:.2e")

                    # periodic eval + checkpoint
                    if global_step % eval_and_save_every_steps == 0:
                        ckpt_path = os.path.join(save_dir, f"step_{global_step}.pth")
                        save_checkpoint(
                            ckpt_path, model, optimizer, scheduler, epoch, global_step,
                            extra={"best_val": best_val}
                        )
                        print(f"Saved checkpoint: {ckpt_path}")

                        # Eval
                        model.eval()
                        val_loss_sum = 0.0
                        val_pairs_sum = 0
                        acc_sum = 0.0
                        acc_ref_sum = 0.0

                        with torch.no_grad():
                            for batch_v in val_loader:
                                inputs_v = batch_v["input_ids"].to(device, non_blocking=True).long()
                                labels_v = batch_v["labels"].to(device, non_blocking=True).long()
                                is_chosen_v = batch_v["is_chosen"].to(device, non_blocking=True).bool()

                                ref_logits_v = ref_model(inputs_v, False, None, None, None)
                                logits_v = model(inputs_v,False, None, None, None)

                                ch_logits = logits_v[is_chosen_v]
                                rj_logits = logits_v[~is_chosen_v]
                                ch_labels = labels_v[is_chosen_v]
                                rj_labels = labels_v[~is_chosen_v]

                                ref_ch_logits = ref_logits_v[is_chosen_v]
                                ref_rj_logits = ref_logits_v[~is_chosen_v]

                                # DPO loss
                                val_loss = dpo_loss(
                                    ch_logits, rj_logits,
                                    ref_ch_logits, ref_rj_logits,
                                    ch_labels, rj_labels,
                                    C.beta, IGNORE_TOKEN, C.length_normalize,
                                )
                                val_loss_sum += val_loss.item()

                                # Pairwise accuracy (policy preferred)
                                lp_c = seq_logprobs_from_logits(ch_logits, ch_labels, IGNORE_TOKEN, C.length_normalize)
                                lp_r = seq_logprobs_from_logits(rj_logits, rj_labels, IGNORE_TOKEN, C.length_normalize)
                                delta = lp_c - lp_r
                                acc = (delta > 0).float().mean().item()
                                acc_sum += acc

                                # Ref-normalized signal (optional diagnostics)
                                rlp_c = seq_logprobs_from_logits(ref_ch_logits, ch_labels, IGNORE_TOKEN, C.length_normalize)
                                rlp_r = seq_logprobs_from_logits(ref_rj_logits, rj_labels, IGNORE_TOKEN, C.length_normalize)
                                s = C.beta * (delta - (rlp_c - rlp_r))
                                acc_ref = (s > 0).float().mean().item()
                                acc_ref_sum += acc_ref

                                # count pairs in this batch
                                batch_pairs = ch_logits.size(0)
                                val_pairs_sum += 1  # averaging per-batch metrics above

                        mean_val_loss = val_loss_sum / max(1, val_pairs_sum)
                        mean_val_acc = acc_sum / max(1, val_pairs_sum)
                        mean_val_acc_ref = acc_ref_sum / max(1, val_pairs_sum)
                        print(f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

                        wandb.log({
                            "epoch": epoch,
                            "global_step": global_step,
                            "val_loss": mean_val_loss,
                            "val_acc": mean_val_acc,
                            "val_acc_ref": mean_val_acc_ref,
                            "lr": scheduler.get_last_lr()[0],
                            "train_loss": running_loss_sum / running_batches
                        })
                        # we reset training stats to start fresh in next eval.
                        running_loss_sum = 0.0
                        running_batches = 0
                        model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str, help="wandb run name")
    parser.add_argument("sft_name", type=str, default=None, help="existing sft directory")
    parser.add_argument("sft_checkpoint", type=str, default=None, help="sft checkpoint")
    parser.add_argument("--resume-name", type=str, default=None, help="existing run to resume (dir must exist)")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="checkpoint file under *_ckpts/")
    args = parser.parse_args()

    main(
        args.run_name, args.sft_name, args.sft_checkpoint, args.resume_name, args.resume_checkpoint
    )



