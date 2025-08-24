import argparse
import os
import math
import inspect

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
from lmdataset import IndexedLMDataset


"""
# so finish the synchronize and printing every few steps.
# then do hyperparameters below


Now it's time to start thinking hyperparameters.
Come up with some then run them by chatgpt.
First, let's try context_len 1024.
Then, let's do the same for context_len 2048.

Karpathy used these settings:
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 64 # micro batch size
T = 1024 # sequence length
block_size: int = 1024 # max sequence length
vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
n_layer: int = 12 # number of layers
n_head: int = 12 # number of heads
n_embd: int = 768 # embedding dimension

double check: how many steps did Karpathy do?

See if you can fit GPT-medium on your GPU: https://arxiv.org/pdf/2005.14165 (you're willing to train an epoch or two over 10 days.)

All models were trained for a total of 300 billion tokens --> but remember you have high quality tokens.

"""

# step 1: first step is to size your transformer in VRAM
# step 2.1: make sure the checkpointing stuff works!
# step 2: before you train, do a sanity check where you load data and reverse the tokenization to make sure it looks like real language.
# step 1.5 - finish the sft stuff to completion.
# step 5: your inference.py may just want to print out one inference at a time first.
# step 6: hey before you kick off large scale training, overfit one batch.
# step 8: get torch compile in here.


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

PAD_TOKEN = 0

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


def model_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy over all tokens (no PADs expected).
    logits: (B,T,V), labels: (B,T) -> mean over B*T tokens.
    """
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.view(B * T, V),
        labels.view(B * T),
        reduction="mean",
    )

@torch.no_grad()
def model_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean token accuracy over all tokens (no PADs expected)."""
    pred = logits.argmax(dim=-1)  # (B,T)
    return (pred == labels).float().mean().item()

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
            print(f"Creating directory for {run_name}.")
            return run_name, save_dir

def main(run_name: str, resume_name: str, resume_checkpoint: str):
    train_token_path = ""
    train_index_path = ""
    val_token_path = ""
    val_index_path = ""
    tokenizer_dir = ""
    # assert you set these
    assert train_token_path and train_index_path and val_token_path and val_index_path and tokenizer_dir

    config = {
        "weight_decay": 0.1,
        "learning_rate": 0.001,
        "batch_micro_size": 256,
        "batch_effective_size": 1024,
        "num_epochs": 1,
        "num_blocks": 6,
        "dropout_rate": 0.1,
        "embed_dim": 512,
        "context_len": 1024,
        "num_heads": 4,
        "warmup_steps": 1000 # TODO: tune this
    }

    assert config.batch_effective_size % config.batch_micro_size == 0

    run_name, save_dir = resolve_run_name(run_name, resume_name)

    autocast_ctx = (
        torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    with wandb.init(mode="disabled",config=config,project="casallm-pretraining",entity="alancasallas-self",name=run_name) as run:
        C = wandb.config

        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

        # IndexedLMDataset is a custom dataset that loads tokens from memory mapped .bin files. It will return x and y, both of length context_len.
        token_dtype = np.uint16 if tok.vocab_size is not None and tok.vocab_size <= 65535 else np.uint32
        training_set = IndexedLMDataset(train_token_path, train_index_path, dtype=token_dtype)
        validation_set = IndexedLMDataset(val_token_path, val_index_path, dtype=token_dtype)

        # Create data loaders for our datasets; shuffle for training, not for validation
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=C.batch_micro_size, shuffle=True,
            drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=C.batch_micro_size, shuffle=False,
            num_workers=2, pin_memory=True, persistent_workers=True,
        )

        # the forward() function of the transformer takes input_ids: (B, T) and returns logits # (B, T, vocab_size)
        model = Transformer(tokenizer.vocab_size, wandb.config.embed_dim, wandb.config.context_len, wandb.config.num_heads, wandb.config.dropout_rate, wandb.config.num_blocks, PAD_TOKEN)
        model.to(device)

        # let's see number of model parameters
        model.eval()
        summary(model, input_size=(4, C.context_len))

        # Optimizer (fused if available on CUDA)
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        optimizer = AdamW(
            param_groups(model, wd=C.weight_decay),
            lr=C.learning_rate,
            betas=(0.9, 0.98),
            fused=(fused_available and device_type == "cuda"),
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

        # Resume
        start_epoch, global_step, best_val = 0, 0, float("inf")

        if resume_checkpoint:
            ckpt_path = os.path.join(save_dir, resume_checkpoint)
            start_epoch, global_step, extra = load_checkpoint(
                ckpt_path, model, optimizer, scheduler, map_location=str(device)
            )
            best_val = extra.get("best_val", best_val)
            print(f"Resumed from {ckpt_path} at epoch={start_epoch}, global_step={global_step}")

        eval_and_save_every_steps = 1000
        print_every_steps = 100
        # Let's assert gradient_accum_steps is an integer, and also that we print and eval/save when a optimizer step is complete.
        assert eval_and_save_every_steps % grad_accum == 0
        assert print_every_steps % grad_accum == 0

        model.train()
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(start_epoch, C.num_epochs):
            print(f"\nEPOCH {epoch}")
            running_loss_sum = 0.0
            running_batches = 0

            for micro_step, (inputs, labels) in enumerate(train_loader):
                # Start a new accumulation window when the *global* micro step hits a boundary
                if running_micro % grad_accum == 0:
                    optimizer.zero_grad(set_to_none=True)

                inputs = inputs.to(device, non_blocking=True).long()
                labels = labels.to(device, non_blocking=True).long()

                # Pretraining: assert no PADs present
                assert (inputs != PAD_TOKEN).all() and (labels != PAD_TOKEN).all(), \
                    "Pretraining batches must not contain PAD tokens."

                # Basic shape checks
                assert inputs.dim() == 2 and labels.dim() == 2
                assert inputs.size(1) == C.context_len, "context_len mismatch with pretokenized dataset"

                with autocast_ctx:
                    logits = model(inputs)  # (B,T,V)
                    loss = model_loss(logits, labels) / grad_accum

                loss.backward()
                running_micro += 1

                # end of accumulation window -> step
                if running_micro % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), C.clip_norm)
                    optimizer.step()
                    scheduler.step()
                    global_step += 1

                    # lightweight train loss logging (per optimizer step)
                    running_loss_sum += loss.item() * grad_accum  # undo /grad_accum for readability
                    running_batches += 1
                    if global_step % print_every == 0 and running_batches > 0:
                        avg_loss = running_loss_sum / running_batches
                        print(f"step {global_step} | train_loss {avg_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

                    # periodic eval + checkpoint
                    if global_step % eval_every == 0:
                        ckpt_path = os.path.join(save_dir, f"step_{global_step}.pth")
                        save_checkpoint(
                            ckpt_path, model, optimizer, scheduler, epoch, global_step,
                            extra={"best_val": best_val}
                        )
                        print(f"Saved checkpoint: {ckpt_path}")

                        # Eval
                        model.eval()
                        val_loss_sum = 0.0
                        val_batches = 0
                        val_acc_sum = 0.0

                        with torch.no_grad():
                            for inputs_v, labels_v in val_loader:
                                inputs_v = inputs_v.to(device, non_blocking=True).long()
                                labels_v = labels_v.to(device, non_blocking=True).long()
                                assert (inputs_v != PAD_TOKEN).all() and (labels_v != PAD_TOKEN).all()

                                logits_v = model(inputs_v)
                                batch_loss = model_loss(logits_v, labels_v)
                                val_loss_sum += batch_loss.item()
                                val_acc_sum += model_accuracy(logits_v, labels_v)
                                val_batches += 1

                        val_loss = val_loss_sum / max(1, val_batches)
                        val_acc = val_acc_sum / max(1, val_batches)
                        print(f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

                        if val_loss < best_val:
                            best_val = val_loss
                            best_path = os.path.join(save_dir, "best.pth")
                            save_checkpoint(
                                best_path, model, optimizer, scheduler, epoch, global_step,
                                extra={"best_val": best_val}
                            )
                            print(f"New best (val_loss={best_val:.4f}). Saved {best_path}.")

                        wandb.log({
                            "epoch": epoch,
                            "global_step": global_step,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "lr": scheduler.get_last_lr()[0],
                            "train_loss": running_loss_sum / running_batches
                        })
                        running_loss_sum = 0.0
                        running_batches = 0
                        model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str, help="wandb run name")
    parser.add_argument("--resume-name", type=str, default=None, help="existing run to resume (dir must exist)")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="checkpoint file under *_ckpts/")
    args = parser.parse_args()

    main(
        args.run_name, args.resume_name, args.resume_checkpoint
    )
