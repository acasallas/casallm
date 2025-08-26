"""
This script is used to perform both pretraining and SFT of an LLM.
"""

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
from lmdataset import IndexedLMDataset
from transformers import PreTrainedTokenizerFast

# TODO: fix those little bugs
# TODO: start reindexing.
# TODO: you fit, you measure token/sec throughput, and then you think (with torch compile perhaps, how long you will haveto train for.)

# TODO: right off the bat, something's wrong, initial loss should be 10-ish




# consider collecting all TODO's in one main todo file. (to make a final schedule over three days).
# TODO: somehting you should know soon is whether to use 10B or 100B tokens.
# TODO: I wonder if saving the RNG will help you pick up where you left off for tokens.

# step 1: first step is to size your transformer in VRAM
# step 2.1: make sure the checkpointing stuff works!
# step 2: before you train, do a sanity check where you load data and reverse the tokenization to make sure it looks like real language.
# step 6: hey before you kick off large scale training, overfit one batch.
# step 8: get torch compile in here.

# TODO: figure out how not to load too much in validation.

# each training loop takes only a few minutes to run, so by end of day you should be able to run a generate() function on a model that has undergone all three stages.
# note: if torch.compile clamps down batch size, is that ok for SFT and DPO because we're still using the same by-sample batch size?

# TODO: double check that reindex script was legit
#TODO: check for duplication between sft_test and gen_test.

"""
Now it's time to start thinking hyperparameters.
Come up with some then run them by chatgpt.
First, let's try context_len 1024.
Then, let's do the same for context_len 2048.

# once we size transformer in VRAM, we should be able to make a good guess for our context_len

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



# later recommendation: Consider gradient checkpointing inside blocks if you push context length or model size.


"""
Potential DataLoader settings to play around with:

train_loader = DataLoader(
    train_set,
    batch_size=C.batch_micro_size,
    shuffle=True,
    drop_last=True,
    num_workers=12,                 # start ~12–16; tune
    prefetch_factor=4,              # default=2; bump if RAM allows
    pin_memory=True,
    pin_memory_device="cuda",       # PyTorch ≥2.1
    persistent_workers=True,
    multiprocessing_context="fork", # Linux/WSL; try "forkserver" if RAM spikes
    worker_init_fn=lambda _: torch.set_num_threads(1),  # avoid BLAS thread storms
)

val_loader = DataLoader(
    val_set,
    batch_size=C.batch_micro_size,
    shuffle=False,
    num_workers=2,                  # small; eval is bursty
    prefetch_factor=2,
    pin_memory=True,
    pin_memory_device="cuda",
    persistent_workers=True,
    worker_init_fn=lambda _: torch.set_num_threads(1),
)
"""


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"using device {device}")

PAD_TOKEN = None
BOS_TOKEN = None
EOS_TOKEN = None
IGNORE_TOKEN = -100 # token that loss functions will ignore
STAGE_PRETRAINING = "pretraining"
STAGE_SFT = "sft"

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
    Cross-entropy over all tokens
    Note: expect no PAD tokens. (pretraining will not have PAD tokens, and SFT will have IGNORE TOKENS instead).
    logits: (B,T,V), labels: (B,T) -> mean over B*T tokens.
    """
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.view(B * T, V),
        labels.view(B * T),
        reduction="mean",
        ignore_index = IGNORE_TOKEN
    )

@torch.no_grad()
def accuracy_counts(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100):
    pred = logits.argmax(dim=-1)
    valid = labels.ne(ignore_index)
    correct = (pred.eq(labels) & valid).sum()
    total = valid.sum()
    return correct.item(), total.item()

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
    raise ValueError("couldn't resolve run name")

def get_config_for_training_stage(training_stage):
    if training_stage == STAGE_PRETRAINING:
        return {
        "weight_decay": 0.1,
        "learning_rate": 3e-4,
        "min_lr": 1e-4, # TODO: you need to tune this
        "batch_micro_size": 4, # tune this until you max out VRAM.
        "batch_effective_size": 256, #2**19 tokens
        "num_epochs": 1,
        "num_blocks": 24,
        "dropout_rate": 0.1,
        "embed_dim": 1024,
        "context_len": 2048, # if we can get to 2048 that will be preferrable
        "num_heads": 16,
        "warmup_steps": 512 # TODO: tune this, (I think it's 2% of total steps?)
    }
    elif training_stage == STAGE_SFT:
        return {
        "weight_decay": 0.1,
        "learning_rate": 3e-4,
        "min_lr": 1e-4, # TODO: you need to tune this
        "batch_micro_size": 32, # tune this until you max out VRAM.
        "batch_effective_size": 512, #2**19 tokens
        "num_epochs": 1,
        "num_blocks": 24,
        "dropout_rate": 0.0,
        "embed_dim": 1024,
        "context_len": 2048, # if we can get to 2048 that will be preferrable
        "num_heads": 16,
        "warmup_steps": 512 # TODO: tune this, (I think it's 2% of total steps?)
    }
    raise ValueError(f"bad training stage {training_stage}")

def get_datasets_for_stage(training_stage, train_data_dir, val_data_dir, context_len, token_dtype):
    if training_stage == STAGE_PRETRAINING:
        train_token_path = os.path.join(train_data_dir, "tokens.bin")
        train_index_path = os.path.join(train_data_dir, "sample_idx.bin")
        val_token_path = os.path.join(val_data_dir, "tokens.bin")
        val_index_path = os.path.join(val_data_dir, "sample_idx.bin")
        training_set = IndexedLMDataset(train_token_path, train_index_path, context_len, token_dtype)
        validation_set = IndexedLMDataset(val_token_path, val_index_path, context_len, token_dtype)
        return training_set, validation_set
    elif training_stage == STAGE_SFT:
        training_set = SFTMemmapDatasetShifted(train_data_dir, context_len, PAD_TOKEN, IGNORE_TOKEN, token_dtype)
        validation_set = SFTMemmapDatasetShifted(val_data_dir, context_len, PAD_TOKEN, IGNORE_TOKEN, token_dtype)
        return training_set, validation_set
    raise ValueError(f"bad training stage {training_stage}")


def main(training_stage, run_name, pretrained_name, pretrained_checkpoint, resume_name, resume_checkpoint):
    if training_stage ==  STAGE_SFT:
        assert pretrained_name # if we are in SFT, there must be a corresponding pretrained model to start from.

    train_data_dir = "tokenized_pretrain_train_2048"
    val_data_dir = "tokenized_pretrain_validation_2048"
    tokenizer_dir = "casallm_bpe"
    # assert you set these
    assert train_data_dir and val_data_dir and tokenizer_dir

    config = get_config_for_training_stage(training_stage)
    assert config['batch_effective_size'] % config['batch_micro_size'] == 0

    print(f"run name we want {run_name}")
    run_name, save_dir = resolve_run_name(run_name, resume_name)

    print(f"run name is {run_name} in {save_dir}.")

    autocast_ctx = (
        torch.autocast(device_type=str(device), dtype=torch.bfloat16)
        if str(device) == "cuda"
        else nullcontext()
    )

    with wandb.init(mode="disabled",config=config, project=f"casallm-{training_stage}",entity="alancasallas-self",name=run_name) as run:
        C = wandb.config

        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        PAD_TOKEN = tokenizer.pad_token_id
        BOS_TOKEN = tokenizer.bos_token_id
        EOS_TOKEN = tokenizer.eos_token_id

        # IndexedLMDataset is a custom dataset that loads tokens from memory mapped .bin files. It will return x and y, both of length context_len.
        token_dtype = np.uint16 if tokenizer.vocab_size is not None and tokenizer.vocab_size <= 65535 else np.uint32
        train_set, val_set = get_datasets_for_stage(training_stage, train_data_dir, val_data_dir, C.context_len, token_dtype)

        # Create data loaders for our datasets; shuffle for training, not for validation
        # TODO: if gpu-util is not high enough, adjust settings (an example is above).
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=C.batch_micro_size, shuffle=True,
            drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=C.batch_micro_size, shuffle=False,
            num_workers=2, pin_memory=True, persistent_workers=True,
        )

        # the forward() function of the transformer takes input_ids: (B, T) and returns logits # (B, T, vocab_size)
        print(f"vocab size is gonna be: {tokenizer.vocab_size}")
        print(f"but tokenizer len is {len(tokenizer)}")
        model = Transformer(tokenizer.vocab_size, C.embed_dim, C.context_len, C.num_heads, C.dropout_rate, C.num_blocks, PAD_TOKEN)

        # This code was added because torchinfo.summary was double-counting tied weights.
        # 1) Prove it’s the same tensor
        print(model.output.weight is model.embedding.weight)  # -> True
        print(id(model.output.weight) == id(model.embedding.weight))  # -> True

        # 2) Count “true” parameters without double-counting shared objects
        seen = set()
        true_params = 0
        for p in model.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                true_params += p.numel()
        print(f"True parameter count: {true_params:,}")

        #model.eval()
        #summary(model, input_size=(4, C.context_len), dtypes=[torch.long])
        model.to(device)

        model = torch.compile(model)

        # Optimizer (fused if available on CUDA)
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        optimizer = AdamW(
            param_groups(model, wd=C.weight_decay),
            lr=C.learning_rate,
            betas=(0.9, 0.98),
            fused=(fused_available and str(device) == "cuda"),
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
        if resume_name:
            ckpt_path = os.path.join(save_dir, resume_checkpoint)
            start_epoch, global_step, extra = load_checkpoint(
                ckpt_path, model, optimizer, scheduler, map_location=str(device)
            )
            best_val = extra.get("best_val", best_val)
            print(f"Resumed from {ckpt_path} at epoch={start_epoch}, global_step={global_step}")
        elif training_stage == STAGE_SFT:
            # load from a pretrained model
            pretrained_save_dir=f"./{pretrained_name}_ckpts"
            pretrained_ckpt_path = os.path.join(pretrained_save_dir, pretrained_checkpointain)
            pretrained_ckpt = torch.load(pretrained_ckpt_path, map_location=str(device))
            model.load_state_dict(pretrained_ckpt["model"])

        eval_and_save_every_steps = 100
        print_every_steps = 1

        model.train()
        optimizer.zero_grad(set_to_none=True)

        last_print_time = time.time()
        running_micro = 0

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

                if micro_step == 0:
                    assert (inputs != PAD_TOKEN).all(), "Found pad tokens in pretraining inputs"
                    assert (inputs != IGNORE_TOKEN).all(), "Found unknowns in pretraining inputs"
                    assert (inputs < tokenizer.vocab_size).all(), "Found out-of-range labels in pretraining inputs"
                    assert (labels != PAD_TOKEN).all(), "Found pad tokens in pretraining labels"
                    assert (labels != IGNORE_TOKEN).all(), "Found unknowns in pretraining labels"
                    assert (labels < tokenizer.vocab_size).all(), "Found out-of-range labels in pretraining labels"
                    print(f"assertion successful")

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
                        tok_per_sec = (print_every_steps * tok_per_step) / max(dt, 1e-9)
                        avg_loss = running_loss_sum / running_batches
                        print(f"step {global_step} | train_loss {avg_loss:.4f} | tok/s {tok_per_sec:,.0f} | lr {scheduler.get_last_lr()[0]:.2e}")

                    if global_step > 3:
                        return # early exit for now.

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
                        val_batches = 0
                        val_acc_correct = 0.0
                        val_acc_total = 0

                        with torch.no_grad():
                            for inputs_v, labels_v in val_loader:
                                inputs_v = inputs_v.to(device, non_blocking=True).long()
                                labels_v = labels_v.to(device, non_blocking=True).long()
                                assert (inputs_v != PAD_TOKEN).all() and (labels_v != PAD_TOKEN).all()

                                logits_v = model(inputs_v)
                                batch_loss = model_loss(logits_v, labels_v)
                                val_loss_sum += batch_loss.item()
                                acc_correct, acc_total = accuracy_counts(logits_v, labels_v)
                                val_acc_correct += acc_correct
                                val_acc_total += acc_total

                        val_loss = val_loss_sum / max(1, val_batches)
                        val_acc = val_acc_correct / max(1, val_acc_total)
                        print(f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

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
    parser.add_argument("training_stage", choices=[STAGE_PRETRAINING, STAGE_SFT])
    parser.add_argument("run_name", type=str, help="wandb run name")
    parser.add_argument("--pretrained-name", type=str, default=None, help="existing run to resume (dir must exist)")
    parser.add_argument("--pretrained-checkpoint", type=str, default=None, help="checkpoint file under *_ckpts/")
    parser.add_argument("--resume-name", type=str, default=None, help="existing run to resume (dir must exist)")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="checkpoint file under *_ckpts/")
    args = parser.parse_args()

    main(
        args.training_stage, args.run_name, args.pretrained_name, args.pretrained_checkpoint, args.resume_name, args.resume_checkpoint
    )
