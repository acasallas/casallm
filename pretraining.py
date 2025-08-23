import argparse

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb

from common_utils import save_checkpoint, load_checkpoint
from transformer import Transformer

from lmdataset import IndexedLMDataset

# step 2: before you train, do a sanity check where you load data and reverse the tokenization to make sure it looks like real language.
# step 3: use torchsummary to figure out transformer size first.
# step 4: let's figure out hyperparams (first look it up, then double check what Andrej was using)
# step 5: your inference.py may just want to print out one inference at a time first.
# step 6: hey before you kick off large scale training, overfit one batch.
# step 7: figure out tokens/sec training.


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

PAD_TOKEN = 0

# use NVIDIA's tf32.
torch.set_float32_matmul_precision('high')


def transformer_loss(logits,labels,pad_token_id):
    # collapse logits and labels from (B,T,..) to (B*T,...) so they can be passed into cross_entropy function.
    return F.cross_entropy(logits.view(-1,logits.size(-1)), labels.view(-1)).view(B,T)

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
    config = {
        "weight_decay": 0.1,
        "learning_rate": 0.001,
        "batch_micro_size": 512,
        "batch_effective_size": 1024,
        "num_epochs": 10,
        "num_blocks": 6,
        "dropout_rate": 0.1,
        "embed_dim": 512,
        "context_len": 1024,
        "num_heads": 4
    }

    assert wandb.config.batch_effective_size % wandb.config.batch_micro_size == 0

    run_name, save_dir = resolve_run_name(run_name, resume_name)

    tokenizer = None # TODO: load tokenizer

    eval_every_steps = 1000
    save_every_steps = 1000
    print_every_steps = 100

    with wandb.init(mode="disabled",config=config,project="casallm-pretraining",entity="alancasallas-self",name=run_name) as run:
        training_set = IndexedLMDataset(token_path, index_path, dtype=np.uint32) # TODO: are we setting this to 16 b/c vocab size is 40,000???
        validation_set = IndexedLMDataset(token_path, index_path, dtype=np.uint32) # TODO: same as above

        # TODO: we need a loader that will create inputs and labels, where labels will be inputs shifted by one.

        # Create data loaders for our datasets; shuffle for training, not for validation
        training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True,drop_last=True)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

        model = Transformer(tokenizer.vocab_size, wandb.config.embed_dim, wandb.config.context_len, wandb.config.num_heads, wandb.config.dropout_rate, wandb.config.num_blocks, PAD_TOKEN)
        model.to(device)

        # Warmup + cosine (to ~1e-6). ramp up over 2k steps:
        warmup_steps = 2000 # TODO warmup_steps and learninig rate have to be adjusted for an LLM
        total_steps = len(training_loader) * num_epochs
        base_lr = wandb.config.learning_rate

        # the famous cosine annealing with rampup.
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            # cosine from 1.0 -> ~1e-6/base_lr
            min_lr = 1e-6 / base_lr
            return min_lr + 0.5 * (1 - min_lr) * (1 + math.cos(math.pi * t))

        optimizer = torch.optim.AdamW(model.parameters(),lr=wandb.config.learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=wandb.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        start_epoch, global_step, best_val = 0, 0, float("inf")

        # Load saved checkpoint if requested
        if resume_path:
            start_epoch, global_step, extra = load_checkpoint(os.path.join(save_dir, resume_checkpoint), model, optimizer, scheduler, map_location=str(device))
            best_val = extra.get("best_val", best_val)
            print(f"Resumed from {resume_path} at epoch={start_epoch}, global_step={global_step}")

        gradient_accum_steps = wandb.config.batch_effective_size//wandb.config.batch_micro_size
        model.train()
        optimizer.zero_grad()

        for epoch in range(num_epochs):
            metrics = {"epoch": epoch}
            print(f"EPOCH {epoch}")
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            for step, data in enumerate(training_loader):
                model.train()
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # TODO: can we get bf16 in here?

                outputs = model(inputs) # outputs are logits (B,T,vocab_size)
                loss = transformer_loss(outputs, labels)/gradient_accum_steps # labels are B,T

                B = inputs.size(0)
                train_loss += loss.item()*B
                train_total += B

                loss.backward()

                with torch.no_grad():
                    train_correct += (labels==predicted).sum().item()

                # update weights every accum_steps
                if (step + 1) % accum_steps == 0: 
                    global_step += 1
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # TODO: let's put in other eval and printing things here.

                if global_step % print_every_steps == 0:
                    print(f"epoch {step} step {step}")

                # save every few steps
                if global_step % save_every_steps == 0:
                    ckpt_path = os.path.join(save_dir, f"step_{global_step}.pth")
                    save_checkpoint(
                        ckpt_path, model, optimizer, scheduler, epoch, global_step,
                        extra={"best_val": best_val}
                    )
                    print(f"Checkpoint saved {ckpt_path}")

                if global_step % eval_every_steps == 0:
                    model.eval()
                    val_losses, val_correct, val_total = 0.0, 0, 0

                    with torch.no_grad():
                        for i,data in enumerate(validation_loader):
                            inputs, labels = data
                            inputs = inputs.to(device)
                            labels = labels.to(device)
                            outputs = model(inputs)
                            B = inputs.size(0)
                            _, predicted = torch.max(outputs, 1)
                            loss = transformer_loss(outputs, labels)
                            val_losses += loss.item()*B
                            val_correct += (labels==predicted).sum().item()
                            val_total += B

                    print(f"validation loss {val_losses/val_total} accuracy {val_correct/val_total}")
                    metrics.update({"val_loss": train_losses/train_total})
                    wandb.log(metrics)
                    train_loss, train_correct, train_total = 0.0, 0, 0

            if (step + 1) % accum_steps != 0:
                optimizer.zero_grad()  # if there's a leftover micro_batch discard partial window


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple command-line program.")
    parser.add_argument("run_name", type=str, help="wandb run name")
    parser.add_argument("--resume-name", type=str, help="wandb run to resume name")
    parser.add_argument("--resume-checkpoint", type=str, help="wandb run to resume name")
    args = parser.parse_args()


    main(args.run_name, args.resume_name, args.resume_checkpoint)
