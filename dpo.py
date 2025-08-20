import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb

from transformer import Transformer


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def dpo_loss(logits,labels,pad_token_id):
    # TODO: you must implement the DPO loss.
    pass


def main():
    config = {
        "weight_decay": 0.1,
        "learning_rate": 0.001,
        "batch_micro_size": 512,
        "batch_effective_size": 1024,
        "num_epochs": 30
    }

    assert wandb.config.batch_effective_size % wandb.config.batch_micro_size == 0

    # print every steps
    # save every steps
    # run eval every steps


    # Create datasets for training & validation, download if necessary
    training_set = None # this is going to be a tokenized version of the dataset
    validation_set = None # this is going to be a tokenized version of the dataset.

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True,drop_last=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    # Report split sizes
    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    # can we download the finewebedu set, a little sample?
    # define dataloader here
    print(f'Training set has shape {training_set[0][0].shape}')


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

    optimizer = AdamW(param_groups(model, wd=wandb.config.weight_decay), lr=wandb.config.learning_rate, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    # TODO: once you have this loop finished, it will be largely copy-pasted to  sft and dpo
    # TODO: I'm not sure you can just copy paste pretraining loop to dpo like you could with sft.
    with wandb.init(mode="disabled",config=config,project="casallm-dpo",entity="alancasallas-self") as run:

        model = Transformer(wandb.config.neuron_size)
        model.to(device)

        # todo: can you implement batch accumulation?



        # let's try Adam for now with default parameters
        optimizer = torch.optim.AdamW(model.parameters(),lr=wandb.config.learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=wandb.config.weight_decay)
        

        gradient_accum_steps = wandb.config.batch_effective_size//wandb.config.batch_micro_size
        model.train()
        optimizer.zero_grad()

        # for something that takes this long, you'll definetly want to print or even validation after every few steps, not just epochs.

        for epoch in range(num_epochs):
            metrics = {"epoch": epoch}
            print(f"EPOCH {epoch}")
            train_losses = 0
            train_total = 0
            
            for step, data in enumerate(training_loader):
                model.train()
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs) # outputs are logits (B,T,vocab_size)
                loss = transformer_loss(outputs,labels)/gradient_accum_steps # labels are B,T

                train_losses += loss.item()*inputs.size(0)
                train_total += inputs.size(0)
                loss.backward()

                # TODO: let's get accuracy in here.

                # update weights every accum_steps
                 if (step + 1) % accum_steps == 0: 
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                 # TODO: let's put in other eval and printing things here.

                print(f"train loss {train_losses/train_total}")
                metrics.update({"train_loss": train_losses/train_total})


                # remember model.train() and model.eval()!
                model.eval()
                val_losses = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for i,data in enumerate(validation_loader):
                        inputs, labels = data
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs) # this doesn't work because outputs is a 10 vector!
                        _, predicted = torch.max(outputs, 1)
                        loss =loss_fn(outputs,labels)
                        val_losses += loss.item()*inputs.size(0)
                        val_correct += (labels==predicted).sum().item()
                        val_total += labels.size(0)

                print(f"validation loss {val_losses/val_total} accuracy {val_correct/val_total}")
                metrics.update({"val_loss": train_losses/train_total})
                wandb.log(metrics)

            if (step + 1) % accum_steps != 0:
                optimizer.zero_grad()  # if there's a leftover micro_batch discard partial window




if __name__ == "__main__":
    main()