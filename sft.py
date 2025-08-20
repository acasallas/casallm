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



# use NVIDIA's tf32.
torch.set_float32_matmul_precision('high')


# TODO: what should you do about gradient accumulation here?
def transformer_loss(logits,labels,pad_token_id):
	loss_mask = (labels != pad_token_id).float() # (B,T)
	# collapse logits from (B,T,vocab_size) to (B*T,vocab_size) and collapse labels from (B,T) to (B*T)
	# this is done to be able to use the cross_entropy function
	# then we return the shape to B,T
	unreduced_loss = F.cross_entropy(logits.view(-1,logits.size(-1)), labels.view(-1), reduction="none").view(B,T)
	return (unreduced_loss * loss_mask).sum()/loss_mask.sum() # average over only non-padded losses.

# gradient accumulation


def main():
	config = {
		"weight_decay": 0.1,
		"learning_rate": 0.001
	}


	batch_accum_size = 32
	batch_total_size = 512

	num_epochs = 30

	# Create datasets for training & validation, download if necessary
	training_set = None # this is going to be a tokenized version of the dataset
	validation_set = None # this is going to be a tokenized version of the dataset.

	# Create data loaders for our datasets; shuffle for training, not for validation
	training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
	validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

	# Report split sizes
	print('Training set has {} instances'.format(len(training_set)))
	print('Validation set has {} instances'.format(len(validation_set)))

	# can we download the finewebedu set, a little sample?
	# define dataloader here
	print(f'Training set has shape {training_set[0][0].shape}')

	with wandb.init(config=config,project="casallm-sft",entity="alancasallas-self") as run:

		model = Transformer(wandb.config.neuron_size)
		model.to(device)

		# todo: can you implement batch accumulation?

		# let's try Adam for now with default parameters
		# TODO: we may just have a linear decay instead of the whole cosine annearling thing.s
		optimizer = torch.optim.AdamW(model.parameters(),lr=wandb.config.learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=wandb.config.weight_decay)
		

		# for something that takes this long, you'll definetly want to print or even validation after every few steps, not just epochs.

		for epoch in range(num_epochs):
			metrics = {"epoch": epoch}
			print(f"EPOCH {epoch}")
			train_losses = 0
			train_total = 0

			model.train()
			for i,data in enumerate(training_loader):
				inputs, labels = data
				inputs = inputs.to(device)
				labels = labels.to(device)

				optimizer.zero_grad()

				outputs = model(inputs) # outputs are logits (B,T,vocab_size)
				loss = transformer_loss(outputs,labels, 0) # labels are B,T

				train_losses += loss.item()*inputs.size(0)
				train_total += inputs.size(0)
				loss.backward()
				optimizer.step()
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




if __name__ == "__main__":
	main()