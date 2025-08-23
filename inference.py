import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb

from transformer import Transformer
import common_utils


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# TODO: load tokenizer

# TODO: put temperature in here. What about top-p and top-k?


def run_inference():
	# load model



	while True:
		# run inference forever
		new_batch = input("Give me a new batch of data")

		# tokenize batch of data
		# feed into LLM and get result
		# first do all at a time, then stream tokens back.

		pass


if __name__ == "__main__":
	# load model and run inference
	run_inference()

