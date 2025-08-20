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

    