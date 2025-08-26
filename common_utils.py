import random

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb

# load and save model

def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step, extra: dict = None):
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if extra:
        ckpt["extra"] = extra
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cuda"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    # ---- Restore RNG states safely ----
    if "torch_rng_state" in ckpt:
        # must be a CPU ByteTensor
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())

    if "cuda_rng_state" in ckpt and torch.cuda.is_available():
        states = ckpt["cuda_rng_state"]
        # handle both list-of-tensors (usual) and a single tensor (defensive)
        if isinstance(states, torch.Tensor):
            states = [states]
        # torch.cuda.set_rng_state_all expects CPU ByteTensors as well
        torch.cuda.set_rng_state_all([t.cpu() for t in states])

    if "numpy_rng_state" in ckpt:
        np.random.set_state(ckpt["numpy_rng_state"])
    if "python_rng_state" in ckpt:
        random.setstate(ckpt["python_rng_state"])
    start_epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    extra = ckpt.get("extra", {})
    return start_epoch, global_step, extra