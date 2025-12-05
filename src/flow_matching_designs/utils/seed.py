# src/flow_matching_designs/utils/seed.py

from __future__ import annotations

import random
import os

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Seed Python, NumPy and PyTorch (CPU & CUDA) for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Make cuDNN deterministic (at some performance cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
