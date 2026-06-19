"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
