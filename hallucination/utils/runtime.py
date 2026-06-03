"""Runtime helpers shared by experiment entry points."""

from __future__ import annotations

import gc
import json
import os
import random
from datetime import datetime
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def timestamped_output_dir(base_dir: str) -> str:
    out_dir = os.path.join(base_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_json(path: str, payload: Any, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=indent)


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
