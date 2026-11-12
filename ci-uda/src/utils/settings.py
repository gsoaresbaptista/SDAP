import os
import random
from pathlib import Path

import numpy as np
import torch


def dataset_path(name, default_root="datasets"):
    """Absolute path of a dataset, allowing the root to be overridden by CIUDA_DATA_ROOT."""
    root = Path(os.environ.get("CIUDA_DATA_ROOT", default_root))
    return root / name


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
