import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set RNG seeds for Python, NumPy and PyTorch and make cuDNN deterministic.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN flags to improve determinism (may slow training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)