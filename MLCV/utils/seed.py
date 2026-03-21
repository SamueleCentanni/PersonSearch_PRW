import random

import numpy as np
import torch


def fix_random(seed: int) -> None:
    """Fix all possible sources of randomness."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(f"Random seed fixed with seed = {seed}")
