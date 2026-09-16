"""Run the pinned Fair PSG CLI after fixing all available random seeds."""

import os
import random
import runpy

import numpy as np
import torch


SEED = 0


def main() -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    runpy.run_module("fair_psgg", run_name="__main__")


if __name__ == "__main__":
    main()
