"""Reproductibilité : une graine unique propagée à python/numpy/torch."""
from __future__ import annotations

import os
import random
from typing import List

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> int:
    """Fixe la graine de tous les générateurs disponibles et la retourne."""
    seed = int(seed) % (2**32 - 1)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch est optionnel (backtest/validation fonctionnent sans)
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(False)  # cudnn RNN interdit le mode strict
            torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - torch absent
        pass
    return seed


def spawn_seeds(master_seed: int, n: int) -> List[int]:
    """Dérive `n` graines indépendantes (utilisé pour les ensembles multi-seeds)."""
    ss = np.random.SeedSequence(master_seed)
    return [int(s.generate_state(1, dtype=np.uint32)[0]) for s in ss.spawn(n)]
