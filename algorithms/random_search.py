"""Random-search baseline for hyperparameter tuning.

Same Objective protocol as :mod:`algorithms.dp_gibo`: any object with
``random_theta`` and ``mean_loss`` is supported.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch
from tqdm import tqdm

from .dp_gibo import Objective


@dataclasses.dataclass
class RandomSearchResult:
    losses: np.ndarray         # mean validation loss at each trial, shape (n_iter,)
    best_so_far: np.ndarray    # running-best mean loss, shape (n_iter,)
    best_theta: torch.Tensor   # θ that achieved best_so_far[-1]


def run_random_search(
    objective: Objective,
    n_iter: int,
    seed: int = 0,
    progress: bool = True,
) -> RandomSearchResult:
    """Sample θ uniformly inside the objective's bounds and track running-best."""
    gen = torch.Generator().manual_seed(seed)

    losses = np.empty(n_iter)
    best_so_far = np.empty(n_iter)
    best = math.inf
    best_theta: torch.Tensor | None = None

    iterator = range(n_iter)
    if progress:
        iterator = tqdm(iterator, desc="Random search")

    for t in iterator:
        theta = objective.random_theta(n=1, generator=gen)
        loss = objective.mean_loss(theta)
        losses[t] = loss
        if loss < best:
            best = loss
            best_theta = theta[0].detach().clone()
        best_so_far[t] = best

    assert best_theta is not None, "n_iter must be positive"
    return RandomSearchResult(losses=losses, best_so_far=best_so_far, best_theta=best_theta)
