"""Differentiable optimization utilities used in tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import random

try:  # pragma: no cover - torch is optional
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - gracefully degrade when torch is missing
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def toy_slice(num_nodes: int = 50, seed: int = 0) -> List[List[float]]:
    """Return a synthetic feedforward slice as a list-of-lists matrix."""

    rng = random.Random(seed)
    weights = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for src in range(num_nodes - 1):
        weights[src][src + 1] = 0.6 + 0.1 * (src % 3)
    extra_edges = num_nodes // 2
    for _ in range(extra_edges):
        src = rng.randrange(0, num_nodes - 1)
        dst = rng.randrange(src + 1, num_nodes)
        weights[src][dst] += rng.uniform(0.05, 0.2)
    return weights


if torch is not None:

    @dataclass
    class ActivationMaximizer:
        weights: "torch.Tensor"
        depth: int = 3

        def __post_init__(self) -> None:
            if self.weights.ndim != 2 or self.weights.shape[0] != self.weights.shape[1]:
                raise ValueError("weights must be a square matrix")
            if self.depth <= 0:
                raise ValueError("depth must be positive")

        def forward(self, params: "torch.Tensor") -> "torch.Tensor":
            activation = torch.sigmoid(params)
            for _ in range(self.depth):
                activation = F.relu(activation @ self.weights)
            return activation

        def maximize(
            self,
            target_index: int,
            *,
            steps: int = 40,
            lr: float = 0.1,
        ) -> List[float]:
            param = torch.nn.Parameter(torch.zeros(self.weights.shape[0], device=self.weights.device))
            opt = torch.optim.Adam([param], lr=lr)
            history: List[float] = []
            for _ in range(steps):
                opt.zero_grad()
                activation = self.forward(param)
                target_value = activation[target_index]
                (-target_value).backward()
                opt.step()
                history.append(float(target_value.detach()))
            return history

else:  # pragma: no cover - executed only when torch is missing

    class ActivationMaximizer:  # type: ignore[override]
        def __init__(self, *_, **__):
            raise RuntimeError("torch is required for ActivationMaximizer")

        def forward(self, *_: object) -> None:
            raise RuntimeError("torch is required for ActivationMaximizer")

        def maximize(self, *_: object, **__: object) -> None:
            raise RuntimeError("torch is required for ActivationMaximizer")
