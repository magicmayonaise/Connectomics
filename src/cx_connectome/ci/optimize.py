"""Typed stubs for optimization routines."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol

__all__ = ["Optimizer"]


class Optimizer(Protocol):
    """Protocol describing an optimizer for CI models."""

    def minimize(self, objective: Callable[[Mapping[str, Any]], float], *, max_steps: int) -> Mapping[str, Any]:
        """Return optimized parameters for the provided objective function."""
        ...
