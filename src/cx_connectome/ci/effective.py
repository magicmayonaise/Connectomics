"""Typed stubs for effective connectivity calculations."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

__all__ = ["EffectiveConnectivitySolver"]


class EffectiveConnectivitySolver(Protocol):
    """Protocol describing an effective connectivity solver."""

    def compute(self, adjacency: Any, *, k_hops: Sequence[int]) -> Any:
        """Return an effective connectivity tensor for the given adjacency graph."""
        ...
