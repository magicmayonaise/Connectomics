"""Typed stubs for evaluation metrics."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

__all__ = ["Metric"]


class Metric(Protocol):
    """Protocol describing a metric for CI evaluations."""

    def compute(self, predictions: Sequence[Any], targets: Sequence[Any]) -> float:
        """Return a scalar score summarizing predictive quality."""
        ...
