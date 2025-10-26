"""Typed stubs for non-linear network dynamics."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

__all__ = ["DynamicsModel"]


class DynamicsModel(Protocol):
    """Protocol describing a neural dynamics simulator."""

    def run(self, initial_state: Mapping[str, Any], *, time_steps: int) -> Mapping[str, Any]:
        """Simulate the system for the given number of time steps."""
        ...
