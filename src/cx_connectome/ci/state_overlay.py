"""Typed stubs for state overlay utilities."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

__all__ = ["StateOverlay"]


class StateOverlay(Protocol):
    """Protocol describing overlay of state variables on connectome structures."""

    def render(self, state: Mapping[str, Any]) -> Any:
        """Render a representation of the provided state."""
        ...
