"""Typed stubs for connectome integration helpers."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

__all__ = ["ConnectomeIntegrator"]


class ConnectomeIntegrator(Protocol):
    """Protocol describing integration of CI outputs with connectome data."""

    def integrate(self, connectome: Mapping[str, Any], ci_outputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return an enriched connectome representation."""
        ...
