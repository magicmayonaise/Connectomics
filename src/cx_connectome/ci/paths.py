"""Typed stubs for graph path enumeration utilities."""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence

__all__ = ["PathEnumerator"]


class PathEnumerator(Protocol):
    """Protocol describing path enumeration in a directed graph."""

    def enumerate_paths(self, sources: Sequence[int], targets: Sequence[int]) -> Iterable[Sequence[int]]:
        """Yield index paths between the given source and target populations."""
        ...
