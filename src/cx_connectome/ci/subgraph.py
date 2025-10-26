"""Utilities for working with subgraphs extracted from circuit isolations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Set, Tuple

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
    _PANDAS_ERROR = exc
else:  # pragma: no cover - optional dependency
    _PANDAS_ERROR = None


@dataclass
class Subgraph:
    """Container for a sparse subgraph.

    Parameters
    ----------
    nodes:
        Node identifiers that participate in the subgraph.
    edges:
        Mapping from ``(source, target)`` tuples to edge weights. The
        weights are optional, but when provided they are stored as floats
        to make serialisation consistent.
    """

    nodes: Set[int] = field(default_factory=set)
    edges: Dict[Tuple[int, int], float] = field(default_factory=dict)

    def add_edge(self, source: int, target: int, weight: float | None = None) -> None:
        """Add an edge to the subgraph.

        The ``weight`` argument is optional. When omitted the edge is
        stored with a default weight of ``1.0`` so that downstream code
        can still rely on numeric values.
        """

        if weight is None:
            weight = 1.0
        self.nodes.add(source)
        self.nodes.add(target)
        self.edges[(source, target)] = float(weight)

    def update_edges(self, items: Iterable[Tuple[int, int, float]]) -> None:
        """Bulk-add weighted edges to the subgraph."""

        for source, target, weight in items:
            self.add_edge(source, target, weight)

    @property
    def node_ids(self) -> Set[int]:
        """Return the node identifiers contained in the subgraph."""

        return set(self.nodes)

    def iter_edges(self) -> Iterator[Tuple[int, int, float]]:
        """Iterate over the stored edges as ``(source, target, weight)`` tuples."""

        for (source, target), weight in self.edges.items():
            yield source, target, weight

    def to_nodes_frame(self) -> pd.DataFrame:
        """Return a dataframe describing the nodes in the subgraph."""

        if pd is None:  # pragma: no cover - optional dependency
            raise ImportError("pandas is required to export node tables") from _PANDAS_ERROR
        return pd.DataFrame({"node_id": sorted(self.nodes)})

    def to_edges_frame(self) -> pd.DataFrame:
        """Return a dataframe describing the edges in the subgraph."""

        if pd is None:  # pragma: no cover - optional dependency
            raise ImportError("pandas is required to export edge tables") from _PANDAS_ERROR
        if not self.edges:
            return pd.DataFrame(columns=["source", "target", "weight"])
        sources, targets, weights = zip(*self.iter_edges())
        return pd.DataFrame({"source": sources, "target": targets, "weight": weights})

    def __len__(self) -> int:  # pragma: no cover - trivial proxy
        return len(self.nodes)


__all__ = ["Subgraph"]
