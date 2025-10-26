"""Graph search utilities for identifying compact connectome subcircuits."""
from __future__ import annotations

from collections import deque
from numbers import Integral
from typing import Deque, Dict, Iterable, Mapping, Set, Tuple

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
    _PANDAS_ERROR = exc
else:  # pragma: no cover - optional dependency
    _PANDAS_ERROR = None

try:  # pragma: no cover - optional dependency
    from scipy.sparse import csr_matrix
except ImportError as exc:  # pragma: no cover - optional dependency
    csr_matrix = None  # type: ignore
    _SCIPY_ERROR = exc
else:  # pragma: no cover - optional dependency
    _SCIPY_ERROR = None

from .subgraph import Subgraph

Edge = Tuple[int, int]


def _build_adjacency(edges_df, *, source_col: str, target_col: str) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]], Set[Edge]]:
    """Return forward/backward adjacency maps and the unique edge set."""

    forward: Dict[int, Set[int]] = {}
    backward: Dict[int, Set[int]] = {}
    edges: Set[Edge] = set()

    for row in edges_df.itertuples(index=False):
        source = getattr(row, source_col)
        target = getattr(row, target_col)
        if not isinstance(source, Integral) or not isinstance(target, Integral):
            raise TypeError("edge endpoints must be integer identifiers")
        source = int(source)
        target = int(target)
        edges.add((source, target))
        forward.setdefault(source, set()).add(target)
        backward.setdefault(target, set()).add(source)

    return forward, backward, edges


def _multi_source_bfs(adjacency: Mapping[int, Set[int]], start: Iterable[int], k_max: int) -> Dict[int, int]:
    """Breadth-first search from multiple starting nodes up to ``k_max`` steps."""

    visited: Dict[int, int] = {}
    frontier: Deque[Tuple[int, int]] = deque()

    for node in start:
        node = int(node)
        visited[node] = 0
        frontier.append((node, 0))

    while frontier:
        node, depth = frontier.popleft()
        if depth >= k_max:
            continue
        for neighbour in adjacency.get(node, ()):  # type: ignore[arg-type]
            if neighbour not in visited:
                visited[neighbour] = depth + 1
                frontier.append((neighbour, depth + 1))

    return visited


def meet_in_middle(
    edges_df,
    sources: Set[int],
    targets: Set[int],
    k_max: int,
    *,
    source_col: str = "source",
    target_col: str = "target",
) -> Dict[str, Set]:
    """Compute nodes and edges that participate in any path up to length ``k_max``.

    The function performs a bi-directional breadth-first search. The forward
    and backward passes produce distance tables that are then intersected to
    recover the minimal set of nodes/edges that lie on a path from any source
    to any target whose total length does not exceed ``k_max``.

    Parameters
    ----------
    edges_df:
        Edge list containing at least two integer columns that identify the
        source and target nodes of each connection.
    sources / targets:
        Sets of integer node identifiers defining the entry and exit points
        of the search.
    k_max:
        Maximum hop length to consider.
    source_col / target_col:
        Column names in ``edges_df`` that contain the source and target node
        identifiers.

    Returns
    -------
    dict
        Dictionary with ``"nodes"`` and ``"edges"`` entries containing the
        nodes involved in qualifying paths and the edges connecting them.
    """

    if pd is None:  # pragma: no cover - optional dependency
        raise ImportError("pandas is required for meet_in_middle") from _PANDAS_ERROR
    if k_max < 0:
        raise ValueError("k_max must be non-negative")
    if not sources or not targets:
        return {"nodes": set(), "edges": set()}

    forward_adj, backward_adj, all_edges = _build_adjacency(
        edges_df, source_col=source_col, target_col=target_col
    )

    dist_from_sources = _multi_source_bfs(forward_adj, sources, k_max)
    dist_to_targets = _multi_source_bfs(backward_adj, targets, k_max)

    qualifying_nodes: Set[int] = set()
    qualifying_edges: Set[Edge] = set()

    for source in sources:
        if source in dist_from_sources and dist_from_sources[source] <= k_max:
            qualifying_nodes.add(source)
    for target in targets:
        if target in dist_to_targets and dist_to_targets[target] <= k_max:
            qualifying_nodes.add(target)

    for u, v in all_edges:
        if u not in dist_from_sources or v not in dist_to_targets:
            continue
        total_length = dist_from_sources[u] + 1 + dist_to_targets[v]
        if total_length <= k_max:
            qualifying_nodes.add(u)
            qualifying_nodes.add(v)
            qualifying_edges.add((u, v))

    # Include intermediate meeting nodes that satisfy the distance constraint.
    for node, forward_depth in dist_from_sources.items():
        if node in dist_to_targets:
            if forward_depth + dist_to_targets[node] <= k_max:
                qualifying_nodes.add(node)

    return {"nodes": qualifying_nodes, "edges": qualifying_edges}


def slice_from_effective(
    eff_k,
    sources: Set[int],
    targets: Set[int],
    thr: float = 0.01,
) -> Subgraph:
    """Extract a compact subgraph from an effective connectivity matrix.

    The effective connectivity matrix ``eff_k`` is assumed to encode
    influence scores where larger values indicate stronger relationships.
    Nodes that both receive strong input from any source and project strongly
    to any target (according to ``thr``) are retained. The resulting
    subgraph only contains edges whose weights also satisfy the threshold.
    """

    if csr_matrix is None:  # pragma: no cover - optional dependency
        raise ImportError("scipy is required for slice_from_effective") from _SCIPY_ERROR
    if thr < 0:
        raise ValueError("thr must be non-negative")
    if eff_k.shape[0] != eff_k.shape[1]:
        raise ValueError("eff_k must be a square matrix")
    if not sources or not targets:
        return Subgraph()

    eff_k = eff_k.tocsr()
    eff_k.eliminate_zeros()

    n_nodes = eff_k.shape[0]

    def _validate(nodes: Iterable[int]) -> None:
        for node in nodes:
            if node < 0 or node >= n_nodes:
                raise IndexError(f"node id {node} is outside the matrix bounds")

    _validate(sources)
    _validate(targets)

    strong_from_sources: Set[int] = set()
    for source in sources:
        row = eff_k.getrow(source)
        mask = row.data >= thr
        strong_from_sources.update(int(idx) for idx in row.indices[mask])

    strong_to_targets: Set[int] = set()
    eff_t = eff_k.transpose().tocsr()
    for target in targets:
        row = eff_t.getrow(target)
        mask = row.data >= thr
        strong_to_targets.update(int(idx) for idx in row.indices[mask])

    candidate_nodes = (strong_from_sources & strong_to_targets) | set(sources) | set(targets)
    if not candidate_nodes:
        return Subgraph(nodes=set(sources) | set(targets))

    subgraph = Subgraph(nodes=set(candidate_nodes))

    candidate_lookup = set(candidate_nodes)
    for source in sorted(candidate_nodes):
        row = eff_k.getrow(source)
        for target, weight in zip(row.indices, row.data):
            target = int(target)
            weight = float(weight)
            if target in candidate_lookup and weight >= thr:
                subgraph.add_edge(int(source), target, weight)

    return subgraph


__all__ = ["meet_in_middle", "slice_from_effective"]
