"""Utilities for path enumeration in directed graphs."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set


def _build_reverse(graph: Mapping[int, Sequence[int]]) -> Dict[int, List[int]]:
    reverse: Dict[int, List[int]] = {node: [] for node in graph}
    for src, dsts in graph.items():
        for dst in dsts:
            reverse.setdefault(dst, []).append(src)
            reverse.setdefault(src, [])
    return reverse


def _bfs_layers(graph: Mapping[int, Sequence[int]], starts: Iterable[int], depth: int) -> Dict[int, Set[int]]:
    layers: Dict[int, Set[int]] = {0: set(starts)}
    frontier = deque((node, 0) for node in starts)
    while frontier:
        node, dist = frontier.popleft()
        if dist == depth:
            continue
        for nxt in graph.get(node, ()):  # type: ignore[arg-type]
            if nxt not in layers.setdefault(dist + 1, set()):
                layers[dist + 1].add(nxt)
                frontier.append((nxt, dist + 1))
    return layers


def meet_in_the_middle(
    graph: Mapping[int, Sequence[int]],
    sources: Iterable[int],
    targets: Iterable[int],
    *,
    max_depth: int,
) -> MutableMapping[str, object]:
    """Compute nodes participating in source→target paths of length ≤ ``max_depth``."""

    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    sources = set(sources)
    targets = set(targets)
    forward_layers = _bfs_layers(graph, sources, max_depth)
    reverse_graph = _build_reverse(graph)
    backward_layers = _bfs_layers(reverse_graph, targets, max_depth)

    within: Set[int] = set()
    forward_dists: Dict[int, int] = {}
    backward_dists: Dict[int, int] = {}
    for depth, nodes in forward_layers.items():
        for node in nodes:
            forward_dists.setdefault(node, depth)
    for depth, nodes in backward_layers.items():
        for node in nodes:
            backward_dists.setdefault(node, depth)
    for node in set(forward_dists) | set(backward_dists):
        f = forward_dists.get(node)
        b = backward_dists.get(node)
        if f is not None and b is not None and f + b <= max_depth:
            within.add(node)
    return {
        "forward_layers": forward_layers,
        "backward_layers": backward_layers,
        "within_k": within,
    }
