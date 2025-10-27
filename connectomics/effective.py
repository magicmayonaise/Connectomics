"""Routines for working with effective connectivity matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, MutableMapping, Optional, Sequence, Tuple

from . import _matrix

Matrix = _matrix.Matrix


@dataclass(frozen=True)
class EffectiveSlice:
    """Summary of a slice extracted from an effective connectivity matrix."""

    selected: Tuple[int, ...]
    scores: Tuple[float, ...]

    def to_mapping(self) -> MutableMapping[int, float]:
        return {idx: score for idx, score in zip(self.selected, self.scores)}


def dense_matrix_power(matrix: Sequence[Sequence[float]], exponent: int) -> Matrix:
    return _matrix.matrix_power(matrix, exponent)


def chunked_matrix_power(
    matrix: Sequence[Sequence[float]], exponent: int, *, chunk_size: int = 64
) -> Matrix:
    if exponent < 0:
        raise ValueError("Exponent must be non-negative.")
    _matrix.ensure_square(matrix)
    if exponent == 0:
        return _matrix.identity(len(matrix))
    result = _matrix.identity(len(matrix))
    base = _matrix.clone(matrix)
    for _ in range(exponent):
        result = _matrix.chunked_matmul(result, base, chunk_size)
    return result


def slice_from_effective(
    effective: Sequence[Sequence[float]],
    sources: Sequence[int],
    targets: Sequence[int],
    *,
    min_score: float = 0.0,
    drop: Optional[Iterable[int]] = None,
    top_k: Optional[int] = None,
) -> EffectiveSlice:
    _matrix.ensure_square(effective)
    if not sources or not targets:
        raise ValueError("sources and targets must be non-empty")
    forward_sum = _matrix.row_sums(effective, sorted(set(sources)))
    backward_sum = _matrix.column_sums(effective, sorted(set(targets)))
    scores = [f * b for f, b in zip(forward_sum, backward_sum)]
    blocked = set(drop or ()) | set(sources) | set(targets)
    candidates: List[Tuple[int, float]] = []
    for idx, score in enumerate(scores):
        if idx in blocked:
            continue
        if score > min_score:
            candidates.append((idx, score))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        candidates = candidates[:top_k]
    selected, weights = zip(*candidates) if candidates else ((), ())
    return EffectiveSlice(tuple(selected), tuple(float(w) for w in weights))
