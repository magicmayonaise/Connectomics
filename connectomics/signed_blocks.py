"""Block algebra helpers for signed adjacency matrices."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from . import _matrix

Block = Dict[str, _matrix.Matrix]


def _indices_from_mask(mask: Iterable[bool]) -> Tuple[List[int], List[int]]:
    excitatory: List[int] = []
    inhibitory: List[int] = []
    for idx, is_exc in enumerate(mask):
        (excitatory if is_exc else inhibitory).append(idx)
    return excitatory, inhibitory


def _slice(matrix: _matrix.Matrix, rows: Iterable[int], cols: Iterable[int]) -> _matrix.Matrix:
    row_idx = list(rows)
    col_idx = list(cols)
    return [[matrix[i][j] for j in col_idx] for i in row_idx]


def blocks_from_partition(matrix: _matrix.Matrix, mask: Iterable[bool]) -> Block:
    _matrix.ensure_square(matrix)
    excitatory, inhibitory = _indices_from_mask(mask)
    return {
        "EE": _slice(matrix, excitatory, excitatory),
        "EI": _slice(matrix, excitatory, inhibitory),
        "IE": _slice(matrix, inhibitory, excitatory),
        "II": _slice(matrix, inhibitory, inhibitory),
    }


def block_multiply(left: Block, right: Block) -> Block:
    return {
        "EE": _matrix.add(
            _matrix.matmul(left["EE"], right["EE"]),
            _matrix.matmul(left["EI"], right["IE"]),
        ),
        "EI": _matrix.add(
            _matrix.matmul(left["EE"], right["EI"]),
            _matrix.matmul(left["EI"], right["II"]),
        ),
        "IE": _matrix.add(
            _matrix.matmul(left["IE"], right["EE"]),
            _matrix.matmul(left["II"], right["IE"]),
        ),
        "II": _matrix.add(
            _matrix.matmul(left["IE"], right["EI"]),
            _matrix.matmul(left["II"], right["II"]),
        ),
    }
