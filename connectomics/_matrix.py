"""Light-weight matrix helpers that avoid numpy."""

from __future__ import annotations

from typing import Iterable, List, Sequence

Matrix = List[List[float]]


def ensure_square(matrix: Sequence[Sequence[float]]) -> int:
    n = len(matrix)
    if n == 0:
        raise ValueError("matrix must be non-empty")
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be square")
    return n


def clone(matrix: Sequence[Sequence[float]]) -> Matrix:
    return [list(row) for row in matrix]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def identity(n: int) -> Matrix:
    eye = zeros(n, n)
    for i in range(n):
        eye[i][i] = 1.0
    return eye


def add(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    if len(left) != len(right) or len(left[0]) != len(right[0]):
        raise ValueError("matrix dimensions do not match")
    out = zeros(len(left), len(left[0]))
    for i, (l_row, r_row) in enumerate(zip(left, right)):
        for j, (l_val, r_val) in enumerate(zip(l_row, r_row)):
            out[i][j] = l_val + r_val
    return out


def matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    left_rows = len(left)
    inner = len(left[0])
    if any(len(row) != inner for row in left):
        raise ValueError("left matrix is ragged")
    if any(len(row) != len(right[0]) for row in right):
        raise ValueError("right matrix is ragged")
    if inner != len(right):
        raise ValueError("matrix dimensions do not align")
    cols = len(right[0])
    out = zeros(left_rows, cols)
    for i, left_row in enumerate(left):
        out_row = out[i]
        for k, left_value in enumerate(left_row):
            if left_value == 0.0:
                continue
            right_row = right[k]
            for j, right_value in enumerate(right_row):
                out_row[j] += left_value * right_value
    return out


def chunked_matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]], chunk_size: int) -> Matrix:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    left_rows = len(left)
    cols = len(right[0])
    out = zeros(left_rows, cols)
    for start in range(0, left_rows, chunk_size):
        end = min(start + chunk_size, left_rows)
        for i in range(start, end):
            left_row = left[i]
            out_row = out[i]
            for k, left_value in enumerate(left_row):
                if left_value == 0.0:
                    continue
                right_row = right[k]
                for j, right_value in enumerate(right_row):
                    out_row[j] += left_value * right_value
    return out


def matrix_power(matrix: Sequence[Sequence[float]], exponent: int) -> Matrix:
    if exponent < 0:
        raise ValueError("exponent must be non-negative")
    n = ensure_square(matrix)
    if exponent == 0:
        return identity(n)
    result = identity(n)
    base = clone(matrix)
    exp = exponent
    while exp > 0:
        if exp & 1:
            result = matmul(result, base)
        base = matmul(base, base)
        exp >>= 1
    return result


def row_sums(matrix: Sequence[Sequence[float]], indices: Iterable[int]) -> List[float]:
    sums = [0.0 for _ in range(len(matrix))]
    index_set = list(indices)
    for idx in index_set:
        row = matrix[idx]
        for j, value in enumerate(row):
            sums[j] += value
    return sums


def column_sums(matrix: Sequence[Sequence[float]], indices: Iterable[int]) -> List[float]:
    sums = [0.0 for _ in range(len(matrix))]
    index_list = list(indices)
    for i, row in enumerate(matrix):
        total = 0.0
        for j in index_list:
            total += row[j]
        sums[i] = total
    return sums
