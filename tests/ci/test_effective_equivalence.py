from random import Random

from connectomics.effective import chunked_matrix_power, dense_matrix_power


def _generate_sparse_matrix(n: int, sparsity: float, seed: int) -> list[list[float]]:
    rng = Random(seed)
    matrix = []
    for _ in range(n):
        row = []
        for _ in range(n):
            if rng.random() < sparsity:
                row.append(rng.random())
            else:
                row.append(0.0)
        row_sum = sum(row)
        if row_sum:
            row = [value / row_sum for value in row]
        matrix.append(row)
    return matrix


def _max_difference(left: list[list[float]], right: list[list[float]]) -> float:
    diff = 0.0
    for row_left, row_right in zip(left, right):
        for a, b in zip(row_left, row_right):
            diff = max(diff, abs(a - b))
    return diff


def test_chunked_matches_dense_for_sparse_graph():
    n = 200
    matrix = _generate_sparse_matrix(n, sparsity=0.05, seed=2)
    exponent = 4
    dense = dense_matrix_power(matrix, exponent)
    chunked = chunked_matrix_power(matrix, exponent, chunk_size=17)
    assert _max_difference(chunked, dense) < 1e-6
