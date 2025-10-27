from random import Random

from connectomics import _matrix
from connectomics.signed_blocks import block_multiply, blocks_from_partition


def _random_matrix(n: int, seed: int) -> list[list[float]]:
    rng = Random(seed)
    return [[rng.gauss(0.0, 0.2) for _ in range(n)] for _ in range(n)]


def _assert_close(left: list[list[float]], right: list[list[float]], tol: float = 1e-9) -> None:
    assert len(left) == len(right)
    for row_l, row_r in zip(left, right):
        assert len(row_l) == len(row_r)
        for a, b in zip(row_l, row_r):
            assert abs(a - b) < tol


def test_block_square_matches_dense():
    n = 12
    matrix = _random_matrix(n, seed=3)
    mask = [True] * 7 + [False] * (n - 7)
    blocks = blocks_from_partition(matrix, mask)
    dense_sq = _matrix.matmul(matrix, matrix)
    block_sq = block_multiply(blocks, blocks)
    ee_idx = [i for i, flag in enumerate(mask) if flag]
    ii_idx = [i for i, flag in enumerate(mask) if not flag]
    ee_dense = [[dense_sq[i][j] for j in ee_idx] for i in ee_idx]
    ei_dense = [[dense_sq[i][j] for j in ii_idx] for i in ee_idx]
    ie_dense = [[dense_sq[i][j] for j in ee_idx] for i in ii_idx]
    ii_dense = [[dense_sq[i][j] for j in ii_idx] for i in ii_idx]
    _assert_close(block_sq["EE"], ee_dense)
    _assert_close(block_sq["EI"], ei_dense)
    _assert_close(block_sq["IE"], ie_dense)
    _assert_close(block_sq["II"], ii_dense)
