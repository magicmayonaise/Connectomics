"""Effective connectivity computation utilities.

This module mirrors the logic used by the Interpreter tooling to compute
higher-order effective connectivity matrices. The primary entry-point is
:func:`effective_k`, which repeatedly multiplies a sparse adjacency matrix by
itself while chunking the dense intermediate representation by column. The
chunked strategy keeps the intermediate dense matrices small enough to fit in
memory for very large graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

__all__ = ["effective_k", "compute_suite"]


@dataclass(slots=True)
class _ChunkSpec:
    """Metadata describing the location of a dense column chunk."""

    start: int
    stop: int


def _normalize_sparse_columns(matrix: csr_matrix, eps: float) -> csr_matrix:
    """Column-normalise a sparse matrix.

    Parameters
    ----------
    matrix:
        Input matrix in CSR format.
    eps:
        Minimum magnitude required for a column to be considered non-empty.

    Returns
    -------
    csr_matrix
        Column-normalised matrix in CSR format.
    """

    if matrix.nnz == 0:
        return matrix.copy()

    csc = matrix.tocsc(copy=True)
    column_sums = np.asarray(csc.sum(axis=0)).ravel()
    scale = np.zeros_like(column_sums, dtype=csc.dtype)
    valid = np.abs(column_sums) > eps
    if np.any(valid):
        scale[valid] = (1.0 / column_sums[valid]).astype(csc.dtype, copy=False)

    for col in range(csc.shape[1]):
        start, stop = csc.indptr[col], csc.indptr[col + 1]
        if start == stop:
            continue
        factor = scale[col]
        if factor != 0:
            csc.data[start:stop] *= factor
        else:
            csc.data[start:stop] = 0

    csc.eliminate_zeros()
    return csc.tocsr()


def _normalize_dense_columns(block: np.ndarray, eps: float) -> np.ndarray:
    """Normalise the columns of a dense block in-place."""

    if block.size == 0:
        return block

    column_sums = block.sum(axis=0)
    scale = np.zeros_like(column_sums, dtype=block.dtype)
    valid = np.abs(column_sums) > eps
    if np.any(valid):
        scale[valid] = (1.0 / column_sums[valid]).astype(block.dtype, copy=False)
        block *= scale
    else:
        block[...] = 0
    return block


def _column_chunks(num_cols: int, chunk_size: int) -> Iterable[_ChunkSpec]:
    """Yield column chunk boundaries for the given matrix width."""

    for start in range(0, num_cols, chunk_size):
        yield _ChunkSpec(start=start, stop=min(start + chunk_size, num_cols))


def _prune_inplace(matrix: csr_matrix, eps: float) -> None:
    """Drop entries whose magnitude is below ``eps``."""

    if matrix.nnz == 0:
        return
    mask = np.abs(matrix.data) < eps
    if np.any(mask):
        matrix.data[mask] = 0
        matrix.eliminate_zeros()


def effective_k(
    adj: csr_matrix,
    k: int,
    *,
    chunk_size_cols: int = 4096,
    normalize: str | None = "post_total",
    eps: float = 1e-12,
    dtype: np.dtype | type[np.floating] = np.float32,
) -> csr_matrix:
    """Compute the effective connectivity matrix ``A^k``.

    Parameters
    ----------
    adj:
        Input adjacency matrix in CSR format.
    k:
        Target power. Must be greater than zero.
    chunk_size_cols:
        Number of columns to materialise at once during the sparse×dense
        multiplication.
    normalize:
        Column normalisation strategy. Only ``"post_total"`` is supported.
        Set to ``None`` to disable normalisation.
    eps:
        Values below this threshold are discarded when re-sparsifying results.
    dtype:
        Floating point dtype used for intermediate dense blocks.
    """

    if k < 1:
        raise ValueError("k must be >= 1")
    if chunk_size_cols < 1:
        raise ValueError("chunk_size_cols must be >= 1")
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency matrix must be square")

    if not sparse.isspmatrix_csr(adj):
        adj = adj.tocsr()

    dtype = np.dtype(dtype)
    base = adj.astype(dtype, copy=True)

    if normalize not in {None, "post_total"}:
        raise ValueError(f"Unsupported normalization mode: {normalize}")

    if normalize == "post_total":
        base = _normalize_sparse_columns(base, eps)

    result = base.copy()
    if k == 1:
        _prune_inplace(result, eps)
        return result

    for _ in range(2, k + 1):
        chunk_results: list[csr_matrix] = []
        for chunk in _column_chunks(result.shape[1], chunk_size_cols):
            dense_block = result[:, chunk.start : chunk.stop].toarray()
            if dense_block.size == 0:
                chunk_results.append(
                    sparse.csr_matrix((base.shape[0], chunk.stop - chunk.start), dtype=dtype)
                )
                continue

            dense_product = base.dot(dense_block)
            if normalize == "post_total":
                dense_product = _normalize_dense_columns(dense_product, eps)

            block_sparse = sparse.csr_matrix(dense_product, dtype=dtype)
            _prune_inplace(block_sparse, eps)
            chunk_results.append(block_sparse)

        result = sparse.hstack(chunk_results, format="csr", dtype=dtype)
        _prune_inplace(result, eps)

    return result


def compute_suite(
    adj: csr_matrix,
    ks: Sequence[int],
    **kwargs,
) -> Dict[int, csr_matrix]:
    """Compute a suite of effective connectivity matrices.

    Parameters
    ----------
    adj:
        Input adjacency matrix in CSR format.
    ks:
        Iterable of powers to compute.

    Returns
    -------
    dict[int, csr_matrix]
        Mapping from ``k`` to the corresponding effective matrix.
    """

    unique_ks = sorted({int(k) for k in ks})
    return {k: effective_k(adj, k, **kwargs) for k in unique_ks}

