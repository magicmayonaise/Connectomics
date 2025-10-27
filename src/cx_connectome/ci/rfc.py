<<<<<<< HEAD
"""Contextualized receptive field calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csr


@dataclass(frozen=True)
class _ContributionMatrix:
    """Helper container storing sign-specific matrices.

    Attributes
    ----------
    label:
        Human readable label describing the contribution slice. ``None``
        represents the aggregate (unsigned) contribution.
    matrix:
        Sparse matrix holding the magnitude of the contributions for the
        chosen slice. The matrix is expected to store *non-negative* values
        because normalisation is performed on the magnitude of each column.
    """

    label: str | None
    matrix: csc_matrix


def _prepare_matrix(matrix: csr_matrix) -> csc_matrix:
    """Normalise the provided matrix to a CSC representation."""

    if not isspmatrix_csr(matrix):
        matrix = csr_matrix(matrix)
    return matrix.tocsc()


def _split_by_sign(matrix: csc_matrix, signed: bool) -> List[_ContributionMatrix]:
    """Generate matrices representing the requested sign configuration."""

    if not signed:
        magnitude = matrix.copy()
        magnitude.data = np.abs(magnitude.data)
        return [_ContributionMatrix(label=None, matrix=magnitude)]

    positive = matrix.copy()
    positive.data = np.clip(positive.data, 0.0, None)

    negative = matrix.copy()
    negative.data = -np.clip(negative.data, None, 0.0)

    return [
        _ContributionMatrix(label="positive", matrix=positive),
        _ContributionMatrix(label="negative", matrix=negative),
    ]


def _validated_indices(
    indices: Iterable[int],
    size: int,
    *,
    label: str,
    allow_empty: bool = False,
) -> np.ndarray:
    """Return validated, sorted indices as an ``np.ndarray``.

    Parameters
    ----------
    indices:
        Iterable of indices to validate.
    size:
        Size of the dimension these indices refer to.
    label:
        Human readable label used in the error message.

    Returns
    -------
    numpy.ndarray
        Sorted array of distinct indices.
    """

    index_array = np.fromiter(sorted(set(indices)), dtype=int)
    if index_array.size == 0:
        if allow_empty:
            return index_array
        raise ValueError(f"No {label} were provided.")

    if np.any(index_array < 0) or np.any(index_array >= size):
        raise IndexError(
            f"{label.capitalize()} {index_array[(index_array < 0) | (index_array >= size)]} "
            f"fall outside the permissible range [0, {size})."
        )

    return index_array


def _column_normalised_share(
    matrix: csc_matrix,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
) -> np.ndarray:
    """Calculate the column normalised share for ``row_indices``.

    The routine computes the sum of the requested rows for every column and
    divides it by the total magnitude present in the column. Columns without
    any magnitude are assigned a share of zero to avoid division errors.
    """

    if column_indices.size == 0:
        return np.zeros(0, dtype=float)

    # Aggregate the magnitude contributed by the chosen set of rows.
    subset = matrix[row_indices][:, column_indices]
    numerator = np.asarray(subset.sum(axis=0)).ravel()

    # Total magnitude present in each requested column.
    column_slice = matrix[:, column_indices]
    denominator = np.asarray(column_slice.sum(axis=0)).ravel()

    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)

    return share


def compute_rfc(
    eff_per_k: Dict[int, csr_matrix],
    source_ids: Iterable[int],
    target_ids: Iterable[int] | str,
    *,
    signed: bool = False,
) -> pd.DataFrame:
    """Compute contextualised receptive field (RFc) summaries.

    Parameters
    ----------
    eff_per_k:
        Mapping from hop distance ``k`` to a sparse effective connectivity
        matrix. The matrices are assumed to use the same node ordering.
    source_ids:
        Collection of indices representing the source neurons whose influence
        should be quantified.
    target_ids:
        Collection of indices representing the targets that receive the
        influence. The special string ``"all"`` considers every node that is in
        scope for the matrix.
    signed:
        If ``True`` the inflow is decomposed into positive and negative
        components. Otherwise the calculation is performed on the magnitude of
        the effective weights.

    Returns
    -------
    pandas.DataFrame
        Long-form table containing one row per ``(k, target_id)`` pair and an
        optional ``sign`` column. The ``share`` column reports the aggregated
        column-normalised inflow from ``source_ids`` into ``target_ids`` for the
        corresponding hop distance.
    """

    if not eff_per_k:
        raise ValueError("eff_per_k must not be empty.")

    records: List[dict] = []

    for hop in sorted(eff_per_k):
        matrix = _prepare_matrix(eff_per_k[hop])
        n_rows, n_cols = matrix.shape

        source_index = _validated_indices(source_ids, n_rows, label="source indices")

        if target_ids == "all":
            target_index = np.arange(n_cols, dtype=int)
        else:
            target_index = _validated_indices(
                target_ids, n_cols, label="target indices", allow_empty=True
            )

        for contribution in _split_by_sign(matrix, signed):
            share = _column_normalised_share(contribution.matrix, source_index, target_index)

            for tgt, value in zip(target_index, share, strict=False):
                record = {"k": hop, "target_id": int(tgt), "share": float(value)}
                if contribution.label is not None:
                    record["sign"] = contribution.label
                records.append(record)

    columns: Sequence[str] = ["k", "target_id", "share"]
    if signed:
        columns = ["k", "target_id", "sign", "share"]

    return pd.DataFrame.from_records(records, columns=columns)


__all__ = ["compute_rfc"]
=======
"""Typed stubs for receptive field clustering (RFC)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

__all__ = ["RFCEstimator"]


class RFCEstimator(Protocol):
    """Protocol describing RFC estimation."""

    def fit(self, inputs: Sequence[Any], responses: Sequence[Any]) -> Any:
        """Fit RFC parameters to the provided dataset."""
        ...

    def transform(self, inputs: Sequence[Any]) -> Any:
        """Transform inputs into RFC feature space."""
        ...
>>>>>>> 8f0f588 (Add CI scaffolding and configuration)
