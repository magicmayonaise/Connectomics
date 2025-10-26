"""Utilities for computing signed connectivity blocks.

This module implements helpers that expose the functionality needed for the
"signed" mode of the Connectivity Interpreter (CI) workflows used in the
Connectome Interpreter project.  The implementation is intentionally written to
be lightweight and to minimise external dependencies so that unit tests can
exercise the logic without depending on large production datasets.

The public API focuses on three main tasks:

``classify_neuron_signs``
    Consume baseline chemistry/annotation tables and derive a
    :class:`SignMasks` object containing boolean arrays that flag excitatory,
    inhibitory, and unknown neurons.

``compute_signed_blocks``
    Given an adjacency matrix and a :class:`SignMasks` object, compute the
    excitatory/inhibitory block sub-matrices for a sequence of hop counts
    (matrix powers).  The routine understands how to operate on large sparse
    matrices by chunking the requested rows which mirrors the strategy used in
    the Interpreter paper (Fig. 1G–L).

``combine_blocks``
    Reassemble block matrices into a single signed matrix or return the blocks
    unchanged depending on the requested ``mode``.

The concrete schema of the chemistry/annotation tables varies across material-
isations.  For the purposes of the tests we support a small but expressive set
of heuristics: neurotransmitter annotations containing ``"ach"`` are treated as
excitatory, entries with ``"gaba"`` are inhibitory, while glutamatergic labels
are flagged as *unknown* so that the caller can decide how to handle them.  Any
conflicting annotation is marked as unknown as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "SignMasks",
    "classify_neuron_signs",
    "compute_signed_blocks",
    "combine_blocks",
]


_EXCITATORY_MARKERS = {
    "ach",
    "acetylcholine",
    "cholinergic",
}

_INHIBITORY_MARKERS = {
    "gaba",
    "gabaergic",
    "gamma-aminobutyric acid",
}

_UNKNOWN_MARKERS = {
    "glu",
    "glutamate",
    "glutamatergic",
}


def _normalise_string(value: object) -> str:
    """Normalise user-provided annotations for comparison.

    Parameters
    ----------
    value:
        Arbitrary value retrieved from chemistry/annotation tables.  ``None``
        is treated as the empty string.

    Returns
    -------
    str
        Lower-case, stripped representation.
    """

    if value is None:
        return ""
    text = str(value).strip().lower()
    return text


@dataclass
class SignMasks:
    """Boolean masks describing excitatory/inhibitory cell assignments.

    Attributes
    ----------
    excitatory:
        Boolean array indicating which neurons are treated as excitatory.
    inhibitory:
        Boolean array indicating inhibitory neurons.
    unknown:
        Boolean array flagging neurons that could not be classified (this
        includes cases where conflicting annotations were encountered).
    index:
        Sequence of neuron identifiers matching the order of the masks.
    """

    excitatory: np.ndarray
    inhibitory: np.ndarray
    unknown: np.ndarray
    index: Sequence[int]

    def __post_init__(self) -> None:
        if not (
            len(self.excitatory) == len(self.inhibitory) == len(self.unknown)
        ):
            raise ValueError("SignMasks arrays must share the same length")

    def subset(self, keep: np.ndarray) -> "SignMasks":
        """Return a new :class:`SignMasks` restricted to a subset.

        Parameters
        ----------
        keep:
            Boolean mask specifying which entries to retain.
        """

        return SignMasks(
            excitatory=self.excitatory[keep],
            inhibitory=self.inhibitory[keep],
            unknown=self.unknown[keep],
            index=[self.index[i] for i, flag in enumerate(keep) if flag],
        )


def _resolve_annotation(label: str) -> Optional[str]:
    """Resolve a normalised annotation into ``"E"``, ``"I"`` or ``None``."""

    if not label:
        return None
    if label in _EXCITATORY_MARKERS:
        return "E"
    if label in _INHIBITORY_MARKERS:
        return "I"
    if label in _UNKNOWN_MARKERS:
        return None
    # Fuzzy matching: handle partial matches such as "strongly cholinergic".
    if any(marker in label for marker in _EXCITATORY_MARKERS):
        return "E"
    if any(marker in label for marker in _INHIBITORY_MARKERS):
        return "I"
    if any(marker in label for marker in _UNKNOWN_MARKERS):
        return None
    return None


def _coerce_series(data: Mapping[str, pd.Series], column: str) -> Optional[pd.Series]:
    """Safely extract a column from a mapping of :class:`pandas.Series`.

    Many tables in production are represented as a mapping of column names to
    series objects.  Tests may pass either a DataFrame or a mapping.  This
    helper normalises the access pattern.
    """

    if not data:
        return None
    try:
        series = data[column]
    except KeyError:
        return None
    if isinstance(series, pd.Series):
        return series
    return None


def classify_neuron_signs(
    node_ids: Sequence[int],
    chemistry: pd.DataFrame | Mapping[str, pd.Series] | None,
    annotations: pd.DataFrame | Mapping[str, pd.Series] | None = None,
    *,
    id_column: str = "root_id",
    neurotransmitter_column: str | Sequence[str] = "neurotransmitter",
    annotation_column: str | Sequence[str] = "annotation",
) -> SignMasks:
    """Classify neurons into excitatory/inhibitory masks.

    Parameters
    ----------
    node_ids:
        Sequence describing the order of neurons in the adjacency matrix.
    chemistry:
        Baseline chemistry table.  Either a :class:`pandas.DataFrame` or a
        mapping of column names to :class:`pandas.Series`.
    annotations:
        Additional annotations used to reinforce the classification.
    id_column:
        Name of the column that stores neuron/root identifiers.
    neurotransmitter_column:
        Column(s) containing neurotransmitter annotations.
    annotation_column:
        Column(s) containing free-text annotations.
    """

    def _iter_records(table: Optional[pd.DataFrame | Mapping[str, pd.Series]]):
        if table is None:
            return []
        if isinstance(table, pd.DataFrame):
            return table.to_dict("records")
        if isinstance(table, Mapping):
            index_series = _coerce_series(table, id_column)
            if index_series is None:
                return []
            columns = {name: series for name, series in table.items()}
            length = len(index_series)
            records = []
            for idx in range(length):
                entry = {id_column: index_series.iat[idx]}
                for key, series in columns.items():
                    if len(series) == length:
                        entry[key] = series.iat[idx]
                records.append(entry)
            return records
        raise TypeError("Unsupported table type")

    nt_columns: Sequence[str]
    if isinstance(neurotransmitter_column, str):
        nt_columns = (neurotransmitter_column,)
    else:
        nt_columns = tuple(neurotransmitter_column)

    annotation_columns: Sequence[str]
    if isinstance(annotation_column, str):
        annotation_columns = (annotation_column,)
    else:
        annotation_columns = tuple(annotation_column)

    assignments: MutableMapping[int, str] = {}

    def _register(root_id: int, label: Optional[str]) -> None:
        if label is None:
            return
        previous = assignments.get(root_id)
        if previous is None:
            assignments[root_id] = label
            return
        if previous != label:
            assignments[root_id] = "U"

    for record in _iter_records(chemistry):
        root_id = record.get(id_column)
        if pd.isna(root_id):
            continue
        try:
            root_id = int(root_id)
        except (TypeError, ValueError):
            continue
        for column in nt_columns:
            if column not in record:
                continue
            label = _resolve_annotation(_normalise_string(record[column]))
            if label is not None:
                _register(root_id, label)

    for record in _iter_records(annotations):
        root_id = record.get(id_column)
        if pd.isna(root_id):
            continue
        try:
            root_id = int(root_id)
        except (TypeError, ValueError):
            continue
        for column in annotation_columns:
            if column not in record:
                continue
            label = _resolve_annotation(_normalise_string(record[column]))
            if label is not None:
                _register(root_id, label)

    excitatory = np.zeros(len(node_ids), dtype=bool)
    inhibitory = np.zeros(len(node_ids), dtype=bool)
    unknown = np.zeros(len(node_ids), dtype=bool)

    for idx, root_id in enumerate(node_ids):
        label = assignments.get(int(root_id))
        if label == "E":
            excitatory[idx] = True
        elif label == "I":
            inhibitory[idx] = True
        else:
            unknown[idx] = True

    return SignMasks(
        excitatory=excitatory,
        inhibitory=inhibitory,
        unknown=unknown,
        index=list(node_ids),
    )


def _filter_unknowns(matrix: sp.spmatrix, masks: SignMasks) -> tuple[sp.spmatrix, SignMasks]:
    """Remove rows/columns corresponding to unknown neurons."""

    keep = ~(masks.unknown)
    if keep.all():
        return matrix, masks
    filtered_matrix = matrix[keep][:, keep]
    filtered_masks = masks.subset(keep)
    return filtered_matrix, filtered_masks


def _slice_block(
    matrix: sp.spmatrix,
    row_mask: np.ndarray,
    col_mask: np.ndarray,
    *,
    chunk_size: Optional[int] = None,
) -> sp.spmatrix:
    """Extract a block from ``matrix`` using optional chunking."""

    if chunk_size is None or len(row_mask) <= chunk_size:
        return matrix[row_mask][:, col_mask]

    indices = np.flatnonzero(row_mask)
    blocks = []
    for start in range(0, len(indices), chunk_size):
        batch = indices[start : start + chunk_size]
        blocks.append(matrix[batch][:, col_mask])
    return sp.vstack(blocks, format="csr")


def compute_signed_blocks(
    adjacency: sp.spmatrix,
    masks: SignMasks,
    ks: Iterable[int],
    *,
    chunk_size: Optional[int] = None,
    copy: bool = True,
) -> dict[int, dict[str, sp.spmatrix]]:
    """Compute signed adjacency blocks for each power in ``ks``.

    Parameters
    ----------
    adjacency:
        Square sparse adjacency matrix describing synaptic counts.
    masks:
        :class:`SignMasks` describing the sign of each neuron.
    ks:
        Iterable of positive integers describing the desired hop distances.
    chunk_size:
        Optional chunk size used when extracting row blocks.  ``None`` (the
        default) disables chunking which is adequate for the small matrices
        exercised in unit tests.  The logic mirrors the production pipeline by
        iterating over row chunks and stacking the resulting sparse matrices.
    copy:
        If ``True`` the input matrix is copied before computing successive
        powers.  Disable to operate in-place when the caller can guarantee the
        matrix will not be reused elsewhere.
    """

    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency matrix must be square")

    matrix = adjacency.copy() if copy else adjacency
    matrix = matrix.tocsr()
    matrix, filtered_masks = _filter_unknowns(matrix, masks)

    if not (filtered_masks.excitatory.any() or filtered_masks.inhibitory.any()):
        raise ValueError("No excitatory or inhibitory neurons remain after filtering")

    ordered_ks = sorted({int(k) for k in ks if int(k) >= 1})
    if not ordered_ks:
        raise ValueError("At least one positive hop length is required")

    results: dict[int, dict[str, sp.spmatrix]] = {}

    current_power = None
    for power in range(1, ordered_ks[-1] + 1):
        if power == 1:
            current_power = matrix
        else:
            current_power = (current_power @ matrix).tocsr()  # type: ignore[operator]

        if power in ordered_ks:
            ee = _slice_block(
                current_power,
                filtered_masks.excitatory,
                filtered_masks.excitatory,
                chunk_size=chunk_size,
            )
            ei = _slice_block(
                current_power,
                filtered_masks.excitatory,
                filtered_masks.inhibitory,
                chunk_size=chunk_size,
            )
            ie = _slice_block(
                current_power,
                filtered_masks.inhibitory,
                filtered_masks.excitatory,
                chunk_size=chunk_size,
            )
            ii = _slice_block(
                current_power,
                filtered_masks.inhibitory,
                filtered_masks.inhibitory,
                chunk_size=chunk_size,
            )
            results[power] = {"ee": ee, "ei": ei, "ie": ie, "ii": ii}

    return results


def combine_blocks(
    ee: sp.spmatrix,
    ei: sp.spmatrix,
    ie: sp.spmatrix,
    ii: sp.spmatrix,
    *,
    mode: str = "net",
) -> sp.spmatrix | dict[str, sp.spmatrix]:
    """Combine block matrices using the requested ``mode``.

    Parameters
    ----------
    ee, ei, ie, ii:
        Block matrices corresponding to the E→E, E→I, I→E and I→I projections.
    mode:
        ``"net"`` (default) returns a single signed matrix where inhibitory rows
        contribute negative values.  ``"blocks"`` returns the original block
        matrices grouped in a dictionary for callers that prefer to operate on
        them individually.
    """

    mode = mode.lower()
    if mode == "blocks":
        return {"ee": ee, "ei": ei, "ie": ie, "ii": ii}
    if mode != "net":
        raise ValueError(f"Unsupported combination mode: {mode}")

    top = sp.hstack([ee, ei], format="csr")
    bottom = sp.hstack([-ie, -ii], format="csr")
    return sp.vstack([top, bottom], format="csr")

