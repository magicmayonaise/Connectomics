"""Utilities for building CI adjacency matrices.

This module provides helpers for extracting scoped ID sets from the
materialization tables, loading the corresponding edges, and converting
those edges into sparse adjacency matrices.  The utilities are written in a
way that keeps the CI pipeline independent from any one-off tables that may
be used elsewhere in the project.
"""

from __future__ import annotations

from pathlib import Path
import json
import pickle
from typing import Dict, Iterable, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


DEFAULT_BASELINE_TABLE_CANDIDATES: Tuple[str, ...] = (
    "ci_baseline_annotations",
    "baseline_cell_annotations",
    "baseline_annotations",
)
DEFAULT_SYNAPSE_TABLE_CANDIDATES: Tuple[str, ...] = (
    "ci_synapses",
    "synapses_pni_2",
    "synapses",
)


def _normalize_iterable(value: object) -> Iterable[int]:
    """Return an iterator of integer IDs from ``value``.

    The baseline annotation tables occasionally store stream IDs either as
    scalar values or serialized collections (JSON strings, comma separated
    strings, etc.).  This helper attempts to coerce each entry into a
    collection of integers.
    """

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, (set, tuple, list, np.ndarray, pd.Series)):
        return (int(v) for v in value if v is not None and not (isinstance(v, float) and np.isnan(v)))

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        # Try JSON first; fall back to comma separated values.
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return (
                int(v.strip())
                for v in value.split(",")
                if v.strip()
            )
        else:
            return _normalize_iterable(parsed)

    return [int(value)]


def _extract_ids_from_columns(df: pd.DataFrame, columns: Sequence[str]) -> set[int]:
    ids: set[int] = set()
    for column in columns:
        if column not in df:
            continue
        for value in df[column].dropna():
            ids.update(_normalize_iterable(value))
    return ids


def _infer_baseline_columns(df: pd.DataFrame, prefix: str) -> Sequence[str]:
    lower_prefix = prefix.lower()
    return [col for col in df.columns if col.lower().startswith(lower_prefix)]


def resolve_scope(client, scope: str) -> tuple[set[int], set[int]]:
    """Return the ID sets for the requested scope.

    Parameters
    ----------
    client:
        A ``CAVEclient`` (or compatible) instance that provides access to the
        materialization API.
    scope:
        Either ``"whole"`` or ``"cx"``.  ``"whole"`` indicates that no
        filtering should be applied.  ``"cx"`` resolves to the set of CX stream
        IDs stored in the baseline annotations table.
    """

    normalized_scope = scope.lower()
    if normalized_scope not in {"whole", "cx"}:
        raise ValueError(f"Unsupported scope: {scope!r}")

    if normalized_scope == "whole":
        return set(), set()

    # ``cx`` scope – load the baseline annotations to obtain the stream IDs.
    annotation_df = None
    table_candidates: Sequence[str] = ()
    if hasattr(client, "ci_baseline_table"):
        candidate = getattr(client, "ci_baseline_table")
        if candidate:
            table_candidates = (candidate,)
    elif isinstance(getattr(client, "config", None), MutableMapping):
        candidate = client.config.get("ci_baseline_table")
        if candidate:
            table_candidates = (candidate,)

    table_candidates = table_candidates or DEFAULT_BASELINE_TABLE_CANDIDATES

    for table_name in table_candidates:
        try:
            annotation_df = client.materialize.query_table(table_name)
        except Exception:  # pragma: no cover - best effort across deployments
            continue
        if annotation_df is not None:
            break

    if annotation_df is None:
        raise RuntimeError(
            "Unable to load baseline annotations for CX scope. "
            "Checked tables: " + ", ".join(table_candidates)
        )

    pre_columns = _infer_baseline_columns(annotation_df, "cx_pre")
    post_columns = _infer_baseline_columns(annotation_df, "cx_post")

    if not pre_columns and "direction" in annotation_df.columns:
        pre_columns = [
            col
            for col in annotation_df.columns
            if col.lower().endswith("_id") and "pre" in col.lower()
        ]
    if not post_columns and "direction" in annotation_df.columns:
        post_columns = [
            col
            for col in annotation_df.columns
            if col.lower().endswith("_id") and "post" in col.lower()
        ]

    pre_ids = _extract_ids_from_columns(annotation_df, pre_columns)
    post_ids = _extract_ids_from_columns(annotation_df, post_columns)

    if not pre_ids and not post_ids:
        raise RuntimeError("Failed to extract CX stream IDs from baseline annotations.")

    return pre_ids, post_ids


def _resolve_synapse_table(client) -> str:
    if hasattr(client, "ci_synapse_table") and getattr(client, "ci_synapse_table"):
        return getattr(client, "ci_synapse_table")
    if isinstance(getattr(client, "config", None), MutableMapping):
        table_name = client.config.get("ci_synapse_table") or client.config.get("synapse_table")
        if table_name:
            return table_name
    for candidate in DEFAULT_SYNAPSE_TABLE_CANDIDATES:
        try:
            # Test availability without loading data; a cheap COUNT query is not
            # exposed, so we simply rely on the eventual call to fail if the
            # table does not exist.
            return candidate
        except Exception:  # pragma: no cover - we never expect to hit this branch
            continue
    raise RuntimeError("Unable to determine synapse table for CI edges.")


def fetch_edges(
    client,
    materialization: int | None = None,
    scope: str = "whole",
    columns: Sequence[str] = ("pre_root_id", "post_root_id", "weight"),
    *,
    at_ts: str | None = None,
) -> pd.DataFrame:
    """Load the synapse edges for the requested scope.

    Parameters
    ----------
    client:
        ``CAVEclient`` (or compatible) instance used to issue the materialize
        query.
    materialization:
        Optional materialization version to query.
    scope:
        Either ``"whole"`` (default) or ``"cx"``.
    columns:
        The columns to request from the materialization table.  The first two
        entries must be the pre and post root IDs respectively, while the third
        entry is used as the edge weight.
    at_ts:
        Optional timestamp string to query a historical view of the table.
    """

    if not columns or len(columns) < 2:
        raise ValueError("At least two columns (pre and post IDs) are required.")

    pre_column, post_column = columns[0], columns[1]
    weight_column = columns[2] if len(columns) > 2 else None

    table_name = _resolve_synapse_table(client)

    query_kwargs: Dict[str, object] = {"select_columns": list(dict.fromkeys(columns))}
    if materialization is not None:
        query_kwargs["materialization_version"] = materialization
    if at_ts is not None:
        query_kwargs["timestamp"] = at_ts

    edges = client.materialize.query_table(table_name, **query_kwargs)
    if edges.empty:
        return edges

    edges = edges.loc[:, list(dict.fromkeys(columns))].copy()

    if scope.lower() == "cx":
        pre_ids, post_ids = resolve_scope(client, scope)
        mask = pd.Series(False, index=edges.index)
        if pre_ids:
            mask |= edges[pre_column].isin(pre_ids)
        if post_ids:
            mask |= edges[post_column].isin(post_ids)
        edges = edges.loc[mask].reset_index(drop=True)

    out_dir = Path("out") / "ci"
    out_dir.mkdir(parents=True, exist_ok=True)
    edges_path = out_dir / f"base_edges_{scope.lower()}.parquet"
    edges.to_parquet(edges_path, index=False)

    return edges


def build_index(
    edges: pd.DataFrame,
    scope: str = "whole",
    *,
    pre_column: str = "pre_root_id",
    post_column: str = "post_root_id",
    weight_column: str = "weight",
) -> tuple[csr_matrix, dict[int, int], dict[int, int]]:
    """Construct a CSR adjacency matrix from the synapse edge table."""

    if edges.empty:
        empty_matrix = csr_matrix((0, 0), dtype=float)
        idmaps_path = Path("out") / "ci" / f"idmaps_{scope.lower()}.pkl"
        idmaps_path.parent.mkdir(parents=True, exist_ok=True)
        with idmaps_path.open("wb") as fh:
            pickle.dump({"rows": {}, "cols": {}}, fh)
        return empty_matrix, {}, {}

    if pre_column not in edges or post_column not in edges:
        raise KeyError("Edge DataFrame must contain the pre and post columns.")

    working = edges[[pre_column, post_column]].copy()
    if weight_column in edges:
        working[weight_column] = edges[weight_column].astype(float)
    else:
        working[weight_column] = 1.0

    grouped = (
        working.groupby([pre_column, post_column], sort=True, as_index=False)[weight_column]
        .sum()
        .rename(columns={weight_column: "weight"})
    )

    pre_ids = np.array(sorted(grouped[pre_column].unique()), dtype=np.int64)
    post_ids = np.array(sorted(grouped[post_column].unique()), dtype=np.int64)

    id2row = {int(node_id): idx for idx, node_id in enumerate(pre_ids)}
    id2col = {int(node_id): idx for idx, node_id in enumerate(post_ids)}

    row_indices = grouped[pre_column].map(id2row).to_numpy()
    col_indices = grouped[post_column].map(id2col).to_numpy()
    data = grouped["weight"].to_numpy(dtype=float)

    matrix = csr_matrix((data, (row_indices, col_indices)), shape=(len(pre_ids), len(post_ids)))

    out_dir = Path("out") / "ci"
    out_dir.mkdir(parents=True, exist_ok=True)
    idmaps_path = out_dir / f"idmaps_{scope.lower()}.pkl"
    with idmaps_path.open("wb") as fh:
        pickle.dump({"rows": id2row, "cols": id2col}, fh)

    return matrix, id2row, id2col


__all__ = [
    "build_index",
    "fetch_edges",
    "resolve_scope",
]
