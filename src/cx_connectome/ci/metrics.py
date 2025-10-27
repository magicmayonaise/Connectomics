"""Connectivity Interpreter metrics utilities.

This module provides light-weight analytics helpers that mirror the
exploratory notebooks used with the Connectome Interpreter.  The
functions operate on tidy (long-form) :class:`pandas.DataFrame`
structures so they can be composed in downstream plotting pipelines or
report generation tools.

The three public helpers are intentionally opinionated about their
inputs but attempt to be forgiving by accepting common column naming
conventions that show up in Interpreter exports.

* :func:`ei_ratio_by_hop` – quantify the excitatory / inhibitory balance
  as paths extend outward from a seed population.
* :func:`lateral_bias` – measure ipsilateral versus contralateral
  contributions using hemisphere labels (replicating the
  ``LPLC1 → DNa05`` pattern from Interpreter demos).
* :func:`robustness_curves` – examine how effective connectivity metrics
  change as increasingly strict ``k`` thresholds are applied.

The helpers return tidy :class:`pandas.DataFrame` objects to make them
straightforward to plot with :mod:`matplotlib`, :mod:`seaborn`, or other
visualisation libraries.  Each function raises a :class:`ValueError`
when required information is missing instead of failing silently so that
unit tests can detect malformed inputs early.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd


_SIGN_ALIASES = {
    "e": "exc",
    "exc": "exc",
    "excitatory": "exc",
    "+": "exc",
    "pos": "exc",
    "positive": "exc",
    "1": "exc",
    1: "exc",
    True: "exc",
    "i": "inh",
    "inh": "inh",
    "inhibitory": "inh",
    "-": "inh",
    "neg": "inh",
    "negative": "inh",
    "-1": "inh",
    -1: "inh",
    False: "inh",
}

_SOURCE_ALIASES = ("source", "pre", "presyn", "upstream", "from", "root_id_pre")
_TARGET_ALIASES = ("target", "post", "partner", "downstream", "to", "root_id_post")
_HOP_ALIASES = ("hop", "hops", "distance", "path_length", "pathlen", "step")
_VALUE_ALIASES = ("weight", "value", "count", "n", "size", "strength", "score")
_HEMI_ALIASES = ("hemi", "hemisphere", "side", "lateral", "lr", "hemis")
_NODE_ALIASES = ("node", "id", "root_id", "body", "neuron", "cell")


def _sort_key(value: Any) -> Tuple[int, Any]:
    """Sorting helper that keeps ``None`` last and numbers ordered."""

    if value is None:
        return (2, "")
    if isinstance(value, (int, float, np.number)):
        return (0, float(value))
    return (1, str(value))


def _normalise_mapping(data: Union[Mapping[Any, Any], pd.DataFrame, pd.Series]) -> Dict[Any, Any]:
    """Return a ``dict`` mapping keys to dataframes.

    The Interpreter exports often provide either a dictionary keyed by
    the effective ``k`` value or a single dataframe with a ``k`` column.
    This helper standardises both shapes for downstream processing.
    """

    if isinstance(data, Mapping):
        return dict(data)

    if isinstance(data, pd.Series):
        return {None: data.to_frame(name="value")}

    if isinstance(data, pd.DataFrame):
        if "k" in data.columns:
            return {k: frame.drop(columns=["k"]).copy() for k, frame in data.groupby("k")}
        return {None: data.copy()}

    raise TypeError("Unsupported container type for metrics computation")


def _ensure_dataframe(obj: Any, value_name: str = "value") -> pd.DataFrame:
    """Coerce ``obj`` into a dataframe with a numeric value column."""

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif isinstance(obj, pd.Series):
        df = obj.to_frame(name=value_name)
    else:
        raise TypeError("Metrics expect pandas objects (DataFrame or Series)")

    if df.index.names and any(name is not None for name in df.index.names):
        df = df.reset_index()
    return df


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    """Return the first matching column from ``candidates``.

    Raises ``ValueError`` if no candidate is found.
    """

    lower_cols = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        cand_lower = candidate.lower()
        if cand_lower in lower_cols:
            return lower_cols[cand_lower]
    raise ValueError(f"Required column not found; searched {tuple(candidates)}")


def _value_column(df: pd.DataFrame) -> str:
    """Identify a numeric value column, preferring known aliases."""

    for alias in _VALUE_ALIASES:
        try:
            column = _find_column(df, (alias,))
        except ValueError:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            return column

    numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if numeric_columns:
        return numeric_columns[0]
    raise ValueError("No numeric value column available for metrics computation")


def _normalise_sign(value: Any) -> str | None:
    """Map heterogeneous sign labels to ``'exc'`` or ``'inh'``."""

    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        if value > 0:
            return "exc"
        if value < 0:
            return "inh"
        return None

    key = str(value).strip().lower()
    return _SIGN_ALIASES.get(key)


def _normalise_signed_block(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy dataframe with ``hop``, ``sign``, and ``value`` columns."""

    hop_column = _find_column(df, _HOP_ALIASES)
    sign_column = _find_column(df, ("sign", "polarity", "type", "class"))
    value_column = _value_column(df)

    tidy = df[[hop_column, sign_column, value_column]].copy()
    tidy.columns = ["hop", "sign", "value"]
    tidy["sign"] = tidy["sign"].map(_normalise_sign)
    tidy = tidy.dropna(subset=["sign"])
    tidy["hop"] = tidy["hop"].astype(int, errors="ignore")
    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")
    tidy = tidy.dropna(subset=["value"])
    return tidy


def _normalise_edge_block(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with ``source``, ``target``, and ``weight`` columns."""

    source_column = _find_column(df, _SOURCE_ALIASES)
    target_column = _find_column(df, _TARGET_ALIASES)
    value_column = _value_column(df)

    tidy = df[[source_column, target_column, value_column]].copy()
    tidy.columns = ["source", "target", "weight"]
    tidy["weight"] = pd.to_numeric(tidy["weight"], errors="coerce")
    tidy = tidy.dropna(subset=["source", "target", "weight"])
    return tidy


def _normalise_hemi_labels(hemi_labels: Union[Mapping[Any, Any], pd.DataFrame, pd.Series]) -> Dict[Any, str]:
    """Normalise hemisphere labels to ``{'L', 'R'}``."""

    if isinstance(hemi_labels, Mapping):
        items = hemi_labels.items()
    elif isinstance(hemi_labels, pd.Series):
        items = hemi_labels.items()
    elif isinstance(hemi_labels, pd.DataFrame):
        node_col = _find_column(hemi_labels, _NODE_ALIASES)
        hemi_col = _find_column(hemi_labels, _HEMI_ALIASES)
        items = hemi_labels[[node_col, hemi_col]].itertuples(index=False, name=None)
    else:
        raise TypeError("Hemisphere labels must be a mapping or pandas object")

    def _coerce(value: Any) -> str | None:
        if pd.isna(value):
            return None
        key = str(value).strip().lower()
        if key in {"l", "left"}:
            return "L"
        if key in {"r", "right"}:
            return "R"
        return None

    result: Dict[Any, str] = {}
    for node, hemi in items:
        normalised = _coerce(hemi)
        if normalised is not None:
            result[node] = normalised
    if not result:
        raise ValueError("No valid hemisphere labels were provided")
    return result


def ei_ratio_by_hop(signed_blocks_per_k: Union[Mapping[Any, Any], pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """Summarise the excitatory / inhibitory balance as a function of hop distance.

    Parameters
    ----------
    signed_blocks_per_k:
        A mapping from effective ``k`` values to connectivity blocks.  Each
        block must contain *at least* ``hop``, ``sign``, and a numeric value
        column.  The function is tolerant to common column aliases such as
        ``type`` or ``weight``.

    Returns
    -------
    pandas.DataFrame
        A tidy dataframe with ``k``, ``hop``, ``exc_weight``,
        ``inh_weight``, ``total_weight``, ``balance`` (signed balance in
        ``[-1, 1]``), and ``exc_fraction`` (fraction of excitatory
        contribution).  The structure is intentionally plot-friendly, for
        example ``df.pivot(index='hop', columns='k', values='balance')``
        produces a heatmap showing trends toward excitation/inhibition
        balance.
    """

    blocks = {
        k: _normalise_signed_block(_ensure_dataframe(block))
        for k, block in _normalise_mapping(signed_blocks_per_k).items()
    }

    records = []
    for k, block in sorted(blocks.items(), key=lambda item: _sort_key(item[0])):
        if block.empty:
            continue
        grouped = block.groupby(["hop", "sign"])["value"].sum().unstack(fill_value=0.0)
        excit = grouped.get("exc", pd.Series(0.0, index=grouped.index))
        inhib = grouped.get("inh", pd.Series(0.0, index=grouped.index)).abs()
        total = excit + inhib
        balance = (excit - inhib) / total.replace(0.0, np.nan)
        exc_fraction = excit / total.replace(0.0, np.nan)

        records.append(
            pd.DataFrame(
                {
                    "k": k,
                    "hop": grouped.index,
                    "exc_weight": excit.values,
                    "inh_weight": inhib.values,
                    "total_weight": total.values,
                    "balance": balance.values,
                    "exc_fraction": exc_fraction.values,
                }
            )
        )

    if not records:
        return pd.DataFrame(
            columns=["k", "hop", "exc_weight", "inh_weight", "total_weight", "balance", "exc_fraction"]
        )

    result = pd.concat(records, ignore_index=True)
    return result.sort_values(["k", "hop"]).reset_index(drop=True)


def lateral_bias(
    eff_k: Union[Mapping[Any, Any], pd.DataFrame, pd.Series],
    hemi_labels: Union[Mapping[Any, Any], pd.DataFrame, pd.Series],
) -> pd.DataFrame:
    """Compute ipsilateral versus contralateral connectivity bias.

    Parameters
    ----------
    eff_k:
        Effective connectivity blocks keyed by ``k``.  Each block must
        identify ``source`` and ``target`` neurons (common aliases such as
        ``pre``/``post`` are supported) and contain a numeric weight.
    hemi_labels:
        Hemisphere annotations for neuron bodies.  The helper accepts a
        mapping (``{body: 'L'/'R'}``), a :class:`pandas.Series`, or a
        dataframe with columns matching ``('body', 'hemi')`` aliases.

    Returns
    -------
    pandas.DataFrame
        A dataframe with ``k``, ``ipsilateral``, ``contralateral``,
        ``total_weight``, ``bias`` (signed ipsi-vs-contra bias), and
        ``ipsi_fraction`` columns.  Positive ``bias`` values indicate a
        stronger ipsilateral drive.

    Notes
    -----
    The resulting dataframe can be plotted directly to reproduce the
    ``LPLC1 → DNa05`` interpreter visualisation::

        bias = lateral_bias(eff_blocks, hemi_labels)
        ax = bias.plot(x="k", y=["ipsilateral", "contralateral"], marker="o")
        ax.set_ylabel("Synaptic weight")
    """

    edges_per_k = {
        k: _normalise_edge_block(_ensure_dataframe(block))
        for k, block in _normalise_mapping(eff_k).items()
    }
    hemi_map = _normalise_hemi_labels(hemi_labels)

    records = []
    for k, edges in sorted(edges_per_k.items(), key=lambda item: _sort_key(item[0])):
        if edges.empty:
            records.append(
                {
                    "k": k,
                    "ipsilateral": 0.0,
                    "contralateral": 0.0,
                    "total_weight": 0.0,
                    "bias": np.nan,
                    "ipsi_fraction": np.nan,
                }
            )
            continue

        edges = edges.copy()
        edges["source_hemi"] = edges["source"].map(hemi_map)
        edges["target_hemi"] = edges["target"].map(hemi_map)
        edges = edges.dropna(subset=["source_hemi", "target_hemi"])

        if edges.empty:
            records.append(
                {
                    "k": k,
                    "ipsilateral": 0.0,
                    "contralateral": 0.0,
                    "total_weight": 0.0,
                    "bias": np.nan,
                    "ipsi_fraction": np.nan,
                }
            )
            continue

        ipsilateral = edges.loc[edges["source_hemi"] == edges["target_hemi"], "weight"].sum()
        contralateral = edges.loc[edges["source_hemi"] != edges["target_hemi"], "weight"].sum()
        total = ipsilateral + contralateral

        bias = (ipsilateral - contralateral) / total if total else np.nan
        ipsi_fraction = ipsilateral / total if total else np.nan

        records.append(
            {
                "k": k,
                "ipsilateral": float(ipsilateral),
                "contralateral": float(contralateral),
                "total_weight": float(total),
                "bias": float(bias) if bias == bias else np.nan,
                "ipsi_fraction": float(ipsi_fraction) if ipsi_fraction == ipsi_fraction else np.nan,
            }
        )

    return pd.DataFrame.from_records(records).sort_values("k").reset_index(drop=True)


def robustness_curves(
    eff_k: Union[Mapping[Any, Any], pd.DataFrame, pd.Series],
    quantiles: Sequence[float] = (0.99, 0.95, 0.9, 0.75),
) -> pd.DataFrame:
    """Quantify how sensitive effective connectivity is to thresholding.

    Parameters
    ----------
    eff_k:
        Effective connectivity blocks keyed by ``k``.  Each block must
        include ``source`` and ``target`` identifiers (aliases accepted)
        and a numeric ``weight`` column.
    quantiles:
        Iterable of quantiles in the open interval ``(0, 1)`` used to
        derive thresholds.  Higher quantiles correspond to stricter
        thresholds (retaining only the strongest connections).

    Returns
    -------
    pandas.DataFrame
        A tidy dataframe with ``k``, ``quantile``, ``threshold`` (weight
        cut-off), ``remaining_edges`` (count of surviving edges),
        ``remaining_fraction`` (fraction of edges retained),
        ``weight_fraction`` (fraction of total weight retained), and
        ``mean_weight`` (mean weight after thresholding).

    Examples
    --------
    >>> curves = robustness_curves(eff_blocks)
    >>> curves.query("quantile == 0.95").plot(x="k", y="weight_fraction")
    """

    if not quantiles:
        raise ValueError("At least one quantile must be provided")

    for quantile in quantiles:
        if not 0.0 < quantile < 1.0:
            raise ValueError("Quantiles must lie strictly between 0 and 1")

    edges_per_k = {
        k: _normalise_edge_block(_ensure_dataframe(block))
        for k, block in _normalise_mapping(eff_k).items()
    }

    records = []
    for k, edges in sorted(edges_per_k.items(), key=lambda item: _sort_key(item[0])):
        weights = edges["weight"].abs()
        total_weight = weights.sum()
        total_edges = len(edges)

        for quantile in quantiles:
            if edges.empty:
                records.append(
                    {
                        "k": k,
                        "quantile": quantile,
                        "threshold": np.nan,
                        "remaining_edges": 0,
                        "remaining_fraction": np.nan,
                        "weight_fraction": np.nan,
                        "mean_weight": np.nan,
                    }
                )
                continue

            threshold = float(weights.quantile(quantile))
            survivors = edges.loc[weights >= threshold]
            remaining_edges = len(survivors)
            remaining_fraction = remaining_edges / total_edges if total_edges else np.nan
            weight_fraction = survivors["weight"].abs().sum() / total_weight if total_weight else np.nan
            mean_weight = survivors["weight"].mean() if remaining_edges else np.nan

            records.append(
                {
                    "k": k,
                    "quantile": quantile,
                    "threshold": threshold,
                    "remaining_edges": remaining_edges,
                    "remaining_fraction": remaining_fraction,
                    "weight_fraction": weight_fraction,
                    "mean_weight": mean_weight,
                }
            )

    return (
        pd.DataFrame.from_records(records)
        .sort_values(["quantile", "k"])
        .reset_index(drop=True)
    )


__all__ = ["ei_ratio_by_hop", "lateral_bias", "robustness_curves"]
