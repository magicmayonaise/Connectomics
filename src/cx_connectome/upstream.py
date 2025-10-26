"""Tools for quantifying upstream partners of N1 streams.

This module wraps the CAVE materialization service to pull the synapses where
``post_pt_root_id`` belongs to an N1 stream, aggregates those synapses into an
N0→N1 adjacency (filtered to edges with at least five synapses) and annotates the
presynaptic partners by their ``super_class``.  The resulting counts and
proportions by input category per N1 stream can be exported and plotted as
stacked bars.

The helpers here follow the same timestamped snapshot pattern used in the other
``cx_connectome`` analyses so that each run leaves an immutable record on disk.
Conceptually this recapitulates the "input class" analysis from previous work,
but scales it to the full dataset available via CAVE.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

try:  # pragma: no cover - the package is only required at runtime
    from caveclient import CAVEclient
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "The `caveclient` package is required to run the upstream analysis."
    ) from exc

LOGGER = logging.getLogger(__name__)

CATEGORY_ORDER: tuple[str, ...] = ("sensory", "ascending", "descending", "central")
DEFAULT_SUPER_CLASS_SYNONYMS: Mapping[str, str] = {
    "sensory": "sensory",
    "sensory neuron": "sensory",
    "sensory afferent": "sensory",
    "sensory interneuron": "sensory",
    "ascending": "ascending",
    "ascending neuron": "ascending",
    "ascending interneuron": "ascending",
    "descending": "descending",
    "descending neuron": "descending",
    "descending interneuron": "descending",
    "descending projection neuron": "descending",
}
DEFAULT_SUPER_CLASS_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("sensory", "sensory"),
    ("ascending", "ascending"),
    ("descending", "descending"),
)


def _chunked(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Yield ``values`` in successive chunks of ``size``."""

    iterator = iter(values)
    while True:
        block = list(islice(iterator, size))
        if not block:
            break
        yield block


def _materialization(client: CAVEclient, timestamp: str | None) -> Any:
    """Return the materialization client at ``timestamp`` if provided."""

    materialization = client.materialize
    if timestamp:
        materialization = materialization.get_timestamp(timestamp)
    return materialization


def make_super_class_categorizer(
    synonyms: Mapping[str, str] | None = None,
    keywords: Sequence[tuple[str, str]] | None = None,
    *,
    default: str = "central",
) -> Callable[[Any], str]:
    """Create a function that maps a ``super_class`` value to a category."""

    synonyms = {key.lower(): value for key, value in (synonyms or {}).items()}
    keywords = tuple(keywords or ())

    def categorizer(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        normalized = str(value).strip().lower()
        if not normalized:
            return default
        if normalized in synonyms:
            return synonyms[normalized]
        for key, category in keywords:
            if key in normalized:
                return category
        return default

    return categorizer


CATEGORIZE_SUPER_CLASS = make_super_class_categorizer(
    synonyms=DEFAULT_SUPER_CLASS_SYNONYMS,
    keywords=DEFAULT_SUPER_CLASS_KEYWORDS,
)


def query_n1_streams(
    client: CAVEclient,
    table: str,
    *,
    root_id_column: str = "pt_root_id",
    stream_column: str = "stream",
    timestamp: str | None = None,
    filter_in_dict: Mapping[str, Sequence[Any]] | None = None,
    select_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Fetch the N1 stream membership table."""

    materialization = _materialization(client, timestamp)
    columns = [root_id_column, stream_column]
    if select_columns:
        for column in select_columns:
            if column not in columns:
                columns.append(column)

    LOGGER.info("Querying N1 stream membership from %s", table)
    n1_streams = materialization.query_table(
        table,
        select_columns=columns,
        filter_in_dict=filter_in_dict,
    )
    if root_id_column not in n1_streams.columns:
        raise KeyError(f"Column '{root_id_column}' not returned by {table}")
    if stream_column not in n1_streams.columns:
        raise KeyError(f"Column '{stream_column}' not returned by {table}")
    return n1_streams[[root_id_column, stream_column]].drop_duplicates(
        subset=[root_id_column]
    )


def query_synapse_counts(
    client: CAVEclient,
    n1_ids: Sequence[int],
    synapse_table: str,
    *,
    timestamp: str | None = None,
    chunk_size: int = 50_000,
    select_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load synapses targeting the supplied N1 root IDs and count them."""

    if not n1_ids:
        return pd.DataFrame(
            columns=["pre_pt_root_id", "post_pt_root_id", "synapse_count"]
        )

    materialization = _materialization(client, timestamp)
    columns = ["pre_pt_root_id", "post_pt_root_id"]
    if select_columns:
        for column in select_columns:
            if column not in columns:
                columns.append(column)

    frames: list[pd.DataFrame] = []
    for chunk in _chunked(n1_ids, chunk_size):
        LOGGER.info(
            "Querying synapses from %s for %d postsynaptic IDs", synapse_table, len(chunk)
        )
        frame = materialization.query_table(
            synapse_table,
            select_columns=columns,
            filter_in_dict={"post_pt_root_id": chunk},
        )
        if not frame.empty:
            frames.append(frame[columns])

    if not frames:
        LOGGER.warning("No synapses retrieved for the provided N1 IDs")
        return pd.DataFrame(
            columns=["pre_pt_root_id", "post_pt_root_id", "synapse_count"]
        )

    synapses = pd.concat(frames, ignore_index=True)
    LOGGER.info(
        "Aggregating %d synapses into presynaptic-postsynaptic counts", len(synapses)
    )
    counts = (
        synapses.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False)
        .size()
        .rename(columns={"size": "synapse_count"})
    )
    return counts


def load_super_class_annotations(
    client: CAVEclient,
    root_ids: Sequence[int],
    annotation_table: str,
    *,
    timestamp: str | None = None,
    root_id_column: str = "pt_root_id",
    super_class_column: str = "super_class",
    select_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the ``super_class`` annotations for the provided roots."""

    if not root_ids:
        return pd.DataFrame(columns=[root_id_column, super_class_column])

    materialization = _materialization(client, timestamp)
    columns = [root_id_column, super_class_column]
    if select_columns:
        for column in select_columns:
            if column not in columns:
                columns.append(column)

    LOGGER.info(
        "Fetching super_class annotations for %d presynaptic partners from %s",
        len(root_ids),
        annotation_table,
    )
    annotations = materialization.query_table(
        annotation_table,
        select_columns=columns,
        filter_in_dict={root_id_column: list(root_ids)},
    )
    if annotations.empty:
        LOGGER.warning("No annotations found for presynaptic root IDs")
        return pd.DataFrame(columns=[root_id_column, super_class_column])

    annotations = annotations[columns]
    annotations = annotations.drop_duplicates(subset=[root_id_column])
    return annotations


def build_adjacency(
    synapse_counts: pd.DataFrame,
    annotations: pd.DataFrame,
    n1_streams: pd.DataFrame,
    *,
    min_synapses: int = 5,
    pre_root_column: str = "pre_pt_root_id",
    post_root_column: str = "post_pt_root_id",
    annotation_root_column: str = "pt_root_id",
    super_class_column: str = "super_class",
    stream_column: str = "stream",
    categorize: Callable[[Any], str] = CATEGORIZE_SUPER_CLASS,
) -> pd.DataFrame:
    """Combine synapse counts, annotations and N1 stream metadata."""

    if synapse_counts.empty:
        LOGGER.warning("Synapse count table is empty; returning without edges")
        return pd.DataFrame(
            columns=[
                pre_root_column,
                post_root_column,
                "synapse_count",
                super_class_column,
                "input_category",
                "n1_stream",
            ]
        )

    LOGGER.info("Filtering synapse counts with at least %d synapses", min_synapses)
    adjacency = synapse_counts[synapse_counts["synapse_count"] >= min_synapses].copy()
    if adjacency.empty:
        LOGGER.warning("No edges survive the minimum synapse threshold")
        return pd.DataFrame(
            columns=[
                pre_root_column,
                post_root_column,
                "synapse_count",
                super_class_column,
                "input_category",
                "n1_stream",
            ]
        )

    n1_streams = n1_streams[[annotation_root_column, stream_column]].drop_duplicates(
        subset=[annotation_root_column]
    )
    adjacency = adjacency.merge(
        n1_streams,
        left_on=post_root_column,
        right_on=annotation_root_column,
        how="left",
    )
    adjacency = adjacency.rename(columns={stream_column: "n1_stream"})
    adjacency.drop(columns=[annotation_root_column], inplace=True)

    annotations = annotations[[annotation_root_column, super_class_column]].drop_duplicates(
        subset=[annotation_root_column]
    )
    adjacency = adjacency.merge(
        annotations,
        left_on=pre_root_column,
        right_on=annotation_root_column,
        how="left",
    )
    adjacency.drop(columns=[annotation_root_column], inplace=True)
    adjacency["input_category"] = adjacency[super_class_column].map(categorize)

    column_order = [
        pre_root_column,
        post_root_column,
        "synapse_count",
        super_class_column,
        "input_category",
        "n1_stream",
    ]
    return adjacency[column_order]


def summarise_by_category(
    adjacency: pd.DataFrame,
    *,
    stream_column: str = "n1_stream",
    category_column: str = "input_category",
    weight_column: str = "synapse_count",
    category_order: Sequence[str] = CATEGORY_ORDER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return stacked count and proportion tables by input category."""

    if adjacency.empty:
        index = pd.Index([], name=stream_column)
        empty = pd.DataFrame(columns=category_order, index=index)
        return empty, empty

    grouped = (
        adjacency.groupby([stream_column, category_column], dropna=False)[
            weight_column
        ]
        .sum()
        .reset_index()
    )

    counts = grouped.pivot(
        index=stream_column,
        columns=category_column,
        values=weight_column,
    ).fillna(0)

    if category_order:
        for column in category_order:
            if column not in counts.columns:
                counts[column] = 0
        ordered = [col for col in category_order]
        ordered.extend(col for col in counts.columns if col not in category_order)
        counts = counts[ordered]

    counts = counts.sort_index()
    counts.index.name = stream_column

    totals = counts.sum(axis=1)
    denominators = totals.replace(0, pd.NA)
    proportions = counts.divide(denominators, axis=0)

    return counts, proportions


def ensure_output_directory(
    output_root: Path | str,
    *,
    label: str,
    snapshot: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Create (or reuse) the timestamped snapshot directory."""

    root = Path(output_root)
    timestamp = snapshot or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    directory = root / label / timestamp
    if directory.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory {directory} already exists. Pass overwrite=True to reuse it."
        )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def plot_stacked_bars(
    table: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    ylabel: str,
    category_order: Sequence[str] = CATEGORY_ORDER,
    colors: Mapping[str, str] | None = None,
    proportion: bool = False,
) -> Path | None:
    """Plot a stacked bar chart from the provided table."""

    if table.empty:
        LOGGER.warning("Skipping plot %s because the table is empty", output_path)
        return None

    ordered_columns = [col for col in category_order if col in table.columns]
    ordered_columns.extend([col for col in table.columns if col not in ordered_columns])
    table = table[ordered_columns]

    fig, ax = plt.subplots(figsize=(max(6, len(table.index) * 0.6), 6))
    bottom = pd.Series(0, index=table.index, dtype=float)

    for column in table.columns:
        values = table[column].fillna(0).astype(float)
        ax.bar(
            table.index.astype(str),
            values,
            bottom=bottom,
            label=column,
            color=(colors or {}).get(column),
        )
        bottom += values

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("N1 stream")
    ax.legend(title="Input category")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    if proportion:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved plot to %s", output_path)
    return output_path


def run_upstream_analysis(
    client: CAVEclient,
    *,
    n1_table: str,
    synapse_table: str,
    annotation_table: str,
    timestamp: str | None = None,
    min_synapses: int = 5,
    chunk_size: int = 50_000,
    output_root: Path | str = Path("snapshots"),
    snapshot_label: str = "n0_to_n1_inputs",
    snapshot_name: str | None = None,
    overwrite: bool = False,
    stream_column: str = "stream",
    n1_root_column: str = "pt_root_id",
    super_class_column: str = "super_class",
) -> dict[str, Path | None]:
    """Run the upstream analysis end-to-end and persist its outputs."""

    directory = ensure_output_directory(
        output_root, label=snapshot_label, snapshot=snapshot_name, overwrite=overwrite
    )

    n1_streams = query_n1_streams(
        client,
        n1_table,
        root_id_column=n1_root_column,
        stream_column=stream_column,
        timestamp=timestamp,
    )
    n1_ids = n1_streams[n1_root_column].astype("int64").tolist()

    synapse_counts = query_synapse_counts(
        client,
        n1_ids,
        synapse_table,
        timestamp=timestamp,
        chunk_size=chunk_size,
    )

    annotations = load_super_class_annotations(
        client,
        synapse_counts["pre_pt_root_id"].unique().tolist(),
        annotation_table,
        timestamp=timestamp,
        root_id_column=n1_root_column,
        super_class_column=super_class_column,
    )

    n1_streams_for_merge = n1_streams.rename(columns={n1_root_column: "pt_root_id"})

    if "pt_root_id" in annotations.columns:
        annotation_root_column = "pt_root_id"
    elif n1_root_column in annotations.columns:
        annotations = annotations.rename(columns={n1_root_column: "pt_root_id"})
        annotation_root_column = "pt_root_id"
    else:
        raise KeyError(
            "Unable to locate a pt_root_id column in the annotation table."
        )

    adjacency = build_adjacency(
        synapse_counts,
        annotations,
        n1_streams_for_merge,
        min_synapses=min_synapses,
        pre_root_column="pre_pt_root_id",
        post_root_column="post_pt_root_id",
        annotation_root_column=annotation_root_column,
        super_class_column=super_class_column,
        stream_column=stream_column,
        categorize=CATEGORIZE_SUPER_CLASS,
    )

    counts, proportions = summarise_by_category(
        adjacency,
        stream_column="n1_stream",
        category_column="input_category",
        weight_column="synapse_count",
    )

    adjacency_path = directory / "n0_to_n1_adjacency.csv"
    counts_path = directory / "input_category_counts.csv"
    proportions_path = directory / "input_category_proportions.csv"

    LOGGER.info("Writing adjacency to %s", adjacency_path)
    adjacency.to_csv(adjacency_path, index=False)

    LOGGER.info("Writing counts to %s", counts_path)
    counts.reset_index().to_csv(counts_path, index=False)

    LOGGER.info("Writing proportions to %s", proportions_path)
    proportions.reset_index().to_csv(proportions_path, index=False)

    counts_plot = plot_stacked_bars(
        counts,
        directory / "input_category_counts.png",
        title="N0→N1 inputs by category",
        ylabel="Number of synapses",
    )

    proportions_plot = plot_stacked_bars(
        proportions,
        directory / "input_category_proportions.png",
        title="N0→N1 input mix by category",
        ylabel="Proportion of synapses",
        proportion=True,
    )

    metadata_path = directory / "metadata.json"
    metadata = {
        "n1_table": n1_table,
        "synapse_table": synapse_table,
        "annotation_table": annotation_table,
        "timestamp": timestamp,
        "min_synapses": min_synapses,
        "chunk_size": chunk_size,
        "snapshot_created": datetime.utcnow().isoformat(timespec="seconds"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    LOGGER.info("Wrote run metadata to %s", metadata_path)

    return {
        "adjacency": adjacency_path,
        "counts": counts_path,
        "proportions": proportions_path,
        "counts_plot": counts_plot,
        "proportions_plot": proportions_plot,
        "metadata": metadata_path,
        "output_directory": directory,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the module."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datastack", help="Name of the CAVE datastack to query")
    parser.add_argument("n1_table", help="Materialization table with N1 stream membership")
    parser.add_argument(
        "synapse_table",
        help="Materialization table containing synapses (pre/post root columns required)",
    )
    parser.add_argument(
        "annotation_table",
        help="Materialization table with presynaptic super_class annotations",
    )
    parser.add_argument(
        "--timestamp",
        help="Materialization timestamp to use for all tables (default: latest)",
    )
    parser.add_argument(
        "--min-synapses",
        type=int,
        default=5,
        help="Minimum synapses per edge to retain (default: 5)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Number of N1 IDs to request per materialization query chunk",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("snapshots"),
        help="Root directory for timestamped snapshots",
    )
    parser.add_argument(
        "--snapshot-label",
        default="n0_to_n1_inputs",
        help="Label used in the snapshot directory hierarchy",
    )
    parser.add_argument(
        "--snapshot-name",
        help="Override the timestamp folder name with a custom value",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing snapshot directory",
    )
    parser.add_argument(
        "--stream-column",
        default="stream",
        help="Column in the N1 table describing the stream label",
    )
    parser.add_argument(
        "--n1-root-column",
        default="pt_root_id",
        help="Column in the N1 table containing the N1 pt_root_id values",
    )
    parser.add_argument(
        "--super-class-column",
        default="super_class",
        help="Column in the annotation table holding the super_class labels",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Path | None]:
    """CLI entry-point used by ``python -m cx_connectome.upstream``."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    LOGGER.info("Connecting to CAVE datastack %s", args.datastack)
    client = CAVEclient(args.datastack)

    return run_upstream_analysis(
        client,
        n1_table=args.n1_table,
        synapse_table=args.synapse_table,
        annotation_table=args.annotation_table,
        timestamp=args.timestamp,
        min_synapses=args.min_synapses,
        chunk_size=args.chunk_size,
        output_root=args.output_root,
        snapshot_label=args.snapshot_label,
        snapshot_name=args.snapshot_name,
        overwrite=args.overwrite,
        stream_column=args.stream_column,
        n1_root_column=args.n1_root_column,
        super_class_column=args.super_class_column,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
