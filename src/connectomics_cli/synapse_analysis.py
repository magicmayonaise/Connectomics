"""Synapse analysis routines built on top of the materialization helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import networkx as nx
import pandas as pd

from .config import DEFAULT_CHUNK_SIZE, QueryContext
from .materialization import (
    TableSelectionError,
    find_best_table,
    get_table_schema,
    map_roots_to_snapshot,
    query_table_chunked,
    restore_timestamp_roots,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (late import for backend selection)


SYNAPSE_REQUIRED_COLUMNS = ("pre_pt_root_id", "post_pt_root_id")


@dataclass(slots=True)
class SynapseSummary:
    """Container for aggregated synapse connectivity data."""

    dataframe: pd.DataFrame
    graph: nx.DiGraph
    metadata: dict[str, Any]

    def top_edges(self, n: int = 10) -> pd.DataFrame:
        """Return the ``n`` most strongly connected edges."""

        if self.dataframe.empty:
            return self.dataframe
        return self.dataframe.nlargest(n, "synapse_count")

    def export(self, output_dir: Path) -> None:
        """Write tabular, graphical, and metadata artefacts to ``output_dir``."""

        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "synapse_summary.csv"
        parquet_path = output_dir / "synapse_summary.parquet"
        png_path = output_dir / "synapse_counts.png"
        svg_path = output_dir / "synapse_graph.svg"
        summary_path = output_dir / "run_summary.json"

        self.dataframe.to_csv(csv_path, index=False)
        self.dataframe.to_parquet(parquet_path, index=False, engine="pyarrow")
        _plot_synapse_counts(self.dataframe, png_path)
        _draw_graph(self.graph, svg_path)
        summary_path.write_text(json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf8")


@dataclass(slots=True)
class SynapseAnalyzer:
    """Stream synapse data from CAVE and build connectivity summaries."""

    client: Any
    context: QueryContext
    synapse_table: str
    cell_type_table: str | None = None
    chunk_size: int = DEFAULT_CHUNK_SIZE

    @classmethod
    def auto_configure(
        cls,
        client: Any,
        context: QueryContext,
        *,
        synapse_table: str | None = None,
        cell_type_table: str | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> "SynapseAnalyzer":
        """Instantiate the analyzer while discovering sensible defaults."""

        if synapse_table is None:
            synapse_table = find_best_table(
                client,
                name_hint="synapse",
                required_columns=SYNAPSE_REQUIRED_COLUMNS,
            )
        if cell_type_table is None:
            try:
                cell_type_table = find_best_table(
                    client,
                    name_hint="cell_type",
                    required_columns=("root",),
                )
            except TableSelectionError:
                cell_type_table = None
        return cls(
            client=client,
            context=context,
            synapse_table=synapse_table,
            cell_type_table=cell_type_table,
            chunk_size=chunk_size,
        )

    def summarise(
        self,
        *,
        pre_root_ids: Sequence[int],
        post_root_ids: Sequence[int] | None = None,
        min_synapse_count: int = 1,
    ) -> SynapseSummary:
        """Return an aggregated summary for the provided root IDs."""

        if min_synapse_count < 1:
            msg = "min_synapse_count must be >= 1"
            raise ValueError(msg)
        pre_mappings = map_roots_to_snapshot(
            self.client,
            pre_root_ids,
            timestamp=self.context.timestamp,
            chunk_size=self.chunk_size,
        )
        post_mappings = map_roots_to_snapshot(
            self.client,
            post_root_ids or [],
            timestamp=self.context.timestamp,
            chunk_size=self.chunk_size,
        )
        pre_lookup = {mapping.snapshot: mapping.requested for mapping in pre_mappings}
        post_lookup = {mapping.snapshot: mapping.requested for mapping in post_mappings}
        filter_in: dict[str, Sequence[int]] = {}
        if pre_mappings:
            filter_in["pre_pt_root_id"] = [mapping.snapshot for mapping in pre_mappings]
        if post_mappings:
            filter_in["post_pt_root_id"] = [mapping.snapshot for mapping in post_mappings]

        frames: list[pd.DataFrame] = []
        for chunk in query_table_chunked(
            self.client,
            self.synapse_table,
            context=self.context,
            filter_in=filter_in,
            columns=["pre_pt_root_id", "post_pt_root_id", "id"],
            chunk_size=self.chunk_size,
        ):
            if "pre_pt_root_id" not in chunk.columns or "post_pt_root_id" not in chunk.columns:
                missing = sorted(set(SYNAPSE_REQUIRED_COLUMNS) - set(chunk.columns))
                msg = f"Synapse table '{self.synapse_table}' missing columns {missing}."
                raise TableSelectionError(msg)
            group = (
                chunk.groupby(["pre_pt_root_id", "post_pt_root_id"], dropna=False)
                .size()
                .rename("synapse_count")
                .reset_index()
            )
            frames.append(group)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            summary = (
                combined.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False)
                .agg({"synapse_count": "sum"})
                .sort_values("synapse_count", ascending=False)
                .reset_index(drop=True)
            )
        else:
            summary = pd.DataFrame(columns=["pre_pt_root_id", "post_pt_root_id", "synapse_count"])

        post_snapshot_ids = summary["post_pt_root_id"].astype("int64", copy=False).tolist()
        post_restored = restore_timestamp_roots(
            self.client,
            post_snapshot_ids,
            timestamp=self.context.timestamp,
            chunk_size=self.chunk_size,
        )

        summary["pre_root_id"] = summary["pre_pt_root_id"].map(lambda value: pre_lookup.get(int(value), int(value)))
        summary["post_root_id"] = summary["post_pt_root_id"].map(
            lambda value: post_lookup.get(int(value), post_restored.get(int(value), int(value)))
        )
        summary = summary.loc[summary["synapse_count"] >= min_synapse_count].reset_index(drop=True)

        cell_type_lookup = self._load_cell_types(
            set(summary["pre_root_id"].astype(int)) | set(summary["post_root_id"].astype(int))
        )
        summary["pre_cell_type"] = summary["pre_root_id"].map(cell_type_lookup).fillna("unknown")
        summary["post_cell_type"] = summary["post_root_id"].map(cell_type_lookup).fillna("unknown")

        result = summary[
            [
                "pre_root_id",
                "post_root_id",
                "synapse_count",
                "pre_cell_type",
                "post_cell_type",
                "pre_pt_root_id",
                "post_pt_root_id",
            ]
        ].copy()

        graph = _build_graph(result)
        metadata = {
            "dataset": self.context.dataset,
            "materialization": self.context.materialization,
            "timestamp": None if self.context.timestamp is None else self.context.timestamp.isoformat(),
            "chunk_size": self.chunk_size,
            "min_synapse_count": min_synapse_count,
            "queried_synapse_table": self.synapse_table,
            "queried_cell_type_table": self.cell_type_table,
            "pre_roots_requested": [int(rid) for rid in pre_root_ids],
            "post_roots_requested": None if post_root_ids is None else [int(rid) for rid in post_root_ids],
            "row_count": int(result.shape[0]),
            "total_synapses": int(result["synapse_count"].sum()) if not result.empty else 0,
        }
        return SynapseSummary(dataframe=result, graph=graph, metadata=metadata)

    def _load_cell_types(self, root_ids: Iterable[int]) -> dict[int, str]:
        if self.cell_type_table is None:
            return {}
        roots = sorted({int(rid) for rid in root_ids})
        if not roots:
            return {}
        schema = get_table_schema(self.client, self.cell_type_table)
        root_column = _choose_root_column(schema)
        cell_type_column = _choose_cell_type_column(schema)
        frames: list[pd.DataFrame] = []
        for chunk in query_table_chunked(
            self.client,
            self.cell_type_table,
            context=self.context,
            filter_in={root_column: roots},
            columns=[root_column, cell_type_column],
            chunk_size=self.chunk_size,
        ):
            frames.append(chunk[[root_column, cell_type_column]])
        if not frames:
            return {}
        data = pd.concat(frames, ignore_index=True)
        data = data.dropna(subset=[root_column, cell_type_column])
        data[root_column] = data[root_column].astype("int64", copy=False)
        data[cell_type_column] = data[cell_type_column].astype(str)
        deduped = data.drop_duplicates(subset=[root_column])
        return {int(row[root_column]): str(row[cell_type_column]) for _, row in deduped.iterrows()}


def _plot_synapse_counts(dataframe: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    top = dataframe.nlargest(20, "synapse_count") if not dataframe.empty else dataframe
    ax.barh(
        [f"{row.pre_root_id}→{row.post_root_id}" for row in top.itertuples()],
        top["synapse_count"].to_list(),
        color="steelblue",
    )
    ax.set_xlabel("Synapse count")
    ax.set_ylabel("Connection")
    ax.set_title("Top synaptic connections")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_graph(graph: nx.DiGraph, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    if graph.number_of_nodes() == 0:
        ax.set_axis_off()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return
    pos = nx.spring_layout(graph, seed=42)
    weights = [graph[u][v]["synapse_count"] for u, v in graph.edges]
    if weights:
        max_weight = max(weights)
        width = [0.1 + (weight / max_weight if max_weight else 0.1) for weight in weights]
    else:  # pragma: no cover - empty graph already handled above
        width = 0.5
    nx.draw_networkx(
        graph,
        pos=pos,
        ax=ax,
        with_labels=True,
        node_color="lightgrey",
        edge_color=weights,
        edge_cmap=plt.cm.Blues,
        width=width,
        arrows=True,
        arrowsize=10,
        node_size=400,
        font_size=6,
    )
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _build_graph(dataframe: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    for row in dataframe.itertuples(index=False):
        graph.add_node(row.pre_root_id, cell_type=row.pre_cell_type)
        graph.add_node(row.post_root_id, cell_type=row.post_cell_type)
        graph.add_edge(
            row.pre_root_id,
            row.post_root_id,
            synapse_count=int(row.synapse_count),
            pre_cell_type=row.pre_cell_type,
            post_cell_type=row.post_cell_type,
            snapshot_pre=int(row.pre_pt_root_id),
            snapshot_post=int(row.post_pt_root_id),
        )
    return graph


def _choose_root_column(schema: Sequence[str]) -> str:
    if schema:
        chosen = _choose_column(schema, ("pt_root_id", "root_id", "pre_pt_root_id", "post_pt_root_id"))
        if chosen:
            return chosen
    return "pt_root_id"


def _choose_cell_type_column(schema: Sequence[str]) -> str:
    if schema:
        chosen = _choose_column(schema, ("cell_type", "celltype", "type", "class"))
        if chosen:
            return chosen
    return "cell_type"


def _choose_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        column = lowered.get(candidate)
        if column:
            return column
    for column in columns:
        label = column.lower()
        if any(candidate in label for candidate in candidates):
            return column
    return None
