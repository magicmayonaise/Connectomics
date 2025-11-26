"""Build CX connectivity graphs directly from FlyWire data.

This module wires together existing CAVE helpers to construct a two-hop
connectivity network (N1→N2→N3) from a list of seed neuron IDs.  The resulting
graph captures the synaptic relationships between the seed population (N1),
their immediate partners (N2), and the partners-of-partners (N3).  Convenience
outputs include adjacency parquet files and a pickled NetworkX graph for
downstream analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from cx_connectome import annotations, cave_io, idtools
from cx_connectome.adjacency import build_connectivity

LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET = "flywire_fafb_production"
DEFAULT_MATERIALIZATION = 630
DEFAULT_SECOND_HOP_THRESHOLD = 10
DEFAULT_FIRST_HOP_THRESHOLD = 5
DEFAULT_N1_N2_OUTPUT = Path("out/N1_to_N2_adjacency.parquet")
DEFAULT_N2_N3_OUTPUT = Path("out/N2_to_N3_adjacency.parquet")
DEFAULT_GRAPH_OUTPUT = Path("out/cx_network_N1_N2_N3.gpickle")


def _read_root_ids(path: Path) -> list[int]:
    """Return integer root IDs parsed from ``path``.

    Lines starting with ``#`` and blank lines are ignored. Values can be
    comma-separated or whitespace separated.
    """

    text = path.read_text(encoding="utf8").splitlines()
    roots: list[int] = []
    for line in text:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for chunk in stripped.replace(",", " ").split():
            if not chunk:
                continue
            try:
                roots.append(int(chunk))
            except ValueError as exc:  # pragma: no cover - defensive parsing
                raise ValueError(f"Invalid root ID '{chunk}' in {path}") from exc
    return roots


def _ensure_annotation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Guarantee annotation columns for graph node attributes."""

    columns = {
        "cell_type": pd.Series([pd.NA] * len(frame), dtype="string"),
        "super_class": pd.Series([pd.NA] * len(frame), dtype="string"),
        "side": pd.Series([pd.NA] * len(frame), dtype="string"),
    }
    for name, series in columns.items():
        if name not in frame.columns:
            frame[name] = series
    return frame


def _add_edges(graph: nx.DiGraph, adjacency: pd.DataFrame) -> None:
    """Insert directed edges from an adjacency dataframe into ``graph``."""

    if adjacency.empty:
        return

    for _, row in adjacency.iterrows():
        pre = int(row["pre_root_id"])
        post = int(row["post_root_id"])
        attributes = {
            "synapse_count": int(row["syn_count"]),
            "layer": row.get("layer", pd.NA),
            "pre_cell_type": row.get("pre_cell_type", pd.NA),
            "post_cell_type": row.get("post_cell_type", pd.NA),
        }
        graph.add_edge(pre, post, **attributes)


def build_connectivity_graph(
    seed_path: Path,
    *,
    dataset: str = DEFAULT_DATASET,
    client: Any | None = None,
    materialization: int | None = DEFAULT_MATERIALIZATION,
    first_hop_threshold: int = DEFAULT_FIRST_HOP_THRESHOLD,
    second_hop_threshold: int = DEFAULT_SECOND_HOP_THRESHOLD,
    n1_n2_output: Path = DEFAULT_N1_N2_OUTPUT,
    n2_n3_output: Path = DEFAULT_N2_N3_OUTPUT,
    graph_output: Path = DEFAULT_GRAPH_OUTPUT,
) -> tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame]:
    """Construct the N1→N2→N3 connectivity network from FlyWire data.

    Parameters
    ----------
    seed_path:
        File containing one or more seed root IDs (N1) to expand from.
    dataset:
        FlyWire datastack name used to initialise the CAVE client.
    client:
        Optional pre-created client. When omitted, :func:`cave_io.get_client`
        is used to create one for ``dataset``.
    materialization:
        Materialization version used for lineage resolution and synapse queries.
    first_hop_threshold:
        Minimum synapse count to retain N1→N2 edges.
    second_hop_threshold:
        Minimum synapse count to retain N2→N3 edges.
    n1_n2_output / n2_n3_output:
        Locations where the adjacency parquet files will be written.
    graph_output:
        Destination for the pickled NetworkX graph.
    """

    if client is None:
        LOGGER.info("Creating CAVEclient for dataset %s", dataset)
        client = cave_io.get_client(dataset)

    seeds = _read_root_ids(seed_path)
    if not seeds:
        raise ValueError(f"No seed root IDs found in {seed_path}")

    LOGGER.info("Updating %d seed root IDs using lineage", len(seeds))
    resolved = idtools.update_root_ids(client, seeds, materialization, None)
    n1_roots = [resolved[root] for root in seeds]

    LOGGER.info("Querying first-hop connectivity (N1→N2)")
    n1_n2 = build_connectivity(
        client=client,
        pre_roots=n1_roots,
        threshold=first_hop_threshold,
        layer_tag="N1->N2",
        materialization=materialization,
        output_path=n1_n2_output,
    )

    n2_candidates = (
        []
        if n1_n2.empty
        else sorted(n1_n2["post_root_id"].astype("int64").unique().tolist())
    )

    LOGGER.info("Querying second-hop connectivity (N2→N3)")
    n2_n3 = build_connectivity(
        client=client,
        pre_roots=n2_candidates,
        threshold=second_hop_threshold,
        layer_tag="N2->N3",
        materialization=materialization,
        output_path=n2_n3_output,
    )

    graph = nx.DiGraph()
    _add_edges(graph, n1_n2)
    _add_edges(graph, n2_n3)

    involved_nodes: set[int] = set(n1_roots)
    if not n1_n2.empty:
        involved_nodes.update(n1_n2["post_root_id"].astype("int64"))
    if not n2_n3.empty:
        involved_nodes.update(n2_n3["post_root_id"].astype("int64"))
        involved_nodes.update(n2_n3["pre_root_id"].astype("int64"))

    LOGGER.info("Fetching annotations for %d nodes", len(involved_nodes))
    annotations_df = annotations.fetch_cell_types(
        client,
        involved_nodes,
        materialization=materialization,
    )
    annotations_df = _ensure_annotation_columns(annotations_df).set_index("root_id")

    for node in involved_nodes:
        data = annotations_df.loc[node] if node in annotations_df.index else None
        attributes = {
            "cell_type": data.cell_type if data is not None else pd.NA,
            "super_class": data.super_class if data is not None else pd.NA,
            "side": data.side if data is not None else pd.NA,
        }
        graph.add_node(int(node), **attributes)

    graph_output = Path(graph_output)
    graph_output.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Writing graph with %d nodes and %d edges to %s",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        graph_output,
    )
    nx.write_gpickle(graph, graph_output)

    LOGGER.info(
        "Example node: %s", next(iter(graph.nodes(data=True)), (None, {}))
    )

    return graph, n1_n2, n2_n3


__all__ = [
    "build_connectivity_graph",
]
