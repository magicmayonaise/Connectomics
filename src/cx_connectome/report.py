"""Tools for exporting connectome reports.

This module provides a lightweight data model around a connectome dataset and
helpers for exporting the canonical tables used in reports.  It also exposes a
command line interface under the ``cx-report`` entry point with a ``build``
sub-command.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:  # pragma: no cover - handled at runtime
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    plt = None
    MATPLOTLIB_ERROR = exc
else:  # pragma: no cover - handled at runtime
    MATPLOTLIB_ERROR = None

LOGGER = logging.getLogger(__name__)


def _canonical_layer(layer: Optional[str]) -> str:
    """Normalise a layer label into the canonical ``N#`` form."""

    if not layer:
        return ""
    text = str(layer).strip().upper()
    if text.startswith("N") and len(text) > 1:
        return f"N{text[1:]}"
    return text


def _stringify(value: Any) -> str:
    """Convert values to a human-readable string for table export."""

    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (int, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(_stringify(item) for item in value)
    if isinstance(value, dict):
        return "; ".join(f"{key}={_stringify(val)}" for key, val in sorted(value.items()))
    return str(value)


def _format_properties(properties: Mapping[str, Any]) -> str:
    """Format a property dictionary for export."""

    if not properties:
        return ""
    return "; ".join(
        f"{key}={_stringify(value)}" for key, value in sorted(properties.items())
    )


def _parse_generic_value(value: Any) -> Any:
    """Best-effort conversion for string values from CSV sources."""

    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:  # pragma: no cover - defensive
        return text
    return parsed


def _pop_first(data: MutableMapping[str, Any], *keys: str) -> Optional[Any]:
    """Remove the first matching key from a mapping and return its value."""

    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data.pop(key)
    return None


@dataclass
class Node:
    """Representation of a node within the connectome."""

    identifier: str
    layer: str
    label: str = ""
    properties: Mapping[str, Any] = field(default_factory=dict)

    @property
    def canonical_layer(self) -> str:
        return _canonical_layer(self.layer)

    def display_name(self) -> str:
        return self.label or self.identifier


@dataclass
class Edge:
    """Representation of an edge between two nodes."""

    source: str
    target: str
    weight: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TableSpec:
    """A table description used for exports."""

    name: str
    title: str
    columns: Sequence[str]
    rows: List[Mapping[str, Any]]


class GraphData:
    """In-memory representation of a connectome graph."""

    def __init__(self, nodes: Iterable[Node], edges: Iterable[Edge]):
        self.nodes: Dict[str, Node] = {node.identifier: node for node in nodes}
        self.edges: List[Edge] = list(edges)
        self._out_edges: Dict[str, List[Edge]] = defaultdict(list)
        self._in_edges: Dict[str, List[Edge]] = defaultdict(list)
        for edge in self.edges:
            self._out_edges[edge.source].append(edge)
            self._in_edges[edge.target].append(edge)

    def nodes_by_layer(self, layer: str) -> List[Node]:
        canonical = _canonical_layer(layer)
        return [
            node for node in self.nodes.values() if node.canonical_layer == canonical
        ]

    def edges_between_layers(self, source_layer: str, target_layer: str) -> Iterator[Tuple[Edge, Node, Node]]:
        source_canonical = _canonical_layer(source_layer)
        target_canonical = _canonical_layer(target_layer)
        for edge in self.edges:
            source_node = self.nodes.get(edge.source)
            target_node = self.nodes.get(edge.target)
            if not source_node or not target_node:
                continue
            if (
                source_node.canonical_layer == source_canonical
                and target_node.canonical_layer == target_canonical
            ):
                yield edge, source_node, target_node

    def upstream_nodes(
        self, node_id: str, layer: Optional[str] = None
    ) -> List[Tuple[Node, Edge]]:
        canonical = _canonical_layer(layer) if layer else None
        result: List[Tuple[Node, Edge]] = []
        for edge in self._in_edges.get(node_id, []):
            upstream_node = self.nodes.get(edge.source)
            if not upstream_node:
                continue
            if canonical and upstream_node.canonical_layer != canonical:
                continue
            result.append((upstream_node, edge))
        return result

    def downstream_nodes(
        self, node_id: str, layer: Optional[str] = None
    ) -> List[Tuple[Node, Edge]]:
        canonical = _canonical_layer(layer) if layer else None
        result: List[Tuple[Node, Edge]] = []
        for edge in self._out_edges.get(node_id, []):
            downstream_node = self.nodes.get(edge.target)
            if not downstream_node:
                continue
            if canonical and downstream_node.canonical_layer != canonical:
                continue
            result.append((downstream_node, edge))
        return result


def load_graph(dataset_path: Path) -> Tuple[GraphData, Dict[str, Any]]:
    """Load a dataset from a JSON file or a directory of CSVs."""

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    if dataset_path.is_dir():
        return _load_graph_from_directory(dataset_path)
    return _load_graph_from_file(dataset_path)


def _load_graph_from_directory(directory: Path) -> Tuple[GraphData, Dict[str, Any]]:
    nodes_file = directory / "nodes.csv"
    edges_file = directory / "edges.csv"
    if not nodes_file.exists() or not edges_file.exists():
        raise FileNotFoundError(
            "Expected 'nodes.csv' and 'edges.csv' inside the dataset directory"
        )

    nodes: List[Node] = []
    edges: List[Edge] = []

    with nodes_file.open("r", encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = {key: _parse_generic_value(value) for key, value in raw_row.items()}
            node_id = _pop_first(row, "id", "node_id", "identifier")
            if not node_id:
                raise ValueError("Node entry missing 'id' column")
            layer = _pop_first(row, "layer", "type", "group", "category") or ""
            label = _pop_first(row, "label", "name", "title") or str(node_id)
            nodes.append(
                Node(
                    identifier=str(node_id),
                    layer=str(layer),
                    label=str(label),
                    properties=row,
                )
            )

    with edges_file.open("r", encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = {key: _parse_generic_value(value) for key, value in raw_row.items()}
            source = _pop_first(row, "source", "src", "from")
            target = _pop_first(row, "target", "dst", "to")
            if not source or not target:
                raise ValueError("Edge entry missing source/target columns")
            weight_value = _pop_first(row, "weight", "w", "strength")
            weight: Optional[float] = None
            if weight_value is not None:
                try:
                    weight = float(weight_value)
                except (TypeError, ValueError):
                    row.setdefault("weight", weight_value)
            edges.append(
                Edge(
                    source=str(source),
                    target=str(target),
                    weight=weight,
                    metadata=row,
                )
            )

    metadata: Dict[str, Any] = {}
    metadata_file = directory / "metadata.json"
    if metadata_file.exists():
        with metadata_file.open("r", encoding="utf8") as handle:
            raw_metadata = json.load(handle)
        if isinstance(raw_metadata, dict):
            metadata.update(raw_metadata)

    return GraphData(nodes, edges), metadata


def _load_graph_from_file(dataset_file: Path) -> Tuple[GraphData, Dict[str, Any]]:
    with dataset_file.open("r", encoding="utf8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise ValueError("Dataset file must contain a JSON object")

    metadata: Dict[str, Any] = {}
    metadata.update(payload.get("metadata", {}))
    for key in ("dataset", "materialization", "timestamp", "thresholds"):
        if key in payload and key not in metadata:
            metadata[key] = payload[key]

    raw_nodes = payload.get("nodes", [])
    raw_edges = payload.get("edges", [])

    nodes: List[Node] = []
    for entry in raw_nodes:
        if not isinstance(entry, Mapping):
            continue
        data = dict(entry)
        node_id = data.pop("id", None) or data.pop("node_id", None) or data.pop("identifier", None)
        if not node_id:
            raise ValueError("Node entry missing identifier")
        layer = data.pop("layer", None) or data.pop("type", None) or data.pop("group", None) or ""
        label = data.pop("label", None) or data.pop("name", None) or data.pop("title", None) or str(node_id)
        properties = data.pop("properties", None)
        if properties and isinstance(properties, Mapping):
            data.update(properties)  # merge nested properties
        nodes.append(
            Node(
                identifier=str(node_id),
                layer=str(layer),
                label=str(label),
                properties=data,
            )
        )

    edges: List[Edge] = []
    for entry in raw_edges:
        if not isinstance(entry, Mapping):
            continue
        data = dict(entry)
        source = data.pop("source", None) or data.pop("src", None)
        target = data.pop("target", None) or data.pop("dst", None)
        if not source or not target:
            raise ValueError("Edge entry missing source/target")
        weight_value = data.pop("weight", None) or data.pop("w", None)
        weight: Optional[float] = None
        if weight_value is not None:
            try:
                weight = float(weight_value)
            except (TypeError, ValueError):
                data.setdefault("weight", weight_value)
        metadata_payload = data.pop("metadata", None)
        if isinstance(metadata_payload, Mapping):
            data.update(metadata_payload)
        edges.append(
            Edge(
                source=str(source),
                target=str(target),
                weight=weight,
                metadata=data,
            )
        )

    return GraphData(nodes, edges), metadata


def build_table_n1_roster(graph: GraphData) -> TableSpec:
    rows = [
        {
            "node_id": node.identifier,
            "name": node.display_name(),
            "layer": node.canonical_layer,
            "properties": _format_properties(node.properties),
        }
        for node in sorted(graph.nodes_by_layer("N1"), key=lambda item: item.display_name())
    ]
    return TableSpec(
        name="table_1_1_n1_roster",
        title="Table 1.1 – N1 roster",
        columns=["node_id", "name", "layer", "properties"],
        rows=rows,
    )


def build_table_edge_flow(
    graph: GraphData,
    source_layer: str,
    target_layer: str,
    name: str,
    title: str,
) -> TableSpec:
    rows = []
    for edge, source_node, target_node in graph.edges_between_layers(source_layer, target_layer):
        rows.append(
            {
                "source": source_node.display_name(),
                "target": target_node.display_name(),
                "weight": "" if edge.weight is None else _stringify(edge.weight),
                "metadata": _format_properties(edge.metadata),
            }
        )
    rows.sort(key=lambda row: (row["source"], row["target"]))
    return TableSpec(
        name=name,
        title=title,
        columns=["source", "target", "weight", "metadata"],
        rows=rows,
    )


def build_table_n2n3_properties(graph: GraphData) -> TableSpec:
    target_layers = {"N2", "N3"}
    rows = []
    for node in graph.nodes.values():
        if node.canonical_layer not in target_layers:
            continue
        rows.append(
            {
                "node_id": node.identifier,
                "name": node.display_name(),
                "layer": node.canonical_layer,
                "properties": _format_properties(node.properties),
            }
        )
    rows.sort(key=lambda row: (row["layer"], row["name"]))
    return TableSpec(
        name="table_3_1_n2_n3_properties",
        title="Table 3.1 – N2/N3 properties",
        columns=["node_id", "name", "layer", "properties"],
        rows=rows,
    )


def build_motif_summary(graph: GraphData) -> TableSpec:
    motif_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for n2_node in graph.nodes_by_layer("N2"):
        upstream_n1 = {
            upstream.display_name()
            for upstream, _ in graph.upstream_nodes(n2_node.identifier, "N1")
        }
        downstream_n3 = {
            downstream.display_name()
            for downstream, _ in graph.downstream_nodes(n2_node.identifier, "N3")
        }
        for n1_name in upstream_n1:
            for n3_name in downstream_n3:
                motif_counts[(n1_name, n2_node.display_name(), n3_name)] += 1
    rows = [
        {
            "n1": n1,
            "n2": n2,
            "n3": n3,
            "count": count,
        }
        for (n1, n2, n3), count in sorted(
            motif_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1], item[0][2])
        )
    ]
    return TableSpec(
        name="motif_summary",
        title="Motif summary",
        columns=["n1", "n2", "n3", "count"],
        rows=rows,
    )


def build_topology_summary(graph: GraphData) -> TableSpec:
    rows: List[Dict[str, Any]] = []
    layer_counts: Dict[str, int] = defaultdict(int)
    for node in graph.nodes.values():
        layer_counts[node.canonical_layer] += 1
    for layer, count in sorted(layer_counts.items()):
        rows.append({"category": "nodes", "descriptor": layer or "(unknown)", "count": count})

    edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for edge, source_node, target_node in (
        (edge, graph.nodes.get(edge.source), graph.nodes.get(edge.target))
        for edge in graph.edges
    ):
        if not source_node or not target_node:
            continue
        edge_counts[(source_node.canonical_layer, target_node.canonical_layer)] += 1

    for (source_layer, target_layer), count in sorted(edge_counts.items()):
        descriptor = f"{source_layer or '?'}→{target_layer or '?'}"
        rows.append({"category": "edges", "descriptor": descriptor, "count": count})

    return TableSpec(
        name="topology_summary",
        title="Topology summary",
        columns=["category", "descriptor", "count"],
        rows=rows,
    )


def build_upstream_summary(graph: GraphData) -> TableSpec:
    rows: List[Dict[str, Any]] = []
    for n3_node in sorted(graph.nodes_by_layer("N3"), key=lambda node: node.display_name()):
        incoming_n2 = graph.upstream_nodes(n3_node.identifier, "N2")
        n2_names = sorted({node.display_name() for node, _ in incoming_n2})
        upstream_n1: set[str] = set()
        for n2_node, _ in incoming_n2:
            upstream_n1.update(
                upstream.display_name()
                for upstream, _ in graph.upstream_nodes(n2_node.identifier, "N1")
            )
        direct_n1 = {
            upstream.display_name()
            for upstream, _ in graph.upstream_nodes(n3_node.identifier, "N1")
        }
        all_n1 = sorted(upstream_n1 | direct_n1)
        rows.append(
            {
                "n3": n3_node.display_name(),
                "n2_support": ", ".join(n2_names),
                "n1_count": len(all_n1),
                "n1_sources": ", ".join(all_n1),
            }
        )
    return TableSpec(
        name="upstream_summary",
        title="Upstream summary",
        columns=["n3", "n2_support", "n1_count", "n1_sources"],
        rows=rows,
    )


def export_table(table: TableSpec, output_dir: Path) -> Dict[str, Any]:
    """Export a table to CSV and PNG formats."""

    if MATPLOTLIB_ERROR is not None:
        raise RuntimeError(
            "matplotlib is required for exporting PNG tables"
        ) from MATPLOTLIB_ERROR

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{table.name}.csv"
    png_path = output_dir / f"{table.name}.png"

    normalized_rows = [
        {column: _stringify(row.get(column)) for column in table.columns}
        for row in table.rows
    ]

    with csv_path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table.columns))
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow(row)

    cell_text: List[List[str]]
    if normalized_rows:
        cell_text = [[row[column] for column in table.columns] for row in normalized_rows]
    else:
        cell_text = [["(no data)"] + [""] * (len(table.columns) - 1)]

    fig_width = max(6.0, len(table.columns) * 2.0)
    fig_height = max(2.5, len(cell_text) * 0.6 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table_artist = ax.table(
        cellText=cell_text,
        colLabels=list(table.columns),
        cellLoc="center",
        loc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(8)
    table_artist.scale(1, 1.2)
    ax.set_title(table.title, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    return {
        "name": table.name,
        "title": table.title,
        "csv": str(csv_path),
        "png": str(png_path),
        "row_count": len(table.rows),
    }


def build_report(
    dataset: Path | str,
    output_dir: Path | str = Path("out/fig"),
    thresholds: Optional[Mapping[str, float]] = None,
    *,
    dataset_name: Optional[str] = None,
    materialization: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate the full connectome report.

    Parameters
    ----------
    dataset:
        Path to a dataset JSON file or a directory containing ``nodes.csv`` and
        ``edges.csv``.
    output_dir:
        Directory where the exported artefacts should be created.  Defaults to
        ``out/fig``.
    thresholds:
        Optional dictionary of threshold values used during the run.
    dataset_name, materialization, timestamp:
        Optional overrides for the metadata fields stored in ``run_summary``.
    """

    dataset_path = Path(dataset)
    graph, metadata = load_graph(dataset_path)

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_label = dataset_name or metadata.get("dataset") or dataset_path.stem
    materialization_value = materialization or metadata.get("materialization")
    dataset_timestamp = timestamp or metadata.get("timestamp")

    combined_thresholds: Dict[str, float] = {}
    if isinstance(metadata.get("thresholds"), Mapping):
        for key, value in metadata["thresholds"].items():
            try:
                combined_thresholds[str(key)] = float(value)
            except (TypeError, ValueError):
                LOGGER.debug("Skipping non-numeric threshold %s=%s", key, value)
    if thresholds:
        for key, value in thresholds.items():
            try:
                combined_thresholds[str(key)] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid threshold value for {key}: {value}") from exc

    tables: List[TableSpec] = [
        build_table_n1_roster(graph),
        build_table_edge_flow(
            graph,
            "N1",
            "N2",
            name="table_2_1_n1_to_n2",
            title="Table 2.1 – N1→N2 connectivity",
        ),
        build_table_edge_flow(
            graph,
            "N2",
            "N3",
            name="table_2_1_n2_to_n3",
            title="Table 2.1 – N2→N3 connectivity",
        ),
        build_table_n2n3_properties(graph),
        build_motif_summary(graph),
        build_topology_summary(graph),
        build_upstream_summary(graph),
    ]

    table_outputs = [export_table(table, resolved_output_dir) for table in tables]

    generated_at = datetime.now(timezone.utc).isoformat()
    summary_path = resolved_output_dir / "run_summary.json"
    run_summary = {
        "dataset": dataset_label,
        "materialization": materialization_value,
        "dataset_timestamp": dataset_timestamp,
        "generated_at": generated_at,
        "thresholds": combined_thresholds,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "table_names": [entry["name"] for entry in table_outputs],
        "tables": table_outputs,
        "output_directory": str(resolved_output_dir.resolve()),
        "dataset_source": str(dataset_path.resolve()),
    }

    with summary_path.open("w", encoding="utf8") as handle:
        json.dump(run_summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    LOGGER.info("Report artefacts created in %s", resolved_output_dir)
    return {**run_summary, "run_summary_path": str(summary_path.resolve())}


def _parse_threshold_pairs(pairs: Optional[Sequence[str]]) -> Dict[str, float]:
    if not pairs:
        return {}
    parsed: Dict[str, float] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Threshold '{item}' must be in KEY=VALUE format")
        key, value_text = item.split("=", 1)
        try:
            parsed[key.strip()] = float(value_text)
        except ValueError as exc:
            raise ValueError(f"Invalid threshold value for {key!r}: {value_text!r}") from exc
    return parsed


def _handle_build_command(args: argparse.Namespace) -> Dict[str, Any]:
    thresholds: Dict[str, float]
    try:
        thresholds = _parse_threshold_pairs(args.threshold)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    return build_report(
        dataset=args.dataset,
        output_dir=args.output,
        thresholds=thresholds,
        dataset_name=args.dataset_name,
        materialization=args.materialization,
        timestamp=args.timestamp,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cx-report",
        description="Build combined figure-table exports for connectome datasets.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="cx-report 1.0",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )

    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser(
        "build",
        help="Generate CSV and PNG artefacts for the connectome report.",
    )
    build_parser.add_argument(
        "dataset",
        help="Path to a dataset JSON file or directory with 'nodes.csv' and 'edges.csv'.",
    )
    build_parser.add_argument(
        "--output",
        default=str(Path("out/fig")),
        help="Directory where artefacts should be written (default: out/fig).",
    )
    build_parser.add_argument(
        "--dataset-name",
        dest="dataset_name",
        help="Override the dataset name stored in run_summary.json.",
    )
    build_parser.add_argument(
        "--materialization",
        help="Override the materialization identifier for run_summary.json.",
    )
    build_parser.add_argument(
        "--timestamp",
        help="Override the dataset timestamp stored in run_summary.json.",
    )
    build_parser.add_argument(
        "--threshold",
        action="append",
        metavar="KEY=VALUE",
        help="Threshold definitions applied during the run (may be repeated).",
    )
    build_parser.set_defaults(handler=_handle_build_command)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 1

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        result = args.handler(args)
    except Exception as exc:  # pragma: no cover - CLI error handling
        LOGGER.error("Failed to build report: %s", exc)
        return 1

    LOGGER.debug("Run summary: %s", result)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
