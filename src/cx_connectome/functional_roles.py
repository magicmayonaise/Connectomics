"""Annotate CX neurons with functional roles based on known cell types.

This utility loads a pickled NetworkX graph built from the central complex
connectivity network (N1→N2→N3), tags neurons with functional roles derived
from literature-reported cell types, and prints summaries of sleep-related and
navigation-related neurons by network layer.
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import networkx as nx
import pandas as pd

# Mapping of cell-type names to functional labels informed by Wolff et al.
# and related CX literature. Keys are normalized (see ``_normalize_cell_type``).
DEFAULT_FUNCTIONAL_MAPPING: Mapping[str, str] = {
    # Sleep-promoting / clock-related interneurons
    "hdeltaf": "Sleep-Promoting",
    "pen_b": "Sleep-Promoting",
    "penb": "Sleep-Promoting",
    "pfg": "Sleep-Promoting",
    "smp368": "Sleep-Promoting",
    # Navigation-related compass and path-integration pathways
    "pfna": "Navigation",
    "pfnb": "Navigation",
    "pfnc": "Navigation",
    "pfnd": "Navigation",
    "pfr": "Navigation",
    "epg": "Navigation",
}


def _normalize_cell_type(cell_type: str) -> str:
    """Return a normalized cell-type string for dictionary lookup."""

    return (
        cell_type.strip()
        .lower()
        .replace("Δ", "delta")
        .replace("-", "_")
        .replace(" ", "")
    )


def annotate_functional_roles(
    graph: nx.DiGraph, mapping: Mapping[str, str] = DEFAULT_FUNCTIONAL_MAPPING
) -> None:
    """Attach a ``FunctionalRole`` attribute to each graph node.

    Parameters
    ----------
    graph:
        NetworkX directed graph containing neuron nodes with a ``cell_type``
        node attribute.
    mapping:
        Dictionary mapping normalized cell-type names to functional labels.
    """

    normalized_mapping = {key.lower(): value for key, value in mapping.items()}
    for node_id, data in graph.nodes(data=True):
        cell_type = data.get("cell_type")
        role: str | None = None
        if isinstance(cell_type, str):
            role = normalized_mapping.get(_normalize_cell_type(cell_type))
        graph.nodes[node_id]["FunctionalRole"] = role


def infer_layer_membership(graph: nx.DiGraph) -> dict[str, set[int]]:
    """Infer N1/N2/N3 membership based on edge layer annotations."""

    membership: dict[str, set[int]] = {"N1": set(), "N2": set(), "N3": set()}
    for pre, post, data in graph.edges(data=True):
        layer = data.get("layer")
        if not isinstance(layer, str) or "->" not in layer:
            continue
        pre_label, post_label = (part.strip().upper() for part in layer.split("->", 1))
        if pre_label in membership:
            membership[pre_label].add(int(pre))
        if post_label in membership:
            membership[post_label].add(int(post))
    return membership


def collect_functional_neurons(
    graph: nx.DiGraph, membership: Mapping[str, Iterable[int]]
) -> dict[str, list[dict[str, object]]]:
    """Gather neurons with functional annotations along with their layers."""

    summary: dict[str, list[dict[str, object]]] = defaultdict(list)
    for node_id, data in graph.nodes(data=True):
        role = data.get("FunctionalRole")
        if not isinstance(role, str):
            continue
        cell_type = data.get("cell_type") if isinstance(data.get("cell_type"), str) else pd.NA
        layers = sorted(layer for layer, nodes in membership.items() if int(node_id) in nodes)
        summary[role].append({"id": int(node_id), "cell_type": cell_type, "layers": layers})
    return summary


def _format_role_section(role: str, entries: list[dict[str, object]]) -> str:
    if not entries:
        return f"No neurons annotated as {role}."

    lines = [f"{role} neurons ({len(entries)} total):"]
    for entry in sorted(entries, key=lambda item: item["id"]):
        layer_label = ", ".join(entry["layers"]) if entry["layers"] else "Unassigned"
        lines.append(
            f"  - ID {entry['id']} | Type: {entry['cell_type']} | Layer(s): {layer_label}"
        )
    return "\n".join(lines)


def _print_summary(summary: Mapping[str, list[dict[str, object]]]) -> None:
    sleep_entries = summary.get("Sleep-Promoting", [])
    navigation_entries = summary.get("Navigation", [])

    if not sleep_entries and not navigation_entries:
        print("No sleep- or navigation-related cell types were found at N1, N2, or N3.")
        return

    print(_format_role_section("Sleep-Promoting", sleep_entries))
    print()
    print(_format_role_section("Navigation", navigation_entries))


def _save_graph(graph: nx.DiGraph, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Annotate CX neurons with functional roles.")
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=Path("cx_network_N1_N2_N3.gpickle"),
        help="Path to the input pickled NetworkX graph (default: cx_network_N1_N2_N3.gpickle)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("cx_network_annotated.gpickle"),
        help="Destination for the annotated graph (default: cx_network_annotated.gpickle)",
    )

    args = parser.parse_args(argv)

    if not args.graph_path.exists():
        raise FileNotFoundError(f"Input graph not found: {args.graph_path}")

    with args.graph_path.open("rb") as handle:
        graph: nx.DiGraph = pickle.load(handle)

    annotate_functional_roles(graph)
    membership = infer_layer_membership(graph)
    summary = collect_functional_neurons(graph, membership)

    _print_summary(summary)
    _save_graph(graph, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
