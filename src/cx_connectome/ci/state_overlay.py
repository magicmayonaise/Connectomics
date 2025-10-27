"""State overlay analysis utilities.

This module provides the logic backing the ``state-overlay`` CI command. It
loads curated state seed sets, collapses a toy CX connectome into high level
streams, computes directional flow metrics (including a simple recurrent flow
coefficient), and renders a summary diagram showing the dominant influence from
state seeds into the CX and vice versa.

The implementation is intentionally data driven: the command expects
``data/state_seeds.csv`` and uses the reference ``data/cx_streams.csv`` and
``data/connectivity.csv`` tables bundled with the repository. The functions can
be reused in notebooks or tests by injecting alternative paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from xml.etree.ElementTree import Element, ElementTree, SubElement
import csv


@dataclass(frozen=True)
class SeedNeuron:
    """Representation of a neuron that participates in a seed set."""

    neuron_id: str
    neuron_type: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class SeedSet:
    """A named collection of neurons representing a state."""

    name: str
    neurons: List[SeedNeuron]

    def neuron_ids(self) -> List[str]:
        return [neuron.neuron_id for neuron in self.neurons]


@dataclass(frozen=True)
class StreamNeuron:
    """Metadata describing a neuron that forms part of a CX stream."""

    neuron_id: str
    hemisphere: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class StreamDefinition:
    """A CX stream grouping together relevant neurons."""

    name: str
    members: List[StreamNeuron]

    def neuron_ids(self) -> List[str]:
        return [member.neuron_id for member in self.members]


@dataclass(frozen=True)
class Edge:
    """A single directed connection in the connectivity table."""

    pre: str
    post: str
    weight: float
    sign: int


@dataclass
class FlowMetrics:
    """Directional flow statistics between a seed set and a CX stream."""

    forward_net: float
    reverse_net: float
    forward_abs: float
    reverse_abs: float
    forward_count: int
    reverse_count: int

    @property
    def total_abs(self) -> float:
        return self.forward_abs + self.reverse_abs

    @property
    def rfc(self) -> float:
        """Recurrent flow coefficient (forward vs reverse bias).

        Defined as (|forward| - |reverse|) / (|forward| + |reverse|).
        When both directions are silent the value is zero.
        """

        if self.total_abs == 0:
            return 0.0
        return (self.forward_abs - self.reverse_abs) / self.total_abs

    @property
    def net_signed_total(self) -> float:
        return self.forward_net - self.reverse_net

    @property
    def dominant_direction(self) -> str:
        if self.forward_abs == self.reverse_abs:
            return "balanced"
        return "forward" if self.forward_abs > self.reverse_abs else "reverse"

    def to_row(self, seed_set: str, stream: str) -> Dict[str, object]:
        return {
            "seed_set": seed_set,
            "stream": stream,
            "forward_net": round(self.forward_net, 6),
            "reverse_net": round(self.reverse_net, 6),
            "forward_abs": round(self.forward_abs, 6),
            "reverse_abs": round(self.reverse_abs, 6),
            "forward_count": self.forward_count,
            "reverse_count": self.reverse_count,
            "rfc": round(self.rfc, 6),
            "net_signed_total": round(self.net_signed_total, 6),
            "dominant_direction": self.dominant_direction,
        }


@dataclass
class StateOverlayResult:
    """Results returned by :func:`compute_state_overlay`."""

    seed_sets: Mapping[str, SeedSet]
    streams: Mapping[str, StreamDefinition]
    flows: Mapping[str, Mapping[str, FlowMetrics]]
    metrics_path: Path
    figure_path: Path


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Return the root directory of the repository."""

    return Path(__file__).resolve().parents[3]


def load_state_seed_sets(csv_path: Path) -> Dict[str, SeedSet]:
    """Load curated state seed sets from a CSV file."""

    seed_sets: Dict[str, SeedSet] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"seed_set", "neuron"}
        missing = required_fields - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
        for row in reader:
            seed_name = row["seed_set"].strip()
            neuron_id = row["neuron"].strip()
            neuron_type = row.get("type", "").strip() or None
            notes = row.get("notes", "").strip() or None
            if not seed_name or not neuron_id:
                continue
            seed_sets.setdefault(seed_name, SeedSet(seed_name, [])).neurons.append(
                SeedNeuron(neuron_id=neuron_id, neuron_type=neuron_type, notes=notes)
            )
    return seed_sets


def load_cx_streams(streams_csv: Optional[Path] = None) -> Dict[str, StreamDefinition]:
    """Load CX stream definitions from the reference CSV."""

    if streams_csv is None:
        streams_csv = _repo_root() / "data" / "cx_streams.csv"
    stream_map: Dict[str, StreamDefinition] = {}
    with streams_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"stream", "neuron"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {streams_csv}: {sorted(missing)}")
        for row in reader:
            stream_name = row["stream"].strip()
            neuron_id = row["neuron"].strip()
            hemisphere = row.get("hemisphere", "").strip() or None
            notes = row.get("notes", "").strip() or None
            if not stream_name or not neuron_id:
                continue
            stream_map.setdefault(stream_name, StreamDefinition(stream_name, [])).members.append(
                StreamNeuron(neuron_id=neuron_id, hemisphere=hemisphere, notes=notes)
            )
    return stream_map


def load_connectivity_edges(connectivity_csv: Optional[Path] = None) -> List[Edge]:
    """Load directed connectivity edges from the reference CSV."""

    if connectivity_csv is None:
        connectivity_csv = _repo_root() / "data" / "connectivity.csv"
    edges: List[Edge] = []
    with connectivity_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"pre", "post", "weight", "sign"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {connectivity_csv}: {sorted(missing)}")
        for row in reader:
            try:
                weight = float(row["weight"].strip())
            except ValueError as exc:  # pragma: no cover - validation guard
                raise ValueError(f"Invalid weight for edge {row}") from exc
            try:
                sign = int(row["sign"].strip())
            except ValueError as exc:  # pragma: no cover - validation guard
                raise ValueError(f"Invalid sign for edge {row}") from exc
            edges.append(
                Edge(pre=row["pre"].strip(), post=row["post"].strip(), weight=weight, sign=sign)
            )
    return edges


# ---------------------------------------------------------------------------
# Flow computation
# ---------------------------------------------------------------------------

def _build_edges_by_source(edges: Sequence[Edge]) -> Dict[str, List[Edge]]:
    lookup: Dict[str, List[Edge]] = {}
    for edge in edges:
        lookup.setdefault(edge.pre, []).append(edge)
    return lookup


def _sum_flow(edges: Iterable[Edge]) -> Tuple[float, float, int]:
    net = 0.0
    total = 0.0
    count = 0
    for edge in edges:
        net += edge.weight * edge.sign
        total += abs(edge.weight)
        count += 1
    return net, total, count


def _compute_pair_metrics(
    seed_ids: Sequence[str],
    stream_ids: Sequence[str],
    edges_by_source: Mapping[str, Sequence[Edge]],
) -> FlowMetrics:
    forward_edges: List[Edge] = []
    reverse_edges: List[Edge] = []

    stream_set = set(stream_ids)
    seed_set = set(seed_ids)

    for seed in seed_ids:
        forward_edges.extend(edge for edge in edges_by_source.get(seed, []) if edge.post in stream_set)
    for stream_neuron in stream_ids:
        reverse_edges.extend(
            edge for edge in edges_by_source.get(stream_neuron, []) if edge.post in seed_set
        )

    forward_net, forward_abs, forward_count = _sum_flow(forward_edges)
    reverse_net, reverse_abs, reverse_count = _sum_flow(reverse_edges)
    return FlowMetrics(
        forward_net=forward_net,
        reverse_net=reverse_net,
        forward_abs=forward_abs,
        reverse_abs=reverse_abs,
        forward_count=forward_count,
        reverse_count=reverse_count,
    )


def compute_state_overlay(
    seed_sets: Mapping[str, SeedSet],
    streams: Mapping[str, StreamDefinition],
    edges: Sequence[Edge],
) -> Dict[str, Dict[str, FlowMetrics]]:
    """Compute directional flow metrics for every seed/stream combination."""

    edges_by_source = _build_edges_by_source(edges)
    flows: Dict[str, Dict[str, FlowMetrics]] = {}
    for seed_name, seed_set in seed_sets.items():
        flows[seed_name] = {}
        for stream_name, stream_def in streams.items():
            metrics = _compute_pair_metrics(seed_set.neuron_ids(), stream_def.neuron_ids(), edges_by_source)
            if metrics.forward_count == 0 and metrics.reverse_count == 0:
                continue
            flows[seed_name][stream_name] = metrics
    return flows


# ---------------------------------------------------------------------------
# Rendering and reporting helpers
# ---------------------------------------------------------------------------

def _flow_color(value: float) -> str:
    if value > 1e-9:
        return "#2166ac"  # blue for net excitatory
    if value < -1e-9:
        return "#b2182b"  # red for net inhibitory
    return "#777777"


def _normalise(value: float, maximum: float, base: float = 0.4, scale: float = 2.6) -> float:
    if maximum <= 0:
        return base
    return base + scale * (value / maximum)


def render_summary_circuit(
    flows: Mapping[str, Mapping[str, FlowMetrics]],
    seed_sets: Mapping[str, SeedSet],
    streams: Mapping[str, StreamDefinition],
    output_path: Path,
    title: str = "State → CX stream overlay",
) -> Path:
    """Render a summary diagram visualising the computed flows as an SVG."""

    if not flows:
        raise ValueError("No flows available to render")

    seed_names = sorted(seed_sets.keys())
    stream_names = sorted({stream for stream_map in flows.values() for stream in stream_map.keys()})
    if not stream_names:
        raise ValueError("Flows do not reference any streams")

    spacing = 120.0
    top_padding = 160.0
    bottom_padding = 160.0
    left_x = 200.0
    right_x = 760.0
    width = 960.0
    count = max(len(seed_names), len(stream_names), 1)
    height = top_padding + spacing * (count - 1) + bottom_padding

    y_seed_positions = {
        seed: top_padding + index * spacing for index, seed in enumerate(seed_names)
    }
    y_stream_positions = {
        stream: top_padding + index * spacing for index, stream in enumerate(stream_names)
    }

    magnitudes = [
        max(metrics.forward_abs, metrics.reverse_abs)
        for stream_map in flows.values()
        for metrics in stream_map.values()
        if metrics.total_abs > 0
    ]
    max_flow = max(magnitudes) if magnitudes else 1.0

    svg = Element(
        "svg",
        attrib={
            "xmlns": "http://www.w3.org/2000/svg",
            "width": f"{width:.0f}",
            "height": f"{height:.0f}",
            "viewBox": f"0 0 {width:.0f} {height:.0f}",
        },
    )

    defs = SubElement(svg, "defs")
    marker = SubElement(
        defs,
        "marker",
        attrib={
            "id": "arrowhead",
            "markerWidth": "12",
            "markerHeight": "8",
            "refX": "10",
            "refY": "4",
            "orient": "auto",
            "markerUnits": "strokeWidth",
        },
    )
    SubElement(marker, "path", attrib={"d": "M 0 0 L 10 4 L 0 8 z", "fill": "context-stroke"})

    SubElement(
        svg,
        "text",
        attrib={
            "x": f"{width / 2:.0f}",
            "y": "70",
            "text-anchor": "middle",
            "font-size": "28",
            "font-weight": "600",
        },
    ).text = title

    def _add_node(x: float, y: float, label: str) -> None:
        SubElement(
            svg,
            "circle",
            attrib={
                "cx": f"{x:.1f}",
                "cy": f"{y:.1f}",
                "r": "38",
                "fill": "#f7f7f7",
                "stroke": "#333333",
                "stroke-width": "1.4",
            },
        )
        SubElement(
            svg,
            "text",
            attrib={
                "x": f"{x:.1f}",
                "y": f"{y + 4:.1f}",
                "text-anchor": "middle",
                "font-size": "18",
                "font-family": "Helvetica, Arial, sans-serif",
            },
        ).text = label

    for seed, y in y_seed_positions.items():
        _add_node(left_x, y, seed)
    for stream, y in y_stream_positions.items():
        _add_node(right_x, y, stream)

    def _draw_curve(
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        value: float,
        magnitude: float,
        above: bool,
        opacity: float,
    ) -> None:
        curve = 80.0
        offset = curve if above else -curve
        control1_x = start_x + (end_x - start_x) * 0.35
        control2_x = start_x + (end_x - start_x) * 0.65
        control1_y = start_y - offset
        control2_y = end_y - offset
        color = _flow_color(value)
        stroke_width = _normalise(magnitude, max_flow, base=2.0, scale=8.0)
        path = SubElement(
            svg,
            "path",
            attrib={
                "d": (
                    f"M {start_x:.1f} {start_y:.1f} C {control1_x:.1f} {control1_y:.1f}, "
                    f"{control2_x:.1f} {control2_y:.1f}, {end_x:.1f} {end_y:.1f}"
                ),
                "fill": "none",
                "stroke": color,
                "stroke-width": f"{stroke_width:.2f}",
                "opacity": f"{opacity:.2f}",
                "stroke-linecap": "round",
                "marker-end": "url(#arrowhead)",
            },
        )
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2 - (offset * 0.35)
        SubElement(
            svg,
            "text",
            attrib={
                "x": f"{mid_x:.1f}",
                "y": f"{mid_y:.1f}",
                "text-anchor": "middle",
                "font-size": "16",
                "font-family": "Helvetica, Arial, sans-serif",
                "fill": color,
            },
        ).text = f"{value:+.2f}"

    for seed_name, stream_map in flows.items():
        for stream_name, metrics in stream_map.items():
            start_y = y_seed_positions[seed_name]
            end_y = y_stream_positions[stream_name]
            if metrics.forward_abs > 0:
                _draw_curve(left_x + 36, start_y, right_x - 36, end_y, metrics.forward_net, metrics.forward_abs, True, 0.9)
            if metrics.reverse_abs > 0:
                _draw_curve(right_x - 36, end_y, left_x + 36, start_y, metrics.reverse_net, metrics.reverse_abs, False, 0.75)

    SubElement(
        svg,
        "text",
        attrib={
            "x": f"{width / 2:.0f}",
            "y": f"{height - bottom_padding / 2:.1f}",
            "text-anchor": "middle",
            "font-size": "16",
            "font-family": "Helvetica, Arial, sans-serif",
            "fill": "#444444",
        },
    ).text = "Arrow width ∝ |flow| · colour encodes net sign (blue = excitatory, red = inhibitory)."

    if output_path.suffix.lower() != ".svg":
        output_path = output_path.with_suffix(".svg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree(svg).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def format_flow_table(flows: Mapping[str, Mapping[str, FlowMetrics]]) -> str:
    """Return a formatted table summarising flow metrics."""

    lines: List[str] = []
    header = (
        f"{'Seed Set':<22} {'Stream':<12} {'Fwd Net':>8} {'Rev Net':>8}"
        f" {'|Fwd|':>8} {'|Rev|':>8} {'RFc':>7} {'Dominant':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for seed_name in sorted(flows.keys()):
        stream_map = flows[seed_name]
        for stream_name in sorted(stream_map.keys()):
            metrics = stream_map[stream_name]
            lines.append(
                f"{seed_name:<22} {stream_name:<12} {metrics.forward_net:>8.2f} {metrics.reverse_net:>8.2f}"
                f" {metrics.forward_abs:>8.2f} {metrics.reverse_abs:>8.2f} {metrics.rfc:>7.2f}"
                f" {metrics.dominant_direction:>10}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public orchestration helper
# ---------------------------------------------------------------------------

def run_state_overlay(
    scope: str,
    seeds_csv: Path,
    output_dir: Optional[Path] = None,
    figure_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
    connectivity_csv: Optional[Path] = None,
    streams_csv: Optional[Path] = None,
) -> StateOverlayResult:
    """Execute the state overlay workflow for the requested scope."""

    scope = scope.lower().strip()
    if scope != "cx":
        raise ValueError(f"Unsupported scope '{scope}'. Only 'cx' is available.")

    seed_sets = load_state_seed_sets(seeds_csv)
    streams = load_cx_streams(streams_csv)
    edges = load_connectivity_edges(connectivity_csv)

    flows = compute_state_overlay(seed_sets, streams, edges)

    if output_dir is None:
        output_dir = _repo_root() / "output"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if metrics_path is None:
        metrics_path = output_dir / "state_overlay_metrics.csv"
    if figure_path is None:
        figure_path = output_dir / "state_overlay_summary.svg"

    metrics_path = metrics_path.resolve()
    figure_path = figure_path.resolve()

    # Write metrics CSV
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed_set",
                "stream",
                "forward_net",
                "reverse_net",
                "forward_abs",
                "reverse_abs",
                "forward_count",
                "reverse_count",
                "rfc",
                "net_signed_total",
                "dominant_direction",
            ],
        )
        writer.writeheader()
        for seed_name, stream_map in flows.items():
            for stream_name, metrics in stream_map.items():
                writer.writerow(metrics.to_row(seed_name, stream_name))

    figure_path = render_summary_circuit(flows, seed_sets, streams, figure_path)

    return StateOverlayResult(
        seed_sets=seed_sets,
        streams=streams,
        flows=flows,
        metrics_path=metrics_path,
        figure_path=figure_path,
    )


__all__ = [
    "Edge",
    "FlowMetrics",
    "SeedNeuron",
    "SeedSet",
    "StateOverlayResult",
    "StreamDefinition",
    "StreamNeuron",
    "compute_state_overlay",
    "format_flow_table",
    "load_connectivity_edges",
    "load_cx_streams",
    "load_state_seed_sets",
    "render_summary_circuit",
    "run_state_overlay",
]
