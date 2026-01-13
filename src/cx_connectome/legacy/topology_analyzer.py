"""Topology analysis utilities for CX synaptic connectivity.

This module provides plotting utilities that mirror the weighted analyses
used in previous taste circuit studies while operating solely on the CX data.
Only the Python standard library is required so that the analyses remain
portable in restricted environments.

The key entry point is :class:`TopologyAnalyzer`, which accepts a table of
synapse counts (``syn_count``). The table can be a pandas ``DataFrame``
(when pandas is available) or any sequence of mappings with at least the
following logical fields:

* ``pre`` neuron identifier and class (e.g. N1)
* ``post`` neuron identifier and class (e.g. N2 or N3)
* ``syn_count`` indicating the number of synapses/weight for the edge

From this information the analyzer can generate

* Histograms and complementary cumulative distribution functions (CCDFs)
  of synapse weights for N1→N2 and N2→N3 connectivity.
* A convergence versus mean synaptic strength scatter plot for N2 neurons
  that receive input from N1, alongside exported correlation statistics.

The figures are emitted as SVG files so no third-party rendering libraries
(such as ``matplotlib``) are required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import logging
import math
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from xml.sax.saxutils import escape as xml_escape

try:  # Optional pandas support when available
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover - pandas is optional
    _pd = None


logger = logging.getLogger(__name__)

# Default labels for the CX populations.
DEFAULT_N1_LABEL = "N1"
DEFAULT_N2_LABEL = "N2"
DEFAULT_N3_LABEL = "N3"

# Candidate column names used when auto-detecting schema fields.
_PRE_CLASS_CANDIDATES: Tuple[str, ...] = (
    "pre_class",
    "pre_type",
    "pre_population",
    "pre_layer",
    "source_class",
    "source_type",
    "pre_category",
)
_POST_CLASS_CANDIDATES: Tuple[str, ...] = (
    "post_class",
    "post_type",
    "post_population",
    "post_layer",
    "target_class",
    "target_type",
    "post_category",
)
_PRE_ID_CANDIDATES: Tuple[str, ...] = (
    "pre_root_id",
    "pre_id",
    "pre_neuron",
    "pre",
    "source_id",
    "source",
)
_POST_ID_CANDIDATES: Tuple[str, ...] = (
    "post_root_id",
    "post_id",
    "post_neuron",
    "post",
    "target_id",
    "target",
)
_WEIGHT_CANDIDATES: Tuple[str, ...] = (
    "syn_count",
    "synapses",
    "weight",
    "n_synapses",
    "strength",
    "count",
)


@dataclass
class TopologyAnalyzer:
    """Analyze CX synapse topology and emit summary plots.

    Parameters
    ----------
    syn_count:
        Input synapse table. May be a pandas ``DataFrame`` (when pandas is
        installed) or any iterable of mapping objects describing edges.
    n1_label / n2_label / n3_label:
        Population labels used to identify N1, N2, and N3 neurons.
    pre_class_col / post_class_col / pre_id_col / post_id_col / weight_col:
        Optional overrides for the column names used when extracting data.
        When omitted the analyzer attempts to infer the appropriate columns
        by scanning the provided records for a known candidate name.
    """

    syn_count: Sequence[Mapping[str, object]]
    n1_label: str = DEFAULT_N1_LABEL
    n2_label: str = DEFAULT_N2_LABEL
    n3_label: str = DEFAULT_N3_LABEL
    pre_class_col: Optional[str] = None
    post_class_col: Optional[str] = None
    pre_id_col: Optional[str] = None
    post_id_col: Optional[str] = None
    weight_col: Optional[str] = None
    _records: List[Dict[str, object]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._records = self._normalize_records(self.syn_count)
        if not self._records:
            raise ValueError("syn_count is empty; nothing to analyze")
        columns = self._available_columns()
        self.pre_class_col = self.pre_class_col or self._resolve_column(
            _PRE_CLASS_CANDIDATES, columns, "pre-class"
        )
        self.post_class_col = self.post_class_col or self._resolve_column(
            _POST_CLASS_CANDIDATES, columns, "post-class"
        )
        self.pre_id_col = self.pre_id_col or self._resolve_column(
            _PRE_ID_CANDIDATES, columns, "pre neuron id"
        )
        self.post_id_col = self.post_id_col or self._resolve_column(
            _POST_ID_CANDIDATES, columns, "post neuron id"
        )
        self.weight_col = self.weight_col or self._resolve_column(
            _WEIGHT_CANDIDATES, columns, "synaptic weight"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plot_synapse_weight_distributions(self, output_dir: Path) -> Dict[str, List[float]]:
        """Generate histograms and CCDF plots for N1→N2 and N2→N3 synapses.

        Parameters
        ----------
        output_dir:
            Directory in which the SVG figures will be saved. The directory
            is created if it does not already exist.

        Returns
        -------
        Dict[str, List[float]]
            The raw weight sequences used for the plots keyed by the
            connection label (``"N1→N2"`` and ``"N2→N3"``). These lists are
            useful for unit tests or downstream statistics.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n1_n2_weights = self._extract_weights(self.n1_label, self.n2_label)
        n2_n3_weights = self._extract_weights(self.n2_label, self.n3_label)

        weight_map = {
            f"{self.n1_label}→{self.n2_label}": n1_n2_weights,
            f"{self.n2_label}→{self.n3_label}": n2_n3_weights,
        }

        for label, weights in weight_map.items():
            safe_label = label.replace("→", "_to_")
            hist_path = output_dir / f"{safe_label.lower()}_weight_histogram.svg"
            ccdf_path = output_dir / f"{safe_label.lower()}_weight_ccdf.svg"
            _create_histogram_svg(
                weights,
                hist_path,
                title=f"Synapse weight histogram ({label})",
                x_label="Synapse weight",
                y_label="Frequency",
            )
            _create_ccdf_svg(
                weights,
                ccdf_path,
                title=f"Synapse weight CCDF ({label})",
                x_label="Synapse weight",
                y_label="CCDF (log₁₀)",
            )

        return weight_map

    def plot_convergence_vs_strength(
        self, output_dir: Path
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        """Plot convergence versus mean synaptic strength for N2 neurons.

        The method produces a scatter plot and exports two CSV files:

        * ``..._convergence_strength.csv`` contains the per-neuron metrics.
        * ``..._correlations.csv`` stores Pearson and Spearman coefficients.

        Parameters
        ----------
        output_dir:
            Directory where the scatter plot SVG and CSV exports will live.

        Returns
        -------
        Tuple[List[Dict[str, float]], List[Dict[str, float]]]
            Two lists of dictionaries mirroring the contents of the CSV
            exports. The first list contains the per-neuron metrics (including
            convergence and mean strength). The second list provides the
            correlation statistics.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = self._compute_convergence_metrics()
        metrics_path = (
            output_dir
            / f"{self.n1_label.lower()}_to_{self.n2_label.lower()}_convergence_strength.csv"
        )
        _write_csv(metrics_path, metrics)

        correlations = _compute_correlations(metrics)
        corr_path = (
            output_dir
            / f"{self.n1_label.lower()}_to_{self.n2_label.lower()}_convergence_strength_correlations.csv"
        )
        _write_csv(corr_path, correlations)

        scatter_path = (
            output_dir
            / f"{self.n1_label.lower()}_to_{self.n2_label.lower()}_convergence_vs_mean_strength.svg"
        )
        _create_scatter_svg(
            metrics,
            scatter_path,
            title=(
                f"Convergence vs mean strength ({self.n1_label}→{self.n2_label})"
            ),
            x_label=f"Inputs from {self.n1_label} (convergence)",
            y_label="Mean synaptic strength",
            annotations=correlations,
        )

        return metrics, correlations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_records(
        self, data: Sequence[Mapping[str, object]]
    ) -> List[Dict[str, object]]:
        if _pd is not None and isinstance(data, _pd.DataFrame):  # type: ignore
            return [dict(row) for row in data.to_dict(orient="records")]
        if isinstance(data, Mapping):
            return [dict(data)]

        normalized: List[Dict[str, object]] = []
        for record in data:
            if isinstance(record, Mapping):
                normalized.append(dict(record))
            else:
                raise TypeError(
                    "Each syn_count entry must be a mapping. Received: "
                    f"{type(record)!r}"
                )
        return normalized

    def _available_columns(self) -> List[str]:
        columns = set()
        for record in self._records:
            columns.update(record.keys())
        return sorted(columns)

    def _resolve_column(
        self, candidates: Tuple[str, ...], columns: Sequence[str], descriptor: str
    ) -> str:
        for name in candidates:
            if name in columns:
                return name
        raise KeyError(
            f"Unable to determine {descriptor} column. "
            f"Available columns: {', '.join(columns)}"
        )

    def _extract_weights(self, pre_label: str, post_label: str) -> List[float]:
        weights: List[float] = []
        for record in self._records:
            if (
                record.get(self.pre_class_col) == pre_label
                and record.get(self.post_class_col) == post_label
            ):
                weight = record.get(self.weight_col)
                if weight is None:
                    continue
                try:
                    weights.append(float(weight))
                except (TypeError, ValueError):
                    logger.debug("Skipping non-numeric weight %r", weight)
        return weights

    def _compute_convergence_metrics(self) -> List[Dict[str, float]]:
        groups: Dict[object, Dict[str, object]] = {}
        for record in self._records:
            if (
                record.get(self.pre_class_col) != self.n1_label
                or record.get(self.post_class_col) != self.n2_label
            ):
                continue

            post_id = record.get(self.post_id_col)
            pre_id = record.get(self.pre_id_col)
            weight_raw = record.get(self.weight_col)
            if post_id is None or pre_id is None or weight_raw is None:
                continue

            try:
                weight = float(weight_raw)
            except (TypeError, ValueError):
                logger.debug("Skipping non-numeric weight %r", weight_raw)
                continue

            group = groups.setdefault(
                post_id,
                {
                    "post_neuron": post_id,
                    "pre_ids": set(),
                    "weights": [],
                },
            )
            group["pre_ids"].add(pre_id)  # type: ignore[index]
            group["weights"].append(weight)  # type: ignore[index]

        metrics: List[Dict[str, float]] = []
        for post_id, info in groups.items():
            weights = info.get("weights", [])  # type: ignore[assignment]
            if not weights:
                continue
            pre_ids = info.get("pre_ids", set())  # type: ignore[assignment]
            convergence = float(len(pre_ids))
            mean_strength = float(fmean(weights))
            total_strength = float(sum(weights))
            metrics.append(
                {
                    "post_neuron": post_id,  # type: ignore[arg-type]
                    "convergence": convergence,
                    "mean_strength": mean_strength,
                    "total_strength": total_strength,
                    "input_count": float(len(weights)),
                }
            )

        metrics.sort(key=lambda row: row["convergence"])  # type: ignore[index]
        return metrics


# ----------------------------------------------------------------------
# CSV helpers
# ----------------------------------------------------------------------
def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Still create the file with a placeholder header so downstream tools
        # know which metrics are expected.
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["value"])
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ----------------------------------------------------------------------
# Correlation utilities
# ----------------------------------------------------------------------
def _compute_correlations(
    metrics: Sequence[Mapping[str, float]]
) -> List[Dict[str, float]]:
    if not metrics:
        return []

    convergence = [float(row["convergence"]) for row in metrics]
    mean_strength = [float(row["mean_strength"]) for row in metrics]

    pearson = _pearson_correlation(convergence, mean_strength)
    spearman = _spearman_correlation(convergence, mean_strength)

    correlations = [
        {"metric": "pearson", "correlation": pearson},
        {"metric": "spearman", "correlation": spearman},
    ]
    return correlations


def _pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n != len(y) or n < 2:
        return math.nan
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = 0.0
    sum_sq_x = 0.0
    sum_sq_y = 0.0
    for xi, yi in zip(x, y):
        dx = xi - mean_x
        dy = yi - mean_y
        numerator += dx * dy
        sum_sq_x += dx * dx
        sum_sq_y += dy * dy
    denominator = math.sqrt(sum_sq_x * sum_sq_y)
    if denominator == 0:
        return math.nan
    return numerator / denominator


def _spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return math.nan
    ranks_x = _average_ranks(x)
    ranks_y = _average_ranks(y)
    return _pearson_correlation(ranks_x, ranks_y)


def _average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][0] == indexed[i][0]:
            j += 1
        average_rank = (i + j - 1) / 2 + 1
        for k in range(i, j):
            _, idx = indexed[k]
            ranks[idx] = average_rank
        i = j
    return ranks


# ----------------------------------------------------------------------
# SVG rendering utilities
# ----------------------------------------------------------------------
SVG_WIDTH = 720
SVG_HEIGHT = 480
SVG_MARGIN = {
    "left": 80,
    "right": 30,
    "top": 60,
    "bottom": 70,
}


def _create_histogram_svg(
    values: Sequence[float],
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    if not values:
        _write_empty_svg(path, title, message="No data available")
        return

    bins = _build_histogram_bins(values)
    min_val = bins[0][0]
    max_val = bins[-1][1]
    max_count = max(bin_[2] for bin_ in bins)

    x_ticks = _compute_ticks(min_val, max_val)
    y_ticks = _compute_ticks(0, max_count)

    elements: List[str] = []
    elements.extend(
        _svg_axes(
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            min_x=min_val,
            max_x=max_val,
            min_y=0,
            max_y=max_count,
            title=title,
            x_label=x_label,
            y_label=y_label,
        )
    )

    plot_width = SVG_WIDTH - SVG_MARGIN["left"] - SVG_MARGIN["right"]
    plot_height = SVG_HEIGHT - SVG_MARGIN["top"] - SVG_MARGIN["bottom"]
    axis_x0 = SVG_MARGIN["left"]
    axis_y0 = SVG_HEIGHT - SVG_MARGIN["bottom"]

    num_bins = len(bins)
    bar_width = plot_width / num_bins
    for idx, (start, end, count) in enumerate(bins):
        bar_height = 0.0 if max_count == 0 else (count / max_count) * plot_height
        x = axis_x0 + idx * bar_width
        y = axis_y0 - bar_height
        elements.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 1:.2f}" '
            f'height="{bar_height:.2f}" fill="#2a5599" opacity="0.75" />'
        )

    _write_svg(path, elements)


def _create_ccdf_svg(
    values: Sequence[float],
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    if not values:
        _write_empty_svg(path, title, message="No data available")
        return

    points = _compute_ccdf_points(values)
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)

    # Guard against identical probabilities to avoid division by zero.
    if math.isclose(min_y, max_y):
        min_y = max(min_y / 10 if min_y > 0 else 1e-3, 1e-6)
        max_y = max_y if max_y > min_y else min_y * 10

    x_ticks = _compute_ticks(min_x, max_x)
    y_ticks = _log_ticks(min_y, max_y)

    elements: List[str] = []
    elements.extend(
        _svg_axes(
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            title=title,
            x_label=x_label,
            y_label=y_label,
            log_y=True,
        )
    )

    axis_x0 = SVG_MARGIN["left"]
    axis_y0 = SVG_HEIGHT - SVG_MARGIN["bottom"]
    plot_width = SVG_WIDTH - SVG_MARGIN["left"] - SVG_MARGIN["right"]
    plot_height = SVG_HEIGHT - SVG_MARGIN["top"] - SVG_MARGIN["bottom"]

    log_min_y = math.log10(min_y)
    log_max_y = math.log10(max_y)
    span_x = max_x - min_x if not math.isclose(max_x, min_x) else 1.0

    prev_x: Optional[float] = None
    prev_y: Optional[float] = None
    for value, prob in points:
        norm_x = (value - min_x) / span_x
        x = axis_x0 + norm_x * plot_width
        log_prob = math.log10(prob)
        y = axis_y0 - ((log_prob - log_min_y) / (log_max_y - log_min_y)) * plot_height
        if prev_x is not None and prev_y is not None:
            elements.append(
                f'<line x1="{prev_x:.2f}" y1="{prev_y:.2f}" '
                f'x2="{x:.2f}" y2="{y:.2f}" stroke="#d17c0b" '
                f'stroke-width="2" />'
            )
        elements.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="#d17c0b" />'
        )
        prev_x, prev_y = x, y

    _write_svg(path, elements)


def _create_scatter_svg(
    metrics: Sequence[Mapping[str, float]],
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    annotations: Sequence[Mapping[str, float]],
) -> None:
    if not metrics:
        _write_empty_svg(path, title, message="No matching synapses")
        return

    x_values = [float(row["convergence"]) for row in metrics]
    y_values = [float(row["mean_strength"]) for row in metrics]

    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    x_ticks = _compute_ticks(min_x, max_x)
    y_ticks = _compute_ticks(min_y, max_y)

    elements: List[str] = []
    elements.extend(
        _svg_axes(
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            title=title,
            x_label=x_label,
            y_label=y_label,
        )
    )

    axis_x0 = SVG_MARGIN["left"]
    axis_y0 = SVG_HEIGHT - SVG_MARGIN["bottom"]
    plot_width = SVG_WIDTH - SVG_MARGIN["left"] - SVG_MARGIN["right"]
    plot_height = SVG_HEIGHT - SVG_MARGIN["top"] - SVG_MARGIN["bottom"]

    span_x = max_x - min_x if not math.isclose(max_x, min_x) else 1.0
    span_y = max_y - min_y if not math.isclose(max_y, min_y) else 1.0

    for x_val, y_val in zip(x_values, y_values):
        x = axis_x0 + ((x_val - min_x) / span_x) * plot_width
        y = axis_y0 - ((y_val - min_y) / span_y) * plot_height
        elements.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#1b7837" opacity="0.8" />'
        )

    if annotations:
        lines = []
        for item in annotations:
            metric = item.get("metric", "")
            corr = item.get("correlation")
            if corr is None or isinstance(corr, float) and math.isnan(corr):
                corr_text = "nan"
            else:
                corr_text = f"{float(corr):.3f}"
            lines.append(f"{metric}: {corr_text}")
        text = "\n".join(lines)
        elements.append(
            _svg_annotation(
                text,
                x=SVG_WIDTH - SVG_MARGIN["right"] - 150,
                y=SVG_MARGIN["top"] + 20,
            )
        )

    _write_svg(path, elements)


# ----------------------------------------------------------------------
# SVG primitive helpers
# ----------------------------------------------------------------------
def _svg_axes(
    *,
    x_ticks: Sequence[float],
    y_ticks: Sequence[float],
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    title: str,
    x_label: str,
    y_label: str,
    log_y: bool = False,
) -> List[str]:
    axis_x0 = SVG_MARGIN["left"]
    axis_y0 = SVG_HEIGHT - SVG_MARGIN["bottom"]
    plot_width = SVG_WIDTH - SVG_MARGIN["left"] - SVG_MARGIN["right"]
    plot_height = SVG_HEIGHT - SVG_MARGIN["top"] - SVG_MARGIN["bottom"]

    elements = [
        f'<rect x="0" y="0" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="#ffffff" />',
        f'<text x="{SVG_WIDTH / 2:.2f}" y="{SVG_MARGIN["top"] / 2:.2f}" '
        f'text-anchor="middle" font-size="18" fill="#222">{xml_escape(title)}</text>',
        f'<text x="{SVG_WIDTH / 2:.2f}" y="{SVG_HEIGHT - 20:.2f}" '
        f'text-anchor="middle" font-size="14" fill="#222">{xml_escape(x_label)}</text>',
        f'<text x="{SVG_MARGIN["left"] / 4:.2f}" y="{SVG_HEIGHT / 2:.2f}" '
        f'text-anchor="middle" font-size="14" fill="#222" '
        f'transform="rotate(-90 {SVG_MARGIN["left"] / 4:.2f} {SVG_HEIGHT / 2:.2f})">'
        f'{xml_escape(y_label)}</text>',
        f'<line x1="{axis_x0:.2f}" y1="{axis_y0:.2f}" x2="{axis_x0 + plot_width:.2f}" '
        f'y2="{axis_y0:.2f}" stroke="#333" stroke-width="1.5" />',
        f'<line x1="{axis_x0:.2f}" y1="{axis_y0:.2f}" x2="{axis_x0:.2f}" '
        f'y2="{axis_y0 - plot_height:.2f}" stroke="#333" stroke-width="1.5" />',
    ]

    span_x = max_x - min_x if not math.isclose(max_x, min_x) else 1.0
    span_y = max_y - min_y if not math.isclose(max_y, min_y) else 1.0

    for tick in x_ticks:
        norm = (tick - min_x) / span_x if span_x else 0.5
        x = axis_x0 + norm * plot_width
        elements.append(
            f'<line x1="{x:.2f}" y1="{axis_y0:.2f}" x2="{x:.2f}" y2="{axis_y0 + 6:.2f}" '
            f'stroke="#333" />'
        )
        elements.append(
            f'<text x="{x:.2f}" y="{axis_y0 + 22:.2f}" text-anchor="middle" '
            f'font-size="12" fill="#333">{_format_tick(tick)}</text>'
        )

    for tick in y_ticks:
        if log_y and tick <= 0:
            continue
        if log_y:
            log_min = math.log10(min_y)
            log_max = math.log10(max_y)
            norm = (math.log10(tick) - log_min) / (log_max - log_min) if log_max != log_min else 0.5
        else:
            norm = (tick - min_y) / span_y if span_y else 0.5
        y = axis_y0 - norm * plot_height
        elements.append(
            f'<line x1="{axis_x0 - 6:.2f}" y1="{y:.2f}" x2="{axis_x0:.2f}" '
            f'y2="{y:.2f}" stroke="#333" />'
        )
        anchor = "end"
        elements.append(
            f'<text x="{axis_x0 - 10:.2f}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="12" fill="#333">{_format_tick(tick, log_scale=log_y)}</text>'
        )

    return elements


def _svg_annotation(text: str, *, x: float, y: float) -> str:
    lines = xml_escape(text).split("\n")
    tspan = "".join(
        f'<tspan x="{x:.2f}" dy="{(i != 0) * 18}">{line}</tspan>'
        for i, line in enumerate(lines)
    )
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="12" fill="#333" '
        f'font-family="monospace" text-anchor="start" '
        f'background="#fff">{tspan}</text>'
    )


def _write_empty_svg(path: Path, title: str, *, message: str) -> None:
    elements = _svg_axes(
        x_ticks=[0.0],
        y_ticks=[0.0],
        min_x=0.0,
        max_x=1.0,
        min_y=0.0,
        max_y=1.0,
        title=title,
        x_label="",
        y_label="",
    )
    elements.append(
        f'<text x="{SVG_WIDTH / 2:.2f}" y="{SVG_HEIGHT / 2:.2f}" '
        f'text-anchor="middle" font-size="16" fill="#666">{xml_escape(message)}</text>'
    )
    _write_svg(path, elements)


def _write_svg(path: Path, elements: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as fh:
        fh.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" '
            f'height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">'
        )
        for element in elements:
            fh.write(element)
        fh.write("</svg>")


def _build_histogram_bins(values: Sequence[float]) -> List[Tuple[float, float, int]]:
    sorted_values = sorted(float(v) for v in values)
    min_val = sorted_values[0]
    max_val = sorted_values[-1]
    if math.isclose(min_val, max_val):
        return [(min_val, max_val, len(sorted_values))]

    span = max_val - min_val
    num_bins = min(max(int(math.sqrt(len(sorted_values))), 5), 40)
    bin_width = span / num_bins if span else 1.0

    bins: List[Tuple[float, float, int]] = []
    counts = [0 for _ in range(num_bins)]
    edges = [min_val + i * bin_width for i in range(num_bins + 1)]

    for value in sorted_values:
        if value == max_val:
            index = num_bins - 1
        else:
            index = int((value - min_val) / bin_width)
        counts[index] += 1

    for idx in range(num_bins):
        bins.append((edges[idx], edges[idx + 1], counts[idx]))
    return bins


def _compute_ccdf_points(values: Sequence[float]) -> List[Tuple[float, float]]:
    sorted_values = sorted(float(v) for v in values)
    n = len(sorted_values)
    points: List[Tuple[float, float]] = []
    remaining = n
    last_value: Optional[float] = None
    for value in sorted_values:
        if last_value is None or not math.isclose(value, last_value):
            ccdf = max(remaining / n, 1.0 / (n * 10))
            points.append((value, ccdf))
            last_value = value
        remaining -= 1
    # Ensure the final point is represented.
    if points and not math.isclose(points[-1][0], sorted_values[-1]):
        points.append((sorted_values[-1], max(1.0 / n, 1.0 / (n * 10))))
    return points


def _compute_ticks(min_val: float, max_val: float, desired: int = 5) -> List[float]:
    if math.isclose(min_val, max_val):
        return [min_val]
    raw_range = _nice_number(max_val - min_val, round_up=False)
    step = _nice_number(raw_range / max(desired - 1, 1), round_up=True)
    tick_min = math.floor(min_val / step) * step
    tick_max = math.ceil(max_val / step) * step
    ticks: List[float] = []
    value = tick_min
    while value <= tick_max + step / 2:
        ticks.append(value)
        value += step
    return ticks


def _nice_number(value: float, round_up: bool) -> float:
    if value == 0:
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    fraction = abs(value) / 10 ** exponent
    if round_up:
        if fraction < 1.5:
            nice_fraction = 1.0
        elif fraction < 3:
            nice_fraction = 2.0
        elif fraction < 7:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0
    else:
        if fraction <= 1:
            nice_fraction = 1.0
        elif fraction <= 2:
            nice_fraction = 2.0
        elif fraction <= 5:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0
    return math.copysign(nice_fraction * (10 ** exponent), value)


def _log_ticks(min_val: float, max_val: float) -> List[float]:
    if min_val <= 0:
        min_val = 1e-6
    exp_min = math.floor(math.log10(min_val))
    exp_max = math.ceil(math.log10(max_val))
    ticks = []
    for exponent in range(exp_min, exp_max + 1):
        tick = 10 ** exponent
        if min_val <= tick <= max_val:
            ticks.append(tick)
    if not ticks:
        ticks.append(max_val)
    return ticks


def _format_tick(value: float, *, log_scale: bool = False) -> str:
    if log_scale:
        exponent = int(round(math.log10(value))) if value > 0 else 0
        return f"1e{exponent}"
    magnitude = abs(value)
    if magnitude >= 1000 or magnitude < 0.01:
        return f"{value:.2e}"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


__all__ = [
    "TopologyAnalyzer",
    "_compute_correlations",  # exported for unit testing convenience
]
