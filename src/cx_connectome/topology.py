"""cx_connectome.topology
==========================

Tools for summarizing central complex (CX) connectivity motifs.  The metrics follow
Walker et al.'s modality overlap and fan-in/fan-out visualizations but are
reinterpreted for CX "computational streams" (PFN, h\u0394, etc.): we examine how
putative input neuron (N1) cell types distribute their outputs onto downstream
N2 and N3 partners, and how those downstream populations converge back onto
upstream streams.  The outputs mirror Walker et al.'s overlap matrices and degree
histograms, providing comparable summaries tailored to CX circuitry analyses.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence

try:  # pragma: no cover - import guard for optional plotting dependency
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - provide actionable error message
    raise ImportError(
        "matplotlib is required for cx_connectome.topology visualizations. "
        "Install it with 'pip install matplotlib'."
    ) from exc

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class TopologyConfig:
    """Configuration for summarizing a single adjacency table."""

    stage: str
    table: pd.DataFrame
    pre_col: str
    post_col: str
    group_by: Optional[str]
    post_group_by: Optional[str]
    output_dir: Path

    def normalize_stage(self) -> str:
        return slugify(self.stage)


def slugify(value: str) -> str:
    """Return a filesystem-friendly identifier."""

    cleaned = [ch.lower() if ch.isalnum() else "_" for ch in value]
    slug = "".join(cleaned).strip("_")
    return slug or "stage"


def parse_stage_arguments(entries: Sequence[str]) -> Dict[str, Path]:
    """Parse ``--adj`` CLI entries into a mapping of stage name to path.

    ``entries`` can be a collection of strings of the form ``stage=path`` or
    simple paths.  When a bare path is supplied we derive the stage name from the
    stem of the file (e.g., ``N1_to_N2_adjacency.parquet`` -> ``N1_to_N2``).
    """

    results: Dict[str, Path] = {}
    for entry in entries:
        if "=" in entry:
            stage, path_str = entry.split("=", 1)
            stage = stage.strip()
            path = Path(path_str.strip())
        else:
            path = Path(entry)
            stage = infer_stage_name(path)
        if stage in results:
            raise ValueError(f"Duplicate stage '{stage}' provided")
        results[stage] = path
    return results


def infer_stage_name(path: Path) -> str:
    """Infer a stage label from a file path."""

    stem = path.stem
    if stem.endswith("_adjacency"):
        stem = stem[: -len("_adjacency")]
    return stem


def load_adjacency_table(path: Path) -> pd.DataFrame:
    """Load an adjacency table from disk.

    Parquet is preferred but CSV files are also supported to ease prototyping.
    """

    if not path.exists():
        raise FileNotFoundError(path)

    LOGGER.info("Loading adjacency table: %s", path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".feather", ".ft"}:
        return pd.read_feather(path)
    raise ValueError(f"Unsupported adjacency format for {path}")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_degrees(table: pd.DataFrame, source_col: str, target_col: str) -> pd.Series:
    """Compute unique partner counts for each neuron."""

    if source_col not in table.columns or target_col not in table.columns:
        missing = {source_col, target_col} - set(table.columns)
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    unique_pairs = table[[source_col, target_col]].dropna().drop_duplicates()
    return unique_pairs.groupby(source_col)[target_col].nunique().sort_values(ascending=False)


def write_series(series: pd.Series, index_name: str, value_name: str, path: Path) -> pd.DataFrame:
    """Persist a numeric series as a two-column CSV."""

    df = series.reset_index()
    df.columns = [index_name, value_name]
    df.to_csv(path, index=False)
    LOGGER.info("Wrote %s", path)
    return df


def write_summary(series: pd.Series, value_name: str, path: Path) -> pd.DataFrame:
    summary = series.describe()
    summary_df = summary.to_frame(name=value_name)
    summary_df.to_csv(path)
    LOGGER.info("Wrote %s", path)
    return summary_df


def attach_metadata(
    series: pd.Series,
    table: pd.DataFrame,
    id_column: str,
    metadata_column: Optional[str],
    value_name: str,
) -> Optional[pd.DataFrame]:
    """Attach metadata (e.g., cell type) to a degree series."""

    if not metadata_column:
        return None
    if metadata_column not in table.columns:
        LOGGER.warning("Column '%s' not present; skipping metadata attachment", metadata_column)
        return None

    mapping = (
        table[[id_column, metadata_column]]
        .dropna(subset=[metadata_column])
        .drop_duplicates()
        .set_index(id_column)[metadata_column]
    )
    annotated = series.to_frame(value_name)
    annotated[metadata_column] = annotated.index.map(mapping)
    missing = annotated[metadata_column].isna().sum()
    if missing:
        LOGGER.warning(
            "%d neurons lack '%s' annotations; results will omit those rows", missing, metadata_column
        )
    return annotated


def write_group_summary(
    annotated: Optional[pd.DataFrame],
    metadata_column: Optional[str],
    value_name: str,
    path: Path,
) -> Optional[pd.DataFrame]:
    if annotated is None or metadata_column is None:
        return None
    subset = annotated.dropna(subset=[metadata_column])
    if subset.empty:
        LOGGER.warning("No annotated rows available to summarize for %s", metadata_column)
        return None
    grouped = subset.groupby(metadata_column)[value_name].describe().sort_index()
    grouped.to_csv(path)
    LOGGER.info("Wrote %s", path)
    return grouped


def plot_histogram(series: pd.Series, title: str, xlabel: str, path: Path) -> None:
    if series.empty:
        LOGGER.warning("Series empty; skipping histogram %s", title)
        return

    values = series.values
    bin_count = min(75, max(10, int(math.sqrt(len(values)))))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=bin_count, color="#4c72b0", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Neuron count")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    LOGGER.info("Wrote %s", path)


def compute_cell_type_overlap(
    table: pd.DataFrame,
    cell_type_col: str,
    target_col: str,
    metric: str = "jaccard",
) -> pd.DataFrame:
    """Compute a cell-type overlap matrix using downstream partner sets."""

    if cell_type_col not in table.columns:
        raise KeyError(f"Column '{cell_type_col}' not present in adjacency table")
    if target_col not in table.columns:
        raise KeyError(f"Column '{target_col}' not present in adjacency table")

    pairs = table[[cell_type_col, target_col]].dropna().drop_duplicates()
    partner_sets = pairs.groupby(cell_type_col)[target_col].apply(set)
    partner_sets = partner_sets[partner_sets.map(len) > 0]
    cell_types = partner_sets.index.tolist()

    if not cell_types:
        raise ValueError("No cell-type partner sets available for overlap computation")

    data = np.zeros((len(cell_types), len(cell_types)), dtype=float)
    for i, cell_i in enumerate(cell_types):
        partners_i = partner_sets[cell_i]
        for j, cell_j in enumerate(cell_types):
            partners_j = partner_sets[cell_j]
            intersection = len(partners_i & partners_j)
            if metric == "percent_shared":
                denom = min(len(partners_i), len(partners_j))
                score = (intersection / denom * 100.0) if denom else 0.0
            else:  # default to Jaccard
                union = len(partners_i | partners_j)
                score = (intersection / union) if union else 0.0
            data[i, j] = score

    matrix = pd.DataFrame(data, index=cell_types, columns=cell_types)
    return matrix


def plot_overlap_heatmap(matrix: pd.DataFrame, metric: str, path: Path) -> None:
    fig_size = max(6, 0.4 * len(matrix))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    cmap = "viridis" if metric == "jaccard" else "magma"
    im = ax.imshow(matrix.values, cmap=cmap)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    color_label = "Jaccard index" if metric == "jaccard" else "% shared downstream"
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(color_label)

    # Overlay numeric values for readability.
    for (i, j), value in np.ndenumerate(matrix.values):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)

    ax.set_title(f"Cell-type overlap ({color_label})")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    LOGGER.info("Wrote %s", path)


def summarize_stage(config: TopologyConfig) -> Mapping[str, Path]:
    """Summarize divergence and convergence metrics for a single stage."""

    stage_slug = config.normalize_stage()
    outputs: MutableMapping[str, Path] = {}

    fan_out = compute_degrees(config.table, config.pre_col, config.post_col)
    fan_out_path = config.output_dir / f"topology_{stage_slug}_divergence.csv"
    write_series(fan_out, config.pre_col, "fan_out", fan_out_path)
    outputs["divergence_csv"] = fan_out_path

    divergence_summary_path = config.output_dir / f"topology_{stage_slug}_divergence_summary.csv"
    write_summary(fan_out, "fan_out", divergence_summary_path)
    outputs["divergence_summary"] = divergence_summary_path

    divergence_hist_path = config.output_dir / f"topology_{stage_slug}_divergence_hist.png"
    plot_histogram(
        fan_out,
        title=f"Divergence (fan-out) for {config.stage}",
        xlabel="Unique downstream partners",
        path=divergence_hist_path,
    )
    outputs["divergence_hist"] = divergence_hist_path

    annotated_out = attach_metadata(fan_out, config.table, config.pre_col, config.group_by, "fan_out")
    if annotated_out is not None and config.group_by:
        divergence_group_path = (
            config.output_dir
            / f"topology_{stage_slug}_divergence_by_{slugify(config.group_by)}.csv"
        )
        write_group_summary(annotated_out, config.group_by, "fan_out", divergence_group_path)
        outputs["divergence_group"] = divergence_group_path

    fan_in = compute_degrees(config.table, config.post_col, config.pre_col)
    fan_in_path = config.output_dir / f"topology_{stage_slug}_convergence.csv"
    write_series(fan_in, config.post_col, "fan_in", fan_in_path)
    outputs["convergence_csv"] = fan_in_path

    convergence_summary_path = config.output_dir / f"topology_{stage_slug}_convergence_summary.csv"
    write_summary(fan_in, "fan_in", convergence_summary_path)
    outputs["convergence_summary"] = convergence_summary_path

    convergence_hist_path = config.output_dir / f"topology_{stage_slug}_convergence_hist.png"
    plot_histogram(
        fan_in,
        title=f"Convergence (fan-in) for {config.stage}",
        xlabel="Unique upstream partners",
        path=convergence_hist_path,
    )
    outputs["convergence_hist"] = convergence_hist_path

    annotated_in = attach_metadata(fan_in, config.table, config.post_col, config.post_group_by, "fan_in")
    if annotated_in is not None and config.post_group_by:
        convergence_group_path = (
            config.output_dir
            / f"topology_{stage_slug}_convergence_by_{slugify(config.post_group_by)}.csv"
        )
        write_group_summary(annotated_in, config.post_group_by, "fan_in", convergence_group_path)
        outputs["convergence_group"] = convergence_group_path

    return outputs


def summarize_topology(
    adjacency_tables: Mapping[str, pd.DataFrame],
    *,
    output_dir: Path,
    pre_col: str,
    post_col: str,
    group_by: Optional[str],
    post_group_by: Optional[str],
    overlap_metric: str,
) -> None:
    """Run the full topology summary workflow."""

    ensure_output_dir(output_dir)

    overlap_source_stage: Optional[str] = None

    for stage, table in adjacency_tables.items():
        LOGGER.info("Summarizing stage '%s'", stage)
        config = TopologyConfig(
            stage=stage,
            table=table,
            pre_col=pre_col,
            post_col=post_col,
            group_by=group_by,
            post_group_by=post_group_by,
            output_dir=output_dir,
        )
        summarize_stage(config)
        if group_by and group_by in table.columns:
            overlap_source_stage = stage

    if not group_by:
        LOGGER.info("No --group-by column provided; skipping cell-type overlap matrix")
        return

    if overlap_source_stage is None:
        LOGGER.warning(
            "None of the adjacency tables contained '%s'; skipping overlap heatmap", group_by
        )
        return

    LOGGER.info(
        "Computing cell-type overlap matrix for stage '%s' using metric '%s'",
        overlap_source_stage,
        overlap_metric,
    )
    overlap_table = adjacency_tables[overlap_source_stage]
    overlap_matrix = compute_cell_type_overlap(overlap_table, group_by, post_col, metric=overlap_metric)

    metric_slug = slugify(overlap_metric)
    overlap_csv = output_dir / f"topology_{slugify(overlap_source_stage)}_{group_by}_overlap_{metric_slug}.csv"
    overlap_matrix.to_csv(overlap_csv)
    LOGGER.info("Wrote %s", overlap_csv)

    overlap_png = output_dir / f"topology_{slugify(overlap_source_stage)}_{group_by}_overlap_{metric_slug}.png"
    plot_overlap_heatmap(overlap_matrix, overlap_metric, overlap_png)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize CX topology statistics.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )

    subparsers = parser.add_subparsers(dest="command")

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize fan-in/fan-out distributions and cell-type overlaps",
    )
    summarize_parser.add_argument(
        "--adj",
        dest="adjacencies",
        action="append",
        required=True,
        help=(
            "Adjacency table(s). Accepts PATH or stage=PATH. Provide N1->N2 and N2->N3 "
            "tables by repeating this option"
        ),
    )
    summarize_parser.add_argument(
        "--group-by",
        default="pre_cell_type",
        help="Column representing N1 cell types for grouping",
    )
    summarize_parser.add_argument(
        "--post-group-by",
        default="post_cell_type",
        help="Column representing postsynaptic cell type annotations",
    )
    summarize_parser.add_argument(
        "--pre-col",
        default="pre_pt_root_id",
        help="Column name for presynaptic neuron identifiers",
    )
    summarize_parser.add_argument(
        "--post-col",
        default="post_pt_root_id",
        help="Column name for postsynaptic neuron identifiers",
    )
    summarize_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out"),
        help="Directory where summary files will be written",
    )
    summarize_parser.add_argument(
        "--overlap-metric",
        choices=["jaccard", "percent_shared"],
        default="jaccard",
        help="Similarity metric for the cell-type overlap matrix",
    )
    summarize_parser.set_defaults(func=run_summarize)

    return parser


def run_summarize(args: argparse.Namespace) -> int:
    adjacency_map = parse_stage_arguments(args.adjacencies)
    adjacency_tables = {
        stage: load_adjacency_table(path) for stage, path in adjacency_map.items()
    }

    summarize_topology(
        adjacency_tables,
        output_dir=args.output_dir,
        pre_col=args.pre_col,
        post_col=args.post_col,
        group_by=args.group_by,
        post_group_by=args.post_group_by,
        overlap_metric=args.overlap_metric,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
