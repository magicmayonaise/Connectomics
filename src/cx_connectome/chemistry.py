"""Utilities for consolidating neurotransmitter metadata for the N2/N3 atlas.

This module reads CAVE exported tables together with an optional CX FISH
annotation file to determine the best available neurotransmitter (NT) and
neuropeptide assignments for Kenyon cell layers N2 and N3.

The workflow gives precedence to the information that originates from the CAVE
export (cell table, synapse table, or a dedicated NT reference table).  The FISH
annotations are only used to fill in gaps that remain after exhausting the CAVE
sources, mirroring the manual workflow used during curation.

Typical usage::

    from pathlib import Path
    from cx_connectome import chemistry

    chemistry.build_n2_n3_properties(
        cell_table="tables/cell_table.parquet",
        fish_annotations="data/cx_fish_annotations.csv",
        synapse_table="tables/synapse_table.parquet",
        nt_reference_table="tables/neurotransmitter_reference.parquet",
        output_path=Path("out/N2_N3_properties.parquet"),
    )

The :func:`build_n2_n3_properties` helper can be used as a library function or
as a small command-line tool via ``python -m cx_connectome.chemistry``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Column aliases that frequently appear in CAVE exports.  The lists are ordered
# by preference (earlier entries take precedence when multiple candidates are
# present in the data).
CANDIDATE_CELL_NT_COLUMNS: Sequence[str] = (
    "neurotransmitter",
    "primary_neurotransmitter",
    "primary_nt",
    "putative_nt",
    "nt_type",
    "nt_prediction",
    "predicted_nt",
    "neurotransmitter_prediction",
    "cell_nt",
    "cell_nt_prediction",
)

CANDIDATE_CELL_NP_COLUMNS: Sequence[str] = (
    "neuropeptides",
    "primary_neuropeptide",
    "neuropeptide",
    "predicted_neuropeptide",
    "neuropeptide_prediction",
)

CANDIDATE_SYNAPSE_NT_COLUMNS: Sequence[str] = (
    "neurotransmitter",
    "synapse_nt_prediction",
    "nt_prediction",
    "predicted_nt",
)

REQUIRED_OUTPUT_COLUMNS: Sequence[str] = (
    "root_id",
    "layer",
    "cell_type",
    "super_class",
    "projection_class",
    "output_neuropils",
)


def build_n2_n3_properties(
    cell_table: pd.DataFrame | str | Path,
    fish_annotations: str | Path | None = None,
    synapse_table: pd.DataFrame | str | Path | None = None,
    nt_reference_table: pd.DataFrame | str | Path | None = None,
    output_path: str | Path = Path("out/N2_N3_properties.parquet"),
) -> pd.DataFrame:
    """Create the consolidated N2/N3 properties table.

    Parameters
    ----------
    cell_table:
        CAVE cell table or a DataFrame containing the per-root metadata.  The
        table must include at least the columns required for the final export.
    fish_annotations:
        Path to the user-supplied CX FISH CSV (transcribed from Wolff et al.).
        The file must contain ``cell_type``, ``neurotransmitter``, and
        ``neuropeptides`` columns.  Set to :data:`None` to skip FISH merging.
    synapse_table:
        Optional synapse table.  When present, the code will try to resolve the
        dominant NT assignment per root using any of the known NT prediction
        columns.
    nt_reference_table:
        Optional CAVE reference table that stores neurotransmitter metadata.
        When supplied, the ``neurotransmitter`` and ``neuropeptides`` columns
        will be used as an additional source during resolution.
    output_path:
        Destination for the Parquet file.  The parent directory will be created
        automatically if it does not exist.

    Returns
    -------
    pandas.DataFrame
        The consolidated N2/N3 table.
    """

    cell_df = _ensure_dataframe(cell_table)
    _validate_required_columns(cell_df, REQUIRED_OUTPUT_COLUMNS)

    nt_reference_df = _ensure_dataframe(nt_reference_table) if nt_reference_table is not None else None
    synapse_df = _ensure_dataframe(synapse_table) if synapse_table is not None else None
    fish_df = _load_fish_annotations(fish_annotations) if fish_annotations is not None else None

    n2_n3_df = cell_df[cell_df["layer"].isin(["N2", "N3"])].copy()
    if n2_n3_df.empty:
        LOGGER.warning("The cell table does not contain any N2/N3 rows.")

    neurotransmitter = _resolve_neurotransmitter(
        n2_n3_df,
        synapse_df=synapse_df,
        reference_df=nt_reference_df,
    )
    neuropeptides = _resolve_neuropeptides(
        n2_n3_df,
        reference_df=nt_reference_df,
    )

    result = n2_n3_df.loc[:, REQUIRED_OUTPUT_COLUMNS].copy()
    result["neurotransmitter"] = neurotransmitter
    result["neuropeptides"] = neuropeptides

    if fish_df is not None:
        result = _merge_fish_annotations(result, fish_df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    LOGGER.info("Exported %d rows to %s", len(result), output_path)

    return result


def _ensure_dataframe(table: pd.DataFrame | str | Path | None) -> pd.DataFrame:
    """Return a DataFrame given an optional path-like input."""

    if table is None:
        raise ValueError("Expected a table or a path, received None")

    if isinstance(table, pd.DataFrame):
        return table

    path = Path(table)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)

    raise ValueError(f"Unsupported table format: {path.suffix}")


def _validate_required_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Ensure that the DataFrame contains the required columns."""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(
            "The cell table is missing required columns: " + ", ".join(missing)
        )


def _resolve_neurotransmitter(
    cell_df: pd.DataFrame,
    synapse_df: Optional[pd.DataFrame] = None,
    reference_df: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Resolve the best available neurotransmitter annotation per root."""

    root_id_series = cell_df["root_id"]
    resolved = pd.Series(pd.NA, index=cell_df.index, dtype="object")

    if reference_df is not None and "root_id" in reference_df.columns:
        ref_series = _first_available_column(reference_df, CANDIDATE_CELL_NT_COLUMNS)
        if ref_series is not None:
            LOGGER.debug("Using reference NT column '%s'", ref_series.name)
            ref_lookup = _build_lookup_series(reference_df, ref_series)
            resolved = resolved.combine_first(root_id_series.map(ref_lookup))

    cell_series = _first_available_column(cell_df, CANDIDATE_CELL_NT_COLUMNS)
    if cell_series is not None:
        LOGGER.debug("Using cell table NT column '%s'", cell_series.name)
        resolved = resolved.combine_first(cell_series)

    if synapse_df is not None and "root_id" in synapse_df.columns:
        synapse_series = _first_available_column(synapse_df, CANDIDATE_SYNAPSE_NT_COLUMNS)
        if synapse_series is not None:
            LOGGER.debug("Aggregating synapse NT column '%s'", synapse_series.name)
            aggregated = _aggregate_synapse_annotations(synapse_df["root_id"], synapse_series)
            resolved = resolved.combine_first(root_id_series.map(aggregated))

    return resolved


def _resolve_neuropeptides(
    cell_df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Resolve neuropeptide annotations by combining CAVE sources."""

    resolved = pd.Series(pd.NA, index=cell_df.index, dtype="object")

    if reference_df is not None and "root_id" in reference_df.columns:
        ref_series = _first_available_column(reference_df, CANDIDATE_CELL_NP_COLUMNS)
        if ref_series is not None:
            LOGGER.debug("Using reference neuropeptide column '%s'", ref_series.name)
            ref_lookup = _build_lookup_series(reference_df, ref_series)
            resolved = resolved.combine_first(cell_df["root_id"].map(ref_lookup))

    cell_series = _first_available_column(cell_df, CANDIDATE_CELL_NP_COLUMNS)
    if cell_series is not None:
        LOGGER.debug("Using cell table neuropeptide column '%s'", cell_series.name)
        resolved = resolved.combine_first(cell_series)

    return resolved


def _first_available_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[pd.Series]:
    """Return the first present column from ``candidates`` or ``None``."""

    for column in candidates:
        if column in df.columns:
            return df[column]
    return None


def _build_lookup_series(df: pd.DataFrame, series: pd.Series) -> pd.Series:
    """Create a lookup Series indexed by ``root_id`` for ``series``."""

    column_name = series.name
    if column_name and column_name in df.columns:
        return df.set_index("root_id")[column_name]

    temp = df.assign(_value=series)
    return temp.set_index("root_id")['_value']


def _aggregate_synapse_annotations(root_ids: pd.Series, annotation: pd.Series) -> pd.Series:
    """Collapse synapse-level annotations to per-root values using the mode."""

    valid = pd.DataFrame({"root_id": root_ids, "annotation": annotation}).dropna(subset=["annotation"])
    if valid.empty:
        return pd.Series(dtype="object")

    grouped = valid.groupby("root_id")["annotation"].agg(_mode)
    return grouped.astype("object")


def _mode(values: Iterable) -> object:
    """Return the most frequent value in ``values`` (ties broken arbitrarily)."""

    series = pd.Series(list(values))
    if series.empty:
        return pd.NA
    counts = series.value_counts(dropna=True)
    if counts.empty:
        return pd.NA
    return counts.idxmax()


def _load_fish_annotations(path: str | Path) -> pd.DataFrame:
    """Load the CX FISH annotation CSV and validate the schema."""

    df = pd.read_csv(path)
    for column in ("cell_type", "neurotransmitter", "neuropeptides"):
        if column not in df.columns:
            raise KeyError(
                f"FISH annotation file is missing required column: {column}"
            )
    return df[["cell_type", "neurotransmitter", "neuropeptides"]]


def _merge_fish_annotations(
    result: pd.DataFrame,
    fish_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge FISH annotations, filling gaps without overwriting existing data."""

    merged = result.merge(
        fish_df,
        on="cell_type",
        how="left",
        suffixes=("", "_fish"),
    )
    merged["neurotransmitter"] = merged["neurotransmitter"].combine_first(
        merged.pop("neurotransmitter_fish")
    )
    merged["neuropeptides"] = merged["neuropeptides"].combine_first(
        merged.pop("neuropeptides_fish")
    )
    return merged


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell_table", help="Cell table (CSV or Parquet)")
    parser.add_argument(
        "--fish-annotations",
        help="CX FISH annotations CSV (Wolff et al.)",
    )
    parser.add_argument(
        "--synapse-table",
        help="Synapse table with NT predictions (CSV or Parquet)",
    )
    parser.add_argument(
        "--nt-reference-table",
        help="CAVE NT reference table (CSV or Parquet)",
    )
    parser.add_argument(
        "--output",
        default=str(Path("out/N2_N3_properties.parquet")),
        help="Destination Parquet path",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _create_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    build_n2_n3_properties(
        cell_table=args.cell_table,
        fish_annotations=args.fish_annotations,
        synapse_table=args.synapse_table,
        nt_reference_table=args.nt_reference_table,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
