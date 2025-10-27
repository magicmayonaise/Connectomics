"""Utilities for building N1-to-N2 connectivity tables from CAVE.

This module exposes a ``build_connectivity`` helper and a small command line
interface that wrap the :mod:`caveclient` materialization APIs.  CAVE synapses
are spatial point annotations bound to pre- and post-synaptic segments; this
script converts the resulting table of synaptic points into a weighted edge
list by grouping the rows between segment pairs and counting them.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, MutableMapping, Optional, Sequence, Set

import pandas as pd

try:
    from caveclient import CAVEclient
except ImportError:  # pragma: no cover - optional dependency for the CLI
    CAVEclient = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT = Path("out/N1_to_N2_adjacency.parquet")
PROMPT3_TAG = "Prompt 3"
ANNOTATION_TABLE = "annotations"


def _read_id_file(path: Path) -> List[int]:
    """Return integer identifiers parsed from ``path``.

    The helper accepts comma or whitespace separated values and ignores blank
    lines as well as comment lines starting with ``#``.
    """

    values: List[int] = []
    text = path.read_text().splitlines()
    for line in text:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for chunk in re.split(r"[\s,]+", stripped):
            if not chunk:
                continue
            try:
                values.append(int(chunk))
            except ValueError as exc:  # pragma: no cover - defensive programming
                raise ValueError(f"Invalid identifier '{chunk}' in {path}") from exc
    return values


def _normalize_annotation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise annotation column names to ``root_id``, ``cell_type`` and
    ``super_class``.

    Parameters
    ----------
    df:
        DataFrame returned by the materialization service.
    """

    if df.empty:
        return pd.DataFrame(
            {
                "root_id": pd.Series(dtype="int64"),
                "cell_type": pd.Series(dtype="string"),
                "super_class": pd.Series(dtype="string"),
            }
        )

    root_candidates = (
        "pt_root_id",
        "target_id",
        "root_id",
        "segment_id",
        "id",
    )
    cell_type_candidates = (
        "cell_type",
        "cell_type_full",
        "cell_type_label",
        "annotation",
        "type",
    )
    super_class_candidates = (
        "super_class",
        "cell_super_class",
        "supertype",
        "cell_class",
        "class",
    )

    root_col = next((col for col in root_candidates if col in df.columns), None)
    if root_col is None:
        raise KeyError("No root identifier column found in annotation table")

    cell_type_col = next((col for col in cell_type_candidates if col in df.columns), None)
    if cell_type_col is None:
        LOGGER.debug("No explicit cell type column present; filling with <NA>.")
        df = df.assign(cell_type=pd.Series([pd.NA] * len(df)))
        cell_type_col = "cell_type"

    super_class_col = next((col for col in super_class_candidates if col in df.columns), None)
    if super_class_col is None:
        LOGGER.debug("No explicit super class column present; filling with <NA>.")
        df = df.assign(super_class=pd.Series([pd.NA] * len(df)))
        super_class_col = "super_class"

    normalized = df[[root_col, cell_type_col, super_class_col]].copy()
    normalized.columns = ["root_id", "cell_type", "super_class"]
    normalized["root_id"] = pd.to_numeric(normalized["root_id"], errors="coerce")
    normalized = normalized.dropna(subset=["root_id"]).copy()
    normalized["root_id"] = normalized["root_id"].astype("int64")
    normalized = normalized.drop_duplicates(subset="root_id", keep="last")
    normalized["cell_type"] = normalized["cell_type"].astype("string")
    normalized["super_class"] = normalized["super_class"].astype("string")
    return normalized


def _fetch_prompt3_annotations(
    client: object,
    root_ids: Iterable[int],
    query_kwargs: Optional[MutableMapping[str, object]] = None,
) -> pd.DataFrame:
    """Fetch Prompt 3 annotations for ``root_ids`` using the materialization API."""

    root_ids = list(dict.fromkeys(int(r) for r in root_ids))
    if not root_ids:
        return pd.DataFrame(
            {
                "root_id": pd.Series(dtype="int64"),
                "cell_type": pd.Series(dtype="string"),
                "super_class": pd.Series(dtype="string"),
            }
        )

    if query_kwargs is None:
        query_kwargs = {}

    mat = getattr(client, "materialize", None)
    if mat is None:
        raise AttributeError("Client does not expose a materialize attribute")

    LOGGER.debug("Querying Prompt 3 annotations for %d roots", len(root_ids))
    annotations = mat.query_table(  # type: ignore[call-arg]
        ANNOTATION_TABLE,
        filter_in_dict={"target_id": root_ids, "tag": [PROMPT3_TAG]},
        **query_kwargs,
    )

    normalized = _normalize_annotation_columns(annotations)
    return normalized


def build_connectivity(
    client: object,
    pre_roots: Sequence[int],
    post_exclude: Optional[Iterable[int]] = None,
    threshold: int = 5,
    layer_tag: str = "N1->N2",
    *,
    materialization: Optional[int] = None,
    at_ts: Optional[int] = None,
    output_path: Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """Build and persist an N1→N2 weighted edge list from CAVE synapses.

    Parameters
    ----------
    client:
        Instance of :class:`caveclient.CAVEclient` (or a compatible stub) used
        to query the materialization service.
    pre_roots:
        Sequence of presynaptic root identifiers (``pre_pt_root_id``) that define
        the population of interest.
    post_exclude:
        Optional sequence of postsynaptic root identifiers to be filtered out
        from the results.
    threshold:
        Minimum synapse count required for an edge to be retained.  The default
        of ``5`` matches the N1→N2 heuristic requested by biology partners.
    layer_tag:
        Value stored in the ``layer`` column of the resulting adjacency table.
    materialization / at_ts:
        Optional materialization version or timestamp used to anchor the query.
        Only one of the two options can be provided at a time.
    output_path:
        Location where the parquet file will be written.  By default the file is
        saved to ``out/N1_to_N2_adjacency.parquet``.

    Returns
    -------
    pandas.DataFrame
        The filtered and annotated adjacency table.
    """

    if materialization is not None and at_ts is not None:
        raise ValueError("Only one of materialization or at_ts may be provided")

    if not pre_roots:
        raise ValueError("pre_roots must not be empty")

    mat = getattr(client, "materialize", None)
    if mat is None:
        raise AttributeError("Client does not expose a materialize attribute")

    pre_root_list = [int(r) for r in pre_roots]
    query_kwargs: MutableMapping[str, object] = {}
    if materialization is not None:
        query_kwargs["materialization_version"] = materialization
    if at_ts is not None:
        query_kwargs["timestamp"] = at_ts

    LOGGER.info("Querying synapses for %d presynaptic roots", len(pre_root_list))
    synapses = mat.query_table(  # type: ignore[call-arg]
        "synapses",
        filter_in_dict={"pre_pt_root_id": pre_root_list},
        **query_kwargs,
    )
    if synapses.empty:
        LOGGER.warning("Synapse query returned an empty table")
        grouped = pd.DataFrame(
            {
                "pre_root_id": pd.Series(dtype="int64"),
                "post_root_id": pd.Series(dtype="int64"),
                "syn_count": pd.Series(dtype="int64"),
            }
        )
    else:
        synapses = synapses.copy()
        synapses["pre_pt_root_id"] = pd.to_numeric(
            synapses["pre_pt_root_id"], errors="coerce"
        )
        synapses["post_pt_root_id"] = pd.to_numeric(
            synapses["post_pt_root_id"], errors="coerce"
        )
        synapses = synapses.dropna(subset=["pre_pt_root_id", "post_pt_root_id"])
        synapses["pre_pt_root_id"] = synapses["pre_pt_root_id"].astype("int64")
        synapses["post_pt_root_id"] = synapses["post_pt_root_id"].astype("int64")

        grouped = (
            synapses.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False)
            .size()
            .rename(columns={"pre_pt_root_id": "pre_root_id", "post_pt_root_id": "post_root_id", "size": "syn_count"})
        )

    LOGGER.debug("Applying synapse count threshold >= %d", threshold)
    mask = grouped["syn_count"] >= threshold
    mask &= grouped["pre_root_id"] != grouped["post_root_id"]
    post_exclude_set: Set[int] = set(int(p) for p in post_exclude or [])
    if post_exclude_set:
        mask &= ~grouped["post_root_id"].isin(post_exclude_set)

    filtered = grouped.loc[mask].reset_index(drop=True)
    LOGGER.info("Retained %d edges after filtering", len(filtered))

    all_roots: Set[int] = set(filtered["pre_root_id"]).union(filtered["post_root_id"]) if not filtered.empty else set()
    annotations = _fetch_prompt3_annotations(client, all_roots, query_kwargs)
    ann_map = annotations.set_index("root_id") if not annotations.empty else pd.DataFrame()

    if not annotations.empty:
        filtered["pre_cell_type"] = filtered["pre_root_id"].map(ann_map["cell_type"])  # type: ignore[index]
        filtered["post_cell_type"] = filtered["post_root_id"].map(ann_map["cell_type"])  # type: ignore[index]
        filtered["pre_super_class"] = filtered["pre_root_id"].map(ann_map["super_class"])  # type: ignore[index]
        filtered["post_super_class"] = filtered["post_root_id"].map(ann_map["super_class"])  # type: ignore[index]
    else:
        filtered["pre_cell_type"] = pd.Series([pd.NA] * len(filtered), dtype="string")
        filtered["post_cell_type"] = pd.Series([pd.NA] * len(filtered), dtype="string")
        filtered["pre_super_class"] = pd.Series([pd.NA] * len(filtered), dtype="string")
        filtered["post_super_class"] = pd.Series([pd.NA] * len(filtered), dtype="string")

    filtered["layer"] = layer_tag

    ordered_columns = [
        "pre_root_id",
        "post_root_id",
        "pre_cell_type",
        "post_cell_type",
        "pre_super_class",
        "post_super_class",
        "syn_count",
        "layer",
    ]
    for column in ordered_columns:
        if column not in filtered.columns:
            filtered[column] = pd.Series([pd.NA] * len(filtered))

    filtered = filtered[ordered_columns]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing adjacency table to %s", output_path)
    filtered.to_parquet(output_path, index=False)

    return filtered


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cx-cave", description="Build N1→N2 adjacency tables from the CAVE materialization service.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level (default: INFO)")
    subparsers = parser.add_subparsers(dest="command")

    n1n2 = subparsers.add_parser("n1-n2", help="Generate the N1→N2 adjacency parquet file")
    n1n2.add_argument("--n1", required=True, type=Path, help="Text file containing presynaptic root ids")
    n1n2.add_argument("--post-exclude", type=Path, help="Optional text file of postsynaptic roots to exclude")
    n1n2.add_argument("--threshold", type=int, default=5, help="Minimum synapse count to keep an edge")
    n1n2.add_argument("--layer", default="N1->N2", help="Layer tag recorded in the output parquet")
    n1n2.add_argument("--materialization", type=int, help="Materialization version for the query")
    n1n2.add_argument("--at-ts", type=int, help="Timestamp for the query (mutually exclusive with --materialization)")
    n1n2.add_argument("--datastack", type=str, help="CAVE datastack name (or set CAVE_DATASTACK environment variable)")
    n1n2.add_argument("--server", type=str, help="Optional CAVE server address")
    n1n2.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination parquet file")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    if args.command != "n1-n2":
        parser.print_help()
        return 1

    if args.materialization and args.at_ts:
        parser.error("--materialization and --at-ts are mutually exclusive")

    if CAVEclient is None:
        parser.error("caveclient is required for this command")

    datastack = args.datastack or os.getenv("CAVE_DATASTACK")
    if not datastack:
        parser.error("No datastack supplied. Use --datastack or set CAVE_DATASTACK")

    client_kwargs = {}
    if args.server:
        client_kwargs["server_address"] = args.server

    LOGGER.info("Connecting to CAVE datastack '%s'", datastack)
    client = CAVEclient(datastack, **client_kwargs)

    pre_roots = _read_id_file(args.n1)
    post_exclude = _read_id_file(args.post_exclude) if args.post_exclude else None

    build_connectivity(
        client=client,
        pre_roots=pre_roots,
        post_exclude=post_exclude,
        threshold=args.threshold,
        layer_tag=args.layer,
        materialization=args.materialization,
        at_ts=args.at_ts,
        output_path=args.output,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
