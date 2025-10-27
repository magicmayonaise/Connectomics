"""cx_connectome.annotations
================================

Utilities to look up FlyWire annotations from the CAVE materialization API.

Schlegel et al. shared hierarchical annotations (superclass, class, cell_type,
and lineage) with the Codex/CAVE data repository; this module retrieves those
labels programmatically so that they can be associated with local analyses.
"""

from __future__ import annotations

import argparse
import logging
import math
from numbers import Integral
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence

import pandas as pd

try:  # pragma: no cover - handled dynamically when caveclient is available.
    from caveclient import CAVEclient  # type: ignore
except ImportError as exc:  # pragma: no cover - we want informative error at runtime.
    CAVEclient = None  # type: ignore
    _IMPORT_ERROR = exc
else:  # pragma: no cover - only executed when import succeeds.
    _IMPORT_ERROR = None

LOGGER = logging.getLogger(__name__)

# Known synonyms for the annotation columns we care about. The values are ordered
# by preference so that the most canonical name will be selected when multiple
# options are present in a table.
COLUMN_SYNONYMS: Mapping[str, Sequence[str]] = {
    "root_id": (
        "root_id",
        "pt_root_id",
        "pt_root",
        "root",
        "id",
        "entity_id",
        "neuron_id",
    ),
    "cell_type": ("cell_type", "type", "celltype", "cell_type_name"),
    "class": ("class", "cell_class", "type_class", "celltype_class"),
    "super_class": (
        "super_class",
        "superclass",
        "super_type",
        "supertyper",
        "supertype",
        "cell_superclass",
    ),
    "side": ("side", "hemisphere", "hemi", "laterality"),
    "lineage": (
        "lineage",
        "lineage_class",
        "lineage_full",
        "hemilineage",
        "neuron_lineage",
    ),
}

DEFAULT_DATASET = "flywire"
DEFAULT_OUTPUT = Path("out/N1_annotations.parquet")
DEFAULT_ROOTS_PATH = Path("out/N1_roots.parquet")
CHUNK_SIZE = 2048


class AnnotationTableNotFoundError(RuntimeError):
    """Raised when no suitable annotation table could be discovered."""


def _normalize_casefolded_mapping(columns: Sequence[str]) -> MutableMapping[str, str]:
    """Return a mapping from casefolded column names to their canonical form."""

    mapping: MutableMapping[str, str] = {}
    for column in columns:
        mapping[column.casefold()] = column
    return mapping


def _identify_column(columns: Sequence[str], synonyms: Sequence[str]) -> Optional[str]:
    """Return the first column present in *columns* that matches any synonym."""

    if not columns:
        return None

    lookup = _normalize_casefolded_mapping(columns)
    for synonym in synonyms:
        key = synonym.casefold()
        if key in lookup:
            return lookup[key]
    return None


def _extract_columns_from_metadata(metadata: Mapping[str, object]) -> List[str]:
    """Extract a flat list of column names from table metadata."""

    def _flatten_schema(schema: object) -> Iterator[str]:
        if isinstance(schema, Mapping):
            if "columns" in schema and isinstance(schema["columns"], Sequence):
                yield from _flatten_schema(schema["columns"])
            elif "items" in schema:
                yield from _flatten_schema(schema["items"])
            else:
                name = schema.get("name") if isinstance(schema.get("name"), str) else None  # type: ignore[attr-defined]
                if name:
                    yield name
        elif isinstance(schema, Sequence) and not isinstance(schema, (str, bytes)):
            for item in schema:
                yield from _flatten_schema(item)
        elif isinstance(schema, str):
            yield schema

    columns: List[str] = []
    if metadata is None:
        return columns

    for key in ("columns", "schema", "fields", "field_list"):
        value = metadata.get(key) if isinstance(metadata, Mapping) else None
        if value is None:
            continue
        for column in _flatten_schema(value):
            if isinstance(column, str) and column not in columns:
                columns.append(column)
    return columns


def _extract_table_names(raw_tables: object) -> List[str]:
    """Normalise the output of ``get_tables`` into a list of names."""

    if raw_tables is None:
        return []

    names: List[str] = []
    if isinstance(raw_tables, pd.DataFrame):
        for candidate in ("name", "table_name", "tablename"):
            if candidate in raw_tables.columns:
                names.extend(raw_tables[candidate].astype(str).tolist())
                break
        else:
            names.extend(raw_tables.iloc[:, 0].astype(str).tolist())
        return names

    if isinstance(raw_tables, Mapping):
        if "tables" in raw_tables and isinstance(raw_tables["tables"], Sequence):
            raw_tables = raw_tables["tables"]
        else:
            raw_tables = list(raw_tables.values())

    if isinstance(raw_tables, Sequence) and not isinstance(raw_tables, (str, bytes)):
        for entry in raw_tables:
            if isinstance(entry, str):
                names.append(entry)
            elif isinstance(entry, Mapping):
                for key in ("name", "table_name", "tablename"):
                    if key in entry and isinstance(entry[key], str):
                        names.append(entry[key])
                        break
            else:
                names.append(str(entry))
    else:
        names.append(str(raw_tables))

    return names


def discover_annotation_table(client, table_name: Optional[str] = None) -> str:
    """Discover the FlyWire annotation table to use.

    Parameters
    ----------
    client:
        Connected :class:`~caveclient.CAVEclient` instance.
    table_name:
        Optional override. When provided, the value is returned unmodified.

    Returns
    -------
    str
        The name of the chosen table.

    Raises
    ------
    AnnotationTableNotFoundError
        If no table could be inferred.
    """

    if table_name:
        LOGGER.info("Using user supplied annotation table: %s", table_name)
        return table_name

    try:
        available = client.materialize.get_tables()  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover - depends on external service.
        raise AnnotationTableNotFoundError(
            "Failed to retrieve table listing from the materialization service"
        ) from exc

    names = _extract_table_names(available)
    if not names:
        raise AnnotationTableNotFoundError("Materialization service returned no tables")

    scored: List[tuple[int, str, List[str]]] = []
    for name in names:
        lowered = name.casefold()
        score = 0
        if "cell_type" in lowered:
            score += 100
        if lowered == "cell_type":
            score += 25
        if lowered.startswith("flytable-info"):
            score += 90
        if "flytable-info" in lowered and "v783" in lowered:
            score += 40
        if "v783" in lowered:
            score += 20
        if "flywire" in lowered:
            score += 15

        columns: List[str] = []
        try:
            metadata = client.materialize.get_table_metadata(name)  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - optional metadata.
            metadata = {}
        else:
            columns = _extract_columns_from_metadata(metadata)

        if columns:
            column_lookup = {col.casefold() for col in columns}
            if "cell_type" in column_lookup:
                score += 60
            if "super_class" in column_lookup or "superclass" in column_lookup:
                score += 20
            if "class" in column_lookup:
                score += 15
            if "lineage" in column_lookup:
                score += 10

        scored.append((score, name, columns))

    scored.sort(reverse=True)
    if not scored or scored[0][0] == 0:
        raise AnnotationTableNotFoundError(
            "Could not identify a cell type annotation table automatically; "
            "please supply --table-name explicitly."
        )

    best_score, best_name, _ = scored[0]
    LOGGER.info(
        "Selected annotation table \"%s\" (score=%s)",
        best_name,
        best_score,
    )
    return best_name


def _chunked(iterable: Sequence[int], size: int) -> Iterator[Sequence[int]]:
    """Yield successive ``size`` sized chunks from *iterable*."""

    for start in range(0, len(iterable), size):
        yield iterable[start : start + size]


def _coerce_root_ids(roots: Iterable[object]) -> List[int]:
    """Normalise a collection of root identifiers to integers."""

    normalised: List[int] = []
    seen = set()
    for root in roots:
        if root is None:
            continue
        if isinstance(root, Integral):
            value = int(root)
        elif isinstance(root, float):
            if math.isnan(root):
                continue
            value = int(root)
        elif isinstance(root, str):
            stripped = root.strip()
            if not stripped:
                continue
            value = int(stripped, 10)
        else:
            try:
                value = int(root)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                LOGGER.debug("Skipping non-integral root id: %r", root)
                continue
        if value in seen:
            continue
        seen.add(value)
        normalised.append(value)
    return normalised


def fetch_cell_types(
    client,
    roots: Iterable[object],
    *,
    materialization: Optional[int] = None,
    at_ts: Optional[int] = None,
    table_name: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """Fetch hierarchical annotation metadata for FlyWire roots.

    Parameters
    ----------
    client:
        Connected :class:`~caveclient.CAVEclient` or compatible client.
    roots:
        Iterable of root ids to fetch annotations for.
    materialization:
        Optional materialization version to query. If omitted, the latest
        version visible to the user is used.
    at_ts:
        Optional timestamp to query against. Mutually exclusive with
        ``materialization`` according to CAVE semantics.
    table_name:
        Optional override for the materialized table name.
    chunk_size:
        Number of root ids to query per request.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing at least ``root_id``, ``cell_type``, ``class``,
        ``super_class``, ``side``, and ``lineage`` columns. Missing columns will
        be populated with ``pd.NA``.
    """

    root_ids = _coerce_root_ids(roots)
    if not root_ids:
        LOGGER.warning("No root ids provided; returning empty annotations dataframe")
        return pd.DataFrame(columns=list(COLUMN_SYNONYMS.keys()))

    table = discover_annotation_table(client, table_name=table_name)

    metadata: Mapping[str, object] = {}
    try:
        metadata = client.materialize.get_table_metadata(table)  # type: ignore[union-attr]
    except Exception:  # pragma: no cover - metadata optional.
        metadata = {}

    columns = _extract_columns_from_metadata(metadata)

    if not columns:
        try:
            sample = client.materialize.query_table(  # type: ignore[union-attr]
                table,
                limit=1,
                materialization_version=materialization,
                timestamp=at_ts,
            )
        except Exception:  # pragma: no cover - optional sample.
            sample = None
        else:
            if isinstance(sample, pd.DataFrame):
                columns = list(sample.columns)
            elif sample is not None:
                columns = list(sample) if isinstance(sample, Sequence) else []

    root_column = _identify_column(columns, COLUMN_SYNONYMS["root_id"])
    if root_column is None:
        raise AnnotationTableNotFoundError(
            "The discovered annotation table does not expose a recognised root id column"
        )

    desired_columns = {root_column}
    column_map = {"root_id": root_column}
    for canonical, synonyms in COLUMN_SYNONYMS.items():
        if canonical == "root_id":
            continue
        column = _identify_column(columns, synonyms)
        if column:
            desired_columns.add(column)
            column_map[canonical] = column

    results: List[pd.DataFrame] = []
    for chunk in _chunked(root_ids, chunk_size):
        filter_dict = {root_column: chunk}
        try:
            df = client.materialize.query_table(  # type: ignore[union-attr]
                table,
                filter_in_dict=filter_dict,
                materialization_version=materialization,
                timestamp=at_ts,
                split_query=False,
                desired_columns=list(desired_columns),
            )
        except TypeError:
            # Some deployments may not support ``desired_columns``; retry without it.
            df = client.materialize.query_table(  # type: ignore[union-attr]
                table,
                filter_in_dict=filter_dict,
                materialization_version=materialization,
                timestamp=at_ts,
                split_query=False,
            )
        except Exception as exc:  # pragma: no cover - external dependency
            LOGGER.warning("Annotation query failed for chunk %s..%s: %s", chunk[0], chunk[-1], exc)
            continue

        if df is None:
            continue
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        if df.empty:
            continue
        results.append(df)

    if not results:
        LOGGER.warning("Annotation query returned no rows; returning empty dataframe")
        return pd.DataFrame(columns=list(COLUMN_SYNONYMS.keys()))

    annotations = pd.concat(results, ignore_index=True)

    rename_map = {source: canonical for canonical, source in column_map.items()}
    annotations = annotations.rename(columns=rename_map)

    for canonical in COLUMN_SYNONYMS.keys():
        if canonical not in annotations.columns:
            annotations[canonical] = pd.NA

    annotations = annotations[list(COLUMN_SYNONYMS.keys())]
    annotations = annotations.drop_duplicates(subset=["root_id"]).reset_index(drop=True)
    annotations = annotations[annotations["root_id"].isin(root_ids)]

    return annotations


def load_root_ids(path: Path, *, column: str = "root_id") -> List[int]:
    """Load root ids from *path*.

    The loader understands ``.parquet``, ``.csv``, and ``.tsv`` files. When the
    requested column is not present, the first column in the file will be used.
    """

    if not path.exists():
        raise FileNotFoundError(f"Root id file does not exist: {path}")

    loader = None
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        loader = pd.read_parquet
    elif suffix == ".csv":
        loader = lambda p: pd.read_csv(p, sep=",")  # noqa: E731
    elif suffix == ".tsv":
        loader = lambda p: pd.read_csv(p, sep="\t")  # noqa: E731
    else:
        # Try parquet first and fall back to CSV.
        def loader(p: Path) -> pd.DataFrame:  # type: ignore[redefinition]
            try:
                return pd.read_parquet(p)
            except Exception:
                return pd.read_csv(p)

    data = loader(path)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    available_columns = list(data.columns)
    if column not in data.columns:
        for candidate in available_columns:
            if candidate.casefold() == column.casefold():
                column = candidate
                break
        else:
            if len(available_columns) != 1:
                raise KeyError(
                    f"Column '{column}' not found in {path}; available columns: {available_columns}"
                )
            column = available_columns[0]

    return _coerce_root_ids(data[column].tolist())


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch FlyWire hierarchical annotations")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="CAVE dataset identifier to connect to (default: %(default)s)",
    )
    parser.add_argument(
        "--roots-path",
        type=Path,
        default=DEFAULT_ROOTS_PATH,
        help="Path to a parquet/CSV file listing root ids (default: %(default)s)",
    )
    parser.add_argument(
        "--roots-column",
        default="root_id",
        help="Column in the roots file that contains root ids (default: %(default)s)",
    )
    parser.add_argument(
        "--materialization",
        type=int,
        help="Materialization version to query (mutually exclusive with --at-ts)",
    )
    parser.add_argument(
        "--at-ts",
        type=int,
        help="Timestamp to query (mutually exclusive with --materialization)",
    )
    parser.add_argument(
        "--table-name",
        help="Override the materialized table name used for annotations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination parquet file (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Number of root ids per materialization query (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: %(default)s)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "caveclient is required to fetch annotations but could not be imported"
        ) from _IMPORT_ERROR

    if args.materialization and args.at_ts:
        parser.error("--materialization and --at-ts are mutually exclusive")

    LOGGER.info("Loading root ids from %s", args.roots_path)
    roots = load_root_ids(args.roots_path, column=args.roots_column)

    LOGGER.info("Connecting to CAVE dataset '%s'", args.dataset)
    client = CAVEclient(args.dataset)  # type: ignore[call-arg]

    annotations = fetch_cell_types(
        client,
        roots,
        materialization=args.materialization,
        at_ts=args.at_ts,
        table_name=args.table_name,
        chunk_size=args.chunk_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing annotations to %s", args.output)
    annotations.to_parquet(args.output, index=False)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
