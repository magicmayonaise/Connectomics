"""Utilities for enumerating directed paths in connectome adjacency tables."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Protocol, Sequence, Tuple

import pandas as pd

__all__ = [
    "PathEnumerator",
    "GraphPathEnumerator",
    "extract_paths",
    "load_adjacency_table",
    "write_path_outputs",
    "main",
]


class PathEnumerator(Protocol):
    """Protocol describing path enumeration in a directed graph."""

    def enumerate_paths(self, sources: Sequence[int], targets: Sequence[int]) -> Iterable[Sequence[int]]:
        """Yield index paths between the given source and target populations."""
        ...


_SOURCE_ALIASES = (
    "source",
    "pre",
    "pre_root_id",
    "pre_pt_root_id",
    "upstream",
    "from",
)
_TARGET_ALIASES = (
    "target",
    "post",
    "post_root_id",
    "post_pt_root_id",
    "downstream",
    "to",
)


@dataclass(frozen=True)
class PathExtractionResult:
    """Container describing the paths discovered during enumeration."""

    paths: List[Tuple[int, ...]]
    source_column: str
    target_column: str

    def to_frame(self) -> pd.DataFrame:
        """Return a dataframe summarising each path as a single row."""

        rows = [
            {
                "path_id": idx,
                "hop_count": len(path) - 1,
                "nodes": ";".join(str(node) for node in path),
            }
            for idx, path in enumerate(self.paths)
        ]
        return pd.DataFrame(rows, columns=["path_id", "hop_count", "nodes"])


class GraphPathEnumerator(PathEnumerator):
    """Enumerate simple paths up to ``max_hops`` in a directed graph."""

    def __init__(self, adjacency: Mapping[int, Iterable[int]], *, max_hops: int = 3) -> None:
        if max_hops <= 0:
            raise ValueError("max_hops must be a positive integer")
        graph: Dict[int, List[int]] = {}
        for node, neighbours in adjacency.items():
            normalized_node = _normalise_node_id(node)
            values = {_normalise_node_id(neighbour) for neighbour in neighbours}
            graph[normalized_node] = sorted(values)
        self._graph = graph
        self._max_hops = max_hops

    def enumerate_paths(self, sources: Sequence[int], targets: Sequence[int]) -> Iterable[Tuple[int, ...]]:
        """Yield all simple paths connecting ``sources`` to ``targets``."""

        if not sources:
            raise ValueError("At least one source node must be provided.")
        if not targets:
            raise ValueError("At least one target node must be provided.")

        source_order = _deduplicate_preserving_order(sources)
        target_set = {_normalise_node_id(node) for node in targets}
        if not target_set:
            raise ValueError("Targets collapsed to an empty set after normalisation.")

        for source in source_order:
            src = _normalise_node_id(source)
            if src not in self._graph:
                continue
            yield from self._dfs(src, target_set)

    def _dfs(self, source: int, targets: set[int]) -> Iterator[Tuple[int, ...]]:
        yield from self._visit(source, [source], {source}, 0, targets)

    def _visit(
        self,
        node: int,
        path: List[int],
        visited: set[int],
        depth: int,
        targets: set[int],
    ) -> Iterator[Tuple[int, ...]]:
        if depth >= self._max_hops:
            return
        neighbours = sorted(
            self._graph.get(node, ()),
            key=lambda value: (value not in targets, value),
        )
        for neighbour in neighbours:
            if neighbour in visited:
                continue
            next_path = path + [neighbour]
            next_depth = depth + 1
            next_visited = visited | {neighbour}
            if neighbour in targets:
                yield tuple(next_path)
            if next_depth < self._max_hops:
                yield from self._visit(neighbour, next_path, next_visited, next_depth, targets)


def extract_paths(
    adjacency: pd.DataFrame,
    *,
    sources: Sequence[int],
    targets: Sequence[int],
    max_hops: int = 3,
    source_column: str | None = None,
    target_column: str | None = None,
) -> PathExtractionResult:
    """Enumerate paths connecting ``sources`` and ``targets``."""

    src_col = source_column or _find_column(adjacency, _SOURCE_ALIASES)
    tgt_col = target_column or _find_column(adjacency, _TARGET_ALIASES)

    adjacency = adjacency.dropna(subset=[src_col, tgt_col])
    if adjacency.empty:
        return PathExtractionResult(paths=[], source_column=src_col, target_column=tgt_col)

    adjacency_map = _build_adjacency(adjacency, src_col, tgt_col)
    enumerator = GraphPathEnumerator(adjacency_map, max_hops=max_hops)
    paths = list(enumerator.enumerate_paths(sources, targets))
    return PathExtractionResult(paths=paths, source_column=src_col, target_column=tgt_col)


def load_adjacency_table(path: Path) -> pd.DataFrame:
    """Load an adjacency table from ``path`` supporting CSV and Parquet."""

    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported adjacency format: {path.suffix}")


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Enumerate directed connectome paths.")
    parser.add_argument("--adjacency", type=Path, required=True, help="Path to a CSV or Parquet adjacency table.")
    parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        type=int,
        required=True,
        help="Source root ID. Provide multiple times to specify multiple sources.",
    )
    parser.add_argument(
        "--target",
        dest="targets",
        action="append",
        type=int,
        required=True,
        help="Target root ID. Provide multiple times to specify multiple targets.",
    )
    parser.add_argument("--max-hops", type=int, default=3, help="Maximum hop distance to explore.")
    parser.add_argument(
        "--source-column",
        dest="source_column",
        help="Override automatic detection of the source column name.",
    )
    parser.add_argument(
        "--target-column",
        dest="target_column",
        help="Override automatic detection of the target column name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./out/ci-paths"),
        help="Directory where path summaries should be written.",
    )
    return parser


def write_path_outputs(
    result: PathExtractionResult, output_dir: Path, *, metadata: Mapping[str, object]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths_frame = result.to_frame()
    paths_frame.to_csv(output_dir / "paths.csv", index=False)

    payload = {"paths": [list(path) for path in result.paths]}
    (output_dir / "paths.json").write_text(json.dumps(payload, indent=2), encoding="utf8")

    summary: MutableMapping[str, object] = {
        "path_count": len(result.paths),
        "max_hops": metadata.get("max_hops"),
        "sources": metadata.get("sources"),
        "targets": metadata.get("targets"),
        "source_column": result.source_column,
        "target_column": result.target_column,
        "adjacency_path": metadata.get("adjacency_path"),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf8")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by ``python -m cx_connectome.ci.paths``."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    adjacency = load_adjacency_table(args.adjacency)
    result = extract_paths(
        adjacency,
        sources=args.sources,
        targets=args.targets,
        max_hops=args.max_hops,
        source_column=args.source_column,
        target_column=args.target_column,
    )

    metadata = {
        "adjacency_path": str(args.adjacency.resolve()),
        "sources": args.sources,
        "targets": args.targets,
        "max_hops": args.max_hops,
    }
    write_path_outputs(result, args.output, metadata=metadata)
    return 0


def _find_column(table: pd.DataFrame, candidates: Sequence[str]) -> str:
    lower_map = {column.lower(): column for column in table.columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]
    raise ValueError(
        "Required column not found. Provide explicit column names via --source-column/--target-column."
    )


def _build_adjacency(table: pd.DataFrame, source_col: str, target_col: str) -> Dict[int, List[int]]:
    adjacency: Dict[int, List[int]] = {}
    grouped = table.groupby(source_col)
    for source, frame in grouped:
        source_id = _normalise_node_id(source)
        neighbours = {_normalise_node_id(value) for value in frame[target_col].tolist()}
        adjacency[source_id] = sorted(neighbours)
    return adjacency


def _deduplicate_preserving_order(values: Sequence[int]) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for value in values:
        node = _normalise_node_id(value)
        if node not in seen:
            seen.add(node)
            ordered.append(node)
    return ordered


def _normalise_node_id(value: object) -> int:
    if value is None:
        raise ValueError("Node identifiers cannot be None")
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid node identifiers")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, float):
        if pd.isna(value):
            raise ValueError("NaN is not a valid node identifier")
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError("Empty node identifier encountered")
    return int(text)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
