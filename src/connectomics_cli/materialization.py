"""Utilities for materialization-aware queries."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import pyarrow as pa

from .config import DEFAULT_CHUNK_SIZE, DEFAULT_MATERIALIZATION, QueryContext


class TableSelectionError(RuntimeError):
    """Raised when no table matches the discovery heuristics."""


@dataclass(slots=True, frozen=True)
class RootMapping:
    """Mapping between requested root IDs and snapshot-aligned IDs."""

    requested: int
    snapshot: int


def resolve_query_context(
    client: Any,
    dataset: str,
    *,
    materialization: int | None,
    timestamp: datetime | None,
) -> QueryContext:
    """Derive the :class:`QueryContext` from user inputs.

    If ``timestamp`` is provided the closest snapshot is discovered via helper
    methods offered by :mod:`caveclient`. When no method is available the
    default materialization is used as a safe fallback.
    """

    if materialization is not None and timestamp is not None:
        msg = "Only one of materialization or timestamp may be provided."
        raise ValueError(msg)
    if timestamp is not None:
        snapshot = _closest_materialization(client, timestamp)
        return QueryContext(dataset=dataset, materialization=snapshot, timestamp=timestamp)
    resolved = materialization if materialization is not None else DEFAULT_MATERIALIZATION
    return QueryContext(dataset=dataset, materialization=resolved, timestamp=None)


def _closest_materialization(client: Any, timestamp: datetime) -> int:
    materialize = getattr(client, "materialize", None)
    if materialize is None:
        return DEFAULT_MATERIALIZATION
    candidates = (
        "get_closest_materialization",
        "get_closest_snapshot",
        "get_timestamp_info",
    )
    for attr in candidates:
        if not hasattr(materialize, attr):
            continue
        method = getattr(materialize, attr)
        try:
            info = method(timestamp=timestamp)
        except TypeError:
            info = method(timestamp.isoformat())
        candidate = _extract_materialization_id(info)
        if candidate is not None:
            return candidate
    listing = getattr(materialize, "get_materialization_info", None)
    if listing is not None:
        info = listing()
        if isinstance(info, Mapping):
            candidate = _extract_materialization_id(info)
            if candidate is not None:
                return candidate
        if isinstance(info, Sequence) and info:
            candidate = _extract_materialization_id(info[0])
            if candidate is not None:
                return candidate
    return DEFAULT_MATERIALIZATION


def _extract_materialization_id(info: Any) -> int | None:
    if info is None:
        return None
    if isinstance(info, int):
        return info
    if isinstance(info, Mapping):
        for key in ("id", "materialization_id", "materialization_version"):
            value = info.get(key)
            if isinstance(value, int):
                return value
    return None


def map_roots_to_snapshot(
    client: Any,
    root_ids: Sequence[int],
    *,
    timestamp: datetime | None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[RootMapping]:
    """Align root IDs to the snapshot associated with ``timestamp``.

    When ``timestamp`` is ``None`` the identity mapping is returned.
    """

    if not root_ids:
        return []
    if timestamp is None:
        return [RootMapping(int(rid), int(rid)) for rid in root_ids]
    chunked = getattr(client, "chunkedgraph", None)
    if chunked is None:
        return [RootMapping(int(rid), int(rid)) for rid in root_ids]
    method_name = next(
        (attr for attr in ("get_roots", "get_root_id", "get_root_ids") if hasattr(chunked, attr)),
        None,
    )
    if method_name is None:
        return [RootMapping(int(rid), int(rid)) for rid in root_ids]
    method = getattr(chunked, method_name)
    mappings: list[RootMapping] = []
    for chunk in _chunk_sequence(root_ids, chunk_size):
        try:
            result = method(chunk, timestamp=timestamp)
        except TypeError:
            result = method(chunk, timestamp=timestamp.isoformat())
        mappings.extend(_normalise_root_response(chunk, result))
    seen = {mapping.requested for mapping in mappings}
    for rid in root_ids:
        if int(rid) not in seen:
            mappings.append(RootMapping(int(rid), int(rid)))
    return mappings


def restore_timestamp_roots(
    client: Any,
    snapshot_root_ids: Sequence[int],
    *,
    timestamp: datetime | None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[int, int]:
    """Map snapshot-aligned root IDs back to the requested timestamp."""

    if not snapshot_root_ids:
        return {}
    if timestamp is None:
        return {int(rid): int(rid) for rid in snapshot_root_ids}
    chunked = getattr(client, "chunkedgraph", None)
    if chunked is None:
        return {int(rid): int(rid) for rid in snapshot_root_ids}
    method_name = next(
        (
            attr
            for attr in ("get_latest_roots", "get_roots_at_timestamp", "get_roots")
            if hasattr(chunked, attr)
        ),
        None,
    )
    if method_name is None:
        return {int(rid): int(rid) for rid in snapshot_root_ids}
    method = getattr(chunked, method_name)
    mapping: dict[int, int] = {}
    for chunk in _chunk_sequence(snapshot_root_ids, chunk_size):
        try:
            result = method(chunk, timestamp=timestamp)
        except TypeError:
            result = method(chunk, timestamp=timestamp.isoformat())
        mapping.update(_normalise_reverse_root_response(chunk, result))
    for rid in snapshot_root_ids:
        mapping.setdefault(int(rid), int(rid))
    return mapping


def _normalise_root_response(chunk: Sequence[int], result: Any) -> list[RootMapping]:
    if result is None:
        return [RootMapping(int(rid), int(rid)) for rid in chunk]
    if isinstance(result, Mapping):
        return [
            RootMapping(int(key), int(value))
            for key, value in result.items()
            if value is not None
        ]
    if isinstance(result, Sequence):
        normalised = []
        for requested, snapshot in zip(chunk, result, strict=False):
            if snapshot is None:
                continue
            normalised.append(RootMapping(int(requested), int(snapshot)))
        return normalised
    return [RootMapping(int(rid), int(rid)) for rid in chunk]


def _normalise_reverse_root_response(chunk: Sequence[int], result: Any) -> dict[int, int]:
    if result is None:
        return {int(rid): int(rid) for rid in chunk}
    if isinstance(result, Mapping):
        normalised: dict[int, int] = {}
        for key, value in result.items():
            if value is None:
                continue
            normalised[int(value)] = int(key)
        return normalised
    if isinstance(result, Sequence):
        mapping: dict[int, int] = {}
        for snapshot, requested in zip(chunk, result, strict=False):
            if requested is None:
                continue
            mapping[int(snapshot)] = int(requested)
        return mapping
    return {int(rid): int(rid) for rid in chunk}


def query_table_chunked(
    client: Any,
    table: str,
    *,
    context: QueryContext,
    filter_in: Mapping[str, Sequence[int]] | None = None,
    columns: Sequence[str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Yield dataframes containing the results of chunked materialization queries."""

    materialize = getattr(client, "materialize")
    filter_in = filter_in or {}
    for chunk_filter in _chunk_filter_in(filter_in, chunk_size):
        query_kwargs: dict[str, Any] = {"table": table}
        if chunk_filter:
            query_kwargs["filter_in_dict"] = chunk_filter
        if columns:
            query_kwargs["select_columns"] = list(columns)
        if context.materialization is not None:
            query_kwargs["materialization_version"] = context.materialization
        if context.timestamp is not None:
            query_kwargs["timestamp"] = context.timestamp.isoformat()
        data = _execute_materialize_query(materialize, query_kwargs)
        if data.empty:
            continue
        yield data


def _chunk_filter_in(
    filter_in: Mapping[str, Sequence[int]],
    chunk_size: int,
) -> Iterator[Mapping[str, Sequence[int]]]:
    if not filter_in:
        yield {}
        return
    keys = list(filter_in)
    longest_key = max(keys, key=lambda key: len(filter_in[key]))
    longest_values = list(filter_in[longest_key])
    for chunk in _chunk_sequence(longest_values, chunk_size):
        chunk_filter: dict[str, Sequence[int]] = {longest_key: chunk}
        for key in keys:
            if key == longest_key:
                continue
            chunk_filter[key] = filter_in[key]
        yield chunk_filter


def _chunk_sequence(values: Sequence[int], chunk_size: int) -> Iterator[list[int]]:
    total = len(values)
    for start in range(0, total, chunk_size):
        chunk = [int(value) for value in values[start : start + chunk_size]]
        if not chunk:
            continue
        yield chunk


def _execute_materialize_query(materialize: Any, query_kwargs: Mapping[str, Any]) -> pd.DataFrame:
    try:
        result = materialize.query_table(format="arrow", **query_kwargs)
    except TypeError:
        result = materialize.query_table(**query_kwargs)
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pa.Table):
        return result.to_pandas()
    if isinstance(result, Sequence):
        return pd.DataFrame(result)
    if isinstance(result, Mapping):
        return pd.DataFrame(result)
    return pd.DataFrame()


def list_tables(client: Any) -> list[dict[str, Any]]:
    """Return a normalised description of annotation tables."""

    materialize = getattr(client, "materialize")
    tables = materialize.get_tables()
    normalised: list[dict[str, Any]] = []
    for entry in tables:
        if isinstance(entry, str):
            normalised.append({"name": entry, "schema": []})
            continue
        if isinstance(entry, Mapping):
            name = entry.get("name", "")
            schema = entry.get("schema") or entry.get("columns") or []
            normalised.append({"name": str(name), "schema": schema})
            continue
    return normalised


def get_table_schema(client: Any, table: str) -> list[str]:
    """Return the column names advertised for ``table``."""

    for entry in list_tables(client):
        if entry.get("name") != table:
            continue
        schema = entry.get("schema", [])
        columns: list[str] = []
        if isinstance(schema, Sequence) and not isinstance(schema, (str, bytes)):
            for column in schema:
                if isinstance(column, str):
                    columns.append(column)
                elif isinstance(column, Mapping):
                    name = column.get("name")
                    if isinstance(name, str):
                        columns.append(name)
        return columns
    return []


def find_best_table(
    client: Any,
    *,
    name_hint: str,
    required_columns: Iterable[str],
) -> str:
    """Select a table whose schema matches ``required_columns``.

    The heuristic favours tables whose names contain ``name_hint``.
    """

    tables = list_tables(client)
    required = {column.lower() for column in required_columns}
    scored: list[tuple[int, str]] = []
    for table in tables:
        name = table.get("name", "")
        schema_columns = _normalise_schema(table.get("schema", []))
        if not required.issubset(schema_columns):
            continue
        score = 0
        lowered = name.lower()
        if name_hint.lower() in lowered:
            score += 2
        if lowered.startswith(name_hint.lower()):
            score += 1
        if lowered.endswith(name_hint.lower()):
            score += 1
        scored.append((score, name))
    if not scored:
        raise TableSelectionError(
            f"No table matching hint '{name_hint}' with columns {sorted(required)} was found."
        )
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _normalise_schema(schema: Any) -> set[str]:
    columns: set[str] = set()
    if isinstance(schema, Sequence) and not isinstance(schema, (str, bytes)):
        for entry in schema:
            if isinstance(entry, str):
                columns.add(entry.lower())
            elif isinstance(entry, Mapping):
                name = entry.get("name")
                if isinstance(name, str):
                    columns.add(name.lower())
    return columns
