"""Utilities for fetching neuron cell type annotations with caching and retries."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

logger = logging.getLogger(__name__)

CAVECLIENT_SPEC = importlib.util.find_spec("caveclient")

if CAVECLIENT_SPEC is not None:  # pragma: no cover - imported dynamically when available
    from caveclient import CAVEclient
else:  # pragma: no cover - imported dynamically when available
    CAVEclient = None  # type: ignore[assignment]

DEFAULT_DATASTACK = "flywire_fafb_production"
DEFAULT_CELL_TYPE_TABLE = "cell_types_v1"
DEFAULT_ROOT_ID_FIELD = "pt_root_id"
DEFAULT_CACHE_TEMPLATE = "cell_types_m{materialization}.json"


class CellTypeCacheWarning(UserWarning):
    """Indicates that a cell type cache could not be used safely."""


CacheData = Dict[int, Optional[Mapping[str, Any]]]
ResultData = Dict[int, Optional[Mapping[str, Any]]]


def fetch_cell_types(
    root_ids: Iterable[int],
    materialization: int = 783,
    *,
    datastack: str = DEFAULT_DATASTACK,
    cache_dir: Optional[Union[str, Path]] = None,
    cache_filename_template: str = DEFAULT_CACHE_TEMPLATE,
    table: str = DEFAULT_CELL_TYPE_TABLE,
    root_id_field: str = DEFAULT_ROOT_ID_FIELD,
    client: Optional[Any] = None,
    max_retries: int = 3,
    retry_delay: float = 0.5,
) -> ResultData:
    """Fetch cell type annotations for ``root_ids``.

    The function caches results to disk to support offline operation and retries
    queries to the remote annotation layer when transient failures occur.
    """

    normalized_root_ids = _normalize_root_ids(root_ids)
    if not normalized_root_ids:
        return {}

    cache_path = _cache_path(materialization, cache_dir, cache_filename_template)
    cache = _load_cache(cache_path, materialization)

    results: ResultData = {}
    missing: List[int] = []

    for root_id in normalized_root_ids:
        if root_id in cache:
            results[root_id] = cache[root_id]
        else:
            missing.append(root_id)

    if missing:
        fetched = _fetch_remote_annotations(
            missing,
            materialization=materialization,
            datastack=datastack,
            table=table,
            root_id_field=root_id_field,
            client=client,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        cache.update(fetched)

        for root_id in missing:
            if root_id in fetched:
                results[root_id] = fetched[root_id]
            else:
                cache[root_id] = None
                results[root_id] = None

        _save_cache(cache_path, materialization, cache)

    return {root_id: results.get(root_id) for root_id in normalized_root_ids}


def _cache_path(
    materialization: int,
    cache_dir: Optional[Union[str, Path]],
    template: str,
) -> Path:
    directory = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    return directory / template.format(materialization=materialization)


def _default_cache_dir() -> Path:
    env = os.environ.get("CONNECTOMICS_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "connectomics_gui"


def _normalize_root_ids(root_ids: Iterable[int]) -> List[int]:
    normalized: List[int] = []
    seen = set()
    for root_id in root_ids:
        try:
            normalized_id = int(root_id)
        except (TypeError, ValueError):
            logger.debug("Skipping invalid root id value: %r", root_id)
            continue
        if normalized_id not in seen:
            seen.add(normalized_id)
            normalized.append(normalized_id)
    normalized.sort()
    return normalized


def _load_cache(path: Path, materialization: int) -> CacheData:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive programming
        warnings.warn(
            f"Failed to read cell type cache at {path}: {exc}",
            CellTypeCacheWarning,
            stacklevel=2,
        )
        return {}

    stored_materialization = payload.get("materialization")
    if stored_materialization != materialization:
        warnings.warn(
            "Cell type cache materialization mismatch: "
            f"cached={stored_materialization} requested={materialization}. Ignoring cache.",
            CellTypeCacheWarning,
            stacklevel=2,
        )
        return {}

    values = payload.get("values")
    if not isinstance(values, Mapping):
        warnings.warn(
            f"Unexpected cache format in {path!s}; ignoring cache.",
            CellTypeCacheWarning,
            stacklevel=2,
        )
        return {}

    cache: CacheData = {}
    for key, value in values.items():
        try:
            cache[int(key)] = value
        except (TypeError, ValueError):
            logger.debug("Skipping cache entry with invalid root id key: %r", key)
            continue
    return cache


def _save_cache(path: Path, materialization: int, data: CacheData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "materialization": materialization,
        "values": {str(key): value for key, value in data.items()},
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, default=_jsonify)


def _jsonify(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonify(item) for item in value]
    return repr(value)


def _fetch_remote_annotations(
    root_ids: Sequence[int],
    *,
    materialization: int,
    datastack: str,
    table: str,
    root_id_field: str,
    client: Optional[Any],
    max_retries: int,
    retry_delay: float,
) -> CacheData:
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    logger.debug(
        "Fetching cell types for %s root IDs from table %s at materialization %s",
        len(root_ids),
        table,
        materialization,
    )

    active_client = client if client is not None else _create_client(datastack)
    materialize = getattr(active_client, "materialize", None)
    if materialize is None:
        raise RuntimeError("Provided client does not expose a materialize attribute")

    query_fn = getattr(materialize, "query_table", None)
    if query_fn is None:
        raise RuntimeError("Materialize client does not implement query_table")

    attempts = 0
    last_error: Optional[Exception] = None
    while attempts < max_retries:
        attempts += 1
        try:
            result = query_fn(
                table,
                materialization_version=materialization,
                filter_in_dict={root_id_field: list(root_ids)},
            )
            return _index_records(result, root_id_field)
        except Exception as exc:  # pragma: no cover - behavior validated via tests
            last_error = exc
            logger.warning("Cell type query attempt %s/%s failed: %s", attempts, max_retries, exc)
            if attempts >= max_retries:
                break
            if retry_delay > 0:
                time.sleep(retry_delay)

    assert last_error is not None
    raise last_error


def _create_client(datastack: str) -> Any:
    if CAVECLIENT_SPEC is None:  # pragma: no cover - defensive programming
        raise RuntimeError(
            "caveclient is required to fetch annotations. Install fafbseg or caveclient."
        )
    return CAVEclient(datastack)


def _index_records(result: Any, root_id_field: str) -> CacheData:
    records = _normalize_records(result)
    indexed: CacheData = {}
    for record in records:
        if not isinstance(record, Mapping):
            logger.debug("Skipping cell type record with unsupported type: %r", type(record))
            continue
        if root_id_field not in record:
            logger.debug("Skipping record missing %s: %r", root_id_field, record)
            continue
        try:
            root_id = int(record[root_id_field])
        except (TypeError, ValueError):
            logger.debug("Skipping record with non-integer root id: %r", record)
            continue
        indexed[root_id] = dict(record)
    return indexed


def _normalize_records(result: Any) -> List[Mapping[str, Any]]:
    if result is None:
        return []
    if isinstance(result, list):
        normalized: List[Mapping[str, Any]] = []
        for item in result:
            if isinstance(item, Mapping):
                normalized.append(dict(item))
            elif hasattr(item, "_asdict"):
                normalized.append(item._asdict())
            else:
                normalized.append(dict(item))
        return normalized

    if isinstance(result, Mapping):
        return [dict(result)]

    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        try:
            converted = to_dict("records")
        except TypeError:
            converted = to_dict()
        if isinstance(converted, list):
            return [dict(record) if isinstance(record, Mapping) else dict(record) for record in converted]
        if isinstance(converted, Mapping):
            keys = list(converted.keys())
            length = len(converted[keys[0]]) if keys else 0
            normalized_list: List[Mapping[str, Any]] = []
            for index in range(length):
                normalized_list.append({key: converted[key][index] for key in keys})
            return normalized_list

    if isinstance(result, Sequence):
        return [dict(item) if isinstance(item, Mapping) else dict(item) for item in result]

    raise TypeError(f"Unsupported cell type query result type: {type(result)!r}")


__all__ = ["fetch_cell_types", "CellTypeCacheWarning"]
