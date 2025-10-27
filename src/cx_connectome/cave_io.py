"""Utilities for interacting with the CAVE materialization service.

This module provides thin wrappers around :mod:`caveclient` that make it a
little easier to authenticate, list available tables, and issue queries that
return :class:`pandas.DataFrame` instances backed by PyArrow dtypes when
possible.  The functions here are intended to be light-weight helpers for
scripts and command line entry points and they purposely avoid introducing
additional dependencies beyond :mod:`pandas` and :mod:`caveclient`.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

try:  # pragma: no cover - exercised in environments with caveclient installed.
    from caveclient import CAVEclient as _CAVEclient
except ImportError as exc:  # pragma: no cover - handled at runtime when absent.
    _CAVEclient = None
    _CAVE_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised in environments with caveclient installed.
    _CAVE_IMPORT_ERROR = None


__all__ = [
    "get_client",
    "require_auth",
    "list_tables",
    "live_query",
    "mat_snapshot_query",
    "main",
]


def get_client(dataset: str, server: str | None = None) -> "CAVEclient":
    """Create a :class:`caveclient.CAVEclient` for *dataset*.

    Parameters
    ----------
    dataset:
        The Codex/CAVE dataset identifier to connect to.
    server:
        Optional explicit server URL.  When omitted the default server for the
        dataset is used.

    Returns
    -------
    CAVEclient
        An authenticated (or at least instantiated) client ready for use.

    Raises
    ------
    ImportError
        If :mod:`caveclient` is not installed in the current environment.
    ValueError
        If *dataset* is empty.
    """

    if not dataset:
        raise ValueError("A dataset name is required to create a CAVEclient")

    if _CAVEclient is None:  # pragma: no cover - depends on optional dependency.
        raise ImportError(
            "caveclient is required to create a CAVEclient instance. Install it "
            "from https://github.com/seung-lab/caveclient or ensure it is "
            "available in your environment."
        ) from _CAVE_IMPORT_ERROR

    kwargs: dict[str, Any] = {}
    if server:
        # Older releases expect ``server_address`` while newer releases accept
        # ``server_url``.  Try both forms to remain compatible.
        for key in ("server_address", "server_url", "base_url"):
            kwargs[key] = server
            try:
                return _CAVEclient(dataset, **kwargs)
            except TypeError:
                kwargs.pop(key, None)
                continue
            except Exception:
                kwargs.pop(key, None)
                raise
        raise TypeError(
            "Unable to determine the correct keyword argument for overriding "
            "the CAVE server address."
        )

    return _CAVEclient(dataset)


def require_auth(client: Any) -> None:
    """Ensure that *client* has a usable authentication token.

    The helper inspects the ``client.auth`` object and checks for a cached
    middle-auth token.  When no token is present a clear error is raised that
    explains how to authenticate using Google OAuth2 via the Codex/CAVE login
    flow.  Tokens issued by middle-auth are cached locally (typically in
    ``~/.cloudvolume/secrets``) and will be reused automatically once saved.

    Parameters
    ----------
    client:
        A :class:`caveclient.CAVEclient` instance.

    Raises
    ------
    RuntimeError
        If the client does not expose an authentication helper or no cached
        token is available.
    """

    auth = getattr(client, "auth", None)
    if auth is None:
        raise RuntimeError(
            "The provided client does not expose an 'auth' helper. Ensure you "
            "created it with caveclient.CAVEclient."
        )

    token = None
    last_error: Exception | None = None

    candidate = getattr(auth, "token", None)
    if callable(candidate):
        try:
            token = candidate()
        except Exception as exc:  # pragma: no cover - defensive.
            last_error = exc
            token = None
    elif candidate is not None:
        token = candidate

    if not token and hasattr(auth, "get_token"):
        try:
            token = auth.get_token()
        except Exception as exc:  # pragma: no cover - defensive.
            last_error = exc
            token = None

    if not token:
        message = textwrap.dedent(
            """
            No middle-auth token is cached for this CAVEclient instance.

            To authenticate:
              1. Run the Codex/CAVE login flow (Google OAuth2) by executing
                 `python -m caveclient.auth.save_token --dataset <dataset>` or
                 use the `caveclient-auth` CLI if it is available.
              2. Follow the browser prompt to approve access.  The resulting
                 token will be cached locally so subsequent runs can reuse it.
            """
        ).strip()
        if last_error is not None:  # pragma: no cover - depends on runtime.
            raise RuntimeError(message) from last_error
        raise RuntimeError(message)


def list_tables(client: Any, materialization: int | None = None) -> pd.DataFrame:
    """Return metadata about available materialization tables.

    The resulting :class:`pandas.DataFrame` contains the schema name, table
    name, reported row count (when provided by the service), and a boolean flag
    indicating whether the table can be accessed at the requested
    materialization version.  When *materialization* is ``None`` all known
    tables are considered available.
    """

    require_auth(client)
    materialize = getattr(client, "materialize", None)
    if materialize is None:
        raise RuntimeError("The provided client does not expose materialize APIs")

    tables = _materialize_tables(materialize)

    records: list[dict[str, Any]] = []
    for table_info in tables:
        raw_schema = _first_not_none(
            table_info,
            "schema",
            "schema_name",
            "schemaname",
            "table_schema",
        )
        raw_name = _first_not_none(
            table_info,
            "table",
            "tablename",
            "name",
            "table_name",
        )
        raw_row_count = _first_not_none(
            table_info,
            "row_count",
            "rows",
            "n_rows",
            "num_rows",
        )

        versions = _extract_versions(table_info)
        available: Any
        if materialization is None:
            available = True
        else:
            available = _check_availability(
                materialize,
                raw_name,
                materialization,
                versions,
            )

        records.append(
            {
                "schema": raw_schema if raw_schema is not None else pd.NA,
                "table": raw_name if raw_name is not None else pd.NA,
                "row_count": raw_row_count if raw_row_count is not None else pd.NA,
                "available": available,
            }
        )

    df = pd.DataFrame.from_records(records)
    try:
        df = df.convert_dtypes(dtype_backend="pyarrow")
    except TypeError:  # pragma: no cover - pandas < 2.0
        df = df.convert_dtypes()

    return df


def live_query(
    client: Any,
    table: str,
    *,
    timestamp: int | None = None,
    materialization: int | None = None,
    dtype_backend: str | None = "pyarrow",
    **kwargs: Any,
) -> pd.DataFrame:
    """Query the live endpoint for *table* and return a DataFrame.

    When *materialization* is provided this function delegates to
    :func:`mat_snapshot_query` and therefore issues a snapshot query.  When no
    materialization is supplied the live endpoint is used with the provided
    *timestamp* (if any).  CAVE maps the timestamp to the nearest available
    snapshot and merges in any pending deltas so that arbitrary points in time
    can be queried.
    """

    if materialization is not None:
        return mat_snapshot_query(
            client,
            table,
            materialization=materialization,
            dtype_backend=dtype_backend,
            **kwargs,
        )

    require_auth(client)
    materialize = getattr(client, "materialize", None)
    if materialize is None:
        raise RuntimeError("The provided client does not expose materialize APIs")

    params = {k: v for k, v in kwargs.items() if v is not None}
    if timestamp is not None:
        params.setdefault("timestamp", timestamp)

    result = _invoke_materialize(materialize, "live_query", table, params)
    return _ensure_dataframe(result, dtype_backend=dtype_backend)


def mat_snapshot_query(
    client: Any,
    table: str,
    *,
    materialization: int | None = None,
    dtype_backend: str | None = "pyarrow",
    **kwargs: Any,
) -> pd.DataFrame:
    """Query a materialized snapshot for *table*.

    The helper calls :meth:`caveclient.materialization.MaterializationClient`
    ``query_table`` method with the provided arguments and returns the result as
    a DataFrame that prefers the PyArrow backed nullable dtypes.
    """

    require_auth(client)
    materialize = getattr(client, "materialize", None)
    if materialize is None:
        raise RuntimeError("The provided client does not expose materialize APIs")

    params = {k: v for k, v in kwargs.items() if v is not None}
    if materialization is not None:
        params.setdefault("materialization_version", materialization)

    result = _invoke_materialize(materialize, "query_table", table, params)
    return _ensure_dataframe(result, dtype_backend=dtype_backend)


def _materialize_tables(materialize: Any) -> list[Mapping[str, Any]]:
    """Return a list of table metadata dictionaries."""

    if hasattr(materialize, "get_tables"):
        tables = materialize.get_tables()
    elif hasattr(materialize, "tables"):
        tables = materialize.tables()
    else:  # pragma: no cover - defensive programming.
        raise RuntimeError("Materialization client does not provide table listings")

    if isinstance(tables, pd.DataFrame):
        return list(tables.to_dict(orient="records"))
    if isinstance(tables, Mapping):  # pragma: no cover - defensive.
        return [tables]
    return list(tables)


def _first_not_none(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _extract_versions(data: Mapping[str, Any]) -> Iterable[int] | None:
    for key in (
        "valid_materialization_versions",
        "versions",
        "materialization_versions",
        "materialized_versions",
    ):
        value = data.get(key)
        if value:
            return _normalize_versions(value)
    return None


def _normalize_versions(value: Any) -> Iterable[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set, frozenset)):
        return {int(v) for v in value if v is not None}
    if isinstance(value, Mapping):
        return {int(k) for k in value.keys() if k is not None}
    try:
        return {int(value)}
    except (TypeError, ValueError):  # pragma: no cover - defensive.
        return None


def _check_availability(
    materialize: Any,
    table: str,
    materialization: int,
    versions: Iterable[int] | None,
) -> Any:
    if table is None:
        return False

    if versions is not None:
        try:
            return materialization in set(int(v) for v in versions)
        except Exception:  # pragma: no cover - defensive.
            pass

    for method_name in (
        "check_table_availability",
        "table_available",
        "has_table_version",
    ):
        method = getattr(materialize, method_name, None)
        if callable(method):
            try:
                return bool(method(table_name=table, materialization_version=materialization))
            except TypeError:
                try:
                    return bool(method(table, materialization))
                except Exception:  # pragma: no cover - defensive.
                    continue
            except Exception:  # pragma: no cover - defensive.
                continue

    probe_kwargs = {"table_name": table, "limit": 0, "materialization_version": materialization}
    method = getattr(materialize, "query_table", None)
    if callable(method):
        try:
            method(**probe_kwargs)
        except TypeError:
            try:
                method(table, limit=0, materialization_version=materialization)
            except Exception:  # pragma: no cover - defensive.
                return False
        except Exception:  # pragma: no cover - defensive.
            return False
        else:
            return True

    return pd.NA


def _invoke_materialize(materialize: Any, method_name: str, table: str, kwargs: Mapping[str, Any]):
    method = getattr(materialize, method_name, None)
    if not callable(method):
        raise RuntimeError(f"Materialization client does not provide '{method_name}'")

    cleaned_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        return method(table_name=table, **cleaned_kwargs)
    except TypeError:
        return method(table, **cleaned_kwargs)


def _ensure_dataframe(result: Any, *, dtype_backend: str | None = "pyarrow") -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        df = result
    elif hasattr(result, "to_pandas"):
        df = result.to_pandas()
    else:
        df = pd.DataFrame(result)

    if dtype_backend:
        try:
            df = df.convert_dtypes(dtype_backend=dtype_backend)
        except TypeError:  # pragma: no cover - pandas < 2.0
            df = df.convert_dtypes()
    return df


def main(argv: Sequence[str] | None = None) -> int:
    """Entry-point used by the ``cx-cave`` console script."""

    parser = argparse.ArgumentParser(prog="cx-cave", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    tables_parser = subparsers.add_parser(
        "tables",
        help="List candidate synapse and cell type tables for a dataset",
    )
    tables_parser.add_argument(
        "--dataset",
        required=True,
        help="Codex/CAVE dataset identifier (e.g. 'minnie65_public_v117')",
    )
    tables_parser.add_argument(
        "--server",
        help="Optional override for the CAVE server URL",
    )
    tables_parser.add_argument(
        "--materialization",
        type=int,
        help="Materialization version to filter on",
    )

    args = parser.parse_args(argv)

    if args.command == "tables":
        try:
            client = get_client(args.dataset, server=args.server)
            table_df = list_tables(client, materialization=args.materialization)
        except Exception as exc:  # pragma: no cover - CLI error path.
            parser.error(str(exc))

        table_series = table_df.get("table")
        if table_series is None:
            parser.error("Table listing did not include a 'table' column")

        mask = table_series.fillna("").str.contains("synapse|cell_type", case=False, na=False)
        filtered = table_df.loc[mask, ["schema", "table", "row_count", "available"]]
        filtered = filtered.sort_values(by=["schema", "table"], ignore_index=True)

        if filtered.empty:
            print("No synapse or cell_type tables were found for the requested dataset.")
            return 0

        print(filtered.to_string(index=False))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised via CLI only.
    sys.exit(main())
