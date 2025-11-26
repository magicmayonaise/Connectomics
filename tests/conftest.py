from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

from connectomics_cli.config import QueryContext


@dataclass
class _FakeAuth:
    def get_token(self) -> str:
        return "fake-token"


class _FakeChunkedGraph:
    def get_roots(self, root_ids: list[int], *, timestamp: Any | None = None) -> dict[int, int]:
        return {int(rid): int(rid) for rid in root_ids}

    def get_latest_roots(self, root_ids: list[int], *, timestamp: Any | None = None) -> dict[int, int]:
        return {int(rid): int(rid) for rid in root_ids}


class _FakeMaterialize:
    def __init__(self, synapse_df: pd.DataFrame, cell_type_df: pd.DataFrame) -> None:
        self._tables = [
            {
                "name": "synapse_table",
                "schema": [
                    {"name": "pre_pt_root_id"},
                    {"name": "post_pt_root_id"},
                    {"name": "id"},
                ],
            },
            {
                "name": "cell_type_table",
                "schema": [
                    {"name": "pt_root_id"},
                    {"name": "cell_type"},
                ],
            },
        ]
        self._data = {
            "synapse_table": synapse_df.reset_index(drop=True),
            "synapses": synapse_df.reset_index(drop=True),
            "annotations": pd.DataFrame(
                {
                    "target_id": pd.Series(dtype="int64"),
                    "tag": pd.Series(dtype="string"),
                    "cell_type": pd.Series(dtype="string"),
                    "super_class": pd.Series(dtype="string"),
                }
            ),
            "cell_type_table": cell_type_df.reset_index(drop=True),
        }

    def get_tables(self) -> list[dict[str, Any]]:
        return self._tables

    def get_closest_materialization(self, *, timestamp: Any | None = None) -> int:
        return 783

    def query_table(self, *args: Any, table: str | None = None, filter_in_dict: dict[str, list[int]] | None = None, select_columns: list[str] | None = None, format: str | None = None, **_: Any) -> pd.DataFrame:
        if table is None and args:
            table = args[0]
        if table is None:
            raise ValueError("A table name must be supplied")

        data = self._data[table]
        if filter_in_dict:
            filtered = data
            for column, values in filter_in_dict.items():
                if not values:
                    return data.head(0)
                filtered = filtered[filtered[column].isin(values)]
            data = filtered
        if select_columns is not None:
            data = data.loc[:, select_columns]
        return data.reset_index(drop=True)


class FakeCAVEclient:
    def __init__(self, synapse_df: pd.DataFrame, cell_type_df: pd.DataFrame) -> None:
        self.materialize = _FakeMaterialize(synapse_df, cell_type_df)
        self.chunkedgraph = _FakeChunkedGraph()
        self.auth = _FakeAuth()


@pytest.fixture()
def fake_client() -> FakeCAVEclient:
    synapse_df = pd.DataFrame(
        [
            {"pre_pt_root_id": 1001, "post_pt_root_id": 2001, "id": 1},
            {"pre_pt_root_id": 1001, "post_pt_root_id": 2001, "id": 2},
            {"pre_pt_root_id": 1001, "post_pt_root_id": 2002, "id": 3},
            {"pre_pt_root_id": 1002, "post_pt_root_id": 2002, "id": 4},
        ]
    )
    cell_type_df = pd.DataFrame(
        [
            {"pt_root_id": 1001, "cell_type": "TypeA"},
            {"pt_root_id": 1002, "cell_type": "TypeB"},
            {"pt_root_id": 2001, "cell_type": "TypeC"},
            {"pt_root_id": 2002, "cell_type": "TypeD"},
        ]
    )
    return FakeCAVEclient(synapse_df, cell_type_df)


@pytest.fixture()
def fake_context() -> QueryContext:
    return QueryContext(dataset="fake", materialization=783, timestamp=None)
