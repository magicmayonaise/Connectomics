from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import pytest

pd = pytest.importorskip("pandas")

from cx_connectome.legacy import trace_next_layer


class _StubMaterialize:
    def __init__(self, records: Sequence[Mapping[str, Any]]):
        self._records = records
        self.calls: list[Dict[str, Any]] = []

    def query_table(self, table: str, **kwargs: Any) -> Sequence[Mapping[str, Any]]:
        self.calls.append({"table": table, "kwargs": kwargs})
        return self._records


class _StubClient:
    def __init__(self, records: Sequence[Mapping[str, Any]]):
        self.materialize = _StubMaterialize(records)


def _fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False, **_: Any) -> None:
    Path(path).write_text("placeholder parquet output", encoding="utf-8")


def _annotation_fetcher(root_ids: Iterable[int], **_: Any) -> Mapping[int, Mapping[str, Any]]:
    return {int(root_id): {"cell_type": f"type_{int(root_id)}"} for root_id in root_ids}


def test_trace_next_layer_builds_outputs(tmp_path: Path, monkeypatch: Any) -> None:
    previous_layer = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 2],
            "post_pt_root_id": [10, 11, 10],
        }
    )

    synapse_records = [
        {"pre_pt_root_id": 10, "post_pt_root_id": 20, "synapse_count": 12},
        {"pre_pt_root_id": 10, "post_pt_root_id": 30, "synapse_count": 5},
        {"pre_pt_root_id": 11, "post_pt_root_id": 20, "synapse_count": 11},
        {"pre_pt_root_id": 11, "post_pt_root_id": 40, "synapse_count": 15},
    ]

    client = _StubClient(synapse_records)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet, raising=False)

    result = trace_next_layer(
        client,
        previous_layer,
        materialization=5,
        threshold=10,
        output_dir=tmp_path,
        annotation_fetcher=_annotation_fetcher,
    )

    # ``synapse_count`` below the threshold (10) is filtered out.
    assert list(result.adjacency.columns) == ["pre_pt_root_id", "post_pt_root_id", "synapse_count"]
    assert result.adjacency.to_dict("records") == [
        {"pre_pt_root_id": 10, "post_pt_root_id": 20, "synapse_count": 12},
        {"pre_pt_root_id": 11, "post_pt_root_id": 40, "synapse_count": 15},
        {"pre_pt_root_id": 11, "post_pt_root_id": 20, "synapse_count": 11},
    ]

    # Verify that ``query_table`` was invoked once with the deduplicated N2 set.
    assert len(client.materialize.calls) == 1
    call = client.materialize.calls[0]
    assert call["table"] == "synapses_nt_v1"
    assert call["kwargs"]["filter_in_dict"]["pre_pt_root_id"] == [10, 11]
    assert call["kwargs"]["materialization_version"] == 5

    adjacency_path = tmp_path / "N2_to_N3_adjacency.parquet"
    n2_path = tmp_path / "N2_nodes.csv"
    n3_path = tmp_path / "N3_nodes.csv"

    assert adjacency_path.exists()
    assert n2_path.exists()
    assert n3_path.exists()

    n2_nodes = pd.read_csv(n2_path)
    n3_nodes = pd.read_csv(n3_path)

    # ``N2`` nodes come from unique posts in the previous layer, deduplicated from N1.
    assert n2_nodes.to_dict("records") == [
        {"pt_root_id": 10, "cell_type": "type_10"},
        {"pt_root_id": 11, "cell_type": "type_11"},
    ]

    # ``N3`` nodes exclude already seen layers and include merged annotations.
    assert n3_nodes.to_dict("records") == [
        {"pt_root_id": 20, "cell_type": "type_20"},
        {"pt_root_id": 40, "cell_type": "type_40"},
    ]

