"""Smoke tests for the connectivity reporting workflow."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pytest

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from connectomics.reporting import ConnectivityAnalyzer


def _read_csv(name: str) -> List[Dict[str, str]]:
    with (DATA_DIR / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class _MockMaterialize:
    def __init__(self, tables: Mapping[str, Sequence[Dict[str, str]]]):
        self._tables = tables
        self.calls: List[Dict[str, Any]] = []

    def query_table(
        self,
        table: str,
        *,
        materialization_version: int | None = None,
        filter_in_dict: Mapping[str, Sequence[Any]] | None = None,
    ) -> List[Dict[str, str]]:
        call_record = {
            "table": table,
            "materialization_version": materialization_version,
            "filter_in_dict": filter_in_dict,
        }
        self.calls.append(call_record)

        rows = [dict(row) for row in self._tables.get(table, ())]
        if filter_in_dict:
            normalized_filter: Dict[str, set[str]] = {}
            for key, values in filter_in_dict.items():
                normalized_filter[key] = {str(value) for value in values}
            filtered: List[Dict[str, str]] = []
            for row in rows:
                include = True
                for key, allowed in normalized_filter.items():
                    value = row.get(key)
                    if value is None or str(value) not in allowed:
                        include = False
                        break
                if include:
                    filtered.append(row)
            rows = filtered
        return rows


class _MockChunkedgraph:
    def __init__(self, mapping: Mapping[int, int]):
        self._mapping = {int(key): int(value) for key, value in mapping.items()}
        self.calls: List[Sequence[int]] = []

    def get_latest_roots(self, root_ids: Iterable[int]) -> List[int]:
        normalized = [int(root_id) for root_id in root_ids]
        self.calls.append(tuple(normalized))
        return [self._mapping.get(root_id, root_id) for root_id in normalized]


class _MockCAVEclient:
    def __init__(self, tables: Mapping[str, Sequence[Dict[str, str]]], mapping: Mapping[int, int]):
        self.materialize = _MockMaterialize(tables)
        self.chunkedgraph = _MockChunkedgraph(mapping)


@pytest.fixture()
def tables() -> Dict[str, List[Dict[str, str]]]:
    return {
        "id_updates": _read_csv("id_updates"),
        "synapses": _read_csv("synapses"),
        "overlaps": _read_csv("overlaps"),
    }


@pytest.fixture()
def mock_client(tables: Mapping[str, Sequence[Dict[str, str]]]) -> _MockCAVEclient:
    # chunkedgraph returns a differing value for 1002 so the table mapping wins.
    chunkedgraph_mapping = {1001: 2001, 1002: 2202, 1003: 2003}
    return _MockCAVEclient(tables, chunkedgraph_mapping)


@pytest.fixture()
def analyzer(mock_client: _MockCAVEclient) -> ConnectivityAnalyzer:
    return ConnectivityAnalyzer(
        client=mock_client,
        synapse_table="synapses",
        overlap_table="overlaps",
        id_update_table="id_updates",
        id_update_source_field="old_root_id",
        id_update_target_field="new_root_id",
    )


def test_id_update_mapping_uses_table_and_chunkedgraph(analyzer: ConnectivityAnalyzer, mock_client: _MockCAVEclient) -> None:
    mapping = analyzer.map_ids([1001, 1002, 1003, 1002], materialization=17)

    assert mapping == {1001: 2001, 1002: 2002, 1003: 2003}
    assert mock_client.materialize.calls[0]["table"] == "id_updates"
    assert set(mock_client.materialize.calls[0]["filter_in_dict"]["old_root_id"]) == {1001, 1002, 1003}
    # Only the unmatched ID should require a chunkedgraph lookup.
    assert mock_client.chunkedgraph.calls == [(1003,)]


def test_adjacency_thresholds_filter_edges(analyzer: ConnectivityAnalyzer, mock_client: _MockCAVEclient) -> None:
    updated = analyzer.map_ids([1001, 1002, 1003]).values()

    strict = analyzer.compute_adjacency(updated, min_synapses=3, materialization=21)
    assert [(edge.pre_id, edge.post_id, edge.synapse_count) for edge in strict] == [(2001, 2002, 4)]
    assert mock_client.materialize.calls[-1]["materialization_version"] == 21

    loose = analyzer.compute_adjacency(updated, min_synapses=2)
    assert [(edge.pre_id, edge.post_id) for edge in loose] == [(2001, 2002), (2001, 2003)]
    assert [edge.synapse_count for edge in loose] == [4, 2]


def test_overlap_metrics_match_synapse_counts(analyzer: ConnectivityAnalyzer) -> None:
    updated = analyzer.map_ids([1001, 1002, 1003]).values()
    adjacency = analyzer.compute_adjacency(updated, min_synapses=2)

    overlaps = analyzer.compute_overlap_metrics(adjacency, materialization=42)
    metrics = {(record.pre_id, record.post_id): record for record in overlaps}

    first = metrics[(2001, 2002)]
    second = metrics[(2001, 2003)]
    total_overlap = first.overlap_volume + second.overlap_volume

    assert first.overlap_volume == pytest.approx(500.0)
    assert first.overlap_per_synapse == pytest.approx(125.0)
    assert first.normalized_overlap == pytest.approx(500.0 / total_overlap)
    assert first.pre_fraction == pytest.approx(0.5)
    assert first.post_fraction == pytest.approx(0.4)

    assert second.overlap_volume == pytest.approx(150.0)
    assert second.overlap_per_synapse == pytest.approx(75.0)
    assert second.normalized_overlap == pytest.approx(150.0 / total_overlap)
    assert second.pre_fraction == pytest.approx(0.25)
    assert second.post_fraction == pytest.approx(0.2)


def test_report_assembly_produces_summary(analyzer: ConnectivityAnalyzer, mock_client: _MockCAVEclient) -> None:
    report = analyzer.build_report([1001, 1002, 1003], min_synapses=2, materialization=99)

    assert report.id_mapping == {1001: 2001, 1002: 2002, 1003: 2003}
    assert [(edge.pre_id, edge.post_id, edge.synapse_count) for edge in report.adjacency] == [
        (2001, 2002, 4),
        (2001, 2003, 2),
    ]
    assert [(record.pre_id, record.post_id) for record in report.overlaps] == [(2001, 2002), (2001, 2003)]

    totals = report.totals
    assert totals["total_synapses"] == pytest.approx(6.0)
    assert totals["edge_count"] == pytest.approx(2.0)
    assert totals["mean_synapses_per_edge"] == pytest.approx(3.0)
    assert report.parameters["materialization"] == 99
    assert report.parameters["min_synapses"] == 2
    assert report.summary().startswith("3 neurons mapped to 2 high-confidence edges totalling 6 synapses")

    report_dict = report.as_dict()
    assert tuple(report_dict["parameters"]["root_ids"]) == (1001, 1002, 1003)
    assert [entry["synapse_count"] for entry in report_dict["adjacency"]] == [4, 2]

    queried_tables = [call["table"] for call in mock_client.materialize.calls]
    assert queried_tables == ["id_updates", "synapses", "overlaps"]
