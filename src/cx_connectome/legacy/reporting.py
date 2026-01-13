"""Reporting utilities for summarizing small connectomics data slices."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AdjacencyRecord:
    """Weighted connection between two neurons."""

    pre_id: int
    post_id: int
    synapse_count: int


@dataclass(frozen=True)
class OverlapRecord:
    """Morphological overlap metrics for a neuron pair."""

    pre_id: int
    post_id: int
    overlap_volume: float
    pre_fraction: float
    post_fraction: float
    overlap_per_synapse: float
    normalized_overlap: float


@dataclass(frozen=True)
class ConnectivityReport:
    """Structured summary of connectivity for a neuron set."""

    id_mapping: Mapping[int, int]
    adjacency: Tuple[AdjacencyRecord, ...]
    overlaps: Tuple[OverlapRecord, ...]
    totals: Mapping[str, float]
    parameters: Mapping[str, Any]

    def summary(self) -> str:
        """Return a short textual summary of the report."""

        neuron_count = len(self.id_mapping)
        edge_count = int(self.totals.get("edge_count", 0))
        total_synapses = int(self.totals.get("total_synapses", 0))
        min_synapses = self.parameters.get("min_synapses")
        materialization = self.parameters.get("materialization")
        materialization_str = f", materialization={materialization}" if materialization is not None else ""
        return (
            f"{neuron_count} neurons mapped to {edge_count} high-confidence edges "
            f"totalling {total_synapses} synapses (min_synapses={min_synapses}{materialization_str})."
        )

    def as_dict(self) -> Dict[str, Any]:
        """Convert the report into a serialisable dictionary."""

        return {
            "id_mapping": dict(self.id_mapping),
            "adjacency": [asdict(edge) for edge in self.adjacency],
            "overlaps": [asdict(overlap) for overlap in self.overlaps],
            "totals": dict(self.totals),
            "parameters": dict(self.parameters),
        }


class ConnectivityAnalyzer:
    """Construct connectivity summaries from a ``caveclient`` instance."""

    def __init__(
        self,
        *,
        client: Any,
        synapse_table: str,
        overlap_table: str,
        id_update_table: Optional[str] = None,
        id_update_source_field: str = "old_root_id",
        id_update_target_field: str = "new_root_id",
    ) -> None:
        self._client = client
        self.synapse_table = synapse_table
        self.overlap_table = overlap_table
        self.id_update_table = id_update_table
        self.id_update_source_field = id_update_source_field
        self.id_update_target_field = id_update_target_field

    # ------------------------------------------------------------------
    # Public API
    def map_ids(self, root_ids: Iterable[int], *, materialization: Optional[int] = None) -> Dict[int, int]:
        """Return a mapping from requested to latest root IDs."""

        normalized: List[int] = []
        seen = set()
        for value in root_ids:
            try:
                candidate = int(value)
            except (TypeError, ValueError):
                continue
            if candidate not in seen:
                seen.add(candidate)
                normalized.append(candidate)

        mapping: Dict[int, int] = {}

        if self.id_update_table and normalized:
            filter_dict = {self.id_update_source_field: normalized}
            records = self._query_table(
                self.id_update_table,
                materialization_version=materialization,
                filter_in_dict=filter_dict,
            )
            for record in records:
                source_value = record.get(self.id_update_source_field)
                target_value = record.get(self.id_update_target_field)
                try:
                    source = int(source_value)
                    target = int(target_value)
                except (TypeError, ValueError):
                    continue
                mapping[source] = target

        missing = [root_id for root_id in normalized if root_id not in mapping]
        chunkedgraph = getattr(self._client, "chunkedgraph", None)
        if missing and chunkedgraph is not None:
            getter = getattr(chunkedgraph, "get_latest_roots", None)
            if getter is None:
                raise RuntimeError("chunkedgraph client does not expose get_latest_roots")
            latest = getter(missing)
            if isinstance(latest, Mapping):
                items = latest.items()
            else:
                items = zip(missing, latest)
            for original, updated in items:
                try:
                    mapping[int(original)] = int(updated)
                except (TypeError, ValueError):
                    continue

        for root_id in normalized:
            mapping.setdefault(root_id, root_id)

        return mapping

    def compute_adjacency(
        self,
        root_ids: Iterable[int],
        *,
        min_synapses: int = 1,
        materialization: Optional[int] = None,
    ) -> List[AdjacencyRecord]:
        """Compute synapse counts between provided neurons."""

        normalized_ids = self._normalize_id_iterable(root_ids)
        if not normalized_ids:
            return []

        records = self._query_table(self.synapse_table, materialization_version=materialization)
        id_set = set(normalized_ids)
        counts: Dict[Tuple[int, int], int] = {}
        for record in records:
            pre_value = record.get("pre_pt_root_id")
            post_value = record.get("post_pt_root_id")
            try:
                pre_id = int(pre_value)
                post_id = int(post_value)
            except (TypeError, ValueError):
                continue
            if pre_id not in id_set and post_id not in id_set:
                continue
            counts[(pre_id, post_id)] = counts.get((pre_id, post_id), 0) + 1

        adjacency = [
            AdjacencyRecord(pre_id=pre_id, post_id=post_id, synapse_count=count)
            for (pre_id, post_id), count in counts.items()
            if count >= min_synapses
        ]
        adjacency.sort(key=lambda edge: (-edge.synapse_count, edge.pre_id, edge.post_id))
        return adjacency

    def compute_overlap_metrics(
        self,
        adjacency: Sequence[AdjacencyRecord],
        *,
        materialization: Optional[int] = None,
    ) -> List[OverlapRecord]:
        """Lookup morphological overlap metrics for ``adjacency`` pairs."""

        if not adjacency:
            return []

        records = self._query_table(self.overlap_table, materialization_version=materialization)
        lookup: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        for record in records:
            try:
                pre_id = int(record.get("pre_pt_root_id"))
                post_id = int(record.get("post_pt_root_id"))
            except (TypeError, ValueError):
                continue
            overlap_volume = float(
                record.get("overlap_volume_nm3")
                or record.get("contact_voxels")
                or record.get("overlap_volume")
                or 0.0
            )
            pre_fraction = float(
                record.get("pre_fraction")
                or record.get("pre_overlap_fraction")
                or record.get("pre_cable_fraction")
                or 0.0
            )
            post_fraction = float(
                record.get("post_fraction")
                or record.get("post_overlap_fraction")
                or record.get("post_cable_fraction")
                or 0.0
            )
            lookup[(pre_id, post_id)] = (overlap_volume, pre_fraction, post_fraction)

        total_overlap = 0.0
        for edge in adjacency:
            volume = lookup.get((edge.pre_id, edge.post_id), (0.0, 0.0, 0.0))[0]
            total_overlap += volume

        overlaps: List[OverlapRecord] = []
        for edge in adjacency:
            overlap_volume, pre_fraction, post_fraction = lookup.get((edge.pre_id, edge.post_id), (0.0, 0.0, 0.0))
            per_synapse = overlap_volume / edge.synapse_count if edge.synapse_count else 0.0
            normalized_overlap = overlap_volume / total_overlap if total_overlap else 0.0
            overlaps.append(
                OverlapRecord(
                    pre_id=edge.pre_id,
                    post_id=edge.post_id,
                    overlap_volume=overlap_volume,
                    pre_fraction=pre_fraction,
                    post_fraction=post_fraction,
                    overlap_per_synapse=per_synapse,
                    normalized_overlap=normalized_overlap,
                )
            )
        overlaps.sort(key=lambda overlap: (-overlap.overlap_volume, overlap.pre_id, overlap.post_id))
        return overlaps

    def build_report(
        self,
        root_ids: Iterable[int],
        *,
        min_synapses: int = 1,
        materialization: Optional[int] = None,
    ) -> ConnectivityReport:
        """Assemble a ``ConnectivityReport`` for ``root_ids``."""

        id_mapping = self.map_ids(root_ids, materialization=materialization)
        updated_ids = list(dict.fromkeys(id_mapping.values()))
        adjacency = self.compute_adjacency(
            updated_ids,
            min_synapses=min_synapses,
            materialization=materialization,
        )
        overlaps = self.compute_overlap_metrics(adjacency, materialization=materialization)

        total_synapses = sum(edge.synapse_count for edge in adjacency)
        edge_count = len(adjacency)
        totals = {
            "total_synapses": float(total_synapses),
            "edge_count": float(edge_count),
            "mean_synapses_per_edge": float(total_synapses / edge_count) if edge_count else 0.0,
        }
        parameters = {
            "min_synapses": min_synapses,
            "materialization": materialization,
            "root_ids": tuple(sorted(id_mapping.keys())),
        }

        return ConnectivityReport(
            id_mapping=id_mapping,
            adjacency=tuple(adjacency),
            overlaps=tuple(overlaps),
            totals=totals,
            parameters=parameters,
        )

    # ------------------------------------------------------------------
    # Helpers
    def _normalize_id_iterable(self, ids: Iterable[int]) -> List[int]:
        normalized: List[int] = []
        seen = set()
        for value in ids:
            try:
                candidate = int(value)
            except (TypeError, ValueError):
                continue
            if candidate not in seen:
                seen.add(candidate)
                normalized.append(candidate)
        return normalized

    def _query_table(
        self,
        table: str,
        *,
        materialization_version: Optional[int] = None,
        filter_in_dict: Optional[Mapping[str, Sequence[Any]]] = None,
    ) -> List[Mapping[str, Any]]:
        materialize = getattr(self._client, "materialize", None)
        if materialize is None:
            raise RuntimeError("Client does not expose a materialize interface")
        query = getattr(materialize, "query_table", None)
        if query is None:
            raise RuntimeError("Materialize client does not provide query_table")

        normalized_filter: Optional[Dict[str, List[Any]]] = None
        if filter_in_dict:
            normalized_filter = {}
            for key, values in filter_in_dict.items():
                normalized_values: List[Any] = []
                for value in values:
                    try:
                        normalized_values.append(int(value))
                    except (TypeError, ValueError):
                        normalized_values.append(value)
                normalized_filter[key] = normalized_values

        result = query(
            table,
            materialization_version=materialization_version,
            filter_in_dict=normalized_filter,
        )
        return _normalize_records(result)


def _normalize_records(result: Any) -> List[Mapping[str, Any]]:
    """Coerce a materialize query result into a list of mappings."""

    if result is None:
        return []

    if isinstance(result, list):
        normalized: List[Mapping[str, Any]] = []
        for item in result:
            if isinstance(item, Mapping):
                normalized.append(dict(item))
            elif hasattr(item, "_asdict"):
                normalized.append(dict(item._asdict()))
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
            return [dict(item) if isinstance(item, Mapping) else dict(item) for item in converted]
        if isinstance(converted, Mapping):
            keys = list(converted.keys())
            length = len(converted[keys[0]]) if keys else 0
            normalized: List[Mapping[str, Any]] = []
            for index in range(length):
                normalized.append({key: converted[key][index] for key in keys})
            return normalized

    if isinstance(result, Sequence):
        return [dict(item) if isinstance(item, Mapping) else dict(item) for item in result]

    raise TypeError(f"Unsupported materialize result type: {type(result)!r}")


__all__ = [
    "AdjacencyRecord",
    "ConnectivityAnalyzer",
    "ConnectivityReport",
    "OverlapRecord",
]
