"""Classify N2/N3 neurons as central-complex local vs projection neurons.

This module queries presynaptic (output) synapse locations from a CAVE materialization
service and derives a concise projection summary per neuron.  For every neuron,
we record whether its outputs stay inside the central complex ("local_cx") or
project to additional neuropils ("projection") and a semicolon-delimited list of
its observed output neuropils.

The logic is intentionally tolerant to the flexible metadata schemas used in the
CAVE synapse tables.  Neuropil annotations may live under different field names
or be encoded as nested structures.  We therefore walk the metadata recursively
and extract any strings that look like neuropil labels, normalising them to an
uppercase token.  When neuropil metadata is missing, a warning is emitted so the
caller can audit the upstream data quality.
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set

import pandas as pd

try:  # Optional import: callers may provide a mock client during testing.
    from caveclient import CAVEclient  # type: ignore
except Exception:  # pragma: no cover - used only when caveclient is unavailable.
    CAVEclient = Any  # type: ignore[misc,assignment]


LOGGER = logging.getLogger(__name__)

CENTRAL_COMPLEX_NEUROPILS: Set[str] = {"PB", "EB", "FB", "NO"}

# Neuropils associated with projection neurons.  Any neuropil outside the
# central complex is considered projection-supporting, but these labels are
# especially common and therefore enumerated for clarity.
PROJECTION_NEUROPILS: Set[str] = {"SMP", "SLP", "LH", "LAL"}

# Metadata keys that commonly host neuropil information in CAVE tables.
NEUROPIL_KEYWORDS: Sequence[str] = (
    "neuropil",
    "Neuropil",
    "roi",
    "ROI",
    "roi_name",
    "compartment",
    "compartments",
    "super_roi",
    "tags",
    "annotations",
    "location",
)


def _chunked(iterable: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    """Yield fixed-size chunks from *iterable*.

    Parameters
    ----------
    iterable:
        Sequence of values to chunk.
    size:
        Desired chunk size.
    """

    for start in range(0, len(iterable), size):
        yield iterable[start : start + size]


def _normalise_label(value: str) -> Optional[str]:
    """Normalise a neuropil label.

    The metadata is messy – entries may include prefixes/suffixes (e.g. "PB_R"),
    lowercase tags, or comma-separated lists.  We collapse whitespace, split on
    common separators, and return upper-case tokens.
    """

    if not value:
        return None

    # Replace separators with spaces then split.
    cleaned = value.replace("/", " ").replace(",", " ").replace(";", " ")
    cleaned = cleaned.replace("|", " ").replace(":", " ")
    cleaned = cleaned.replace("-", " ").replace("_", " ")
    tokens = [token.strip().upper() for token in cleaned.split() if token.strip()]
    if not tokens:
        return None

    # Return the token that looks like a neuropil label.  The first token usually
    # carries the compartment name (e.g. "PB", "SMP").
    return tokens[0]


def _maybe_parse_json(metadata: Any) -> Any:
    """Attempt to parse JSON encoded metadata if necessary."""

    if isinstance(metadata, str):
        metadata = metadata.strip()
        if metadata.startswith("{") or metadata.startswith("["):
            try:
                return json.loads(metadata)
            except json.JSONDecodeError:
                LOGGER.debug("Failed to decode metadata JSON: %s", metadata)
                return metadata
    return metadata


def _extract_neuropils(metadata: Any) -> Set[str]:
    """Extract neuropil labels from a metadata blob."""

    neuropils: Set[str] = set()
    if metadata is None:
        return neuropils

    metadata = _maybe_parse_json(metadata)

    def _visit(item: Any) -> None:
        if isinstance(item, str):
            label = _normalise_label(item)
            if label:
                neuropils.add(label)
        elif isinstance(item, Mapping):
            for key, value in item.items():
                if key in NEUROPIL_KEYWORDS:
                    _visit(value)
                else:
                    _visit(value)
        elif isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray)):
            for value in item:
                _visit(value)

    _visit(metadata)

    return neuropils


def classify_projection_targets(neuropils: Iterable[str]) -> str:
    """Return ``local_cx`` if all neuropils are within the central complex.

    Parameters
    ----------
    neuropils:
        Iterable of neuropil labels associated with presynaptic outputs.

    Returns
    -------
    str
        Either ``"local_cx"`` or ``"projection"``.  If no neuropil information is
        available the neuron is conservatively labelled as ``"projection"``.
    """

    neuropil_set = {label.upper() for label in neuropils if label}
    if not neuropil_set:
        return "projection"
    if neuropil_set & PROJECTION_NEUROPILS:
        return "projection"
    if neuropil_set.issubset(CENTRAL_COMPLEX_NEUROPILS):
        return "local_cx"
    return "projection"


@dataclass
class ProjectionClassifier:
    """Classify neurons as central-complex local vs projection neurons.

    The classifier fetches all presynaptic synapses associated with the supplied
    neurons, extracts neuropil metadata, and summarises the projection pattern.
    """

    client: "CAVEclient"
    neuron_table: str
    synapse_table: str
    output_path: Path
    neuron_type_values: Sequence[str] = ("N2", "N3")
    neuron_type_columns: Sequence[str] = ("nt_type", "cell_type", "celltype", "type")
    pre_root_column: str = "pre_pt_root_id"
    metadata_column: str = "metadata"
    chunk_size: int = 10_000
    logger: logging.Logger = field(default=LOGGER)

    def run(self) -> pd.DataFrame:
        """Execute the full classification workflow and persist the results."""

        neurons = self._fetch_neurons()
        if neurons.empty:
            raise ValueError("No neurons matched the requested type filters.")

        synapses = self._fetch_synapses(neurons["root_id"].tolist())
        results = self._summarise(neurons, synapses)
        self._persist(results)
        return results

    # ------------------------------------------------------------------
    # Data fetching helpers
    # ------------------------------------------------------------------
    def _fetch_neurons(self) -> pd.DataFrame:
        """Retrieve N2/N3 neurons from the materialisation service."""

        self.logger.info("Querying neuron table '%s'", self.neuron_table)
        neurons = self.client.materialize.query_table(self.neuron_table)
        if not isinstance(neurons, pd.DataFrame):
            neurons = pd.DataFrame(neurons)

        type_column = self._resolve_type_column(neurons)
        filtered = neurons[neurons[type_column].isin(self.neuron_type_values)].copy()
        if "root_id" not in filtered.columns:
            raise KeyError("Neuron table is missing the 'root_id' column required for joins.")

        filtered = filtered[["root_id", type_column]].rename(columns={type_column: "neuron_type"})
        self.logger.info("Identified %d neurons of types %s", len(filtered), self.neuron_type_values)
        return filtered

    def _resolve_type_column(self, neurons: pd.DataFrame) -> str:
        for column in self.neuron_type_columns:
            if column in neurons.columns:
                return column
        raise KeyError(
            "Unable to identify neuron type column. Tried: %s" % ", ".join(self.neuron_type_columns)
        )

    def _fetch_synapses(self, root_ids: Sequence[int]) -> pd.DataFrame:
        """Retrieve all presynaptic synapses for the given neurons."""

        all_synapses: List[pd.DataFrame] = []
        missing: Counter[int] = Counter()

        for chunk in _chunked(root_ids, self.chunk_size):
            self.logger.info(
                "Querying synapse table '%s' for %d neurons", self.synapse_table, len(chunk)
            )
            query = self.client.materialize.query_table(
                self.synapse_table,
                filter_in_dict={self.pre_root_column: list(chunk)},
            )
            if not isinstance(query, pd.DataFrame):
                query = pd.DataFrame(query)
            if query.empty:
                continue

            if self.metadata_column not in query.columns:
                raise KeyError(
                    f"Synapse table '{self.synapse_table}' is missing required column '{self.metadata_column}'."
                )

            extracted_neuropils: List[Set[str]] = []
            for metadata in query[self.metadata_column]:
                neuropils = _extract_neuropils(metadata)
                if not neuropils:
                    # We'll log the counts later; for now track which root_id lacked metadata.
                    # The current dataframe contains the root ids for the chunk; the column may
                    # be named 'pre_pt_root_id' or similar.  We fetch it lazily later.
                    extracted_neuropils.append(set())
                else:
                    extracted_neuropils.append(neuropils)

            query = query.assign(_neuropils=extracted_neuropils)
            all_synapses.append(query)

            # Count missing metadata per neuron.
            if not query.empty:
                for root_id, neuropils in zip(query[self.pre_root_column], extracted_neuropils):
                    if not neuropils:
                        missing[int(root_id)] += 1

        for root_id, count in missing.items():
            self.logger.warning(
                "Missing neuropil metadata for %d presynaptic outputs on neuron %d", count, root_id
            )

        if not all_synapses:
            self.logger.warning("No synapses found for the requested neurons.")
            return pd.DataFrame(columns=[self.pre_root_column, "_neuropils"])

        synapses = pd.concat(all_synapses, ignore_index=True)
        return synapses[[self.pre_root_column, "_neuropils"]]

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
    def _summarise(self, neurons: pd.DataFrame, synapses: pd.DataFrame) -> pd.DataFrame:
        grouped: MutableMapping[int, Set[str]] = defaultdict(set)
        for root_id, neuropils in zip(synapses[self.pre_root_column], synapses["_neuropils"]):
            grouped[int(root_id)].update(neuropils)

        rows = []
        for root_id, neuron_type in neurons[["root_id", "neuron_type"]].itertuples(index=False):
            neuropils = grouped.get(int(root_id), set())
            projection_class = classify_projection_targets(neuropils)
            rows.append(
                {
                    "root_id": int(root_id),
                    "neuron_type": neuron_type,
                    "projection_class": projection_class,
                    "output_neuropils": ";".join(sorted(neuropils)),
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Persistence helper
    # ------------------------------------------------------------------
    def _persist(self, df: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        self.logger.info("Wrote projection classification to %s", self.output_path)


__all__ = ["ProjectionClassifier", "classify_projection_targets"]
