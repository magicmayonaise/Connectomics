"""Centralized configuration constants for the cx_connectome package.

This module provides a single source of truth for default values used across
the connectomics analysis pipeline. Import constants from here rather than
defining them in individual modules.

Constants are grouped by category:
- CAVE/Dataset configuration
- Synapse table names
- Output paths
- Analysis thresholds
- CI (Connectivity Interpreter) parameters

Example
-------
>>> from cx_connectome.constants import DEFAULT_DATASET, DEFAULT_MATERIALIZATION
>>> print(f"Using {DEFAULT_DATASET} at materialization {DEFAULT_MATERIALIZATION}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Tuple


# =============================================================================
# CAVE/Dataset Configuration
# =============================================================================

DEFAULT_DATASET = "flywire_fafb_production"
"""Default CAVE datastack for FlyWire FAFB production data."""

DEFAULT_MATERIALIZATION = 783
"""Default materialization version for reproducible queries."""

DEFAULT_CHUNK_SIZE = 5_000
"""Default chunk size for batched CAVE queries (rows per request)."""


# =============================================================================
# Synapse Table Configuration
# =============================================================================

DEFAULT_SYNAPSE_TABLE = "synapses_nt_v1"
"""Default synapse table in CAVE containing neurotransmitter annotations."""

SYNAPSE_TABLE_CANDIDATES: Tuple[str, ...] = (
    "synapses_nt_v1",
    "synapses_pni_2",
    "synapses",
    "synapse_table",
)
"""Ordered candidate table names for automatic synapse table discovery."""

BASELINE_TABLE_CANDIDATES: Tuple[str, ...] = (
    "classification_system",
    "aibs_soma_nuc_metamodel_preds_v117",
    "nucleus_detection_lookup_v1",
)
"""Ordered candidate table names for baseline/classification lookups."""


# =============================================================================
# Output Path Configuration
# =============================================================================

DEFAULT_OUTPUT_DIR = Path("out")
"""Default directory for analysis outputs."""

DEFAULT_N1_N2_OUTPUT = Path("out/N1_to_N2_adjacency.parquet")
"""Default output path for N1->N2 adjacency table."""

DEFAULT_N2_N3_OUTPUT = Path("out/N2_to_N3_adjacency.parquet")
"""Default output path for N2->N3 adjacency table."""

DEFAULT_GRAPH_OUTPUT = Path("out/cx_network_N1_N2_N3.gpickle")
"""Default output path for pickled NetworkX graph."""

DEFAULT_ANNOTATIONS_OUTPUT = Path("out/N1_annotations.parquet")
"""Default output path for annotation lookups."""

DEFAULT_ROOTS_PATH = Path("out/N1_roots.parquet")
"""Default path for root ID files."""


# =============================================================================
# Analysis Threshold Configuration
# =============================================================================

DEFAULT_MIN_SYNAPSES = 10
"""Default minimum synapse count threshold for edge filtering."""

DEFAULT_FIRST_HOP_THRESHOLD = 5
"""Default synapse threshold for N1->N2 connections."""

DEFAULT_SECOND_HOP_THRESHOLD = 10
"""Default synapse threshold for N2->N3 connections."""


# =============================================================================
# Population Label Configuration
# =============================================================================

DEFAULT_N1_LABEL = "N1"
"""Default label for first-layer neurons."""

DEFAULT_N2_LABEL = "N2"
"""Default label for second-layer neurons."""

DEFAULT_N3_LABEL = "N3"
"""Default label for third-layer neurons."""


# =============================================================================
# Functional Role Mappings
# =============================================================================

DEFAULT_FUNCTIONAL_MAPPING: Mapping[str, str] = {
    # Sleep-promoting cell types
    "ExR1": "Sleep-Promoting",
    "ExR2": "Sleep-Promoting",
    "ExR3": "Sleep-Promoting",
    "ExR4": "Sleep-Promoting",
    "ExR5": "Sleep-Promoting",
    "ExR6": "Sleep-Promoting",
    "ExR7": "Sleep-Promoting",
    "ExR8": "Sleep-Promoting",
    # Navigation cell types
    "EPG": "Navigation",
    "PEG": "Navigation",
    "PEN_a": "Navigation",
    "PEN_b": "Navigation",
    "Delta7": "Navigation",
    "EL": "Navigation",
    "FC1": "Navigation",
    "FC2": "Navigation",
    "hDeltaA": "Navigation",
    "hDeltaB": "Navigation",
    "hDeltaC": "Navigation",
    "hDeltaD": "Navigation",
    "hDeltaE": "Navigation",
    "hDeltaF": "Navigation",
    "hDeltaG": "Navigation",
    "hDeltaH": "Navigation",
    "hDeltaI": "Navigation",
    "hDeltaJ": "Navigation",
    "hDeltaK": "Navigation",
    "hDeltaL": "Navigation",
    "hDeltaM": "Navigation",
    "PFN": "Navigation",
    "vDelta": "Navigation",
}
"""Mapping of cell types to functional roles (Sleep-Promoting, Navigation, etc.)."""


# =============================================================================
# Super-class Synonym Mappings (for upstream analysis)
# =============================================================================

DEFAULT_SUPER_CLASS_SYNONYMS: Mapping[str, str] = {
    "ascending": "ascending",
    "asc": "ascending",
    "descending": "descending",
    "desc": "descending",
    "visual": "visual",
    "visual_projection": "visual",
    "mechanosensory": "mechanosensory",
    "mech": "mechanosensory",
    "central": "central",
    "central_brain": "central",
}
"""Synonym mappings for super-class normalization."""

DEFAULT_SUPER_CLASS_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    ("visual", "visual"),
    ("optic_lobe", "visual"),
    ("ol_intrinsic", "visual"),
    ("descending", "descending"),
    ("ascending", "ascending"),
    ("mechanosensory", "mechanosensory"),
    ("jrc", "central"),
)
"""Keyword patterns for inferring super-class membership."""


# =============================================================================
# CI (Connectivity Interpreter) Configuration
# =============================================================================

DEFAULT_K_HOPS: Tuple[int, ...] = (1, 2, 3, 4, 5)
"""Default hop distances for effective connectivity calculation."""

DEFAULT_CI_CHUNK_SIZE = 4096
"""Default chunk size for CI column operations."""

DEFAULT_NORMALIZE_MODE = "post_total"
"""Default normalization mode for effective connectivity (column-normalize by post-total)."""

DEFAULT_EFFECTIVE_THRESHOLD = 0.01
"""Default threshold for normalized effective connectivity."""

DEFAULT_SIGNED_MODE = "net"
"""Default mode for signed connectivity: 'net' or 'blocks'."""

DEFAULT_TAU = 0.20
"""Default time constant for dynamics simulation."""

DEFAULT_EXCITABILITY = 1.0
"""Default excitability parameter for dynamics."""

DEFAULT_TIME_STEPS = 12
"""Default number of time steps for dynamics simulation."""

DEFAULT_DIVISIVE_NORM = False
"""Whether to apply divisive normalization in dynamics by default."""


# =============================================================================
# File Format Configuration
# =============================================================================

ADJACENCY_FILENAME = "N2_to_N3_adjacency.parquet"
"""Standard filename for N2->N3 adjacency output."""

N2_NODES_FILENAME = "N2_nodes.csv"
"""Standard filename for N2 node list."""

N3_NODES_FILENAME = "N3_nodes.csv"
"""Standard filename for N3 node list."""


__all__ = [
    # Dataset
    "DEFAULT_DATASET",
    "DEFAULT_MATERIALIZATION",
    "DEFAULT_CHUNK_SIZE",
    # Tables
    "DEFAULT_SYNAPSE_TABLE",
    "SYNAPSE_TABLE_CANDIDATES",
    "BASELINE_TABLE_CANDIDATES",
    # Outputs
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_N1_N2_OUTPUT",
    "DEFAULT_N2_N3_OUTPUT",
    "DEFAULT_GRAPH_OUTPUT",
    "DEFAULT_ANNOTATIONS_OUTPUT",
    "DEFAULT_ROOTS_PATH",
    # Thresholds
    "DEFAULT_MIN_SYNAPSES",
    "DEFAULT_FIRST_HOP_THRESHOLD",
    "DEFAULT_SECOND_HOP_THRESHOLD",
    # Labels
    "DEFAULT_N1_LABEL",
    "DEFAULT_N2_LABEL",
    "DEFAULT_N3_LABEL",
    # Mappings
    "DEFAULT_FUNCTIONAL_MAPPING",
    "DEFAULT_SUPER_CLASS_SYNONYMS",
    "DEFAULT_SUPER_CLASS_KEYWORDS",
    # CI
    "DEFAULT_K_HOPS",
    "DEFAULT_CI_CHUNK_SIZE",
    "DEFAULT_NORMALIZE_MODE",
    "DEFAULT_EFFECTIVE_THRESHOLD",
    "DEFAULT_SIGNED_MODE",
    "DEFAULT_TAU",
    "DEFAULT_EXCITABILITY",
    "DEFAULT_TIME_STEPS",
    "DEFAULT_DIVISIVE_NORM",
    # Filenames
    "ADJACENCY_FILENAME",
    "N2_NODES_FILENAME",
    "N3_NODES_FILENAME",
]
