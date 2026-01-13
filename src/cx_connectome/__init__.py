"""cx_connectome - Core library for connectomics analysis.

This package provides tools for analyzing neural connectivity from CAVE
materialization databases, with a focus on the Drosophila central complex (CX).

Key Modules
-----------
adjacency
    Build weighted connectivity tables from synaptic data.
annotations
    Fetch hierarchical cell-type annotations from CAVE.
cave_io
    Low-level CAVE API wrappers with error handling.
cx_network
    Build multi-hop connectivity graphs (N1→N2→N3).
functional_roles
    Annotate neurons with functional roles (Navigation, Sleep-Promoting).
motifs
    Discover canonical network motifs (feedback, lateral, recurrent, skip).
pipeline
    Orchestrate multi-step connectome analysis workflows.
topology
    Summarize connectivity with overlap/fan-in/fan-out metrics.
upstream
    Analyze N0→N1 upstream partners by super-class.

Subpackages
-----------
ci
    Connectivity Interpreter stack for effective connectivity,
    dynamics simulation, and receptive field analysis.
legacy
    Backward-compatible implementations for existing tests.

Example
-------
>>> from cx_connectome import build_connectivity_graph
>>> from cx_connectome.constants import DEFAULT_DATASET
>>> graph = build_connectivity_graph(
...     root_ids=[720575940614097912],
...     dataset=DEFAULT_DATASET,
... )
"""

from __future__ import annotations

__version__ = "0.1.0"

# Import key constants for convenience
from cx_connectome.constants import (
    DEFAULT_DATASET,
    DEFAULT_MATERIALIZATION,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SYNAPSE_TABLE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_MIN_SYNAPSES,
)

# Import main functions for top-level access
from cx_connectome.cx_network import build_connectivity_graph
from cx_connectome.functional_roles import annotate_functional_roles, infer_layer_membership

__all__ = [
    # Version
    "__version__",
    # Constants
    "DEFAULT_DATASET",
    "DEFAULT_MATERIALIZATION",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_SYNAPSE_TABLE",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_MIN_SYNAPSES",
    # Main functions
    "build_connectivity_graph",
    "annotate_functional_roles",
    "infer_layer_membership",
]
