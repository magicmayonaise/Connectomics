"""Legacy modules for backward compatibility.

This package contains earlier implementations that have been superseded by
newer modules in ``cx_connectome``, but are retained for backward compatibility
with existing tests and downstream code.

Modules
-------
adjacency_tracer
    Multi-hop synaptic adjacency tracing (``trace_next_layer``).
topology_analyzer
    SVG-based topology analysis without matplotlib dependency.
reporting
    Connectivity reporting utilities (``ConnectivityAnalyzer``).
"""

from cx_connectome.legacy.adjacency_tracer import (
    trace_next_layer,
    TraceResult,
    DEFAULT_MIN_SYNAPSES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SYNAPSE_TABLE,
    ADJACENCY_FILENAME,
    N2_NODES_FILENAME,
    N3_NODES_FILENAME,
)
from cx_connectome.legacy.topology_analyzer import (
    TopologyAnalyzer,
    _compute_correlations,
)
from cx_connectome.legacy.reporting import (
    AdjacencyRecord,
    ConnectivityAnalyzer,
    ConnectivityReport,
    OverlapRecord,
)

__all__ = [
    # adjacency_tracer
    "trace_next_layer",
    "TraceResult",
    "DEFAULT_MIN_SYNAPSES",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SYNAPSE_TABLE",
    "ADJACENCY_FILENAME",
    "N2_NODES_FILENAME",
    "N3_NODES_FILENAME",
    # topology_analyzer
    "TopologyAnalyzer",
    "_compute_correlations",
    # reporting
    "AdjacencyRecord",
    "ConnectivityAnalyzer",
    "ConnectivityReport",
    "OverlapRecord",
]
