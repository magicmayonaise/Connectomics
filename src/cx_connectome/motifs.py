"""cx_connectome.motifs
=======================

Tools for discovering and visualising small network motifs in layered connectomes.

The module expects pre-computed adjacency tables describing chemical synapses
between three successive neuronal populations (``N1``, ``N2`` and ``N3``).  The
layer labels are agnostic to any specific species or circuit, making the
architecture general.  Functional polarity, however, comes from merging the
adjacency edges with neurotransmitter chemistry annotations provided by
:mod:`cx_connectome.chemistry`.  The polarity-aware motifs can then be related
back to computation in the same language used by sensory systems studies: for
instance, inhibitory lateral edges implement *lateral inhibition* to sharpen
contrast, while excitatory recurrent loops amplify or sustain activity.

The core workflow is as follows:

* Combine the three adjacency tables into a single edge list.
* Detect canonical motifs (feedback, lateral, recurrent same-type and skip
  connections).
* Enrich each edge with presynaptic neurotransmitter sign information.
* Export the resulting inventory to disk or simple network diagrams for figure
  supplements.

The functions in this file operate on :class:`pandas.DataFrame` instances and
return DataFrames so they can be chained with other analytical tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

try:  # Optional dependency; importing lazily avoids circular imports during tests.
    from . import chemistry
except ImportError:  # pragma: no cover - chemistry module may be provided later.
    chemistry = None  # type: ignore[assignment]

__all__ = [
    "NT_COLOR_MAP",
    "MotifEdge",
    "build_motif_inventory",
    "draw_motif_diagram",
    "export_motif_inventory",
    "explain_motif_polarity",
    "summarize_motif_inventory",
]

# Canonical motif names used throughout the module.
MOTIF_LABELS: Tuple[str, ...] = (
    "feedback",
    "lateral",
    "recurrent_same_type",
    "skip",
)

# Palette derived from colour-blind safe matplotlib defaults.
NT_COLOR_MAP: Dict[Optional[str], str] = {
    "excitatory": "#D55E00",
    "inhibitory": "#0072B2",
    "mixed": "#CC79A7",
    "modulatory": "#009E73",
    "unknown": "#999999",
    None: "#999999",
}

# Common neurotransmitter name normalisations for polarity inference.
_TRANSMITTER_POLARITY: Dict[str, str] = {
    "acetylcholine": "excitatory",
    "ach": "excitatory",
    "cholinergic": "excitatory",
    "glutamate": "excitatory",
    "glutamatergic": "excitatory",
    "gaba": "inhibitory",
    "gabaergic": "inhibitory",
    "glycine": "inhibitory",
    "dopamine": "modulatory",
    "octopamine": "modulatory",
    "serotonin": "modulatory",
    "histamine": "modulatory",
}

_SIGN_ALIASES: Dict[str, str] = {
    "exc": "excitatory",
    "e": "excitatory",
    "inhib": "inhibitory",
    "inh": "inhibitory",
    "i": "inhibitory",
    "mod": "modulatory",
    "unk": "unknown",
}


@dataclass(frozen=True)
class MotifEdge:
    """A single directed edge participating in one or more motifs.

    The dataclass is primarily returned for interactive exploration, while the
    public APIs expose :class:`pandas.DataFrame` views that are convenient for
    subsequent grouping and exporting.
    """

    motif: str
    source: str
    target: str
    source_layer: Optional[str]
    target_layer: Optional[str]
    source_type: Optional[str]
    target_type: Optional[str]
    weight: Optional[float]
    nt_sign: Optional[str]
    nt_name: Optional[str]


def build_motif_inventory(
    adjacency_tables: Mapping[Tuple[str, str], pd.DataFrame],
    chemistry_lookup: Optional[Mapping[str, Any]] = None,
    *,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: Optional[str] = "weight",
    source_layer_col: str = "source_layer",
    target_layer_col: str = "target_layer",
    source_type_col: Optional[str] = "source_type",
    target_type_col: Optional[str] = "target_type",
) -> pd.DataFrame:
    """Construct a motif inventory from layered adjacency tables.

    Parameters
    ----------
    adjacency_tables:
        Mapping from ``(source_layer, target_layer)`` tuples to adjacency tables.
        Each table must contain at least the ``source`` and ``target`` columns
        (configurable via ``source_col`` / ``target_col``).  The function copies
        the data to avoid mutating the caller's DataFrames.
    chemistry_lookup:
        Optional mapping from presynaptic neuron identifiers to neurotransmitter
        annotations.  Values can be simple sign strings (``"excitatory"``,
        ``"inhibitory"`` …) or dictionaries containing fields such as
        ``"nt"``/``"neurotransmitter"`` and ``"sign"``.  When ``None`` the
        function attempts to call :func:`cx_connectome.chemistry.load_sign_lookup`
        if it exists; otherwise the sign is marked as ``"unknown"``.
    source_col, target_col, weight_col:
        Column names that describe the presynaptic neuron, postsynaptic neuron
        and optional edge weight (number of synapses or connection probability).
    source_layer_col, target_layer_col:
        Column names that carry layer annotations.  Missing columns are
        automatically populated from the mapping key.
    source_type_col, target_type_col:
        Optional column names encoding neuron type or cell class information.
        They are required for detecting ``recurrent_same_type`` motifs.

    Returns
    -------
    pandas.DataFrame
        A tidy table with one row per edge *per motif* containing the original
        edge attributes plus ``motif``, ``nt_sign`` and ``nt_color`` columns.
    """

    if not adjacency_tables:
        raise ValueError("No adjacency tables provided; expected three layer pairings.")

    normalised_frames: List[pd.DataFrame] = []
    for (src_layer, tgt_layer), table in adjacency_tables.items():
        frame = table.copy()
        if source_col not in frame.columns or target_col not in frame.columns:
            raise KeyError(
                "Adjacency tables must include both presynaptic and postsynaptic columns "
                f"({source_col!r}, {target_col!r})."
            )

        # Populate layer annotations when they are missing or incomplete.
        if source_layer_col not in frame.columns:
            frame[source_layer_col] = src_layer
        else:
            frame[source_layer_col] = frame[source_layer_col].fillna(src_layer)

        if target_layer_col not in frame.columns:
            frame[target_layer_col] = tgt_layer
        else:
            frame[target_layer_col] = frame[target_layer_col].fillna(tgt_layer)

        if source_type_col and source_type_col not in frame.columns:
            frame[source_type_col] = None
        if target_type_col and target_type_col not in frame.columns:
            frame[target_type_col] = None

        if weight_col and weight_col not in frame.columns:
            frame[weight_col] = 1

        normalised_frames.append(frame)

    edges = pd.concat(normalised_frames, ignore_index=True, sort=False)

    motif_masks = _compute_motif_masks(
        edges,
        source_layer_col=source_layer_col,
        target_layer_col=target_layer_col,
        source_type_col=source_type_col,
        target_type_col=target_type_col,
    )

    motif_lists = _assign_motif_labels(edges.index, motif_masks)
    edges = edges.copy()
    edges["motifs"] = motif_lists

    inventory = edges.explode("motifs")
    if inventory.empty:
        # No motifs detected; return a consistent, empty DataFrame with expected columns.
        return pd.DataFrame(
            columns=list(edges.columns) + ["motif", "nt_sign", "nt_color", "nt_name"],
        )

    inventory = inventory[inventory["motifs"].notna()].copy()
    inventory.rename(columns={"motifs": "motif"}, inplace=True)

    sign_lookup = _load_sign_lookup(chemistry_lookup)
    inventory["nt_name"], inventory["nt_sign"] = zip(
        *[
            _resolve_nt_annotation(sign_lookup.get(neuron))
            for neuron in inventory[source_col]
        ]
    )
    inventory["nt_sign"].fillna("unknown", inplace=True)
    inventory["nt_color"] = inventory["nt_sign"].map(
        lambda sign: NT_COLOR_MAP.get(sign, NT_COLOR_MAP["unknown"])
    )

    # Re-order columns for readability.
    preferred_order = [
        "motif",
        source_col,
        target_col,
        source_layer_col,
        target_layer_col,
    ]
    optional_columns = [
        col
        for col in (
            source_type_col,
            target_type_col,
            weight_col,
            "nt_name",
            "nt_sign",
            "nt_color",
        )
        if col and col in inventory.columns
    ]
    other_columns = [
        col
        for col in inventory.columns
        if col not in preferred_order and col not in optional_columns
    ]
    return inventory[preferred_order + optional_columns + other_columns]


def summarize_motif_inventory(
    inventory: pd.DataFrame,
    *,
    weight_col: Optional[str] = "weight",
) -> pd.DataFrame:
    """Aggregate motif counts, optionally weighting by synapse number.

    The summary groups by motif label and presynaptic neurotransmitter sign so
    that downstream users can see, for example, how many inhibitory lateral
    edges exist in the dataset.
    """

    if "motif" not in inventory.columns:
        raise KeyError("Input inventory is missing the 'motif' column.")

    working = inventory.copy()
    if "nt_sign" not in working.columns:
        working["nt_sign"] = "unknown"
    else:
        working["nt_sign"].fillna("unknown", inplace=True)

    group_keys = ["motif", "nt_sign"]

    if weight_col and weight_col in working.columns:
        summary = (
            working.groupby(group_keys, dropna=False)[weight_col]
            .sum()
            .reset_index()
            .rename(columns={weight_col: "total_weight"})
        )
    else:
        summary = (
            working.groupby(group_keys, dropna=False)
            .size()
            .reset_index(name="edge_count")
        )

    return summary.sort_values(group_keys).reset_index(drop=True)


def export_motif_inventory(
    inventory: pd.DataFrame,
    destination: Path | str,
    *,
    format: Optional[str] = None,
    orient: str = "records",
    **to_kwargs: Any,
) -> None:
    """Persist the motif inventory to disk.

    Parameters
    ----------
    inventory:
        DataFrame produced by :func:`build_motif_inventory`.
    destination:
        File path to write to.  The suffix is used to infer the format when
        ``format`` is ``None``.
    format:
        Either ``"csv"`` or ``"json"``.  If omitted the suffix of
        ``destination`` is inspected.
    orient:
        JSON orientation forwarded to :meth:`pandas.DataFrame.to_json` when
        exporting JSON output.
    to_kwargs:
        Additional keyword arguments forwarded to the relevant pandas export
        function.
    """

    path = Path(destination)
    fmt = (format or path.suffix.lstrip(".")).lower()

    if fmt == "csv":
        inventory.to_csv(path, index=False, **to_kwargs)
    elif fmt == "json":
        inventory.to_json(path, orient=orient, **to_kwargs)
    else:
        raise ValueError(
            "Unsupported export format. Use 'csv' or 'json', or specify a path with the appropriate suffix."
        )


def draw_motif_diagram(
    inventory: pd.DataFrame,
    motif: str,
    output_path: Path | str,
    *,
    layout: str = "circular",
    node_size: int = 2200,
    font_size: int = 12,
    width_scale: float = 1.0,
    weight_col: Optional[str] = "weight",
) -> None:
    """Render a simple network diagram for a motif using NetworkX.

    The function creates a directed graph for the selected motif and draws it
    with a circular layout by default.  Edge colours reflect the presynaptic
    neurotransmitter sign (excitatory/inhibitory/etc.).  The diagram is saved to
    ``output_path`` using matplotlib.
    """

    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - depends on optional plotting stack.
        raise RuntimeError(
            "draw_motif_diagram requires both matplotlib and networkx to be installed."
        ) from exc

    filtered = inventory[inventory["motif"] == motif]
    if filtered.empty:
        raise ValueError(f"No edges labelled with motif {motif!r} were found.")

    g = nx.DiGraph()

    for row in filtered.to_dict("records"):
        src = row.get("source", row.get("pre", row.get("presynaptic")))
        tgt = row.get("target", row.get("post", row.get("postsynaptic")))
        if src is None or tgt is None:
            raise KeyError(
                "The inventory must contain either 'source'/'target' columns or 'pre'/'post' aliases."
            )
        sign = row.get("nt_sign", "unknown")
        weight = float(row.get(weight_col, 1.0)) if weight_col and weight_col in row else 1.0
        g.add_edge(src, tgt, nt_sign=sign, weight=weight)

    if layout == "circular":
        positions = nx.circular_layout(g)
    elif layout == "spring":
        positions = nx.spring_layout(g, seed=0)
    else:
        raise ValueError("Unsupported layout. Choose 'circular' or 'spring'.")

    edge_colors = [NT_COLOR_MAP.get(data.get("nt_sign"), NT_COLOR_MAP["unknown"]) for _, _, data in g.edges(data=True)]
    edge_widths = [max(data.get("weight", 1.0) * width_scale, 0.2) for _, _, data in g.edges(data=True)]

    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx(
        g,
        pos=positions,
        ax=ax,
        with_labels=True,
        node_size=node_size,
        font_size=font_size,
        arrows=True,
        edge_color=edge_colors,
        width=edge_widths,
        node_color="#f0f0f0",
    )
    ax.set_axis_off()
    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def explain_motif_polarity() -> str:
    """Return a narrative that links motif polarity to computational roles.

    The explanation mirrors discussions that appear in sensory neuroscience
    papers, connecting canonical circuit motifs to their algorithmic function.
    """

    return (
        "Feedback motifs (N2→N1 or N3→N2) inherit their computational effect from the "
        "polarity of the presynaptic neuron. Excitatory feedback loops can "
        "amplify or stabilise activity, whereas inhibitory feedback mirrors the "
        "predictive coding motifs described in olfactory and visual systems. "
        "Lateral edges (N2→N2, N3→N3) implement local gain control: when the "
        "presynaptic partner is inhibitory the circuit realises lateral "
        "inhibition that sharpens spatial or spectral tuning; when excitatory it "
        "supports cooperative pooling observed in contrast gain control. "
        "Recurrent same-type connections flag ensembles of identical neurons that "
        "often sustain persistent activity—excitatory populations behave like "
        "reverberating amplifiers, while inhibitory populations can realise "
        "winner-take-all competition. Finally, skip connections (N1→N3) show how "
        "feedforward information bypasses intermediate processing stages; their "
        "polarity reveals whether the bypass acts as a fast excitatory drive or "
        "a long-range inhibitory shunt.  By tabulating motif counts by "
        "neurotransmitter sign, the same circuit description can be interpreted "
        "in the algorithmic vocabulary used throughout sensory systems papers."
    )


def _compute_motif_masks(
    edges: pd.DataFrame,
    *,
    source_layer_col: str,
    target_layer_col: str,
    source_type_col: Optional[str],
    target_type_col: Optional[str],
) -> Dict[str, pd.Series]:
    """Compute boolean masks describing motif membership for each edge."""

    src_layer = _normalise_layers(edges[source_layer_col]) if source_layer_col in edges.columns else pd.Series(False, index=edges.index)
    tgt_layer = _normalise_layers(edges[target_layer_col]) if target_layer_col in edges.columns else pd.Series(False, index=edges.index)

    feedback_mask = (
        (src_layer == "N2") & (tgt_layer == "N1")
    ) | (
        (src_layer == "N3") & (tgt_layer == "N2")
    )

    lateral_mask = (src_layer == tgt_layer) & src_layer.isin({"N2", "N3"})

    if source_type_col and target_type_col and source_type_col in edges.columns and target_type_col in edges.columns:
        src_type = edges[source_type_col]
        tgt_type = edges[target_type_col]
        recurrent_mask = src_type.notna() & tgt_type.notna() & (src_type.astype(str) == tgt_type.astype(str))
    else:
        recurrent_mask = pd.Series(False, index=edges.index)

    skip_mask = (src_layer == "N1") & (tgt_layer == "N3")

    return {
        "feedback": feedback_mask,
        "lateral": lateral_mask,
        "recurrent_same_type": recurrent_mask,
        "skip": skip_mask,
    }


def _assign_motif_labels(index: pd.Index, masks: Mapping[str, pd.Series]) -> List[List[str]]:
    """Convert motif boolean masks into per-edge label lists."""

    labels: List[List[str]] = []
    for i in index:
        edge_labels = [name for name, mask in masks.items() if bool(mask.get(i, False))]
        labels.append(edge_labels)
    return labels


def _load_sign_lookup(user_lookup: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Load neurotransmitter annotations from the provided lookup or chemistry module."""

    if user_lookup is not None:
        return dict(user_lookup)

    if chemistry is None:
        return {}

    for attr_name in ("load_sign_lookup", "load_nt_sign_lookup", "get_presynaptic_sign_lookup"):
        loader = getattr(chemistry, attr_name, None)
        if callable(loader):
            lookup = loader()
            if lookup is None:
                continue
            if not isinstance(lookup, Mapping):
                raise TypeError(
                    f"chemistry.{attr_name}() must return a mapping, got {type(lookup)!r}."
                )
            return dict(lookup)

    default_lookup = getattr(chemistry, "PRESYNAPTIC_SIGNS", None)
    if isinstance(default_lookup, Mapping):
        return dict(default_lookup)

    return {}


def _resolve_nt_annotation(raw_value: Any) -> Tuple[Optional[str], Optional[str]]:
    """Extract neurotransmitter name and sign from heterogeneous annotations."""

    if raw_value is None:
        return None, None

    if isinstance(raw_value, str):
        normalised = raw_value.strip().lower()
        sign = _SIGN_ALIASES.get(normalised, normalised)
        if sign in NT_COLOR_MAP:
            return None, sign
        sign_from_transmitter = _TRANSMITTER_POLARITY.get(normalised)
        if sign_from_transmitter:
            return raw_value, sign_from_transmitter
        return raw_value, "unknown"

    if isinstance(raw_value, Mapping):
        nt_name = raw_value.get("nt") or raw_value.get("neurotransmitter")
        sign = raw_value.get("sign") or raw_value.get("nt_sign")
        if isinstance(sign, str):
            sign = _SIGN_ALIASES.get(sign.lower(), sign.lower())
        if sign in NT_COLOR_MAP:
            return nt_name, sign
        if isinstance(nt_name, str):
            inferred = _TRANSMITTER_POLARITY.get(nt_name.strip().lower())
            if inferred:
                return nt_name, inferred
        if sign:
            return nt_name, str(sign)
        return nt_name, "unknown"

    return None, "unknown"


def _normalise_layers(series: pd.Series) -> pd.Series:
    """Normalise layer labels to uppercase strings (N1/N2/N3)."""

    normalised = series.astype(str).str.upper()
    normalised = normalised.where(~series.isna(), None)
    return normalised
