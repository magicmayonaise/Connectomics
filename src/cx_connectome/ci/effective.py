"""Effective connectivity calculations and CLI helpers.

The Connectivity Interpreter (CI) stack repeatedly reasons about *effective*
connectivity: the contribution that a source population makes to downstream
targets when paths up to ``k`` synaptic hops are considered.  Production uses a
heavily chunked sparse pipeline to keep the calculations tractable.  For the
unit tests in this kata we implement a lightweight but fully documented version
of that workflow.  The implementation focuses on three ideas:

``compute_effective_connectivity``
    Given a square adjacency matrix, compute the sparse matrix power for each
    requested hop distance.  Column normalisation mimics "post-total" scaling so
    that each column sums to one and can be interpreted as a share of the total
    inflow for the corresponding target neuron.

Chunk-wise processing
    Normalisation is performed in column chunks.  This matches the behaviour of
    the production pipeline and keeps the helper efficient on small matrices.

CLI integration
    ``python -m cx_connectome.ci.effective`` exposes a tiny command-line
    interface that accepts an adjacency matrix stored via
    :func:`scipy.sparse.save_npz` and emits effective connectivity tensors for
    every requested hop.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Protocol, Sequence

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, load_npz, save_npz

__all__ = ["EffectiveConnectivitySolver", "compute_effective_connectivity", "main"]


class EffectiveConnectivitySolver(Protocol):
    """Simple protocol describing an effective connectivity solver."""

    def compute(self, adjacency: csr_matrix, *, k_hops: Sequence[int]) -> Mapping[int, csr_matrix]:  # pragma: no cover - protocol behaviour exercised via compute_effective_connectivity
        """Return a mapping from hop distance to effective connectivity."""
        ...


def _validate_k_hops(k_hops: Sequence[int]) -> Sequence[int]:
    if not k_hops:
        raise ValueError("k_hops must contain at least one hop distance.")
    normalised = sorted(set(int(k) for k in k_hops))
    if normalised[0] <= 0:
        raise ValueError("Hop distances must be positive integers.")
    return normalised


def _column_normalise(matrix: csr_matrix, chunk_size_cols: int) -> csr_matrix:
    if chunk_size_cols <= 0:
        raise ValueError("chunk_size_cols must be a positive integer.")

    working = csc_matrix(matrix, dtype=float, copy=True)
    n_cols = working.shape[1]
    for start in range(0, n_cols, chunk_size_cols):
        end = min(start + chunk_size_cols, n_cols)
        for column in range(start, end):
            begin = working.indptr[column]
            finish = working.indptr[column + 1]
            column_sum = working.data[begin:finish].sum()
            if column_sum > 0.0:
                working.data[begin:finish] /= column_sum
    return working.tocsr()


def _apply_threshold(matrix: csr_matrix, threshold: float) -> csr_matrix:
    if threshold <= 0.0:
        return matrix
    coo = matrix.tocoo(copy=True)
    if coo.nnz == 0:
        return matrix
    mask = np.abs(coo.data) >= threshold
    if mask.all():
        return matrix
    filtered = csr_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=matrix.shape)
    filtered.eliminate_zeros()
    return filtered


def compute_effective_connectivity(
    adjacency: csr_matrix | np.ndarray,
    k_hops: Sequence[int],
    *,
    normalize: str = "post_total",
    chunk_size_cols: int = 4096,
    threshold_norm_input: float = 0.0,
) -> Dict[int, csr_matrix]:
    """Return effective connectivity tensors for the requested hops.

    Parameters
    ----------
    adjacency:
        Square adjacency matrix describing the structural connectivity graph.
    k_hops:
        Sequence of hop distances to evaluate.
    normalize:
        Either ``"post_total"`` (default) to column-normalise the matrices or
        ``"none"`` to keep the raw matrix powers.
    chunk_size_cols:
        Number of columns that should be processed together during
        normalisation.  The parameter mirrors the behaviour in the production
        CI stack.
    threshold_norm_input:
        Minimum column-normalised weight that should be retained.  Smaller
        entries are pruned which helps downstream summarisation stages.
    """

    matrix = csr_matrix(adjacency, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    hops = _validate_k_hops(k_hops)
    normalise = normalize.lower()
    if normalise not in {"post_total", "none"}:
        raise ValueError("normalize must be either 'post_total' or 'none'.")
    if threshold_norm_input < 0.0:
        raise ValueError("threshold_norm_input must be non-negative.")

    results: Dict[int, csr_matrix] = {}
    current_power = matrix.copy()
    current_hop = 1
    for hop in hops:
        while current_hop < hop:
            current_power = current_power @ matrix
            current_hop += 1

        power_copy = current_power.copy()
        if normalise == "post_total":
            power_copy = _column_normalise(power_copy, chunk_size_cols)
            power_copy = _apply_threshold(power_copy, threshold_norm_input)
        else:
            power_copy.eliminate_zeros()
        results[hop] = power_copy.tocsr()

    return results


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Compute CI effective connectivity tensors.")
    parser.add_argument("--adjacency", type=Path, required=True, help="Path to a CSR matrix saved via scipy.sparse.save_npz.")
    parser.add_argument("--k-hop", dest="k_hops", action="append", type=int, required=True, help="Hop distance to evaluate. Provide multiple times for multiple hops.")
    parser.add_argument("--normalize", choices=["post_total", "none"], default="post_total", help="Column normalisation strategy.")
    parser.add_argument("--chunk-size-cols", type=int, default=4096, help="Number of columns to process per chunk during normalisation.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum normalised weight to retain.")
    parser.add_argument("--output", type=Path, default=Path("./out/ci-effective"), help="Directory where tensors should be written.")
    return parser


def _write_summary(path: Path, args: Namespace, results: Mapping[int, csr_matrix]) -> None:
    summary: MutableMapping[str, object] = {
        "adjacency_path": str(args.adjacency),
        "hops": sorted(results.keys()),
        "normalize": args.normalize,
        "chunk_size_cols": args.chunk_size_cols,
        "threshold": args.threshold,
    }
    if results:
        sample = next(iter(results.values()))
        summary["shape"] = sample.shape
        summary["nnz_per_hop"] = {hop: int(matrix.nnz) for hop, matrix in results.items()}
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf8")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by ``python -m cx_connectome.ci.effective``."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    adjacency = load_npz(args.adjacency)
    results = compute_effective_connectivity(
        adjacency,
        args.k_hops,
        normalize=args.normalize,
        chunk_size_cols=args.chunk_size_cols,
        threshold_norm_input=args.threshold,
    )

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    for hop, matrix in results.items():
        save_npz(output_dir / f"effective_k{hop}.npz", matrix)

    _write_summary(output_dir / "effective_summary.json", args, results)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
