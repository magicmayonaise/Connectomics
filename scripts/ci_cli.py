#!/usr/bin/env python3
"""Command-line helpers for connectivity inference workflows."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cx_connectome.ci import compute_suite  # noqa: E402


def _load_sparse_matrix(path: Path) -> sparse.csr_matrix:
    """Load a sparse CSR matrix saved via ``scipy.sparse.save_npz``."""

    with np.load(path) as data:
        if {"data", "indices", "indptr", "shape"}.issubset(data.files):
            matrix = sparse.csr_matrix(
                (data["data"], data["indices"], data["indptr"]), shape=tuple(data["shape"])
            )
        else:
            matrix = sparse.csr_matrix(data["arr_0"])
    return matrix


def _find_default_adjacency(materialization: str, scope: str) -> Path | None:
    """Search common locations for a stored adjacency matrix."""

    candidates = [
        ROOT / "out" / "materializations" / materialization / f"{scope}.npz",
        ROOT / "out" / "materializations" / f"{materialization}_{scope}.npz",
        ROOT / "out" / "ci" / f"adjacency_{scope}_{materialization}.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _write_top_inflow_csv(path: Path, matrix: sparse.csr_matrix, top_n: int) -> None:
    """Persist the highest inflow columns for quick inspection."""

    column_totals = np.asarray(matrix.sum(axis=0)).ravel()
    order = np.argsort(column_totals)[::-1][:top_n]

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["post", "total_inflow"])
        for column in order:
            writer.writerow([int(column), float(column_totals[column])])


def _parse_dtype(value: str) -> np.dtype:
    try:
        return np.dtype(value)
    except TypeError as exc:  # pragma: no cover - defensive branch
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _run_effective(args: argparse.Namespace) -> None:
    if args.adjacency is not None:
        adjacency_path = Path(args.adjacency)
        if not adjacency_path.exists():
            raise FileNotFoundError(f"Adjacency matrix not found: {adjacency_path}")
    else:
        adjacency_path = _find_default_adjacency(args.materialization, args.scope)
        if adjacency_path is None:
            raise FileNotFoundError(
                "Could not locate adjacency matrix. Use --adjacency to supply an explicit path."
            )

    adj = _load_sparse_matrix(adjacency_path)
    suite = compute_suite(
        adj,
        args.k,
        chunk_size_cols=args.chunk_size_cols,
        normalize=None if args.normalize == "none" else args.normalize,
        eps=args.eps,
        dtype=args.dtype,
    )

    out_dir = ROOT / "out" / "ci"
    out_dir.mkdir(parents=True, exist_ok=True)

    for k, matrix in suite.items():
        sparse.save_npz(out_dir / f"effective_k{k}_{args.scope}.npz", matrix)
        _write_top_inflow_csv(out_dir / f"effective_k{k}_{args.scope}_top_inflow.csv", matrix, args.top_n)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    effective = subparsers.add_parser("effective", help="Compute effective connectivity powers")
    effective.add_argument("--scope", required=True, help="Materialisation scope identifier")
    effective.add_argument("--k", required=True, nargs="+", type=int, help="Powers to compute")
    effective.add_argument("--materialization", required=True, help="Materialisation ID")
    effective.add_argument(
        "--chunk-size-cols",
        default=4096,
        type=int,
        help="Number of columns to materialise in each dense chunk",
    )
    effective.add_argument(
        "--normalize",
        choices=["post_total", "none"],
        default="post_total",
        help="Column normalisation mode",
    )
    effective.add_argument("--eps", type=float, default=1e-12, help="Drop values below this threshold")
    effective.add_argument("--dtype", type=_parse_dtype, default=np.float32, help="Intermediate dtype")
    effective.add_argument(
        "--adjacency",
        help="Explicit path to the adjacency matrix npz (defaults to common materialisation locations)",
    )
    effective.add_argument("--top-n", type=int, default=50, help="Rows to keep in the top-inflow CSV")
    effective.set_defaults(func=_run_effective)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
