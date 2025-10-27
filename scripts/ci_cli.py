#!/usr/bin/env python3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

"""Command line entry point for Connectome Integration utilities."""


import argparse
from typing import Sequence

from cx_connectome.ci.cx_integration import (
    DEFAULT_K_VALUES,
    DEFAULT_SCOPE,
    build_cx_signed_effective_connectivity_atlas,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Connectome Integration helper commands")
    subparsers = parser.add_subparsers(dest="command", required=True)

    atlas_parser = subparsers.add_parser(
        "cx-atlas",
        help="Generate the CX signed effective-connectivity atlas",
    )
    atlas_parser.add_argument("--scope", default=DEFAULT_SCOPE, help="Dataset scope identifier")
    atlas_parser.add_argument(
        "--out",
        required=True,
        help="Directory where atlas artefacts will be written",
    )
    atlas_parser.add_argument(
        "--primitives",
        default=None,
        help="Optional path (file or directory) for signed effective-connectivity primitives",
    )
    atlas_parser.add_argument(
        "--baseline",
        default=None,
        help="Optional path to baseline annotations JSON file",
    )
    atlas_parser.add_argument(
        "--k-max",
        type=int,
        default=max(DEFAULT_K_VALUES),
        help="Maximum k-order to include (inclusive)",
    )
    atlas_parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Override the seed category order (defaults to visual ascending central drive_state)",
    )
    return parser


def _resolve_k_values(k_max: int | None) -> Sequence[int]:
    if not k_max:
        return DEFAULT_K_VALUES
    k_max = max(int(k_max), 1)
    return tuple(range(1, k_max + 1))


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "cx-atlas":
        k_values = _resolve_k_values(args.k_max)
        categories = tuple(args.categories) if args.categories else None
        result = build_cx_signed_effective_connectivity_atlas(
            scope=args.scope,
            out_dir=Path(args.out),
            primitives=args.primitives,
            baseline=args.baseline,
            k_values=k_values,
            categories=categories,
        )
        print(f"CX atlas written to {result.output_paths['base_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
