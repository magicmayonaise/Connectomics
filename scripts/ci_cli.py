"""Command line interface for connectome CI helpers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from cx_connectome.ci.state_overlay import format_flow_table, run_state_overlay


def _state_overlay_command(args: argparse.Namespace) -> None:
    result = run_state_overlay(
        scope=args.scope,
        seeds_csv=Path(args.seeds),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        figure_path=Path(args.figure) if args.figure else None,
        metrics_path=Path(args.metrics) if args.metrics else None,
    )

    print(format_flow_table(result.flows))
    print()
    print(f"Metrics written to: {result.metrics_path}")
    print(f"Summary circuit saved to: {result.figure_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Connectome CI helper CLI")
    subparsers = parser.add_subparsers(dest="command")

    state_parser = subparsers.add_parser(
        "state-overlay",
        help="Generate a CX state overlay summary from curated seed sets.",
    )
    state_parser.add_argument(
        "--scope",
        default="cx",
        choices=["cx"],
        help="Analysis scope. Only 'cx' is currently supported.",
    )
    state_parser.add_argument(
        "--seeds",
        required=True,
        help="Path to the state seed CSV (e.g. data/state_seeds.csv)",
    )
    state_parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where artefacts should be written (default: output).",
    )
    state_parser.add_argument(
        "--figure",
        help="Optional explicit output path for the rendered summary figure.",
    )
    state_parser.add_argument(
        "--metrics",
        help="Optional explicit output path for the computed metrics CSV.",
    )
    state_parser.set_defaults(func=_state_overlay_command)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
