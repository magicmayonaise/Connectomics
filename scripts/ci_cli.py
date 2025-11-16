"""CI helper command line interface."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from cx_connectome.ci import paths as ci_paths

try:
    import typer
except ModuleNotFoundError:  # pragma: no cover - fallback to argparse
    typer = None  # type: ignore


@dataclass(frozen=True)
class RunContext:
    command: str
    materialization: Optional[str]
    at_timestamp: Optional[str]
    scope: str
    outdir: Path

    def __post_init__(self) -> None:
        if (self.materialization is None) == (self.at_timestamp is None):
            raise ValueError(
                "Exactly one of 'materialization' or 'at_timestamp' must be provided."
            )
        if self.scope not in {"whole", "cx"}:
            raise ValueError("Scope must be either 'whole' or 'cx'.")

    def ledger(self) -> str:
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        source = (
            f"materialization:{self.materialization}"
            if self.materialization is not None
            else f"timestamp:{self.at_timestamp}"
        )
        return "\n".join(
            [
                "ci-run",
                f"  command={self.command}",
                f"  source={source}",
                f"  scope={self.scope}",
                f"  outdir={self.outdir}",
                f"  issued-at={timestamp}",
            ]
        )


LEDGER_COMMANDS: List[str] = [
    "effective",
    "signed",
    "slice",
    "rfc",
    "simulate",
    "optimal",
    "metrics",
    "cx-atlas",
    "state-overlay",
]

PATHS_OUTPUT_DEFAULT = Path("./out/ci-paths")


def run_paths_workflow(
    *,
    adjacency: Path,
    sources: Sequence[int],
    targets: Sequence[int],
    max_hops: int,
    source_column: Optional[str],
    target_column: Optional[str],
    output: Path,
) -> None:
    """Execute the path extraction workflow used by ``ci-cli paths``."""

    if not sources:
        raise ValueError("At least one --source value must be provided.")
    if not targets:
        raise ValueError("At least one --target value must be provided.")
    if max_hops <= 0:
        raise ValueError("max-hops must be a positive integer.")

    adjacency_table = ci_paths.load_adjacency_table(adjacency)
    result = ci_paths.extract_paths(
        adjacency_table,
        sources=list(sources),
        targets=list(targets),
        max_hops=max_hops,
        source_column=source_column,
        target_column=target_column,
    )
    metadata = {
        "adjacency_path": str(adjacency.resolve()),
        "sources": list(sources),
        "targets": list(targets),
        "max_hops": max_hops,
    }
    ci_paths.write_path_outputs(result, output, metadata=metadata)


def _handle_command(
    *,
    name: str,
    materialization: Optional[str],
    at_timestamp: Optional[str],
    scope: str,
    outdir: Path,
) -> None:
    try:
        context = RunContext(
            command=name,
            materialization=materialization,
            at_timestamp=at_timestamp,
            scope=scope,
            outdir=outdir,
        )
    except ValueError as err:
        raise RuntimeError(str(err))
    print(context.ledger())


if typer is not None:  # pragma: no branch
    app = typer.Typer(help="Utilities for running CI-related workflows.")

    def validate_scope(value: str) -> str:
        normalized = value.lower()
        if normalized not in {"whole", "cx"}:
            raise typer.BadParameter("Scope must be either 'whole' or 'cx'.")
        return normalized

    def command_factory(name: str) -> Callable[[], None]:
        def _command(
            materialization: Optional[str] = typer.Option(
                None,
                "--materialization",
                "-materialization",
                help="Materialization identifier to execute against.",
                show_default=False,
            ),
            at_timestamp: Optional[str] = typer.Option(
                None,
                "--at-timestamp",
                "-at-timestamp",
                help="Source data timestamp (RFC3339) to execute against.",
                show_default=False,
            ),
            scope: str = typer.Option(
                "whole",
                "--scope",
                "-scope",
                callback=validate_scope,
                help="Execution scope (whole or cx).",
            ),
            outdir: Path = typer.Option(
                Path("./ci-artifacts"),
                "--outdir",
                "-outdir",
                help="Directory where outputs should be written.",
            ),
        ) -> None:
            try:
                _handle_command(
                    name=name,
                    materialization=materialization,
                    at_timestamp=at_timestamp,
                    scope=scope,
                    outdir=outdir,
                )
            except RuntimeError as err:
                raise typer.BadParameter(str(err))

        _command.__name__ = name.replace("-", "_")
        _command.__doc__ = f"Run the '{name}' CI workflow."
        return _command

    for command in LEDGER_COMMANDS:
        app.command(command)(command_factory(command))

    @app.command("paths")
    def paths_command(
        adjacency: Path = typer.Option(
            ..., "--adjacency", help="Path to a CSV or Parquet adjacency table."
        ),
        sources: List[int] = typer.Option(
            ..., "--source", "-s", help="Source root ID; repeat for multiple sources."
        ),
        targets: List[int] = typer.Option(
            ..., "--target", "-t", help="Target root ID; repeat for multiple targets."
        ),
        max_hops: int = typer.Option(3, "--max-hops", help="Maximum hop distance to explore."),
        source_column: Optional[str] = typer.Option(
            None, "--source-column", help="Override automatic source column detection."
        ),
        target_column: Optional[str] = typer.Option(
            None, "--target-column", help="Override automatic target column detection."
        ),
        output: Path = typer.Option(
            PATHS_OUTPUT_DEFAULT,
            "--output",
            help="Directory where path summaries should be written.",
        ),
    ) -> None:
        try:
            run_paths_workflow(
                adjacency=adjacency,
                sources=sources,
                targets=targets,
                max_hops=max_hops,
                source_column=source_column,
                target_column=target_column,
                output=output,
            )
        except ValueError as err:
            raise typer.BadParameter(str(err))

    def main() -> None:
        app()

else:
    import argparse

    class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
        pass

    def main() -> None:
        parser = argparse.ArgumentParser(
            prog="ci-cli",
            description="Utilities for running CI-related workflows.",
            formatter_class=_HelpFormatter,
        )
        subparsers = parser.add_subparsers(dest="command", required=True)
        for name in LEDGER_COMMANDS:
            sub = subparsers.add_parser(
                name,
                help=f"Run the '{name}' CI workflow.",
                formatter_class=_HelpFormatter,
            )
            sub.add_argument(
                "--materialization",
                "-materialization",
                dest="materialization",
                help="Materialization identifier to execute against.",
            )
            sub.add_argument(
                "--at-timestamp",
                "-at-timestamp",
                dest="at_timestamp",
                help="Source data timestamp (RFC3339) to execute against.",
            )
            sub.add_argument(
                "--scope",
                "-scope",
                default="whole",
                choices=["whole", "cx"],
                help="Execution scope (whole or cx).",
            )
            sub.add_argument(
                "--outdir",
                "-outdir",
                default="./ci-artifacts",
                type=Path,
                help="Directory where outputs should be written.",
            )
        paths_parser = subparsers.add_parser(
            "paths",
            help="Enumerate directed connectome paths.",
            formatter_class=_HelpFormatter,
        )
        paths_parser.add_argument(
            "--adjacency",
            required=True,
            type=Path,
            help="Path to a CSV or Parquet adjacency table.",
        )
        paths_parser.add_argument(
            "--source",
            dest="sources",
            action="append",
            required=True,
            type=int,
            help="Source root ID; repeat for multiple sources.",
        )
        paths_parser.add_argument(
            "--target",
            dest="targets",
            action="append",
            required=True,
            type=int,
            help="Target root ID; repeat for multiple targets.",
        )
        paths_parser.add_argument(
            "--max-hops",
            dest="max_hops",
            default=3,
            type=int,
            help="Maximum hop distance to explore.",
        )
        paths_parser.add_argument(
            "--source-column",
            dest="source_column",
            help="Override automatic source column detection.",
        )
        paths_parser.add_argument(
            "--target-column",
            dest="target_column",
            help="Override automatic target column detection.",
        )
        paths_parser.add_argument(
            "--output",
            dest="output",
            type=Path,
            default=PATHS_OUTPUT_DEFAULT,
            help="Directory where path summaries should be written.",
        )
        args = parser.parse_args()
        if args.command == "paths":
            try:
                run_paths_workflow(
                    adjacency=args.adjacency,
                    sources=args.sources,
                    targets=args.targets,
                    max_hops=args.max_hops,
                    source_column=args.source_column,
                    target_column=args.target_column,
                    output=args.output,
                )
            except ValueError as err:
                parser.error(str(err))
            return
        materialization = getattr(args, "materialization")
        at_timestamp = getattr(args, "at_timestamp")
        try:
            _handle_command(
                name=args.command,
                materialization=materialization,
                at_timestamp=at_timestamp,
                scope=args.scope,
                outdir=args.outdir,
            )
        except RuntimeError as err:
            parser.error(str(err))


if __name__ == "__main__":
    main()
