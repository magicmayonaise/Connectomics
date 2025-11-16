"""CI helper command line interface."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

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


COMMANDS: List[str] = [
    "effective",
    "signed",
    "slice",
    "rfc",
    "simulate",
    "optimal",
    "metrics",
    "paths",
    "cx-atlas",
    "state-overlay",
]


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

    for command in COMMANDS:
        app.command(command)(command_factory(command))

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
        for name in COMMANDS:
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
        args = parser.parse_args()
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
