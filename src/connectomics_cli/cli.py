"""Typer-powered command-line interface for connectomics analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .auth import CaveAuthenticationError, build_cave_client
from .config import DEFAULT_CHUNK_SIZE, DEFAULT_DATASET, DEFAULT_MATERIALIZATION, parse_timestamp
from .materialization import resolve_query_context
from .synapse_analysis import SynapseAnalyzer, SynapseSummary

app = typer.Typer(help="Reproducible FlyWire/FANC synapse analysis with CAVE materializations.")
console = Console()


@app.command("synapse-report")
def synapse_report(
    dataset: str = typer.Option(
        DEFAULT_DATASET,
        "--dataset",
        help="Datastack name passed to CAVEclient.",
        rich_help_panel="Data selection",
    ),
    materialization: Optional[int] = typer.Option(
        DEFAULT_MATERIALIZATION,
        "--materialization",
        help=(
            "Materialization ID to query. Ignored when --at-timestamp is provided. "
            "Default corresponds to the FlyWire 2022 public release."
        ),
        rich_help_panel="Data selection",
    ),
    at_timestamp: Optional[str] = typer.Option(
        None,
        "--at-timestamp",
        help="ISO8601 timestamp for a live query against the lineage graph.",
        rich_help_panel="Data selection",
    ),
    pre_root_ids: list[int] = typer.Option(
        ...,
        "--pre-root-ids",
        help="One or more presynaptic root IDs to include in the summary.",
        rich_help_panel="Query parameters",
    ),
    post_root_ids: Optional[list[int]] = typer.Option(
        None,
        "--post-root-ids",
        help="Optional postsynaptic root IDs to restrict the summary.",
        rich_help_panel="Query parameters",
    ),
    min_synapse_count: int = typer.Option(
        1,
        "--min-synapse-count",
        help="Minimum number of synapses required for a connection to be reported.",
        rich_help_panel="Query parameters",
    ),
    output_dir: Path = typer.Option(
        Path("out") / "synapse_report",
        "--output-dir",
        help="Directory where outputs will be written.",
        rich_help_panel="Output",
    ),
    synapse_table: Optional[str] = typer.Option(
        None,
        "--synapse-table",
        help="Override automatic discovery of the synapse annotation table.",
        rich_help_panel="Advanced",
    ),
    cell_type_table: Optional[str] = typer.Option(
        None,
        "--cell-type-table",
        help="Override automatic discovery of the cell type annotation table.",
        rich_help_panel="Advanced",
    ),
    chunk_size: int = typer.Option(
        DEFAULT_CHUNK_SIZE,
        "--chunk-size",
        help="Maximum size of each chunked materialization query.",
        rich_help_panel="Advanced",
    ),
) -> None:
    """Generate a connectivity summary for the requested root IDs."""

    if not pre_root_ids:
        console.print("[bold red]At least one --pre-root-ids value is required.[/bold red]")
        raise typer.Exit(code=1)
    if min_synapse_count < 1:
        console.print("[bold red]min-synapse-count must be >= 1[/bold red]")
        raise typer.Exit(code=1)
    if chunk_size < 1:
        console.print("[bold red]chunk-size must be >= 1[/bold red]")
        raise typer.Exit(code=1)
    try:
        client = build_cave_client(dataset)
    except CaveAuthenticationError as exc:
        console.print(f"[bold red]Authentication failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    timestamp = parse_timestamp(at_timestamp)
    materialization_arg = None if timestamp is not None else materialization
    context = resolve_query_context(
        client,
        dataset,
        materialization=materialization_arg,
        timestamp=timestamp,
    )

    analyzer = SynapseAnalyzer.auto_configure(
        client,
        context,
        synapse_table=synapse_table,
        cell_type_table=cell_type_table,
        chunk_size=chunk_size,
    )
    summary = analyzer.summarise(
        pre_root_ids=pre_root_ids,
        post_root_ids=post_root_ids,
        min_synapse_count=min_synapse_count,
    )
    summary.metadata["output_directory"] = str(output_dir)
    summary.export(output_dir)

    console.print("[bold green]Analysis complete.[/bold green]")
    console.print(f"Results written to [cyan]{output_dir}[/cyan]")
    _display_preview(summary)


def _display_preview(summary: SynapseSummary) -> None:
    table = Table(title="Top connections", show_lines=True)
    table.add_column("Pre root", justify="right")
    table.add_column("Post root", justify="right")
    table.add_column("Synapses", justify="right")
    table.add_column("Pre type")
    table.add_column("Post type")
    preview = summary.top_edges(10)
    for row in preview.itertuples(index=False):
        table.add_row(
            str(row.pre_root_id),
            str(row.post_root_id),
            str(row.synapse_count),
            row.pre_cell_type,
            row.post_cell_type,
        )
    console.print(table)


def main() -> None:
    """Entry point for ``python -m connectomics_cli``."""

    app()


if __name__ == "__main__":
    main()
