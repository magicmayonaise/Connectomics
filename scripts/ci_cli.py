#!/usr/bin/env python3
from __future__ import annotations
import importlib
from pathlib import Path
import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)

def _call(modname: str, config: Path | None):
    try:
        mod = importlib.import_module(modname)
    except Exception as e:
        typer.secho(f"[import] {modname}: {e}", fg="red")
        raise typer.Exit(1)

    for fn_name in ("cli", "main", "run", "compute_suite"):
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            try:
                # prefer keyword style
                return fn(config=config)
            except TypeError:
                try:
                    # positional style
                    return fn(str(config) if config else None)
                except TypeError:
                    # no-arg style
                    return fn()
    typer.secho(f"No callable entrypoint found in {modname}", fg="red")
    raise typer.Exit(1)

@app.command("effective")
def cmd_effective(
    config: Path = typer.Option(None, "--config", exists=True, readable=True, help="YAML/JSON config")
):
    """Column-chunked effective connectivity tools."""
    _call("cx_connectome.ci.effective", config)

@app.command("signed")
def cmd_signed(
    config: Path = typer.Option(None, "--config", exists=True, readable=True)
):
    """Signed connectivity blocks / masks."""
    _call("cx_connectome.ci.signed", config)

@app.command("slice")
def cmd_slice(
    config: Path = typer.Option(None, "--config", exists=True, readable=True)
):
    """Path extraction, slicing, BFS utilities."""
    _call("cx_connectome.ci.paths", config)

@app.command("dynamics")
def cmd_dynamics(
    config: Path = typer.Option(None, "--config", exists=True, readable=True)
):
    """Lightweight simulator / dynamics (imports torch lazily inside module)."""
    _call("cx_connectome.ci.dynamics", config)

@app.command("optimize")
def cmd_optimize(
    config: Path = typer.Option(None, "--config", exists=True, readable=True)
):
    """Activation-maximization / optimization routines."""
    _call("cx_connectome.ci.optimize", config)

@app.command("metrics")
def cmd_metrics(
    config: Path = typer.Option(None, "--config", exists=True, readable=True)
):
    """Interpreter metrics / summaries."""
    _call("cx_connectome.ci.metrics", config)

@app.command("state-overlay")
def cmd_state_overlay(
    config: Path = typer.Option(None, "--config", exists=True, readable=True)
):
    """Overlay state/activity onto CX structures."""
    _call("cx_connectome.ci.state_overlay", config)

if __name__ == "__main__":
    app()
