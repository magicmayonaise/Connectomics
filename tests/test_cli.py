from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from typer.testing import CliRunner

from connectomics_cli import cli


def test_cli_synapse_report(tmp_path: Path, fake_client, monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr(cli, "build_cave_client", lambda dataset: fake_client)
    result = runner.invoke(
        cli.app,
        [
            "synapse-report",
            "--dataset",
            "fake",
            "--pre-root-ids",
            "1001",
            "--pre-root-ids",
            "1002",
            "--output-dir",
            str(tmp_path),
        ],
    )
    if result.exception:
        raise result.exception
    assert result.exit_code == 0

    csv_path = tmp_path / "synapse_summary.csv"
    parquet_path = tmp_path / "synapse_summary.parquet"
    png_path = tmp_path / "synapse_counts.png"
    svg_path = tmp_path / "synapse_graph.svg"
    summary_path = tmp_path / "run_summary.json"

    assert csv_path.exists()
    assert parquet_path.exists()
    assert png_path.exists()
    assert svg_path.exists()
    assert summary_path.exists()

    expected = pd.read_csv(Path("tests/data/golden_synapse_summary.csv"))
    actual = pd.read_csv(csv_path)
    assert_frame_equal(actual, expected)

    summary_data = json.loads(summary_path.read_text(encoding="utf8"))
    assert summary_data["total_synapses"] == 4
    assert summary_data["row_count"] == 3
