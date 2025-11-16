from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cx_connectome.ci import paths


def _load_adjacency() -> pd.DataFrame:
    return pd.read_csv(Path("tests/data/path_adjacency.csv"))


def test_extract_paths_returns_simple_paths() -> None:
    adjacency = _load_adjacency()
    result = paths.extract_paths(
        adjacency,
        sources=[1],
        targets=[5, 6],
        max_hops=3,
        source_column="source",
        target_column="target",
    )

    assert result.source_column == "source"
    assert result.target_column == "target"
    assert result.paths == [
        (1, 2, 5),
        (1, 2, 3, 5),
        (1, 2, 3, 6),
        (1, 2, 4, 5),
        (1, 2, 4, 6),
        (1, 3, 5),
        (1, 3, 6),
    ]


def test_paths_cli_writes_expected_outputs(tmp_path: Path) -> None:
    adjacency_path = Path("tests/data/path_adjacency.csv")
    output_dir = tmp_path / "paths"

    exit_code = paths.main(
        [
            "--adjacency",
            str(adjacency_path),
            "--source",
            "1",
            "--source",
            "2",
            "--target",
            "5",
            "--target",
            "6",
            "--max-hops",
            "3",
            "--output",
            str(output_dir),
        ]
    )
    assert exit_code == 0

    csv_path = output_dir / "paths.csv"
    json_path = output_dir / "paths.json"
    summary_path = output_dir / "run_summary.json"

    assert csv_path.exists()
    assert json_path.exists()
    assert summary_path.exists()

    dataframe = pd.read_csv(csv_path)
    assert list(dataframe.columns) == ["path_id", "hop_count", "nodes"]
    assert len(dataframe) == 12
    assert dataframe.iloc[0].to_dict() == {"path_id": 0, "hop_count": 2, "nodes": "1;2;5"}

    payload = json.loads(json_path.read_text(encoding="utf8"))
    assert len(payload["paths"]) == 12
    assert payload["paths"][0] == [1, 2, 5]

    summary = json.loads(summary_path.read_text(encoding="utf8"))
    assert summary["path_count"] == 12
    assert summary["sources"] == [1, 2]
    assert summary["targets"] == [5, 6]
