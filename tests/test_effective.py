from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.sparse import csr_matrix, load_npz, save_npz

from cx_connectome.ci import effective


def test_compute_effective_connectivity_normalises_and_thresholds() -> None:
    adjacency = csr_matrix(
        np.array(
            [
                [0, 2, 0],
                [1, 0, 3],
                [0, 1, 0],
            ],
            dtype=float,
        )
    )

    results = effective.compute_effective_connectivity(
        adjacency,
        [1, 2],
        chunk_size_cols=2,
        normalize="post_total",
        threshold_norm_input=0.25,
    )

    assert set(results) == {1, 2}
    hop_one = results[1].toarray()
    hop_two = results[2].toarray()

    expected_one = np.array(
        [
            [0.0, 2.0 / 3.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0 / 3.0, 0.0],
        ]
    )
    expected_two = np.array(
        [
            [2.0 / 3.0, 0.0, 2.0 / 3.0],
            [0.0, 1.0, 0.0],
            [1.0 / 3.0, 0.0, 1.0 / 3.0],
        ]
    )

    np.testing.assert_allclose(hop_one, expected_one)
    np.testing.assert_allclose(hop_two, expected_two)


def test_compute_effective_connectivity_validates_inputs() -> None:
    with pytest.raises(ValueError):
        effective.compute_effective_connectivity(csr_matrix(np.ones((2, 3))), [1])

    with pytest.raises(ValueError):
        effective.compute_effective_connectivity(csr_matrix(np.eye(2)), [])

    with pytest.raises(ValueError):
        effective.compute_effective_connectivity(csr_matrix(np.eye(2)), [-1])

    with pytest.raises(ValueError):
        effective.compute_effective_connectivity(csr_matrix(np.eye(2)), [1], normalize="bad")


def test_effective_cli_writes_outputs(tmp_path) -> None:
    adjacency = csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
    adjacency_path = tmp_path / "adjacency.npz"
    save_npz(adjacency_path, adjacency)

    output_dir = tmp_path / "outputs"

    exit_code = effective.main(
        [
            "--adjacency",
            str(adjacency_path),
            "--k-hop",
            "1",
            "--k-hop",
            "2",
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 0

    hop_one_path = output_dir / "effective_k1.npz"
    hop_two_path = output_dir / "effective_k2.npz"
    summary_path = output_dir / "effective_summary.json"

    assert hop_one_path.exists()
    assert hop_two_path.exists()
    assert summary_path.exists()

    hop_one = load_npz(hop_one_path).toarray()
    hop_two = load_npz(hop_two_path).toarray()

    np.testing.assert_allclose(hop_one, np.array([[0.0, 1.0], [1.0, 0.0]]))
    np.testing.assert_allclose(hop_two, np.array([[1.0, 0.0], [0.0, 1.0]]))

    summary = json.loads(summary_path.read_text(encoding="utf8"))
    assert summary["hops"] == [1, 2]
    assert summary["nnz_per_hop"]["1"] == 2
