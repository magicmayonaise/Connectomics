import math
from pathlib import Path

import pytest

from topology import TopologyAnalyzer, _compute_correlations


def sample_syn_count():
    """Synthetic synapse table mirroring the CX schema."""
    return [
        # N1 -> N2 edges
        {"pre_class": "N1", "post_class": "N2", "pre_id": 1, "post_id": 10, "syn_count": 5},
        {"pre_class": "N1", "post_class": "N2", "pre_id": 2, "post_id": 10, "syn_count": 7},
        {"pre_class": "N1", "post_class": "N2", "pre_id": 3, "post_id": 11, "syn_count": 2},
        {"pre_class": "N1", "post_class": "N2", "pre_id": 4, "post_id": 11, "syn_count": 4},
        {"pre_class": "N1", "post_class": "N2", "pre_id": 5, "post_id": 11, "syn_count": 1},
        {"pre_class": "N1", "post_class": "N2", "pre_id": 6, "post_id": 12, "syn_count": 9},
        {"pre_class": "N1", "post_class": "N2", "pre_id": 7, "post_id": 12, "syn_count": 6},
        {"pre_class": "N1", "post_class": "N2", "pre_id": 8, "post_id": 12, "syn_count": 5},
        {"pre_class": "N1", "post_class": "N2", "pre_id": 9, "post_id": 12, "syn_count": 4},
        {"pre_class": "N1", "post_class": "N2", "pre_id": 10, "post_id": 13, "syn_count": 3},
        {"pre_class": "N1", "post_class": "N2", "pre_id": 11, "post_id": 13, "syn_count": 2},
        # N2 -> N3 edges (ensure N2->N3 distributions are populated)
        {"pre_class": "N2", "post_class": "N3", "pre_id": 30, "post_id": 40, "syn_count": 8},
        {"pre_class": "N2", "post_class": "N3", "pre_id": 31, "post_id": 41, "syn_count": 3},
        {"pre_class": "N2", "post_class": "N3", "pre_id": 32, "post_id": 41, "syn_count": 6},
    ]


def test_synapse_weight_plots(tmp_path: Path) -> None:
    analyzer = TopologyAnalyzer(sample_syn_count())
    weights = analyzer.plot_synapse_weight_distributions(tmp_path)

    assert "N1→N2" in weights
    assert "N2→N3" in weights
    assert len(weights["N1→N2"]) == 11
    assert len(weights["N2→N3"]) == 3

    # SVG artefacts for both connection directions should exist
    expected_files = [
        "n1_to_n2_weight_histogram.svg",
        "n1_to_n2_weight_ccdf.svg",
        "n2_to_n3_weight_histogram.svg",
        "n2_to_n3_weight_ccdf.svg",
    ]
    for name in expected_files:
        assert (tmp_path / name).exists(), f"Missing SVG output {name}"


def test_convergence_vs_strength_exports(tmp_path: Path) -> None:
    analyzer = TopologyAnalyzer(sample_syn_count())
    metrics, correlations = analyzer.plot_convergence_vs_strength(tmp_path)

    metrics_path = tmp_path / "n1_to_n2_convergence_strength.csv"
    corr_path = tmp_path / "n1_to_n2_convergence_strength_correlations.csv"
    scatter_path = tmp_path / "n1_to_n2_convergence_vs_mean_strength.svg"

    assert metrics_path.exists()
    assert corr_path.exists()
    assert scatter_path.exists()

    expected = {
        10: {"convergence": 2.0, "mean_strength": 6.0},
        11: {"convergence": 3.0, "mean_strength": 7 / 3},
        12: {"convergence": 4.0, "mean_strength": 6.0},
        13: {"convergence": 2.0, "mean_strength": 2.5},
    }

    assert len(metrics) == len(expected)
    for row in metrics:
        neuron = row["post_neuron"]
        assert neuron in expected
        assert math.isclose(row["convergence"], expected[neuron]["convergence"], rel_tol=1e-6)
        assert math.isclose(row["mean_strength"], expected[neuron]["mean_strength"], rel_tol=1e-6)

    pearson = next(item for item in correlations if item["metric"] == "pearson")
    spearman = next(item for item in correlations if item["metric"] == "spearman")
    assert math.isclose(pearson["correlation"], 0.28733220440430024, rel_tol=1e-9)
    assert math.isclose(spearman["correlation"], 0.05555555555555555, rel_tol=1e-9)


def test_correlations_empty_metrics() -> None:
    assert _compute_correlations([]) == []
