"""Tests for the CI metrics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cx_connectome.ci.metrics import (
    ei_ratio_by_hop,
    lateral_bias,
    robustness_curves,
)


class TestEIRatioByHop:
    """Test excitatory/inhibitory ratio calculations."""

    def test_balanced_ei_returns_zero(self) -> None:
        """Equal E and I contributions should give EI ratio of 0."""
        data = pd.DataFrame({
            "hop": [1, 1, 2, 2],
            "sign": ["exc", "inh", "exc", "inh"],
            "weight": [10.0, 10.0, 20.0, 20.0],
        })
        result = ei_ratio_by_hop(data)

        assert len(result) == 2
        assert all(np.isclose(result["ei_ratio"], 0.0))

    def test_pure_excitatory_returns_one(self) -> None:
        """Pure excitatory input should give EI ratio of 1."""
        data = pd.DataFrame({
            "hop": [1, 2],
            "sign": ["exc", "exc"],
            "weight": [10.0, 20.0],
        })
        result = ei_ratio_by_hop(data)

        assert len(result) == 2
        assert all(np.isclose(result["ei_ratio"], 1.0))

    def test_pure_inhibitory_returns_negative_one(self) -> None:
        """Pure inhibitory input should give EI ratio of -1."""
        data = pd.DataFrame({
            "hop": [1, 2],
            "sign": ["inh", "inh"],
            "weight": [10.0, 20.0],
        })
        result = ei_ratio_by_hop(data)

        assert len(result) == 2
        assert all(np.isclose(result["ei_ratio"], -1.0))

    def test_sign_aliases_recognized(self) -> None:
        """Various sign naming conventions should be normalized."""
        data = pd.DataFrame({
            "hop": [1, 1, 1, 1],
            "sign": ["e", "+", "i", "-"],
            "weight": [10.0, 10.0, 5.0, 5.0],
        })
        result = ei_ratio_by_hop(data)

        # 20 exc, 10 inh -> ratio = (20-10)/(20+10) = 1/3
        expected_ratio = (20 - 10) / (20 + 10)
        assert len(result) == 1
        assert np.isclose(result["ei_ratio"].iloc[0], expected_ratio)

    def test_raises_on_missing_sign_column(self) -> None:
        """Should raise ValueError if no sign column found."""
        data = pd.DataFrame({
            "hop": [1, 2],
            "weight": [10.0, 20.0],
        })
        with pytest.raises(ValueError, match="sign"):
            ei_ratio_by_hop(data)

    def test_raises_on_missing_hop_column(self) -> None:
        """Should raise ValueError if no hop column found."""
        data = pd.DataFrame({
            "sign": ["exc", "inh"],
            "weight": [10.0, 20.0],
        })
        with pytest.raises(ValueError, match="hop"):
            ei_ratio_by_hop(data)


class TestLateralBias:
    """Test ipsilateral/contralateral bias calculations."""

    def test_pure_ipsilateral_returns_one(self) -> None:
        """Pure ipsilateral connections should give bias of 1."""
        data = pd.DataFrame({
            "source_hemi": ["L", "L", "R", "R"],
            "target_hemi": ["L", "L", "R", "R"],
            "weight": [10.0, 20.0, 15.0, 25.0],
        })
        result = lateral_bias(data)

        assert np.isclose(result["bias"].iloc[0], 1.0)

    def test_pure_contralateral_returns_negative_one(self) -> None:
        """Pure contralateral connections should give bias of -1."""
        data = pd.DataFrame({
            "source_hemi": ["L", "L", "R", "R"],
            "target_hemi": ["R", "R", "L", "L"],
            "weight": [10.0, 20.0, 15.0, 25.0],
        })
        result = lateral_bias(data)

        assert np.isclose(result["bias"].iloc[0], -1.0)

    def test_balanced_returns_zero(self) -> None:
        """Equal ipsi and contra should give bias of 0."""
        data = pd.DataFrame({
            "source_hemi": ["L", "L", "R", "R"],
            "target_hemi": ["L", "R", "R", "L"],
            "weight": [10.0, 10.0, 10.0, 10.0],
        })
        result = lateral_bias(data)

        assert np.isclose(result["bias"].iloc[0], 0.0)

    def test_handles_hop_grouping(self) -> None:
        """Should compute bias per hop when hop column present."""
        data = pd.DataFrame({
            "hop": [1, 1, 2, 2],
            "source_hemi": ["L", "L", "L", "L"],
            "target_hemi": ["L", "R", "L", "L"],
            "weight": [10.0, 10.0, 30.0, 0.0],
        })
        result = lateral_bias(data)

        # hop 1: 10 ipsi, 10 contra -> bias = 0
        # hop 2: 30 ipsi, 0 contra -> bias = 1
        assert len(result) == 2
        hop1 = result[result["hop"] == 1]["bias"].iloc[0]
        hop2 = result[result["hop"] == 2]["bias"].iloc[0]
        assert np.isclose(hop1, 0.0)
        assert np.isclose(hop2, 1.0)


class TestRobustnessCurves:
    """Test robustness curve calculations."""

    def test_returns_cumulative_metrics(self) -> None:
        """Should return metrics for each k threshold."""
        data = pd.DataFrame({
            "k": [1, 2, 3, 4, 5],
            "weight": [100.0, 50.0, 25.0, 12.5, 6.25],
        })
        result = robustness_curves(data)

        # Should have entries for each k value
        assert len(result) >= 5

    def test_decreasing_connectivity_with_k(self) -> None:
        """Higher k thresholds should reduce connectivity."""
        data = pd.DataFrame({
            "k": [1, 1, 2, 2, 3, 3],
            "source": [1, 2, 1, 2, 1, 2],
            "target": [3, 4, 3, 4, 3, 4],
            "weight": [10.0, 20.0, 5.0, 10.0, 1.0, 2.0],
        })
        result = robustness_curves(data)

        # Total connectivity should decrease as k increases
        totals = result.groupby("k_threshold")["total_weight"].first()
        if len(totals) > 1:
            # More stringent thresholds should show equal or lower connectivity
            assert totals.iloc[-1] <= totals.iloc[0]

    def test_handles_empty_input(self) -> None:
        """Should handle empty dataframe gracefully."""
        data = pd.DataFrame({
            "k": pd.Series(dtype=int),
            "weight": pd.Series(dtype=float),
        })
        result = robustness_curves(data)

        assert isinstance(result, pd.DataFrame)


class TestMetricsIntegration:
    """Integration tests for metrics pipeline."""

    def test_can_chain_metrics_computations(self) -> None:
        """Metrics should be composable in a pipeline."""
        # Create realistic connectivity data
        np.random.seed(42)
        n_edges = 100

        data = pd.DataFrame({
            "hop": np.random.randint(1, 5, n_edges),
            "k": np.random.randint(1, 4, n_edges),
            "sign": np.random.choice(["exc", "inh"], n_edges),
            "source_hemi": np.random.choice(["L", "R"], n_edges),
            "target_hemi": np.random.choice(["L", "R"], n_edges),
            "source": np.random.randint(1000, 2000, n_edges),
            "target": np.random.randint(2000, 3000, n_edges),
            "weight": np.random.exponential(10.0, n_edges),
        })

        # Should be able to compute all metrics
        ei = ei_ratio_by_hop(data)
        bias = lateral_bias(data)
        robust = robustness_curves(data)

        assert len(ei) > 0
        assert len(bias) > 0
        assert len(robust) > 0

        # All results should be valid DataFrames
        assert isinstance(ei, pd.DataFrame)
        assert isinstance(bias, pd.DataFrame)
        assert isinstance(robust, pd.DataFrame)
