from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from connectomics_cli.synapse_analysis import SynapseAnalyzer


def test_synapse_summary_matches_golden(fake_client, fake_context) -> None:
    analyzer = SynapseAnalyzer.auto_configure(fake_client, fake_context)
    summary = analyzer.summarise(pre_root_ids=[1001, 1002])
    expected = pd.read_csv(Path("tests/data/golden_synapse_summary.csv"))
    assert_frame_equal(summary.dataframe.reset_index(drop=True), expected)
    assert summary.metadata["total_synapses"] == 4
    assert summary.graph.number_of_edges() == 3
