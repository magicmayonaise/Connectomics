"""Tests for the centralized constants module."""

from __future__ import annotations

from pathlib import Path

import pytest

from cx_connectome.constants import (
    # Dataset
    DEFAULT_DATASET,
    DEFAULT_MATERIALIZATION,
    DEFAULT_CHUNK_SIZE,
    # Tables
    DEFAULT_SYNAPSE_TABLE,
    SYNAPSE_TABLE_CANDIDATES,
    BASELINE_TABLE_CANDIDATES,
    # Outputs
    DEFAULT_OUTPUT_DIR,
    DEFAULT_N1_N2_OUTPUT,
    DEFAULT_N2_N3_OUTPUT,
    DEFAULT_GRAPH_OUTPUT,
    # Thresholds
    DEFAULT_MIN_SYNAPSES,
    DEFAULT_FIRST_HOP_THRESHOLD,
    DEFAULT_SECOND_HOP_THRESHOLD,
    # Labels
    DEFAULT_N1_LABEL,
    DEFAULT_N2_LABEL,
    DEFAULT_N3_LABEL,
    # Mappings
    DEFAULT_FUNCTIONAL_MAPPING,
    DEFAULT_SUPER_CLASS_SYNONYMS,
    DEFAULT_SUPER_CLASS_KEYWORDS,
    # CI
    DEFAULT_K_HOPS,
    DEFAULT_CI_CHUNK_SIZE,
    DEFAULT_NORMALIZE_MODE,
    DEFAULT_EFFECTIVE_THRESHOLD,
    DEFAULT_SIGNED_MODE,
    DEFAULT_TAU,
    DEFAULT_EXCITABILITY,
    DEFAULT_TIME_STEPS,
    DEFAULT_DIVISIVE_NORM,
    # Filenames
    ADJACENCY_FILENAME,
    N2_NODES_FILENAME,
    N3_NODES_FILENAME,
)


class TestDatasetConstants:
    """Test dataset-related constants."""

    def test_default_dataset_is_flywire(self) -> None:
        assert DEFAULT_DATASET == "flywire_fafb_production"

    def test_default_materialization_is_positive(self) -> None:
        assert DEFAULT_MATERIALIZATION > 0
        assert isinstance(DEFAULT_MATERIALIZATION, int)

    def test_default_chunk_size_is_reasonable(self) -> None:
        assert DEFAULT_CHUNK_SIZE >= 1000
        assert DEFAULT_CHUNK_SIZE <= 100_000


class TestSynapseTableConstants:
    """Test synapse table configuration."""

    def test_default_synapse_table_in_candidates(self) -> None:
        assert DEFAULT_SYNAPSE_TABLE in SYNAPSE_TABLE_CANDIDATES

    def test_synapse_candidates_are_strings(self) -> None:
        assert all(isinstance(t, str) for t in SYNAPSE_TABLE_CANDIDATES)

    def test_baseline_candidates_are_strings(self) -> None:
        assert all(isinstance(t, str) for t in BASELINE_TABLE_CANDIDATES)


class TestOutputPathConstants:
    """Test output path configuration."""

    def test_output_dir_is_path(self) -> None:
        assert isinstance(DEFAULT_OUTPUT_DIR, Path)

    def test_n1_n2_output_is_parquet(self) -> None:
        assert DEFAULT_N1_N2_OUTPUT.suffix == ".parquet"

    def test_n2_n3_output_is_parquet(self) -> None:
        assert DEFAULT_N2_N3_OUTPUT.suffix == ".parquet"

    def test_graph_output_is_pickle(self) -> None:
        assert DEFAULT_GRAPH_OUTPUT.suffix == ".gpickle"


class TestThresholdConstants:
    """Test threshold configuration."""

    def test_min_synapses_is_positive(self) -> None:
        assert DEFAULT_MIN_SYNAPSES > 0

    def test_first_hop_threshold_less_than_second(self) -> None:
        # Generally N1->N2 uses lower threshold than N2->N3
        assert DEFAULT_FIRST_HOP_THRESHOLD <= DEFAULT_SECOND_HOP_THRESHOLD


class TestLabelConstants:
    """Test population label constants."""

    def test_labels_are_distinct(self) -> None:
        labels = {DEFAULT_N1_LABEL, DEFAULT_N2_LABEL, DEFAULT_N3_LABEL}
        assert len(labels) == 3

    def test_labels_are_non_empty(self) -> None:
        assert DEFAULT_N1_LABEL
        assert DEFAULT_N2_LABEL
        assert DEFAULT_N3_LABEL


class TestFunctionalMappingConstants:
    """Test functional role mapping."""

    def test_functional_mapping_has_entries(self) -> None:
        assert len(DEFAULT_FUNCTIONAL_MAPPING) > 0

    def test_functional_mapping_values_are_strings(self) -> None:
        for key, value in DEFAULT_FUNCTIONAL_MAPPING.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_contains_navigation_roles(self) -> None:
        roles = set(DEFAULT_FUNCTIONAL_MAPPING.values())
        assert "Navigation" in roles

    def test_contains_sleep_roles(self) -> None:
        roles = set(DEFAULT_FUNCTIONAL_MAPPING.values())
        assert "Sleep-Promoting" in roles


class TestSuperClassConstants:
    """Test super-class synonyms and keywords."""

    def test_synonyms_normalize_to_standard_forms(self) -> None:
        # 'asc' should map to 'ascending'
        assert DEFAULT_SUPER_CLASS_SYNONYMS.get("asc") == "ascending"
        assert DEFAULT_SUPER_CLASS_SYNONYMS.get("desc") == "descending"

    def test_keywords_are_tuples(self) -> None:
        for item in DEFAULT_SUPER_CLASS_KEYWORDS:
            assert isinstance(item, tuple)
            assert len(item) == 2


class TestCIConstants:
    """Test Connectivity Interpreter configuration."""

    def test_k_hops_is_sequence_of_positive_ints(self) -> None:
        assert len(DEFAULT_K_HOPS) > 0
        assert all(isinstance(k, int) and k > 0 for k in DEFAULT_K_HOPS)

    def test_ci_chunk_size_is_power_of_two(self) -> None:
        # 4096 = 2^12
        assert DEFAULT_CI_CHUNK_SIZE > 0
        assert (DEFAULT_CI_CHUNK_SIZE & (DEFAULT_CI_CHUNK_SIZE - 1)) == 0

    def test_normalize_mode_is_valid(self) -> None:
        valid_modes = {"post_total", "pre_total", "symmetric", "none"}
        assert DEFAULT_NORMALIZE_MODE in valid_modes

    def test_signed_mode_is_valid(self) -> None:
        valid_modes = {"net", "blocks"}
        assert DEFAULT_SIGNED_MODE in valid_modes

    def test_tau_is_positive_fraction(self) -> None:
        assert 0 < DEFAULT_TAU < 1

    def test_excitability_is_positive(self) -> None:
        assert DEFAULT_EXCITABILITY > 0

    def test_time_steps_is_positive(self) -> None:
        assert DEFAULT_TIME_STEPS > 0

    def test_divisive_norm_is_bool(self) -> None:
        assert isinstance(DEFAULT_DIVISIVE_NORM, bool)


class TestFilenameConstants:
    """Test standard filename constants."""

    def test_adjacency_filename_is_parquet(self) -> None:
        assert ADJACENCY_FILENAME.endswith(".parquet")

    def test_node_filenames_are_csv(self) -> None:
        assert N2_NODES_FILENAME.endswith(".csv")
        assert N3_NODES_FILENAME.endswith(".csv")
