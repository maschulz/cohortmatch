"""Tests for edge cases in CohortMatcher participant ID handling.

These tests verify that the matching process correctly handles edge cases
like IDs with special characters, very long IDs, or IDs that look like integers
but are actually strings.
"""

import numpy as np
import pandas as pd
import pytest

from cohortmatch.datatypes import MatcherConfig
from cohortmatch.pipeline import run_match


def test_special_character_ids():
    """Test that IDs with special characters are preserved."""
    # Create data with IDs containing special characters
    special_ids = [
        "patient/001",
        "patient-002",
        "patient+003",
        "patient_004",
        "patient#005",
        "patient@006",
        "patient&007",
        "patient.008",
        "patient!009",
        "patient:010",
    ]

    # Create DataFrame with both treatment and control for each special character type
    data = pd.DataFrame(
        {
            "treatment": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "age": [45, 46, 50, 52, 55, 56, 60, 62, 65, 66],
            "gender": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        },
        index=special_ids,
    )

    # Configure matcher
    config = MatcherConfig(
        treatment_col="treatment",
        covariates=["age", "gender"],
        match_method="greedy",
        caliper_method=None,
        random_state=42,
    )

    # Perform matching
    results = run_match(data, config)

    # Verify match pairs contain the special character IDs
    match_pairs_df = results.get_match_pairs()

    for _, row in match_pairs_df.iterrows():
        assert row["treatment_id"] in special_ids
        assert row["control_id"] in special_ids


def test_numeric_string_ids():
    """Test IDs that are numeric but stored as strings."""
    # Create data with IDs that are numeric but stored as strings
    numeric_str_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008"]

    # Create DataFrame with both treatment and control
    data = pd.DataFrame(
        {
            "treatment": [1, 1, 1, 1, 0, 0, 0, 0],
            "age": [45, 50, 55, 60, 46, 52, 56, 62],
            "gender": [1, 0, 1, 0, 1, 0, 1, 0],
        },
        index=numeric_str_ids,
    )

    # Configure matcher
    config = MatcherConfig(
        treatment_col="treatment",
        covariates=["age", "gender"],
        match_method="greedy",
        caliper_method=None,
        random_state=42,
    )

    # Perform matching
    results = run_match(data, config)

    # Verify match pairs contain the string numeric IDs (not converted to integers)
    match_pairs_df = results.get_match_pairs()

    for _, row in match_pairs_df.iterrows():
        treatment_id = row["treatment_id"]
        control_id = row["control_id"]

        # Verify IDs are in the original list
        assert treatment_id in numeric_str_ids
        assert control_id in numeric_str_ids

        # Verify IDs are strings, not integers
        assert isinstance(treatment_id, str)
        assert isinstance(control_id, str)


def test_very_long_ids():
    """Test unusually long IDs are handled correctly."""
    # Create data with very long IDs
    long_ids = [
        "patient_with_very_long_identifier_for_testing_purposes_001",
        "patient_with_very_long_identifier_for_testing_purposes_002",
        "patient_with_very_long_identifier_for_testing_purposes_003",
        "patient_with_very_long_identifier_for_testing_purposes_004",
        "patient_with_very_long_identifier_for_testing_purposes_005",
        "patient_with_very_long_identifier_for_testing_purposes_006",
        "patient_with_very_long_identifier_for_testing_purposes_007",
        "patient_with_very_long_identifier_for_testing_purposes_008",
    ]

    # Create DataFrame with both treatment and control
    data = pd.DataFrame(
        {
            "treatment": [1, 1, 1, 1, 0, 0, 0, 0],
            "age": [45, 50, 55, 60, 46, 52, 56, 62],
            "gender": [1, 0, 1, 0, 1, 0, 1, 0],
        },
        index=long_ids,
    )

    # Configure matcher
    config = MatcherConfig(
        treatment_col="treatment",
        covariates=["age", "gender"],
        match_method="greedy",
        caliper_method=None,
        random_state=42,
    )

    # Perform matching
    results = run_match(data, config)

    # Verify match pairs contain the long IDs
    match_pairs_df = results.get_match_pairs()

    for _, row in match_pairs_df.iterrows():
        assert row["treatment_id"] in long_ids
        assert row["control_id"] in long_ids


def test_direction_flipping_with_ids():
    """Test participant ID handling when matching direction is flipped."""
    # Create dataset with more treatment than control units
    n_samples = 20
    participant_ids = [f"P{i:03d}" for i in range(1, n_samples + 1)]

    # Deliberately create more treatment than control to force direction flipping
    treatment = np.array([1] * 15 + [0] * 5)  # 15 treatment, 5 control (3:1 ratio)
    age = np.array(
        [40, 45, 50, 55, 60, 65, 42, 47, 52, 57, 62, 67, 43, 48, 53, 44, 49, 54, 59, 64]
    )
    gender = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    # Create DataFrame
    data = pd.DataFrame(
        {"treatment": treatment, "age": age, "gender": gender}, index=participant_ids
    )

    # Configure matcher
    config = MatcherConfig(
        treatment_col="treatment",
        covariates=["age", "gender"],
        match_method="greedy",
        caliper_method=None,
        random_state=42,
    )

    # Perform matching
    results = run_match(data, config)

    # Verify match pairs
    match_pairs_df = results.get_match_pairs()

    for _, row in match_pairs_df.iterrows():
        # Verify IDs are from the original index
        assert row["treatment_id"] in participant_ids
        assert row["control_id"] in participant_ids

        # Verify treatment and control assignments are correct
        # (important when direction is flipped)
        assert data.loc[row["treatment_id"], "treatment"] == 1
        assert data.loc[row["control_id"], "treatment"] == 0


def test_replacement_with_ids():
    """Test matching with replacement using participant IDs."""
    # Create dataset
    n_samples = 15
    participant_ids = [f"P{i:03d}" for i in range(1, n_samples + 1)]

    # Create data with 5 treatment and 10 control
    treatment = np.array([1] * 5 + [0] * 10)
    age = np.array([40, 45, 50, 55, 60, 42, 47, 52, 57, 62, 67, 43, 48, 53, 58])
    gender = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Create DataFrame
    data = pd.DataFrame(
        {"treatment": treatment, "age": age, "gender": gender}, index=participant_ids
    )

    # Configure matcher with replacement
    config = MatcherConfig(
        treatment_col="treatment",
        covariates=["age", "gender"],
        match_method="greedy",
        replace=True,  # Enable replacement
        caliper_method=None,
        random_state=42,
    )

    # Perform matching
    results = run_match(data, config)

    # Verify match pairs
    match_pairs_df = results.get_match_pairs()

    # Make sure treatment_ids are all unique (no replacement for treatment)
    treatment_ids = match_pairs_df["treatment_id"].tolist()
    assert len(treatment_ids) == len(set(treatment_ids))

    # Check if any control is used more than once (may or may not happen)
    # Just verify that the assignments are correct
    for _, row in match_pairs_df.iterrows():
        assert data.loc[row["treatment_id"], "treatment"] == 1
        assert data.loc[row["control_id"], "treatment"] == 0


def test_no_matches_edge_case():
    """Test the edge case where no matches are found due to constraints."""
    # Create dataset
    participant_ids = ["P001", "P002", "P003", "P004", "P005", "P006"]

    # Create data with incompatible groups for exact matching
    data = pd.DataFrame(
        {
            "treatment": [1, 1, 1, 0, 0, 0],
            "age": [40, 45, 50, 60, 65, 70],  # Large age difference
            "gender": [1, 1, 1, 0, 0, 0],  # No overlap in gender (for exact matching)
        },
        index=participant_ids,
    )

    # Configure matcher with exact matching that will find no matches
    config = MatcherConfig(
        treatment_col="treatment",
        covariates=["age"],
        exact_match_cols=["gender"],  # No gender overlap between groups
        match_method="greedy",
        caliper_method="mahalanobis",
        caliper_value=0.001,  # Small caliper that will prevent matches
        random_state=42,
    )

    # Perform matching
    results = run_match(data, config)

    # Verify that no matches were found
    match_pairs_df = results.get_match_pairs()
    assert len(match_pairs_df) == 0

    # Verify matched_data exists but may be empty
    assert results.matched_data is not None


def test_mixed_id_types_fails():
    """Test that mixed ID types (int and str) properly fail."""
    # Create data with mixed ID types
    mixed_ids = [1001, "1002", 1003, "1004", 1005, "1006"]

    # Create DataFrame with mixed index types
    data = pd.DataFrame(
        {
            "treatment": [1, 1, 1, 0, 0, 0],
            "age": [45, 50, 55, 46, 52, 56],
            "gender": [1, 0, 1, 1, 0, 1],
        },
        index=mixed_ids,
    )

    # Configure matcher
    config = MatcherConfig(
        treatment_col="treatment",
        covariates=["age", "gender"],
        match_method="greedy",
        caliper_method=None,
        random_state=42,
    )

    # The validation should catch the mixed index types
    with pytest.raises(TypeError):
        run_match(data, config)


def test_duplicate_ids_error():
    """Test that duplicate IDs in the index raise an appropriate error."""
    # Create data with duplicate IDs
    duplicate_ids = [
        "P001",
        "P002",
        "P003",
        "P004",
        "P002",
        "P005",
    ]  # P002 is duplicated

    # Create DataFrame with duplicate index
    data = pd.DataFrame(
        {
            "treatment": [1, 1, 1, 0, 0, 0],
            "age": [45, 50, 55, 46, 52, 56],
            "gender": [1, 0, 1, 1, 0, 1],
        },
        index=duplicate_ids,
    )

    # Configure matcher
    config = MatcherConfig(
        treatment_col="treatment",
        covariates=["age", "gender"],
        match_method="greedy",
        caliper_method=None,
        random_state=42,
    )

    # Pandas will raise an error for non-unique index during validation
    with pytest.raises(ValueError):
        run_match(data, config)
