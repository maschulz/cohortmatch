"""Tests for the index preservation functionality of CohortMatcher.

These tests verify that the matching process correctly preserves the original
indices of the data, handling various index types and formats.
"""

import numpy as np
import pandas as pd

from cohortmatch.datatypes import MatcherConfig
from cohortmatch.pipeline import run_match


def test_string_participant_ids():
    """Test that string participant IDs are correctly used in match pairs."""
    # Create data with string participant IDs
    n_samples = 50
    rng = np.random.RandomState(42)

    # Create string indices like 'P001', 'P002', etc.
    participant_ids = [f"P{i:03d}" for i in range(1, n_samples + 1)]

    # Generate data
    treatment = rng.binomial(1, 0.4, size=n_samples)
    age = rng.normal(50, 10, size=n_samples)
    gender = rng.binomial(1, 0.5, size=n_samples)

    # Create DataFrame with string index
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

    # Verify the match pairs contain the original string IDs
    match_pairs_df = results.get_match_pairs()

    # Check that IDs in pairs are from the original index
    for _, row in match_pairs_df.iterrows():
        assert row["treatment_id"] in participant_ids
        assert row["control_id"] in participant_ids

        # Verify the IDs correspond to the correct treatment groups
        assert data.loc[row["treatment_id"], "treatment"] == 1
        assert data.loc[row["control_id"], "treatment"] == 0


def test_integer_participant_ids():
    """Test that non-sequential integer IDs are correctly used in match pairs."""
    # Create data with non-sequential integer IDs
    n_samples = 50
    rng = np.random.RandomState(42)

    # Create non-sequential integer IDs
    participant_ids = sorted(
        rng.choice(range(1000, 10000), size=n_samples, replace=False)
    )

    # Generate data
    treatment = rng.binomial(1, 0.4, size=n_samples)
    age = rng.normal(50, 10, size=n_samples)
    gender = rng.binomial(1, 0.5, size=n_samples)

    # Create DataFrame with integer index
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

    # Verify the match pairs contain the original integer IDs
    match_pairs_df = results.get_match_pairs()

    # Check that IDs in pairs are from the original index
    for _, row in match_pairs_df.iterrows():
        assert row["treatment_id"] in participant_ids
        assert row["control_id"] in participant_ids

        # Verify the IDs correspond to the correct treatment groups
        assert data.loc[row["treatment_id"], "treatment"] == 1
        assert data.loc[row["control_id"], "treatment"] == 0


def test_exact_matching_with_participant_ids():
    """Test that exact matching works correctly with participant IDs."""
    # Create data with string participant IDs
    n_samples = 50
    rng = np.random.RandomState(42)

    # Create string indices
    participant_ids = [f"P{i:03d}" for i in range(1, n_samples + 1)]

    # Generate data with a categorical variable for exact matching
    treatment = rng.binomial(1, 0.4, size=n_samples)
    age = rng.normal(50, 10, size=n_samples)
    gender = rng.binomial(1, 0.5, size=n_samples)  # Will be used for exact matching

    # Create DataFrame
    data = pd.DataFrame(
        {"treatment": treatment, "age": age, "gender": gender}, index=participant_ids
    )

    # Configure matcher with exact matching on gender
    config = MatcherConfig(
        treatment_col="treatment",
        covariates=["age"],
        exact_match_cols=["gender"],
        match_method="greedy",
        caliper_method=None,
        random_state=42,
    )

    # Perform matching
    results = run_match(data, config)

    # Verify exact matching constraint is respected
    match_pairs_df = results.get_match_pairs()

    for _, row in match_pairs_df.iterrows():
        treatment_id = row["treatment_id"]
        control_id = row["control_id"]

        # Verify the gender is the same for matched pairs
        assert data.loc[treatment_id, "gender"] == data.loc[control_id, "gender"]


def test_ratio_matching_with_participant_ids():
    """Test that ratio matching works correctly with participant IDs."""
    # Create data with string participant IDs
    n_samples = 60
    rng = np.random.RandomState(42)

    # Create string indices
    participant_ids = [f"P{i:03d}" for i in range(1, n_samples + 1)]

    # Generate data with more controls than treatments
    treatment = np.array([1] * 20 + [0] * 40)  # 20 treatments, 40 controls
    rng.shuffle(treatment)

    age = rng.normal(50, 10, size=n_samples)
    gender = rng.binomial(1, 0.5, size=n_samples)

    # Create DataFrame
    data = pd.DataFrame(
        {"treatment": treatment, "age": age, "gender": gender}, index=participant_ids
    )

    # Configure matcher with 1:2 matching ratio
    config = MatcherConfig(
        treatment_col="treatment",
        covariates=["age", "gender"],
        match_method="greedy",
        ratio=2.0,  # 1:2 matching
        caliper_method=None,
        random_state=42,
    )

    # Perform matching
    results = run_match(data, config)

    # Verify the ratio in the match pairs
    match_pairs_df = results.get_match_pairs()

    # Each treatment should have up to 2 controls
    treatment_ids = set(match_pairs_df["treatment_id"])
    for t_id in treatment_ids:
        control_count = match_pairs_df[match_pairs_df["treatment_id"] == t_id].shape[0]
        assert control_count <= 2, f"Treatment {t_id} has more than 2 controls"


def test_original_and_matched_data_preservation():
    """Test that original_data and matched_data are correctly preserved."""
    # Create data with string participant IDs
    n_samples = 50
    rng = np.random.RandomState(42)

    # Create string indices
    participant_ids = [f"P{i:03d}" for i in range(1, n_samples + 1)]

    # Generate data
    treatment = rng.binomial(1, 0.4, size=n_samples)
    age = rng.normal(50, 10, size=n_samples)
    gender = rng.binomial(1, 0.5, size=n_samples)
    outcome = rng.normal(0, 1, size=n_samples) + treatment * 0.5 + age * 0.1

    # Create DataFrame with an extra column that shouldn't affect matching
    data = pd.DataFrame(
        {
            "treatment": treatment,
            "age": age,
            "gender": gender,
            "outcome": outcome,
            "extra_info": [f"Info-{i}" for i in range(n_samples)],
        },
        index=participant_ids,
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

    # Verify that original_data contains all rows and columns
    assert results.original_data.shape[0] == n_samples
    assert all(col in results.original_data.columns for col in data.columns)

    # Verify that matched_data contains only the matched participants
    match_pairs_df = results.get_match_pairs()
    all_matched_ids = set(match_pairs_df["treatment_id"]).union(
        set(match_pairs_df["control_id"])
    )

    assert results.matched_data.shape[0] == len(all_matched_ids)
    assert all(idx in all_matched_ids for idx in results.matched_data.index)

    # Verify that matched_data preserves all columns from original_data
    assert all(col in results.matched_data.columns for col in data.columns)


def test_optimal_matching_with_participant_ids():
    """Test that optimal matching works correctly with participant IDs."""
    # Create data with string participant IDs
    n_samples = 50
    rng = np.random.RandomState(42)

    # Create string indices
    participant_ids = [f"P{i:03d}" for i in range(1, n_samples + 1)]

    # Generate data
    treatment = rng.binomial(1, 0.4, size=n_samples)
    age = rng.normal(50, 10, size=n_samples)
    gender = rng.binomial(1, 0.5, size=n_samples)

    # Create DataFrame
    data = pd.DataFrame(
        {"treatment": treatment, "age": age, "gender": gender}, index=participant_ids
    )

    # Configure matcher with optimal matching
    config = MatcherConfig(
        treatment_col="treatment",
        covariates=["age", "gender"],
        match_method="optimal",  # Use optimal matching
        caliper_method=None,
        random_state=42,
    )

    # Perform matching
    results = run_match(data, config)

    # Verify the match pairs contain valid participant IDs
    match_pairs_df = results.get_match_pairs()

    for _, row in match_pairs_df.iterrows():
        assert row["treatment_id"] in participant_ids
        assert row["control_id"] in participant_ids

        # Verify the IDs correspond to the correct treatment groups
        assert data.loc[row["treatment_id"], "treatment"] == 1
        assert data.loc[row["control_id"], "treatment"] == 0
