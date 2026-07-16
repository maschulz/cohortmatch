"""Tests for index preservation in CohortMatch.

These tests verify that the matching process correctly preserves the original
indices of the data, handling various index types and formats.
"""

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from cohortmatch.datatypes import MatcherConfig
from cohortmatch.pipeline import run_match
from cohortmatch.validation import validate_dataframe_index


def generate_data(
    index_values: list, n_samples: int = 100, random_state: int = 42
) -> DataFrame:
    """Generate test data with specific index.

    Args:
        index_values: Values to use as index for the returned DataFrame
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with specified index containing synthetic data

    """
    rng = np.random.RandomState(random_state)

    # Generate data
    treatment = rng.binomial(1, 0.4, size=n_samples)
    age = rng.normal(50, 10, size=n_samples)
    gender = rng.binomial(1, 0.5, size=n_samples)
    blood_pressure = rng.normal(120, 20, size=n_samples) + treatment * 5
    disease_risk = rng.normal(10, 5, size=n_samples) + treatment * 3 + age * 0.1
    income = rng.lognormal(10, 1, size=n_samples) * (0.8 + treatment * 0.4)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "treatment": treatment,
            "age": age,
            "gender": gender,
            "blood_pressure": blood_pressure,
            "disease_risk": disease_risk,
            "income": income,
        }
    )

    # Assign index
    if len(index_values) != n_samples:
        raise ValueError(
            f"Index length ({len(index_values)}) must match data length ({n_samples})"
        )

    df.index = index_values
    return df


def get_base_config() -> MatcherConfig:
    """Get a basic configuration for testing."""
    return MatcherConfig(
        treatment_col="treatment",
        covariates=["age", "gender", "blood_pressure", "income"],
        estimate_propensity=True,
        match_method="greedy",
        random_state=42,
    )


def test_integer_index_preservation():
    """Test that non-sequential integer indices are preserved."""
    # Generate non-sequential integer indices
    n_samples = 100
    rng = np.random.RandomState(123)

    # Ensure indices are non-sequential by adding large gaps
    indices = sorted(rng.choice(range(1000, 10000), size=n_samples, replace=False))

    # Generate data and run matching
    data = generate_data(indices, n_samples=n_samples)
    config = get_base_config()

    # Perform matching
    results = run_match(data, config)

    # Check that matched indices are a subset of original indices
    original_indices = set(data.index)
    matched_indices = set(results.matched_data.index)

    assert matched_indices.issubset(original_indices), (
        "Matched indices should be a subset of original indices"
    )

    # Check that match pairs use the original indices
    match_pairs_df = results.get_match_pairs()
    treatment_indices = set(match_pairs_df["treatment_id"])
    control_indices = set(match_pairs_df["control_id"])

    assert treatment_indices.issubset(original_indices), (
        "Treatment indices in match pairs should be original indices"
    )
    assert control_indices.issubset(original_indices), (
        "Control indices in match pairs should be original indices"
    )


def test_string_index_preservation():
    """Test that string indices are preserved."""
    # Generate string indices
    n_samples = 100
    base_strings = ["patient", "subject", "participant", "id", "case"]

    # Create string indices like 'patient_001', 'subject_042', etc.
    indices = []
    for i in range(n_samples):
        base = base_strings[i % len(base_strings)]
        indices.append(f"{base}_{i:03d}")

    # Generate data and run matching
    data = generate_data(indices, n_samples=n_samples)
    config = get_base_config()

    # Perform matching
    results = run_match(data, config)

    # Check that matched indices are a subset of original indices
    original_indices = set(data.index)
    matched_indices = set(results.matched_data.index)

    assert matched_indices.issubset(original_indices), (
        "Matched indices should be a subset of original indices"
    )

    # Check that match pairs use the original indices
    match_pairs_df = results.get_match_pairs()
    treatment_indices = set(match_pairs_df["treatment_id"])
    control_indices = set(match_pairs_df["control_id"])

    assert treatment_indices.issubset(original_indices), (
        "Treatment indices in match pairs should be original indices"
    )
    assert control_indices.issubset(original_indices), (
        "Control indices in match pairs should be original indices"
    )


def test_mixed_index_types():
    """Test that mixed index types (int and str) are preserved."""
    # Generate mixed indices
    n_samples = 100
    indices = []

    for i in range(n_samples):
        # Alternating between int and str indices
        if i % 2 == 0:
            indices.append(i * 10)  # int index
        else:
            indices.append(f"ID_{i:03d}")  # string index

    # Generate data and run matching
    data = generate_data(indices, n_samples=n_samples)

    # This should fail validation
    with pytest.raises(TypeError):
        validate_dataframe_index(data)


def test_unsupported_index_type():
    """Test that non-string, non-integer indices are rejected."""
    # Generate data with float indices
    n_samples = 100
    indices = [float(i) + 0.5 for i in range(n_samples)]

    data = generate_data(indices, n_samples=n_samples)

    # Validate should raise TypeError
    with pytest.raises(TypeError):
        validate_dataframe_index(data)

    # Matching should also fail
    config = get_base_config()
    with pytest.raises(TypeError):
        run_match(data, config)


def test_small_to_large_matching():
    """Test index preservation when small-to-large matching is triggered."""
    # Create data with more treatment than control
    n_samples = 100
    indices = sorted(
        np.random.RandomState(123).choice(
            range(1000, 10000), size=n_samples, replace=False
        )
    )

    # Generate data with treatment bias
    rng = np.random.RandomState(42)
    treatment = rng.binomial(1, 0.7, size=n_samples)  # 70% treatment

    # Rest of the data
    age = rng.normal(50, 10, size=n_samples)
    gender = rng.binomial(1, 0.5, size=n_samples)
    blood_pressure = rng.normal(120, 20, size=n_samples) + treatment * 5
    disease_risk = rng.normal(10, 5, size=n_samples) + treatment * 3 + age * 0.1
    income = rng.lognormal(10, 1, size=n_samples) * (0.8 + treatment * 0.4)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "treatment": treatment,
            "age": age,
            "gender": gender,
            "blood_pressure": blood_pressure,
            "disease_risk": disease_risk,
            "income": income,
        }
    )

    data.index = indices

    # Perform matching
    config = get_base_config()
    results = run_match(data, config)

    # Check that matched indices are a subset of original indices
    original_indices = set(data.index)
    matched_indices = set(results.matched_data.index)

    assert matched_indices.issubset(original_indices), (
        "Matched indices should be a subset of original indices"
    )

    # Check that match pairs use the original indices
    match_pairs_df = results.get_match_pairs()
    treatment_indices = set(match_pairs_df["treatment_id"])
    control_indices = set(match_pairs_df["control_id"])

    assert treatment_indices.issubset(original_indices), (
        "Treatment indices in match pairs should be original indices"
    )
    assert control_indices.issubset(original_indices), (
        "Control indices in match pairs should be original indices"
    )


def test_exact_matching_index_preservation():
    """Test index preservation with exact matching."""
    # Generate non-sequential integer indices
    n_samples = 100
    rng = np.random.RandomState(123)
    indices = sorted(rng.choice(range(1000, 10000), size=n_samples, replace=False))

    # Generate data
    data = generate_data(indices, n_samples=n_samples)

    # Configure with exact matching
    config = get_base_config()
    config.exact_match_cols = ["gender"]

    # Perform matching
    results = run_match(data, config)

    # Check that matched indices are a subset of original indices
    original_indices = set(data.index)
    matched_indices = set(results.matched_data.index)

    assert matched_indices.issubset(original_indices), (
        "Matched indices should be a subset of original indices"
    )

    # Check that match pairs use the original indices
    match_pairs_df = results.get_match_pairs()
    treatment_indices = set(match_pairs_df["treatment_id"])
    control_indices = set(match_pairs_df["control_id"])

    assert treatment_indices.issubset(original_indices), (
        "Treatment indices in match pairs should be original indices"
    )
    assert control_indices.issubset(original_indices), (
        "Control indices in match pairs should be original indices"
    )

    # Verify exact matching worked by checking gender
    for _, row in match_pairs_df.iterrows():
        treat_gender = data.loc[row["treatment_id"], "gender"]
        control_gender = data.loc[row["control_id"], "gender"]
        assert treat_gender == control_gender, (
            "Exact matching should ensure gender matches"
        )


def test_optimal_matching_index_preservation():
    """Test index preservation with optimal matching."""
    # Generate non-sequential integer indices
    n_samples = 100
    rng = np.random.RandomState(123)
    indices = sorted(rng.choice(range(1000, 10000), size=n_samples, replace=False))

    # Generate data
    data = generate_data(indices, n_samples=n_samples)

    # Configure with optimal matching
    config = get_base_config()
    config.match_method = "optimal"

    # Perform matching
    results = run_match(data, config)

    # Check that matched indices are a subset of original indices
    original_indices = set(data.index)
    matched_indices = set(results.matched_data.index)

    assert matched_indices.issubset(original_indices), (
        "Matched indices should be a subset of original indices"
    )

    # Check that match pairs use the original indices
    match_pairs_df = results.get_match_pairs()
    treatment_indices = set(match_pairs_df["treatment_id"])
    control_indices = set(match_pairs_df["control_id"])

    assert treatment_indices.issubset(original_indices), (
        "Treatment indices in match pairs should be original indices"
    )
    assert control_indices.issubset(original_indices), (
        "Control indices in match pairs should be original indices"
    )
