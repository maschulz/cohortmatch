"""Test file for bug fixes in the CohortMatch package.

This module contains tests that verify bug fixes for specific issues.
"""

import numpy as np
import pandas as pd

from cohortmatch.datatypes import MatcherConfig
from cohortmatch.pipeline import run_match


def generate_synthetic_data(n_samples=1000, treatment_prob=0.5, seed=42):
    """Generate synthetic data for testing with a specified treatment probability."""
    np.random.seed(seed)

    # Features
    age = np.random.normal(50, 15, n_samples)
    bmi = np.random.normal(25, 5, n_samples)
    systolic_bp = np.random.normal(120, 15, n_samples)
    cholesterol = np.random.normal(200, 40, n_samples)

    # Binary features
    sex = np.random.binomial(1, 0.5, n_samples)  # 0=female, 1=male
    smoker = np.random.binomial(1, 0.3, n_samples)

    # Categorical feature (converted to dummies)
    blood_type = np.random.choice(
        ["A", "B", "AB", "O"], size=n_samples, p=[0.4, 0.1, 0.1, 0.4]
    )

    # Treatment assignment with specified probability (for more controlled testing)
    if treatment_prob is None:
        # Use the biased assignment from the example script
        logit = -2 + 0.03 * age + 0.5 * sex + 0.8 * smoker + 0.02 * systolic_bp
        prob_treatment = 1 / (1 + np.exp(-logit))
        treatment = np.random.binomial(1, prob_treatment)
    else:
        # Use the specified treatment probability
        treatment = np.random.binomial(1, treatment_prob, n_samples)

    # Outcome with treatment effect = 5.0
    outcome = (
        50
        + 5.0 * treatment
        + 0.1 * age
        + 5 * smoker
        + 0.05 * systolic_bp
        + np.random.normal(0, 5, n_samples)
    )

    # Create DataFrame
    df = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "cholesterol": cholesterol,
            "smoker": smoker,
            "blood_type": blood_type,
            "treatment": treatment,
            "outcome": outcome,
        }
    )

    # Create dummy variables for blood type and convert to int
    blood_type_dummies = pd.get_dummies(df["blood_type"], prefix="blood")
    blood_type_dummies = blood_type_dummies.astype(int)
    df = pd.concat([df, blood_type_dummies], axis=1)

    return df


def test_one_to_one_matching_equal_counts():
    """Test that 1:1 matching results in exactly equal numbers of treatment and control units.

    This test verifies the fix for a bug where 1:1 matching produced unequal counts in the
    matched dataset, even though the match pairs were correctly constructed.
    """
    # Generate data with a 90/10 treatment/control split (similar to the report_generation example)
    data = generate_synthetic_data(n_samples=1000, treatment_prob=None, seed=42)

    # Create covariates list
    covariates = [
        "age",
        "sex",
        "bmi",
        "systolic_bp",
        "cholesterol",
        "smoker",
        "blood_A",
        "blood_B",
        "blood_AB",
        "blood_O",
    ]

    # Create matching configuration with 1:1 ratio
    config = MatcherConfig(
        treatment_col="treatment",
        covariates=covariates,
        match_method="greedy",
        distance_method="propensity",
        standardize=True,
        caliper_method="propensity",
        caliper_value="auto",
        exact_match_cols=["sex"],
        estimate_propensity=True,
        random_state=42,
        ratio=1.0,  # 1:1 matching
    )

    # Perform matching
    results = run_match(data=data, config=config)

    # Get counts
    n_treat_matched = (results.matched_data["treatment"] == 1).sum()
    n_control_matched = (results.matched_data["treatment"] == 0).sum()
    match_pairs = results.get_match_pairs()
    n_pairs = len(match_pairs)
    n_unique_treat_ids = len(match_pairs["treatment_id"].unique())
    n_unique_control_ids = len(match_pairs["control_id"].unique())

    # Print diagnostic information
    print(f"Original data shape: {data.shape}")
    print(f"Original treatment count: {(data['treatment'] == 1).sum()}")
    print(f"Original control count: {(data['treatment'] == 0).sum()}")
    print(f"Matched data shape: {results.matched_data.shape}")
    print(f"Number of pairs: {n_pairs}")
    print(f"Matched treatment count: {n_treat_matched}")
    print(f"Matched control count: {n_control_matched}")
    print(f"Unique treatment IDs in pairs: {n_unique_treat_ids}")
    print(f"Unique control IDs in pairs: {n_unique_control_ids}")

    # DETAILED DEBUGGING
    print("\nDETAILED DEBUGGING")
    # Check if we have any duplicated control units in the matched data
    control_indices = results.matched_data[results.matched_data["treatment"] == 0].index
    print(f"Control indices count: {len(control_indices)}")
    print(f"Unique control indices count: {len(set(control_indices))}")

    # Check which control units are in the matched dataset but not in the match pairs
    control_ids_from_pairs = set(match_pairs["control_id"])
    control_ids_from_data = set(control_indices)
    extra_controls = control_ids_from_data - control_ids_from_pairs

    print(
        f"Extra control units in matched data but not in pairs: {len(extra_controls)}"
    )
    if extra_controls:
        print(f"Extra control IDs: {sorted(extra_controls)}")

        # Verify these are actually in the matched data
        for control_id in sorted(extra_controls):
            in_matched_data = control_id in results.matched_data.index
            print(f"Control ID {control_id} in matched_data: {in_matched_data}")

    # Verify 1:1 matching produced equal counts
    assert n_treat_matched == n_control_matched, (
        f"1:1 matching should produce equal counts, but got {n_treat_matched} treatment and {n_control_matched} control units"
    )

    # Verify the number of pairs matches the treatment and control counts
    assert n_pairs == n_treat_matched, (
        f"Number of pairs ({n_pairs}) should equal number of treatment units ({n_treat_matched})"
    )
    assert n_pairs == n_control_matched, (
        f"Number of pairs ({n_pairs}) should equal number of control units ({n_control_matched})"
    )

    # Verify unique IDs in pairs match the counts
    assert n_unique_treat_ids == n_treat_matched, (
        f"Unique treatment IDs in pairs ({n_unique_treat_ids}) should equal matched treatment count ({n_treat_matched})"
    )
    assert n_unique_control_ids == n_control_matched, (
        f"Unique control IDs in pairs ({n_unique_control_ids}) should equal matched control count ({n_control_matched})"
    )


def test_many_to_one_matching_correct_ratio():
    """Test that many-to-one matching (e.g., 2:1) results in the correct ratio of control to treatment units.

    This test verifies that when a 2:1 matching ratio is specified, the matched dataset
    contains twice as many control units as treatment units.
    """
    # Generate balanced data for more predictable many-to-one matching
    # Using a higher treatment_prob makes the test more robust by ensuring more control units available
    data = generate_synthetic_data(n_samples=1000, treatment_prob=0.25, seed=42)

    # Create covariates list
    covariates = [
        "age",
        "sex",
        "bmi",
        "systolic_bp",
        "cholesterol",
        "smoker",
        "blood_A",
        "blood_B",
        "blood_AB",
        "blood_O",
    ]

    # Create matching configuration with 2:1 ratio
    config = MatcherConfig(
        treatment_col="treatment",
        covariates=covariates,
        match_method="greedy",
        distance_method="mahalanobis",  # Using Mahalanobis for variety
        standardize=True,
        caliper_method="mahalanobis",
        caliper_value=2.0,  # Using a much larger caliper to ensure more potential matches
        exact_match_cols=None,  # Remove exact matching constraint to allow more matches
        estimate_propensity=False,
        random_state=42,
        ratio=2.0,  # 2:1 matching
    )

    # Perform matching
    results = run_match(data=data, config=config)

    # Get counts
    n_treat_matched = (results.matched_data["treatment"] == 1).sum()
    n_control_matched = (results.matched_data["treatment"] == 0).sum()
    match_pairs = results.get_match_pairs()
    n_pairs = len(match_pairs)

    # Print diagnostic information
    print(f"Original data shape: {data.shape}")
    print(f"Original treatment count: {(data['treatment'] == 1).sum()}")
    print(f"Original control count: {(data['treatment'] == 0).sum()}")
    print(f"Matched data shape: {results.matched_data.shape}")
    print(f"Number of pairs: {n_pairs}")
    print(f"Matched treatment count: {n_treat_matched}")
    print(f"Matched control count: {n_control_matched}")

    # In 2:1 matching, we expect 2 control units for each treatment unit
    expected_ratio = 2.0
    actual_ratio = n_control_matched / n_treat_matched

    # Verify ratio is approximately correct (allowing for small deviations)
    assert abs(actual_ratio - expected_ratio) < 0.1, (
        f"Expected a {expected_ratio}:1 ratio, but got {actual_ratio:.2f}:1"
    )

    # Verify number of pairs makes sense for 2:1 matching (pairs is count of treatment-control connections)
    assert n_pairs == n_control_matched, (
        f"Number of pairs ({n_pairs}) should equal number of control units ({n_control_matched}) in 2:1 matching"
    )


def test_internal_flipping_correct_restored():
    """Test that when internal flipping occurs (due to small/large group imbalance),
    the returned matched dataset is properly restored to the original orientation.

    This test creates a dataset with significantly more treatment than control units,
    which triggers internal flipping during matching. It then verifies that the
    final matched dataset has correct treatment indicators and indices.
    """
    # Generate highly imbalanced data with many more treatment than control units
    # to ensure internal flipping occurs
    data = generate_synthetic_data(n_samples=1000, treatment_prob=0.9, seed=42)

    # Verify the imbalance to ensure test conditions
    n_treat_orig = (data["treatment"] == 1).sum()
    n_control_orig = (data["treatment"] == 0).sum()
    assert n_treat_orig > n_control_orig, (
        "Test setup failure: need more treatment than control units to trigger flipping"
    )

    # Create covariates list
    covariates = [
        "age",
        "sex",
        "bmi",
        "systolic_bp",
        "cholesterol",
        "smoker",
        "blood_A",
        "blood_B",
        "blood_AB",
        "blood_O",
    ]

    # Create matching configuration with 1:1 ratio
    config = MatcherConfig(
        treatment_col="treatment",
        covariates=covariates,
        match_method="greedy",
        distance_method="propensity",
        standardize=True,
        caliper_method="propensity",
        caliper_value="auto",
        exact_match_cols=["sex"],
        estimate_propensity=True,
        random_state=42,
        ratio=1.0,  # 1:1 matching
    )

    # Perform matching
    results = run_match(data=data, config=config)

    # Get the matched data
    matched_data = results.matched_data

    # Print diagnostic information
    print(f"Original data shape: {data.shape}")
    print(f"Original treatment count: {n_treat_orig}")
    print(f"Original control count: {n_control_orig}")
    print(f"Matched data shape: {matched_data.shape}")

    # 1. Verify that treatment column is correctly preserved
    n_treat_matched = (matched_data["treatment"] == 1).sum()
    n_control_matched = (matched_data["treatment"] == 0).sum()
    print(f"Matched treatment count: {n_treat_matched}")
    print(f"Matched control count: {n_control_matched}")

    # For 1:1 matching, treatment and control counts should be equal
    assert n_treat_matched == n_control_matched, (
        f"Expected equal treatment and control counts, but got {n_treat_matched} treatment and {n_control_matched} control"
    )

    # 2. Verify match pairs have the correct structure
    match_pairs_df = results.get_match_pairs()
    assert len(match_pairs_df) == n_treat_matched, (
        f"Expected {n_treat_matched} pairs, but got {len(match_pairs_df)}"
    )

    # Ensure all treatment IDs in pairs are in the treatment group
    for treat_id in match_pairs_df["treatment_id"].unique():
        assert data.loc[treat_id, "treatment"] == 1, (
            f"Treatment ID {treat_id} is not actually in the treatment group"
        )

    # Ensure all control IDs in pairs are in the control group
    for control_id in match_pairs_df["control_id"].unique():
        assert data.loc[control_id, "treatment"] == 0, (
            f"Control ID {control_id} is not actually in the control group"
        )

    # 3. Verify that matched_indices contains only and exactly the units in match pairs
    treat_ids_in_pairs = set(match_pairs_df["treatment_id"])
    control_ids_in_pairs = set(match_pairs_df["control_id"])
    expected_matched_indices = pd.Index(
        list(treat_ids_in_pairs) + list(control_ids_in_pairs)
    )

    # Get actual matched indices from the matched data
    actual_matched_indices = matched_data.index

    # Verify the sets match
    assert set(actual_matched_indices) == set(expected_matched_indices), (
        "Matched indices do not correspond exactly to the units in match pairs"
    )

    # 4. Verify that no extra units are included in the matched dataset
    treat_indices_in_matched = matched_data[matched_data["treatment"] == 1].index
    control_indices_in_matched = matched_data[matched_data["treatment"] == 0].index

    assert set(treat_indices_in_matched) == set(treat_ids_in_pairs), (
        "Treatment units in matched data do not match those in pairs"
    )
    assert set(control_indices_in_matched) == set(control_ids_in_pairs), (
        "Control units in matched data do not match those in pairs"
    )

    print("All flipping integrity checks passed!")
