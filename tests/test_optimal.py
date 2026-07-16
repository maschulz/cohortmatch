"""Test suite for optimal matching algorithm in cohortmatch.matching.optimal module.

These tests validate the functionality of the optimal matching algorithm.
"""

import numpy as np
import pandas as pd
import pytest

from cohortmatch.matching.optimal import _apply_exact_matching, optimal_match


class TestOptimalMatching:
    """Test suite for optimal matching algorithm."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create sample dataframe with treatment indicator and covariates
        data = pd.DataFrame(
            {
                "treatment": [1, 1, 1, 0, 0, 0, 0],
                "x1": [1.2, 2.3, 3.4, 1.3, 2.2, 3.1, 4.0],
                "x2": [0.5, 0.7, 0.9, 0.4, 0.6, 0.8, 1.0],
                "category": ["A", "B", "A", "A", "B", "A", "C"],
            }
        )

        # Define treatment mask
        treat_mask = data["treatment"] == 1

        # Simple distance matrix (3 treatment x 4 control)
        distance_matrix = np.array(
            [
                [0.1, 0.3, 0.5, 0.9],  # Distances from treatment unit 0 to controls
                [0.7, 0.2, 0.4, 0.8],  # Distances from treatment unit 1 to controls
                [0.6, 0.5, 0.3, 0.7],  # Distances from treatment unit 2 to controls
            ]
        )

        return data, treat_mask, distance_matrix

    def test_simple_optimal_matching(self, sample_data):
        """Test basic optimal matching without constraints."""
        data, treat_mask, distance_matrix = sample_data

        # Expected results based on Hungarian algorithm
        # The optimal assignment is:
        # - Treatment 0 -> Control 0 (dist=0.1)
        # - Treatment 1 -> Control 1 (dist=0.2)
        # - Treatment 2 -> Control 2 (dist=0.3)
        expected_pairs = {0: [0], 1: [1], 2: [2]}
        expected_distances = [0.1, 0.2, 0.3]

        # Run optimal matching
        pairs, distances = optimal_match(data, distance_matrix, treat_mask)

        # Verify results
        assert pairs == expected_pairs
        assert len(distances) == len(expected_distances)
        np.testing.assert_allclose(sorted(distances), sorted(expected_distances))

    def test_optimal_matching_with_caliper(self, sample_data):
        """Test optimal matching with caliper constraint."""
        data, treat_mask, distance_matrix = sample_data

        # With caliper=0.4, some matches should be excluded
        caliper = 0.4

        # Expected results based on Hungarian algorithm with caliper constraint
        # Only treatment units 0 and 1 can be matched with controls 0, 1 within caliper
        # Treatment 2's best match is control 2 with distance 0.3
        expected_pairs = {0: [0], 1: [1], 2: [2]}
        expected_distances = [0.1, 0.2, 0.3]

        # Run optimal matching with caliper
        distance_matrix[distance_matrix > caliper] = np.inf
        pairs, distances = optimal_match(data, distance_matrix, treat_mask)

        # Verify results
        assert pairs == expected_pairs
        assert len(distances) == len(expected_distances)
        np.testing.assert_allclose(sorted(distances), sorted(expected_distances))

    def test_optimal_matching_with_exact_matching(self, sample_data):
        """Test optimal matching with exact matching constraint."""
        data, treat_mask, _ = sample_data

        # Create a distance matrix for all units
        n_treat = np.sum(treat_mask)
        n_control = len(data) - n_treat
        distance_matrix = np.ones((n_treat, n_control))

        # Set specific distances for testing
        # Distance matrix shape is 3x4 (3 treatment, 4 control)
        distance_matrix = np.array(
            [
                [0.1, 0.4, 0.5, 0.9],  # Treat 0 (category A)
                [0.8, 0.2, 0.6, 0.7],  # Treat 1 (category B)
                [0.6, 0.5, 0.3, 0.8],  # Treat 2 (category A)
            ]
        )

        # Expected results with exact matching on 'category'
        # - Treatment 0 (cat A) can only match with controls 0 and 2 (cat A)
        # - Treatment 1 (cat B) can only match with control 1 (cat B)
        # - Treatment 2 (cat A) can only match with controls 0 and 2 (cat A)
        # Optimal solution:
        # - Treatment 0 -> Control 0 (dist=0.1)
        # - Treatment 1 -> Control 1 (dist=0.2)
        # - Treatment 2 -> Control 2 (dist=0.3)
        expected_pairs = {0: [0], 1: [1], 2: [2]}

        # Run optimal matching with exact matching
        pairs, distances = optimal_match(
            data, distance_matrix, treat_mask, exact_match_cols=["category"]
        )

        # Verify the optimal exact-matched assignment and the constraint
        assert pairs == expected_pairs
        for treat_idx, control_idxs in pairs.items():
            for control_idx in control_idxs:
                # Convert indices to original dataframe rows
                treat_row = np.where(treat_mask)[0][treat_idx]
                control_row = np.where(~treat_mask)[0][control_idx]
                # Verify exact match constraint is satisfied
                assert (
                    data.iloc[treat_row]["category"]
                    == data.iloc[control_row]["category"]
                )

    def test_optimal_matching_with_ratio_without_replacement(self, sample_data):
        """Test optimal matching with ratio > 1 without replacement."""
        data, treat_mask, distance_matrix = sample_data

        # Run optimal matching with ratio=2 and without replacement (default)
        pairs, distances = optimal_match(
            data, distance_matrix, treat_mask, ratio=2.0, replace=False
        )

        # Verify each treatment has up to 2 matches
        assert all(len(controls) <= 2 for controls in pairs.values())

        # Check that the same control is not matched multiple times
        all_controls = []
        for controls in pairs.values():
            all_controls.extend(controls)
        assert len(all_controls) == len(set(all_controls))

    def test_optimal_matching_with_ratio_with_replacement(self, sample_data):
        """Test optimal matching with ratio > 1 with replacement."""
        data, treat_mask, distance_matrix = sample_data

        # Run optimal matching with ratio=2 and with replacement
        pairs, distances = optimal_match(
            data, distance_matrix, treat_mask, ratio=2.0, replace=True
        )

        # Verify each treatment has up to 2 matches
        assert all(len(controls) <= 2 for controls in pairs.values())

        # With replacement, controls can be reused, so we don't check for uniqueness.
        # Instead, we just check that matches were made.
        all_controls = []
        for controls in pairs.values():
            all_controls.extend(controls)
        assert len(all_controls) > 0

    def test_exact_matching_function(self, sample_data):
        """Test the exact matching function directly."""
        data, treat_mask, distance_matrix = sample_data

        # Get treatment and control indices
        treat_indices = np.where(treat_mask)[0]
        control_indices = np.where(~treat_mask)[0]

        # Apply exact matching on 'category'
        result = _apply_exact_matching(
            data, treat_indices, control_indices, distance_matrix.copy(), ["category"]
        )

        # Verify results
        for i, treat_idx in enumerate(treat_indices):
            for j, control_idx in enumerate(control_indices):
                if (
                    data.iloc[treat_idx]["category"]
                    == data.iloc[control_idx]["category"]
                ):
                    assert np.isfinite(result[i, j])
                else:
                    assert np.isinf(result[i, j])

    def test_optimal_matching_complex_scenario(self):
        """Test optimal matching with a more complex scenario involving multiple constraints.

        This test creates a larger dataset with:
        - 5 treatment units, 10 control units
        - Both continuous and binary variables
        - Caliper constraint based on distance
        - Exact matching on binary variable

        The test verifies that the optimal matching correctly finds the minimum-distance matching
        that satisfies all constraints.
        """
        # Create dataset with treatment indicator, 2 continuous variables, and 1 binary variable
        np.random.seed(42)  # For reproducibility

        # Treatment group (5 units)
        treat_data = pd.DataFrame(
            {
                "treatment": [1] * 5,
                "x1": [0.1, 0.3, 0.5, 0.7, 0.9],  # Continuous var 1
                "x2": [0.2, 0.4, 0.6, 0.8, 1.0],  # Continuous var 2
                "binary": [0, 1, 0, 1, 0],  # Binary var for exact matching
            }
        )

        # Control group (10 units)
        control_data = pd.DataFrame(
            {
                "treatment": [0] * 10,
                "x1": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.05],
                "x2": [0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82, 0.92, 1.02, 0.12],
                "binary": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            }
        )

        # Combine datasets
        data = pd.concat([treat_data, control_data], ignore_index=True)
        treat_mask = data["treatment"] == 1

        # Manually calculated distance matrix based on Euclidean distance of (x1, x2)
        # Shape: (5 treatment x 10 control)
        distance_matrix = np.array(
            [
                # Controls: 0    1    2    3    4    5    6    7    8    9
                [
                    0.10,
                    0.22,
                    0.35,
                    0.45,
                    0.58,
                    0.69,
                    0.81,
                    0.92,
                    1.04,
                    0.11,
                ],  # Treatment 0 (binary=0)
                [
                    0.17,
                    0.06,
                    0.13,
                    0.22,
                    0.36,
                    0.48,
                    0.60,
                    0.72,
                    0.84,
                    0.28,
                ],  # Treatment 1 (binary=1)
                [
                    0.39,
                    0.28,
                    0.16,
                    0.05,
                    0.13,
                    0.26,
                    0.39,
                    0.50,
                    0.63,
                    0.50,
                ],  # Treatment 2 (binary=0)
                [
                    0.61,
                    0.50,
                    0.38,
                    0.28,
                    0.15,
                    0.05,
                    0.17,
                    0.29,
                    0.42,
                    0.72,
                ],  # Treatment 3 (binary=1)
                [
                    0.83,
                    0.72,
                    0.60,
                    0.50,
                    0.37,
                    0.27,
                    0.14,
                    0.06,
                    0.20,
                    0.94,
                ],  # Treatment 4 (binary=0)
            ]
        )

        # Description of expected matching with caliper=0.20 and exact matching on 'binary':
        # Treatment units with binary=0 (Treatments 0, 2, 4) can only match with controls with binary=0
        # Treatment units with binary=1 (Treatments 1, 3) can only match with controls with binary=1

        # Run optimal matching with both constraints
        distance_matrix[distance_matrix > 0.20] = np.inf
        pairs, distances = optimal_match(
            data, distance_matrix, treat_mask, exact_match_cols=["binary"]
        )

        # Verify that the matches meet the exact matching constraint
        for treat_idx, control_idxs in pairs.items():
            for control_idx in control_idxs:
                treat_row = np.where(treat_mask)[0][treat_idx]
                control_row = np.where(~treat_mask)[0][control_idx]

                # Check exact matching constraint
                assert (
                    data.iloc[treat_row]["binary"] == data.iloc[control_row]["binary"]
                )

                # Check caliper constraint
                assert distance_matrix[treat_idx, control_idx] <= 0.20

        # More detailed analysis of the expected optimal matching

        # Document the available controls for each treatment unit, considering constraints

        # Controls with binary=0 that are within caliper for each treatment

        # Controls with binary=1 that are within caliper for each treatment

        # Verify that all treatment units have been matched
        # With our specific setup, all 5 treatment units should be matched
        assert len(pairs) == 5

        # The Hungarian algorithm minimizes the total distance while respecting constraints
        # For our specific scenario, we can identify certain optimal assignments

        # Treatment 0 (binary=0) should match with control 0 (dist=0.10) or 9 (dist=0.11)
        # These are the closest controls with binary=0
        assert 0 in pairs
        assert pairs[0][0] in [0, 9]

        # Treatment 2 (binary=0) should match with control 3 (dist=0.05)
        # This is the closest control with binary=0 for treatment 2
        assert 2 in pairs
        assert pairs[2][0] == 3

        # Treatment 4 (binary=0) should match with control 7 (dist=0.06)
        # This is the closest control with binary=0 for treatment 4
        assert 4 in pairs
        assert pairs[4][0] == 7

        # Treatment 1 (binary=1) should match with control 2 (dist=0.13)
        # Control 4 might be assigned to treatment 3, so treatment 1 gets control 2
        assert 1 in pairs
        assert pairs[1][0] in [2, 4]

        # Treatment 3 (binary=1) should match with control 6 (dist=0.17) or 4 (dist=0.15)
        assert 3 in pairs
        assert pairs[3][0] in [4, 6]

        # The total distance should be minimized for the given constraints
        # We can verify this by checking that the sum of distances is close to the expected value
        # Expected optimal matching (approximate):
        # T0 -> C0 (0.10), T1 -> C2 (0.13), T2 -> C3 (0.05), T3 -> C4 (0.15), T4 -> C7 (0.06)
        # Total expected distance: 0.49
        assert 0.45 <= sum(distances) <= 0.55

        # Check that there are no duplicate controls in the matches
        all_controls = []
        for controls in pairs.values():
            all_controls.extend(controls)
        assert len(all_controls) == len(set(all_controls))

        # Verify that all distances are valid
        assert all(d <= 0.20 for d in distances)  # All distances within caliper

        # Get all binary=0 treatment and control indices
        binary_0_treat_indices = [
            i
            for i, idx in enumerate(np.where(treat_mask)[0])
            if data.iloc[idx]["binary"] == 0
        ]
        binary_0_control_indices = [
            i
            for i, idx in enumerate(np.where(~treat_mask)[0])
            if data.iloc[idx]["binary"] == 0
        ]

        # Get all binary=1 treatment and control indices
        binary_1_treat_indices = [
            i
            for i, idx in enumerate(np.where(treat_mask)[0])
            if data.iloc[idx]["binary"] == 1
        ]
        binary_1_control_indices = [
            i
            for i, idx in enumerate(np.where(~treat_mask)[0])
            if data.iloc[idx]["binary"] == 1
        ]

        # Check that we have the expected distribution of binary values
        assert len(binary_0_treat_indices) == 3  # Treatments 0, 2, 4
        assert len(binary_1_treat_indices) == 2  # Treatments 1, 3
        assert len(binary_0_control_indices) == 6  # Controls 0, 1, 3, 5, 7, 9
        assert len(binary_1_control_indices) == 4  # Controls 2, 4, 6, 8
