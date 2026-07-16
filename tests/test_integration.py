"""Integration tests for cohortmatch matching algorithms.

These tests validate the integration between distance calculation and matching algorithms.
"""

import numpy as np
import pandas as pd
import pytest

from cohortmatch.matching.distances import calculate_distance_matrix
from cohortmatch.matching.greedy import greedy_match
from cohortmatch.matching.optimal import optimal_match


class TestMatchingIntegration:
    """Integration tests for matching algorithms."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create a more realistic dataset
        np.random.seed(42)
        n_treat = 10
        n_control = 20

        # Generate covariates
        X_treat = np.random.normal(loc=0.0, scale=1.0, size=(n_treat, 3))
        X_control = np.random.normal(loc=0.2, scale=1.1, size=(n_control, 3))

        # Categorical variables
        cat_treat = np.random.choice(["A", "B", "C"], size=n_treat)
        cat_control = np.random.choice(["A", "B", "C"], size=n_control)

        # Create DataFrame
        treat_df = pd.DataFrame(X_treat, columns=["x1", "x2", "x3"])
        treat_df["category"] = cat_treat
        treat_df["treatment"] = 1

        control_df = pd.DataFrame(X_control, columns=["x1", "x2", "x3"])
        control_df["category"] = cat_control
        control_df["treatment"] = 0

        # Combine
        data = pd.concat([treat_df, control_df], ignore_index=True)

        # Treatment mask
        treat_mask = data["treatment"] == 1

        return data, treat_mask

    def test_euclidean_optimal_match(self, sample_data):
        """Test integration of euclidean distance with optimal matching."""
        data, treat_mask = sample_data

        # Get feature columns (exclude treatment and category)
        feature_cols = ["x1", "x2", "x3"]

        # Extract treatment and control features
        X_treat = data.loc[treat_mask, feature_cols].values
        X_control = data.loc[~treat_mask, feature_cols].values

        # Calculate distance matrix
        distance_matrix = calculate_distance_matrix(
            X_treat, X_control, method="euclidean", standardize=True
        )

        # Verify distance matrix shape
        n_treat = np.sum(treat_mask)
        n_control = np.sum(~treat_mask)
        assert distance_matrix.shape == (n_treat, n_control)

        # Run optimal matching
        pairs, distances = optimal_match(data, distance_matrix, treat_mask)

        # Verify basic matching properties
        assert len(pairs) <= n_treat
        # Keys may be numpy.int64 instead of Python int
        assert all(isinstance(key, (int, np.integer)) for key in pairs.keys())
        assert all(isinstance(val, list) for val in pairs.values())
        assert all(
            isinstance(control_idx, (int, np.integer))
            for controls in pairs.values()
            for control_idx in controls
        )

        # Verify distances match the distance matrix
        for treat_idx, control_idxs in pairs.items():
            for control_idx in control_idxs:
                actual_dist = distance_matrix[treat_idx, control_idx]
                assert actual_dist in distances

    def test_euclidean_greedy_match(self, sample_data):
        """Test integration of euclidean distance with greedy matching."""
        data, treat_mask = sample_data

        # Get feature columns (exclude treatment and category)
        feature_cols = ["x1", "x2", "x3"]

        # Extract treatment and control features
        X_treat = data.loc[treat_mask, feature_cols].values
        X_control = data.loc[~treat_mask, feature_cols].values

        # Calculate distance matrix
        distance_matrix = calculate_distance_matrix(
            X_treat, X_control, method="euclidean", standardize=True
        )

        # Run greedy matching
        pairs, distances = greedy_match(
            data, distance_matrix, treat_mask, random_state=42
        )

        # Verify basic matching properties
        n_treat = np.sum(treat_mask)
        assert len(pairs) <= n_treat

        # Verify distances match the distance matrix
        for treat_idx, control_idxs in pairs.items():
            for control_idx in control_idxs:
                actual_dist = distance_matrix[treat_idx, control_idx]
                assert actual_dist in distances

    def test_exact_matching_integration(self, sample_data):
        """Test integration of distance calculation with exact matching."""
        data, treat_mask = sample_data

        # Get feature columns (exclude treatment and category)
        feature_cols = ["x1", "x2", "x3"]

        # Extract treatment and control features
        X_treat = data.loc[treat_mask, feature_cols].values
        X_control = data.loc[~treat_mask, feature_cols].values

        # Calculate distance matrix
        distance_matrix = calculate_distance_matrix(
            X_treat, X_control, method="euclidean", standardize=True
        )

        # Run optimal matching with exact matching on 'category'
        pairs_optimal, _ = optimal_match(
            data, distance_matrix, treat_mask, exact_match_cols=["category"]
        )

        # Run greedy matching with exact matching on 'category'
        pairs_greedy, _ = greedy_match(
            data,
            distance_matrix,
            treat_mask,
            exact_match_cols=["category"],
            random_state=42,
        )

        # Verify exact matching constraint for optimal matching
        for treat_idx, control_idxs in pairs_optimal.items():
            for control_idx in control_idxs:
                # Convert indices to original dataframe rows
                treat_row = np.where(treat_mask)[0][treat_idx]
                control_row = np.where(~treat_mask)[0][control_idx]
                # Verify exact match constraint is satisfied
                assert (
                    data.iloc[treat_row]["category"]
                    == data.iloc[control_row]["category"]
                )

        # Verify exact matching constraint for greedy matching
        for treat_idx, control_idxs in pairs_greedy.items():
            for control_idx in control_idxs:
                # Convert indices to original dataframe rows
                treat_row = np.where(treat_mask)[0][treat_idx]
                control_row = np.where(~treat_mask)[0][control_idx]
                # Verify exact match constraint is satisfied
                assert (
                    data.iloc[treat_row]["category"]
                    == data.iloc[control_row]["category"]
                )

    def test_ratio_matching_integration(self, sample_data):
        """Test integration with ratio matching."""
        data, treat_mask = sample_data

        # Get feature columns (exclude treatment and category)
        feature_cols = ["x1", "x2", "x3"]

        # Extract treatment and control features
        X_treat = data.loc[treat_mask, feature_cols].values
        X_control = data.loc[~treat_mask, feature_cols].values

        # Calculate distance matrix
        distance_matrix = calculate_distance_matrix(
            X_treat, X_control, method="euclidean", standardize=True
        )

        # Run optimal matching with ratio=2
        pairs_optimal, _ = optimal_match(data, distance_matrix, treat_mask, ratio=2.0)

        # Run greedy matching with ratio=2
        pairs_greedy, _ = greedy_match(
            data, distance_matrix, treat_mask, ratio=2.0, random_state=42
        )

        # Verify ratio constraint for optimal matching
        assert all(len(controls) <= 2 for controls in pairs_optimal.values())

        # Verify ratio constraint for greedy matching
        assert all(len(controls) <= 2 for controls in pairs_greedy.values())

        # Verify controls are unique in greedy matching (without replacement)
        greedy_controls = []
        for controls in pairs_greedy.values():
            greedy_controls.extend(controls)
        assert len(greedy_controls) == len(set(greedy_controls))

        # For optimal matching with ratio > 1 and default (no replacement), controls should be unique
        optimal_controls = []
        for controls in pairs_optimal.values():
            optimal_controls.extend(controls)
        assert len(optimal_controls) == len(set(optimal_controls))

    def test_propensity_score_matching(self, sample_data):
        """Test integration with propensity score matching."""
        data, treat_mask = sample_data

        # For testing purposes, create synthetic propensity scores
        np.random.seed(42)
        n_treat = np.sum(treat_mask)
        n_control = np.sum(~treat_mask)

        # Generate propensity scores
        p_treat = np.random.uniform(0.4, 0.8, size=(n_treat, 1))
        p_control = np.random.uniform(0.2, 0.6, size=(n_control, 1))

        # Calculate distance matrix using propensity method
        distance_matrix = calculate_distance_matrix(
            p_treat, p_control, method="propensity", standardize=False
        )

        # Run optimal matching
        pairs_optimal, distances_optimal = optimal_match(
            data, distance_matrix, treat_mask
        )

        # Run greedy matching
        pairs_greedy, distances_greedy = greedy_match(
            data, distance_matrix, treat_mask, random_state=42
        )

        # Verify distances match the distance matrix
        for treat_idx, control_idxs in pairs_optimal.items():
            for control_idx in control_idxs:
                actual_dist = distance_matrix[treat_idx, control_idx]
                assert (
                    abs(
                        actual_dist
                        - next(
                            d for d in distances_optimal if np.isclose(d, actual_dist)
                        )
                    )
                    < 1e-10
                )

        for treat_idx, control_idxs in pairs_greedy.items():
            for control_idx in control_idxs:
                actual_dist = distance_matrix[treat_idx, control_idx]
                assert (
                    abs(
                        actual_dist
                        - next(
                            d for d in distances_greedy if np.isclose(d, actual_dist)
                        )
                    )
                    < 1e-10
                )
