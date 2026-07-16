"""Test suite for distance calculation functions in cohortmatch.matching.distances module.

These tests validate the functionality of various distance metrics used in matching algorithms.
"""

import numpy as np
import pytest
from scipy.special import logit
from sklearn.preprocessing import StandardScaler

from cohortmatch.matching.distances import (
    _standardize_data,
    calculate_distance_matrix,
)


class TestDistanceCalculations:
    """Test suite for distance calculation functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample treatment and control data for testing."""
        # Simple 2D data for testing distance calculations
        X_treat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_control = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5], [7.5, 8.5]])
        return X_treat, X_control

    def test_euclidean_distance(self, sample_data):
        """Test Euclidean distance calculation without standardization."""
        X_treat, X_control = sample_data

        # Calculate expected distances manually
        expected = np.zeros((3, 4))
        for i in range(3):
            for j in range(4):
                expected[i, j] = np.sqrt(np.sum((X_treat[i] - X_control[j]) ** 2))

        # Calculate distances using the function
        result = calculate_distance_matrix(
            X_treat, X_control, method="euclidean", standardize=False
        )

        # Verify results
        np.testing.assert_allclose(result, expected)
        assert result.shape == (3, 4)

    def test_euclidean_distance_with_standardization(self, sample_data):
        """Test Euclidean distance calculation with standardization."""
        X_treat, X_control = sample_data

        # Use StandardScaler just like the implementation does
        X_combined = np.vstack((X_treat, X_control))
        scaler = StandardScaler()
        scaler.fit(X_combined)

        X_treat_std = scaler.transform(X_treat)
        X_control_std = scaler.transform(X_control)

        # Calculate expected distances with standardized data
        expected = np.zeros((3, 4))
        for i in range(3):
            for j in range(4):
                expected[i, j] = np.sqrt(
                    np.sum((X_treat_std[i] - X_control_std[j]) ** 2)
                )

        # Calculate distances using the function
        result = calculate_distance_matrix(
            X_treat, X_control, method="euclidean", standardize=True
        )

        # Verify results
        np.testing.assert_allclose(result, expected, rtol=1e-8)
        assert result.shape == (3, 4)

    def test_euclidean_distance_with_weights(self, sample_data):
        """Test Euclidean distance calculation with feature weights."""
        X_treat, X_control = sample_data
        weights = np.array([0.7, 0.3])  # More weight on first feature

        # Apply weights manually
        weights_sqrt = np.sqrt(weights)
        X_treat_weighted = X_treat * weights_sqrt
        X_control_weighted = X_control * weights_sqrt

        # Calculate expected distances with weighted data
        expected = np.zeros((3, 4))
        for i in range(3):
            for j in range(4):
                expected[i, j] = np.sqrt(
                    np.sum((X_treat_weighted[i] - X_control_weighted[j]) ** 2)
                )

        # Calculate distances using the function
        result = calculate_distance_matrix(
            X_treat, X_control, method="euclidean", standardize=False, weights=weights
        )

        # Verify results
        np.testing.assert_allclose(result, expected)
        assert result.shape == (3, 4)

    def test_mahalanobis_distance(self, sample_data):
        """Test Mahalanobis distance calculation."""
        X_treat, X_control = sample_data

        # Calculate covariance matrix manually
        X_combined = np.vstack((X_treat, X_control))
        cov_matrix = np.cov(X_combined, rowvar=False)

        # Add regularization term just like the implementation does
        regularized_cov = cov_matrix + 1e-6 * np.eye(cov_matrix.shape[0])
        cov_inv = np.linalg.inv(regularized_cov)

        # Calculate expected Mahalanobis distances
        expected = np.zeros((3, 4))
        for i in range(3):
            for j in range(4):
                diff = X_treat[i] - X_control[j]
                expected[i, j] = np.sqrt(diff @ cov_inv @ diff)

        # Calculate distances using the function
        result = calculate_distance_matrix(
            X_treat, X_control, method="mahalanobis", standardize=False
        )

        # Verify results
        np.testing.assert_allclose(result, expected, rtol=1e-8)
        assert result.shape == (3, 4)

    def test_propensity_distance(self):
        """Test propensity score distance calculation."""
        # Create 1D propensity scores
        p_treat = np.array([[0.3], [0.5], [0.7]])
        p_control = np.array([[0.2], [0.4], [0.6], [0.8]])

        # Calculate expected distances - use the exact same procedure as _calculate_propensity_distances
        # but without applying logit transform
        X_treat_1d = p_treat.ravel()
        X_control_1d = p_control.ravel()

        expected = np.zeros((3, 4))
        for i in range(3):
            for j in range(4):
                expected[i, j] = np.abs(X_treat_1d[i] - X_control_1d[j])

        # Calculate distances using the function
        result = calculate_distance_matrix(
            p_treat, p_control, method="propensity", standardize=False
        )

        # Verify results
        np.testing.assert_allclose(result, expected, rtol=1e-8)
        assert result.shape == (3, 4)

    def test_logit_distance(self):
        """Test logit propensity score distance calculation."""
        # Create 1D propensity scores
        p_treat = np.array([[0.3], [0.5], [0.7]])
        p_control = np.array([[0.2], [0.4], [0.6], [0.8]])

        # Apply logit transform, using same clipping as the implementation
        X_treat_1d = np.clip(p_treat.ravel(), 0.001, 0.999)
        X_control_1d = np.clip(p_control.ravel(), 0.001, 0.999)

        X_treat_1d = logit(X_treat_1d)
        X_control_1d = logit(X_control_1d)

        # Calculate expected distances
        expected = np.zeros((3, 4))
        for i in range(3):
            for j in range(4):
                expected[i, j] = np.abs(X_treat_1d[i] - X_control_1d[j])

        # Calculate distances using the function
        result = calculate_distance_matrix(
            p_treat, p_control, method="logit", standardize=False
        )

        # Verify results
        np.testing.assert_allclose(result, expected, rtol=1e-8)
        assert result.shape == (3, 4)

    def test_input_validation(self):
        """Test input validation for distance calculation."""
        # Invalid shapes
        X_treat_1d = np.array([1.0, 2.0, 3.0])
        X_control_2d = np.array([[1.5, 2.5], [3.5, 4.5]])

        with pytest.raises(ValueError):
            calculate_distance_matrix(X_treat_1d, X_control_2d)

        # Mismatched features
        X_treat_mismatched = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        X_control = np.array([[1.5, 2.5], [3.5, 4.5]])

        with pytest.raises(ValueError):
            calculate_distance_matrix(X_treat_mismatched, X_control)

        # Invalid method
        X_treat = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_control = np.array([[1.5, 2.5], [3.5, 4.5]])

        with pytest.raises(ValueError):
            calculate_distance_matrix(X_treat, X_control, method="invalid_method")

        # Invalid weights length
        weights_wrong_length = np.array([0.5, 0.3, 0.2])

        with pytest.raises(ValueError):
            calculate_distance_matrix(X_treat, X_control, weights=weights_wrong_length)

    def test_standardization_function(self, sample_data):
        """Test the standardization function directly."""
        X_treat, X_control = sample_data

        # Use StandardScaler directly to match implementation
        X_combined = np.vstack((X_treat, X_control))
        scaler = StandardScaler()
        scaler.fit(X_combined)

        expected_X_treat_std = scaler.transform(X_treat)
        expected_X_control_std = scaler.transform(X_control)

        # Use the internal function
        X_treat_std, X_control_std = _standardize_data(X_treat, X_control)

        # Verify results
        np.testing.assert_allclose(X_treat_std, expected_X_treat_std, rtol=1e-10)
        np.testing.assert_allclose(X_control_std, expected_X_control_std, rtol=1e-10)
