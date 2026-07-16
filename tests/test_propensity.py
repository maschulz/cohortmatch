"""Test suite for propensity score estimation in cohortmatch.metrics.propensity module.

These tests validate the functionality of propensity score estimation and evaluation.
"""

import numpy as np
import pandas as pd
import pytest

from cohortmatch.metrics.propensity import (
    assess_common_support,
    assess_propensity_overlap,
    get_propensity_model,
)


class TestPropensityEstimation:
    """Test suite for propensity score estimation functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for propensity score estimation testing."""
        np.random.seed(42)
        n = 100

        # Create synthetic data with known propensity model
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)

        # Treatment assignment based on X1 and X2
        # Higher values of X1 and X2 increase probability of treatment
        propensity = 1 / (1 + np.exp(-(0.5 * X1 + 0.7 * X2)))
        treatment = np.random.binomial(1, propensity)

        # Create dataframe
        data = pd.DataFrame(
            {
                "treatment": treatment,
                "X1": X1,
                "X2": X2,
                "cat_var": np.random.choice(["A", "B", "C"], size=n),
                "outcome": 3
                + 2 * treatment
                + 0.5 * X1
                + 0.3 * X2
                + np.random.normal(0, 1, n),
            }
        )

        return data

    @pytest.fixture
    def propensity_scores(self, sample_data):
        """Create propensity scores for testing."""
        # Simple propensity scores based on synthetic data
        n = len(sample_data)
        ps = np.random.uniform(0.1, 0.9, n)
        # Make treatment units have slightly higher propensity scores
        ps[sample_data["treatment"] == 1] += 0.1
        ps = np.clip(ps, 0.01, 0.99)  # Keep in (0,1) range
        return ps

    @pytest.mark.parametrize("model_type", ["logistic"])
    def test_get_propensity_model(self, model_type):
        """Test getting different types of propensity models."""
        try:
            model = get_propensity_model(model_type=model_type)
            assert model is not None
            # Check that the model has a fit method
            assert hasattr(model, "fit")
            assert hasattr(model, "predict_proba")
        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_get_propensity_model_with_params(self):
        """Test getting a propensity model with custom parameters."""
        try:
            model_params = {"C": 0.5, "class_weight": "balanced"}
            model = get_propensity_model(
                model_type="logistic", model_params=model_params
            )
            assert model is not None
            # Check that the parameters were set correctly
            assert model.C == 0.5
            assert model.class_weight == "balanced"
        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_get_propensity_model_with_random_state(self):
        """Test that random_state is passed correctly to models."""
        try:
            random_state = 42
            model = get_propensity_model(
                model_type="logistic", random_state=random_state
            )
            assert model.random_state == random_state
        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_assess_common_support(self, sample_data, propensity_scores):
        """Test assessing common support between treatment and control propensity distributions."""
        treatment = sample_data["treatment"].values

        # Calculate common support metrics
        support_metrics = assess_common_support(
            propensity_scores=propensity_scores, treatment=treatment, bins=20
        )

        # Check that the function returns the expected keys
        expected_keys = [
            "common_support_min",
            "common_support_max",
            "overlap_coefficient",
            "hist_treated",
            "hist_control",
            "bin_edges",
        ]
        for key in expected_keys:
            assert key in support_metrics

        # Check that the common support range is correct
        treated_ps = propensity_scores[treatment == 1]
        control_ps = propensity_scores[treatment == 0]

        min_treated, max_treated = np.min(treated_ps), np.max(treated_ps)
        min_control, max_control = np.min(control_ps), np.max(control_ps)

        expected_min = max(min_treated, min_control)
        expected_max = min(max_treated, max_control)

        assert support_metrics["common_support_min"] == expected_min
        assert support_metrics["common_support_max"] == expected_max

        # Check that overlap coefficient is between 0 and 1
        assert 0 <= support_metrics["overlap_coefficient"] <= 1

    def test_assess_propensity_overlap(self, sample_data):
        """Test assessing propensity score overlap between treatment and control groups."""
        # Create a copy with propensity scores
        data = sample_data.copy()
        data["propensity"] = np.random.uniform(0.1, 0.9, len(data))

        # Calculate overlap metrics
        overlap_metrics = assess_propensity_overlap(
            data=data, propensity_col="propensity", treatment_col="treatment"
        )

        # Check that the function returns the expected keys
        expected_keys = [
            "ks_statistic",
            "ks_pvalue",
            "overlap_coefficient",
            "common_support_range",
            "treated_range",
            "control_range",
            "prop_in_common_support",
            "prop_treated_in_cs",
            "prop_control_in_cs",
        ]
        for key in expected_keys:
            assert key in overlap_metrics

        # Check that the KS statistic is between 0 and 1
        assert 0 <= overlap_metrics["ks_statistic"] <= 1

        # Check that the p-value is between 0 and 1
        assert 0 <= overlap_metrics["ks_pvalue"] <= 1

        # Check that the overlap coefficient is between 0 and 1
        assert 0 <= overlap_metrics["overlap_coefficient"] <= 1

        # Check that the common support range is a tuple of length 2
        assert isinstance(overlap_metrics["common_support_range"], tuple)
        assert len(overlap_metrics["common_support_range"]) == 2

        # Check that the proportion in common support is between 0 and 1
        assert 0 <= overlap_metrics["prop_in_common_support"] <= 1
