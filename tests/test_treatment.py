"""Test suite for treatment effect estimation in cohortmatch.metrics.treatment module.

These tests validate the functionality of treatment effect estimation and confidence intervals.
"""

import numpy as np
import pandas as pd
import pytest

from cohortmatch.metrics.treatment import (
    estimate_multiple_outcomes,
    estimate_treatment_effect,
)


class TestTreatmentEffect:
    """Test suite for treatment effect estimation functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for treatment effect estimation testing."""
        np.random.seed(42)
        n = 100

        # Create synthetic data with known treatment effect
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)

        # Treatment assignment
        treatment = np.random.binomial(1, 0.5, n)

        # Outcome with treatment effect = 2.0
        outcome = (
            3.0 + 2.0 * treatment + 0.5 * X1 + 0.3 * X2 + np.random.normal(0, 1, n)
        )

        # Create multiple outcomes
        outcome2 = (
            1.0 + 1.5 * treatment + 0.2 * X1 + 0.1 * X2 + np.random.normal(0, 0.5, n)
        )
        outcome3 = (
            5.0 - 0.5 * treatment + 0.1 * X1 + 0.4 * X2 + np.random.normal(0, 0.8, n)
        )

        # Create dataframe
        data = pd.DataFrame(
            {
                "treatment": treatment,
                "X1": X1,
                "X2": X2,
                "outcome1": outcome,
                "outcome2": outcome2,
                "outcome3": outcome3,
            }
        )

        return data

    @pytest.fixture
    def matched_indices(self, sample_data):
        """Create matched indices for testing."""
        np.random.seed(42)
        # Select an equal number of treatment and control units
        n_treat = (sample_data["treatment"] == 1).sum()
        n_control = (sample_data["treatment"] == 0).sum()
        n_to_match = min(n_treat, n_control)

        treat_indices = sample_data[sample_data["treatment"] == 1].index[:n_to_match]
        control_indices = sample_data[sample_data["treatment"] == 0].index[:n_to_match]

        # Combine indices
        matched_indices = pd.Index(np.concatenate([treat_indices, control_indices]))

        return matched_indices

    def test_estimate_treatment_effect_mean_difference(
        self, sample_data, matched_indices
    ):
        """Test estimating treatment effect using mean difference method."""
        # Estimate treatment effect on matched data
        result = estimate_treatment_effect(
            data=sample_data.loc[matched_indices],
            outcome="outcome1",
            treatment_col="treatment",
            method="mean_difference",
            # Skip bootstrapping for speed
        )

        # Check that the function returns the expected keys
        expected_keys = [
            "effect",
            "treat_mean",
            "control_mean",
            "t_statistic",
            "p_value",
            "method",
            "estimand",
            "n_treatment",
            "n_control",
            "n_total",
        ]
        for key in expected_keys:
            assert key in result

        # Check that the treatment effect has the expected sign
        # The true effect is 2.0, but we'll allow some sampling variation
        assert result["effect"] > 0

        # Check that sample sizes are correct
        assert (
            result["n_treatment"]
            == (sample_data.loc[matched_indices, "treatment"] == 1).sum()
        )
        assert (
            result["n_control"]
            == (sample_data.loc[matched_indices, "treatment"] == 0).sum()
        )
        assert result["n_total"] == len(matched_indices)

        # Check effect calculation
        assert np.isclose(
            result["effect"], result["treat_mean"] - result["control_mean"]
        )

    def test_estimate_multiple_outcomes(self, sample_data, matched_indices):
        """Test estimating treatment effects for multiple outcomes."""
        # Outcomes to test
        outcomes = ["outcome1", "outcome2", "outcome3"]

        # Estimate treatment effects for multiple outcomes
        results_df = estimate_multiple_outcomes(
            data=sample_data.loc[matched_indices],
            outcomes=outcomes,
            treatment_col="treatment",
            method="mean_difference",
            confidence_level=0.95,
        )

        # Check that the function returns a DataFrame with the right shape
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == len(outcomes)

        # Check that all outcomes are in the results
        assert set(results_df["outcome"].values) == set(outcomes)

        # Expected columns in the tidy output
        expected_columns = [
            "outcome",
            "effect",
            "measure",
            "standard_error",
            "p_value",
            "ci_lower",
            "ci_upper",
            "method",
            "estimand",
        ]
        for col in expected_columns:
            assert col in results_df.columns

        # Check outcome-specific effects
        # outcome1 has positive effect
        outcome1_row = results_df[results_df["outcome"] == "outcome1"].iloc[0]
        assert outcome1_row["effect"] > 0

        # outcome2 has positive effect
        outcome2_row = results_df[results_df["outcome"] == "outcome2"].iloc[0]
        assert outcome2_row["effect"] > 0

        # outcome3 has negative effect
        outcome3_row = results_df[results_df["outcome"] == "outcome3"].iloc[0]
        assert outcome3_row["effect"] < 0

    def test_regression_adjustment_method(self, sample_data, matched_indices):
        """Test regression adjustment method for treatment effect estimation."""
        pytest.importorskip("statsmodels")
        # Estimate treatment effect with regression adjustment
        result = estimate_treatment_effect(
            data=sample_data.loc[matched_indices],
            outcome="outcome1",
            treatment_col="treatment",
            method="regression_adjustment",
            covariates=["X1", "X2"],
            # Skip bootstrapping for speed
        )

        # Check that the function returns the regression-specific keys
        expected_keys = [
            "effect",
            "standard_error",
            "t_statistic",
            "p_value",
            "r_squared",
            "adj_r_squared",
            "method",
            "estimand",
            "n_treatment",
            "n_control",
            "n_total",
        ]
        for key in expected_keys:
            assert key in result

        # Check that the treatment effect has the expected sign
        assert result["effect"] > 0

        # Check that the R-squared values are between 0 and 1
        assert 0 <= result["r_squared"] <= 1
        assert 0 <= result["adj_r_squared"] <= 1

    def test_estimand_types(self, sample_data, matched_indices):
        """Test different estimand types."""
        # Test ATE (average treatment effect)
        ate_result = estimate_treatment_effect(
            data=sample_data.loc[matched_indices],
            outcome="outcome1",
            treatment_col="treatment",
            method="mean_difference",
            estimand="atc",
        )
        assert ate_result["estimand"] == "atc"

        # Test ATT (average treatment effect on the treated)
        att_result = estimate_treatment_effect(
            data=sample_data.loc[matched_indices],
            outcome="outcome1",
            treatment_col="treatment",
            method="mean_difference",
            estimand="att",
        )
        assert att_result["estimand"] == "att"

        # For matched data with 1:1 matching, ATE and ATT should be similar
        # but might not be exactly equal due to implementation details
        assert np.isclose(ate_result["effect"], att_result["effect"], rtol=0.1)

    def test_no_matched_indices(self, sample_data):
        """Test estimation without matched indices (using all data)."""
        # Estimate treatment effect on all data
        result = estimate_treatment_effect(
            data=sample_data,
            outcome="outcome1",
            treatment_col="treatment",
            method="mean_difference",
        )

        # Check sample sizes
        assert result["n_treatment"] == (sample_data["treatment"] == 1).sum()
        assert result["n_control"] == (sample_data["treatment"] == 0).sum()
        assert result["n_total"] == len(sample_data)

        # The effect should still be positive
        assert result["effect"] > 0

    def test_error_handling(self, sample_data):
        """Test error handling for invalid inputs."""
        # Test invalid method
        with pytest.raises(ValueError, match="Unknown estimation method"):
            estimate_treatment_effect(
                data=sample_data,
                outcome="outcome1",
                treatment_col="treatment",
                method="invalid_method",
            )

        # Test invalid estimand
        with pytest.raises(ValueError, match="Unknown estimand"):
            estimate_treatment_effect(
                data=sample_data,
                outcome="outcome1",
                treatment_col="treatment",
                estimand="invalid_estimand",
            )

        # Test invalid confidence level
        with pytest.raises(
            ValueError, match="Confidence level must be between 0 and 1"
        ):
            estimate_treatment_effect(
                data=sample_data,
                outcome="outcome1",
                treatment_col="treatment",
                confidence_level=1.5,
            )

        # Test regression adjustment without covariates
        with pytest.raises(ValueError, match="Covariates must be provided"):
            estimate_treatment_effect(
                data=sample_data,
                outcome="outcome1",
                treatment_col="treatment",
                method="regression_adjustment",
                covariates=None,
            )

    def test_empty_matched_data(self, sample_data):
        """Test behavior when matched data is empty."""
        with pytest.raises(ValueError):
            estimate_multiple_outcomes(
                data=sample_data.iloc[0:0],
                outcomes=["outcome1"],
                treatment_col="treatment",
            )
