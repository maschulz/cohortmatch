"""Test suite for balance assessment in cohortmatch.metrics.balance module.

These tests validate the functionality of balance assessment metrics and tools.
"""

import numpy as np
import pandas as pd
import pytest

from cohortmatch.metrics.balance import (
    calculate_balance_index,
    calculate_balance_stats,
    calculate_overall_balance,
    calculate_rubin_rules,
    standardized_mean_difference,
    variance_ratio,
)


class TestBalanceMetrics:
    """Test suite for balance metrics functionality."""

    @pytest.fixture
    def unbalanced_data(self):
        """Create unbalanced sample data for balance assessment testing."""
        np.random.seed(42)
        n = 200

        # Create imbalanced data (treatment and control have different distributions)
        # Treatment group - higher values
        treat_size = n // 4  # 25% in treatment
        X1_treat = np.random.normal(1.0, 1.0, treat_size)
        X2_treat = np.random.normal(0.5, 1.2, treat_size)
        X3_treat = np.random.normal(-0.2, 0.8, treat_size)

        # Control group - lower values
        control_size = n - treat_size  # 75% in control
        X1_control = np.random.normal(0.0, 1.0, control_size)
        X2_control = np.random.normal(0.0, 1.0, control_size)
        X3_control = np.random.normal(0.0, 1.0, control_size)

        # Combine data
        treatment = np.concatenate([np.ones(treat_size), np.zeros(control_size)])
        X1 = np.concatenate([X1_treat, X1_control])
        X2 = np.concatenate([X2_treat, X2_control])
        X3 = np.concatenate([X3_treat, X3_control])

        # Create dataframe
        unbalanced = pd.DataFrame(
            {"treatment": treatment, "X1": X1, "X2": X2, "X3": X3}
        )

        return unbalanced

    @pytest.fixture
    def balanced_data(self):
        """Create balanced sample data for testing."""
        np.random.seed(43)
        n = 100

        # Create balanced data (treatment and control have similar distributions)
        # Treatment group
        treat_size = n // 2
        X1_treat = np.random.normal(0.0, 1.0, treat_size)
        X2_treat = np.random.normal(0.0, 1.0, treat_size)
        X3_treat = np.random.normal(0.0, 1.0, treat_size)

        # Control group
        control_size = n - treat_size
        X1_control = np.random.normal(0.0, 1.0, control_size)
        X2_control = np.random.normal(0.0, 1.0, control_size)
        X3_control = np.random.normal(0.0, 1.0, control_size)

        # Combine data
        treatment = np.concatenate([np.ones(treat_size), np.zeros(control_size)])
        X1 = np.concatenate([X1_treat, X1_control])
        X2 = np.concatenate([X2_treat, X2_control])
        X3 = np.concatenate([X3_treat, X3_control])

        # Create dataframe
        balanced = pd.DataFrame({"treatment": treatment, "X1": X1, "X2": X2, "X3": X3})

        return balanced

    def test_standardized_mean_difference(self, unbalanced_data, balanced_data):
        """Test standardized mean difference calculation."""
        # Calculate SMD for unbalanced data
        smd_unbalanced = standardized_mean_difference(
            data=unbalanced_data, var_name="X1", treatment_col="treatment"
        )

        # Calculate SMD for balanced data
        smd_balanced = standardized_mean_difference(
            data=balanced_data, var_name="X1", treatment_col="treatment"
        )

        # |SMD| should be larger for unbalanced data
        assert abs(smd_unbalanced) > 0.5
        assert abs(smd_balanced) < 0.25

    def test_standardized_mean_difference_with_matched_indices(self, unbalanced_data):
        """Test SMD calculation with matched indices."""
        # Create matched indices (subset of data)
        n_treat = (unbalanced_data["treatment"] == 1).sum()
        matched_indices = unbalanced_data.sample(n=n_treat * 2, random_state=42).index

        # Calculate SMD on the subset
        smd_matched = standardized_mean_difference(
            data=unbalanced_data.loc[matched_indices],
            var_name="X1",
            treatment_col="treatment",
        )

        # SMD for matched data should be different from full data
        smd_full = standardized_mean_difference(
            data=unbalanced_data, var_name="X1", treatment_col="treatment"
        )

        # The values should be different (matching changes balance)
        assert smd_matched != smd_full

    def test_variance_ratio(self, unbalanced_data, balanced_data):
        """Test variance ratio calculation."""
        # Calculate variance ratio for unbalanced data
        vr_unbalanced = variance_ratio(
            data=unbalanced_data, var_name="X2", treatment_col="treatment"
        )

        # Calculate variance ratio for balanced data
        vr_balanced = variance_ratio(
            data=balanced_data, var_name="X2", treatment_col="treatment"
        )

        # Variance ratio should be closer to 1 for balanced data
        # and further from 1 for unbalanced data
        assert abs(vr_balanced - 1) < abs(vr_unbalanced - 1)

        # Variance ratio should be positive
        assert vr_unbalanced > 0
        assert vr_balanced > 0

    def test_calculate_balance_stats(self, unbalanced_data, balanced_data):
        """Test calculation of balance statistics."""
        # Define covariates
        covariates = ["X1", "X2", "X3"]

        # Calculate balance statistics
        balance_df = calculate_balance_stats(
            data=unbalanced_data,
            matched_data=balanced_data,
            covariates=covariates,
            treatment_col="treatment",
        )

        # Check that the output has the expected shape and columns
        assert isinstance(balance_df, pd.DataFrame)
        assert len(balance_df) == len(covariates)
        assert set(balance_df["variable"]) == set(covariates)

        expected_columns = [
            "variable",
            "smd_before",
            "smd_after",
            "var_ratio_before",
            "var_ratio_after",
        ]
        for col in expected_columns:
            assert col in balance_df.columns

        # Check that SMD after is lower than SMD before for most variables
        # (this is what we expect when matched_data is more balanced)
        assert (balance_df["smd_after"] < balance_df["smd_before"]).mean() > 0.5

        # Check that variance ratios after are closer to 1 than before
        assert (
            abs(balance_df["var_ratio_after"] - 1)
            < abs(balance_df["var_ratio_before"] - 1)
        ).mean() > 0.5

    def test_calculate_rubin_rules(self, unbalanced_data, balanced_data):
        """Test calculation of Rubin's rules."""
        # Define covariates
        covariates = ["X1", "X2", "X3"]

        # Calculate balance statistics
        balance_df = calculate_balance_stats(
            data=unbalanced_data,
            matched_data=balanced_data,
            covariates=covariates,
            treatment_col="treatment",
        )

        # Calculate Rubin's rules
        rubin_metrics = calculate_rubin_rules(balance_df)

        # Check that the function returns the expected keys
        expected_keys = [
            "n_variables_total",
            "n_smd_small",
            "pct_smd_small",
            "n_var_ratio_good",
            "pct_var_ratio_good",
            "n_both_good",
            "pct_both_good",
        ]
        for key in expected_keys:
            assert key in rubin_metrics

        # Check that the number of variables is correct
        assert rubin_metrics["n_variables_total"] == len(covariates)

        # Check that percentages are between 0 and 100
        assert 0 <= rubin_metrics["pct_smd_small"] <= 100
        assert 0 <= rubin_metrics["pct_var_ratio_good"] <= 100
        assert 0 <= rubin_metrics["pct_both_good"] <= 100

        # Balanced data should have better Rubin's rule metrics than unbalanced data
        # We can check this by calculating Rubin's rules for unbalanced data only
        unbalanced_balance_df = calculate_balance_stats(
            data=unbalanced_data,
            matched_data=unbalanced_data,  # Same data for before and after
            covariates=covariates,
            treatment_col="treatment",
        )
        unbalanced_rubin_metrics = calculate_rubin_rules(unbalanced_balance_df)

        # Balanced data should have higher percentage of variables with SMD < 0.25
        assert (
            rubin_metrics["pct_smd_small"] > unbalanced_rubin_metrics["pct_smd_small"]
        )

    def test_calculate_balance_index(self, unbalanced_data, balanced_data):
        """Test calculation of balance index."""
        # Define covariates
        covariates = ["X1", "X2", "X3"]

        # Calculate balance statistics
        balance_df = calculate_balance_stats(
            data=unbalanced_data,
            matched_data=balanced_data,
            covariates=covariates,
            treatment_col="treatment",
        )

        # Calculate balance index
        balance_index = calculate_balance_index(balance_df)

        # Check that the function returns the expected keys
        expected_keys = [
            "mean_smd_before",
            "mean_smd_after",
            "max_smd_before",
            "max_smd_after",
            "mean_balance_ratio",
            "max_balance_ratio",
            "n_variables_improved",
            "pct_variables_improved",
            "balance_index",
        ]
        for key in expected_keys:
            assert key in balance_index

        # Check that the mean SMD after is lower than before
        assert balance_index["mean_smd_after"] < balance_index["mean_smd_before"]

        # Check that the maximum SMD after is lower than before
        assert balance_index["max_smd_after"] < balance_index["max_smd_before"]

        # Check that the balance ratio is greater than 1 (improvement)
        assert balance_index["mean_balance_ratio"] > 1

        # Check that the balance index is between 0 and 100
        assert 0 <= balance_index["balance_index"] <= 100

    def test_calculate_overall_balance(self, unbalanced_data, balanced_data):
        """Test calculation of overall balance metrics."""
        # Define covariates
        covariates = ["X1", "X2", "X3"]

        # Calculate balance statistics
        balance_df = calculate_balance_stats(
            data=unbalanced_data,
            matched_data=balanced_data,
            covariates=covariates,
            treatment_col="treatment",
        )

        # Calculate overall balance metrics
        overall_balance = calculate_overall_balance(balance_df, threshold=0.1)

        # Check that the function returns the expected keys
        expected_keys = [
            "mean_smd_before",
            "mean_smd_after",
            "max_smd_before",
            "max_smd_after",
            "prop_balanced_before",
            "prop_balanced_after",
            "mean_reduction",
            "mean_reduction_percent",
            "percent_balanced_improved",
        ]
        for key in expected_keys:
            assert key in overall_balance

        # Check that the mean SMD after is lower than before
        assert overall_balance["mean_smd_after"] < overall_balance["mean_smd_before"]

        # Check that the proportion of balanced variables after is higher than before
        assert (
            overall_balance["prop_balanced_after"]
            > overall_balance["prop_balanced_before"]
        )

        # Check that the mean reduction is positive
        assert overall_balance["mean_reduction"] > 0

        # Check that the percent improved is between 0 and 100
        assert 0 <= overall_balance["percent_balanced_improved"] <= 100

    def test_edge_cases(self):
        """Test edge cases for balance metrics."""
        # Create data with extreme cases
        data = pd.DataFrame(
            {
                "treatment": [1, 1, 0, 0],
                "same_var": [10, 10, 10, 10],  # Same value for both groups
                "zero_var_treat": [5, 5, 1, 2],  # Zero variance in treatment
                "zero_var_control": [1, 2, 7, 7],  # Zero variance in control
                "zero_var_both": [3, 3, 3, 3],  # Zero variance in both
            }
        )

        # Test SMD with same values in both groups
        smd_same = standardized_mean_difference(
            data=data, var_name="same_var", treatment_col="treatment"
        )
        assert smd_same == 0.0

        # Test SMD with zero variance in treatment
        smd_zero_treat = standardized_mean_difference(
            data=data, var_name="zero_var_treat", treatment_col="treatment"
        )
        assert np.isfinite(smd_zero_treat)

        # Variance ratio is treated/control (unfolded): zero treated variance -> 0,
        # zero control variance -> inf
        vr_zero_treat = variance_ratio(
            data=data, var_name="zero_var_treat", treatment_col="treatment"
        )
        assert vr_zero_treat == 0.0
        vr_zero_control = variance_ratio(
            data=data, var_name="zero_var_control", treatment_col="treatment"
        )
        assert vr_zero_control == np.inf

        # Test variance ratio with zero variance in both groups
        vr_zero_both = variance_ratio(
            data=data, var_name="zero_var_both", treatment_col="treatment"
        )
        assert vr_zero_both == 1.0
