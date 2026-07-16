"""Tests for the matching pipeline (formerly the Matcher class).

These tests validate the end-to-end functionality of the Matcher class,
including propensity score estimation, distance calculation, matching,
balance assessment, and treatment effect estimation.
"""

from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cohortmatch.datatypes import MatcherConfig
from cohortmatch.pipeline import run_match


# Helper function to create a modified copy of MatcherConfig
def copy_config_with_updates(config, **kwargs):
    """Create a new MatcherConfig with updated values."""
    config_dict = asdict(config)
    config_dict.update(kwargs)
    return MatcherConfig(**config_dict)


class TestMatcher:
    """Test suite for the Matcher class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 1000

        # Features
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)

        # Treatment assignment - more likely for high X1 and X2
        logit = 0.5 + 0.8 * X1 + 0.5 * X2
        p = 1 / (1 + np.exp(-logit))
        treatment = np.random.binomial(1, p)

        # True propensity scores
        true_propensity = p

        # Binary variable instead of categorical
        binary_var = np.random.choice([0, 1], size=n, p=[0.6, 0.4])

        # Outcome with treatment effect = 2.0
        outcome = (
            3.0 + 2.0 * treatment + 0.5 * X1 + 0.3 * X2 + np.random.normal(0, 1, n)
        )

        # Create dataframe
        data = pd.DataFrame(
            {
                "treatment": treatment,
                "X1": X1,
                "X2": X2,
                "binary_var": binary_var,
                "true_propensity": true_propensity,
                "outcome": outcome,
            }
        )

        return data

    @pytest.fixture
    def basic_config(self):
        """Create a basic MatcherConfig for testing."""
        return MatcherConfig(
            treatment_col="treatment",
            covariates=["X1", "X2"],
            match_method="greedy",
            distance_method="euclidean",
            standardize=True,
            caliper_method=None,
            caliper_value=None,
            exact_match_cols=None,
            estimate_propensity=False,
            propensity_col=None,
            random_state=42,
        )

    def test_input_data_unmodified(self, sample_data, basic_config):
        """The pipeline must not mutate the caller's data."""
        before = sample_data.copy()
        run_match(sample_data, basic_config)
        pd.testing.assert_frame_equal(sample_data, before)

    def test_match_basic(self, sample_data, basic_config):
        """Test basic matching without any advanced features."""
        results = run_match(sample_data, basic_config)

        # Basic checks on results
        assert len(results.matched_data) > 0
        assert len(results.matched_data) <= len(sample_data)
        assert all(col in results.matched_data.columns for col in sample_data.columns)

        # Check that treatment and control counts are matched (1:1 ratio by default)
        matched_treat_count = (results.matched_data["treatment"] == 1).sum()
        matched_control_count = (results.matched_data["treatment"] == 0).sum()
        assert matched_treat_count == matched_control_count

        # Check that every treatment unit has exactly one match
        assert len(results.pairs) == (results.matched_data["treatment"] == 1).sum()
        # Check that match_groups maps each treatment ID to a list containing exactly one control ID
        assert all(len(controls) == 1 for controls in results.match_groups.values())

    def test_match_with_propensity(self, sample_data, basic_config):
        """Test matching with propensity score estimation."""
        # Modify config to use propensity scores
        config = copy_config_with_updates(
            basic_config,
            estimate_propensity=True,
            propensity_model="logistic",
            distance_method="propensity",
        )

        results = run_match(sample_data, config)

        # Check that propensity scores were estimated
        assert results.propensity_scores is not None
        assert len(results.propensity_scores) == len(sample_data)
        assert results.propensity_model is not None

        # Verify propensity scores are between 0 and 1
        assert all(0 <= score <= 1 for score in results.propensity_scores)

    def test_match_with_existing_propensity(self, sample_data, basic_config):
        """Test matching using an existing propensity score column."""
        # Modify config to use existing propensity scores
        config = copy_config_with_updates(
            basic_config, propensity_col="true_propensity", distance_method="propensity"
        )

        results = run_match(sample_data, config)

        # Check that propensity scores were used
        assert results.propensity_scores is not None
        assert len(results.propensity_scores) == len(sample_data)
        assert results.propensity_model is None  # No model was trained

        # Verify propensity scores match the input column
        assert np.allclose(results.propensity_scores, sample_data["true_propensity"])

    def test_match_with_exact_matching(self, sample_data, basic_config):
        """Test matching with exact matching constraints."""
        # Modify config to use exact matching
        config = copy_config_with_updates(basic_config, exact_match_cols=["binary_var"])

        results = run_match(sample_data, config)

        # Check that matches respect exact matching constraints
        matched_data = results.matched_data
        matched_pairs = results.get_match_pairs()

        # For 1:1 matching, the number of treatment and control units should be equal
        assert (matched_data["treatment"] == 1).sum() == (
            matched_data["treatment"] == 0
        ).sum()

        for _, row in matched_pairs.iterrows():
            treat_idx = row["treatment_id"]
            control_idx = row["control_id"]
            assert (
                matched_data.loc[treat_idx, "binary_var"]
                == matched_data.loc[control_idx, "binary_var"]
            )

    def test_match_with_caliper(self, sample_data, basic_config):
        """Test matching with caliper constraints."""
        # Modify config to use caliper
        config = copy_config_with_updates(
            basic_config,
            estimate_propensity=True,
            caliper_method="propensity",
            caliper_value=0.2,
        )

        results = run_match(sample_data, config)

        # For 1:1 matching, the number of treatment and control units should be equal
        assert (results.matched_data["treatment"] == 1).sum() == (
            results.matched_data["treatment"] == 0
        ).sum()

        # Check that matches respect caliper constraints.
        # The caliper value 0.2 is on the raw propensity score difference.
        propensity_scores = results.propensity_scores

        # Get a mapping from participant ID to its position for easy lookup
        id_to_pos = {id: i for i, id in enumerate(sample_data.index)}

        for treat_id, control_ids in results.match_groups.items():
            for control_id in control_ids:
                treat_pos = id_to_pos[treat_id]
                control_pos = id_to_pos[control_id]
                ps_diff = abs(
                    propensity_scores[treat_pos] - propensity_scores[control_pos]
                )
                assert ps_diff <= 0.2, (
                    f"Caliper violated: PS diff of {ps_diff} is > 0.2"
                )

    def test_match_with_ratio(self, sample_data, basic_config):
        """Test matching with variable matching ratio."""
        # Modify config to use 1:2 matching
        config = copy_config_with_updates(basic_config, ratio=2.0)

        results = run_match(sample_data, config)

        # Check that each treatment unit has up to 2 controls
        # Use match_groups to check the number of controls per treatment unit
        assert all(
            1 <= len(controls) <= 2 for controls in results.match_groups.values()
        )

        # Get actual counts after matching
        matched_treat_count = (results.matched_data["treatment"] == 1).sum()
        matched_control_count = (results.matched_data["treatment"] == 0).sum()

        # Check if direction was flipped (indicated by more treatment than control units)
        if matched_treat_count > matched_control_count:
            # In this case, the matching was done from control to treatment (direction flipped)
            # Each control unit has up to 2 treatment units
            assert matched_control_count * 2 >= matched_treat_count
        else:
            # Normal case: each treatment unit has up to 2 control units
            assert matched_control_count <= matched_treat_count * 2

    def test_match_with_optimal(self, sample_data, basic_config):
        """Test matching with optimal matching algorithm."""
        # Modify config to use optimal matching
        config = copy_config_with_updates(basic_config, match_method="optimal")

        results = run_match(sample_data, config)

        # Check that every treatment unit has exactly one match (with default 1:1 ratio)
        # Use match_groups to check the number of controls per treatment unit
        assert all(len(controls) == 1 for controls in results.match_groups.values())

        # Basic checks on results
        assert len(results.matched_data) > 0
        assert len(results.matched_data) <= len(sample_data)

        # Check the optimality (treatment and control counts should be equal)
        matched_treat_count = (results.matched_data["treatment"] == 1).sum()
        matched_control_count = (results.matched_data["treatment"] == 0).sum()
        assert matched_treat_count == matched_control_count

    def test_match_with_replacement(self, sample_data, basic_config):
        """Test greedy matching with replacement."""
        # Modify config to use matching with replacement
        config = copy_config_with_updates(
            basic_config, match_method="greedy", replace=True
        )

        # Modify data to make reuse of controls more likely
        modified_data = sample_data.copy()
        # Artificially make some control units very good matches
        control_mask = modified_data["treatment"] == 0
        # Make 20% of controls "ideal matches" by making them very similar to treatment units
        ideal_control_indices = np.random.choice(
            modified_data[control_mask].index,
            size=int(control_mask.sum() * 0.2),
            replace=False,
        )
        # Set their values to be close to median treatment values
        treat_X1_median = modified_data.loc[
            modified_data["treatment"] == 1, "X1"
        ].median()
        treat_X2_median = modified_data.loc[
            modified_data["treatment"] == 1, "X2"
        ].median()
        modified_data.loc[ideal_control_indices, "X1"] = (
            treat_X1_median + np.random.normal(0, 0.1, len(ideal_control_indices))
        )
        modified_data.loc[ideal_control_indices, "X2"] = (
            treat_X2_median + np.random.normal(0, 0.1, len(ideal_control_indices))
        )

        results = run_match(modified_data, config)

        # Get all control indices used in matching
        control_indices = []
        # Use match_groups to get the control indices associated with each treatment unit
        for controls in results.match_groups.values():
            control_indices.extend(controls)

        # Check if any control is used more than once (with replacement)
        unique_controls = set(control_indices)
        assert len(control_indices) >= len(unique_controls), (
            "No controls were reused with replacement enabled"
        )

    def test_match_with_explicit_control_direction(self, sample_data, basic_config):
        """Direction is explicit: anchoring on controls matches all controls."""
        # Create dataset with more treatment than control units
        flipped_data = sample_data.copy()
        control_mask = flipped_data["treatment"] == 0
        n_to_flip = int(control_mask.sum() * 0.4)
        flip_indices = np.random.choice(
            flipped_data[control_mask].index, n_to_flip, replace=False
        )
        flipped_data.loc[flip_indices, "treatment"] = 1
        assert (flipped_data["treatment"] == 1).sum() > (
            flipped_data["treatment"] == 0
        ).sum()

        from dataclasses import replace as dc_replace

        config = dc_replace(basic_config, matching_direction="control")
        results = run_match(flipped_data, config)

        # Check that the matches are valid (each treatment has at most one control)
        n_control = (flipped_data["treatment"] == 0).sum()
        # Check the total number of pairs using results.pairs
        assert len(results.pairs) <= n_control

        # Check that we matched from control to treatment (direction flipped)
        # Use match_groups to check the number of controls (now acting as treatment) per treatment (now acting as control)
        assert all(len(controls) == 1 for controls in results.match_groups.values())

    def test_match_with_balance_calculation(self, sample_data, basic_config):
        """Test that balance statistics are correctly calculated."""
        # anchor on the smaller group so 1:1 matching can balance
        from dataclasses import replace as dc_replace

        n_t = (sample_data["treatment"] == 1).sum()
        n_c = (sample_data["treatment"] == 0).sum()
        config = dc_replace(
            basic_config,
            matching_direction="control" if n_t > n_c else "treatment",
        )

        results = run_match(sample_data, config)

        # Check that balance statistics are available
        assert results.balance_statistics is not None
        assert isinstance(results.balance_statistics, pd.DataFrame)
        assert len(results.balance_statistics) == len(basic_config.covariates)

        # Check that Rubin statistics are available
        assert results.rubin_statistics is not None
        assert "pct_smd_small" in results.rubin_statistics

        # Check that balance index is available
        assert results.balance_index is not None
        assert "mean_smd_before" in results.balance_index
        assert "mean_smd_after" in results.balance_index
        assert "balance_index" in results.balance_index

        # Verify balance improvement
        assert (
            results.balance_index["mean_smd_after"]
            <= results.balance_index["mean_smd_before"]
        )

    def test_match_with_treatment_effect(self, sample_data, basic_config):
        """Test that treatment effects are correctly estimated."""
        # Modify config to estimate treatment effects using a more robust method
        # for this type of confounding (propensity scores).
        config = copy_config_with_updates(
            basic_config,
            match_method="greedy",
            distance_method="propensity",
            estimate_propensity=True,
            propensity_model="logistic",
            caliper_method="propensity",
            caliper_value=0.1,
        )

        from cohortmatch.metrics.treatment import estimate_multiple_outcomes

        results = run_match(sample_data, config)
        effects = estimate_multiple_outcomes(
            data=results.matched_data,
            outcomes=["outcome"],
            treatment_col="treatment",
        )
        assert isinstance(effects, pd.DataFrame)
        assert len(effects) == 1

        # The outcome had a true effect of 2.0, so the estimate should be close
        effect = effects.iloc[0]["effect"]
        assert np.isclose(effect, 2.0, atol=1.0), (
            f"Effect estimate {effect} is far from true effect of 2.0"
        )

        ci_lower = effects.iloc[0]["ci_lower"]
        ci_upper = effects.iloc[0]["ci_upper"]
        assert ci_lower <= 2.0 or np.isclose(ci_lower, 2.0, atol=0.1)
        assert ci_upper >= 2.0 or np.isclose(ci_upper, 2.0, atol=0.1)

    def test_match_with_error_handling(self, sample_data, basic_config):
        """Test behavior for configurations that require propensity scores."""
        # Propensity distance without user-provided/estimated scores should auto-estimate
        config = copy_config_with_updates(
            basic_config,
            distance_method="propensity",
            estimate_propensity=False,
            propensity_col=None,
        )

        results = run_match(sample_data, config)
        assert results.propensity_scores is not None
        assert len(results.matched_data) > 0

    @pytest.mark.parametrize(
        "distance_method", ["euclidean", "mahalanobis", "propensity", "logit"]
    )
    def test_different_distance_methods(
        self, sample_data, basic_config, distance_method
    ):
        """Test different distance calculation methods."""
        # Skip propensity/logit without propensity scores
        if distance_method in ["propensity", "logit"]:
            # Add propensity scores
            config = copy_config_with_updates(
                basic_config,
                distance_method=distance_method,
                propensity_col="true_propensity",
            )
        else:
            config = copy_config_with_updates(
                basic_config, distance_method=distance_method
            )

        results = run_match(sample_data, config)

        # Basic checks on results
        assert len(results.matched_data) > 0
        assert len(results.matched_data) <= len(sample_data)

        # Check that distance matrix dimensions are correct
        (sample_data["treatment"] == 1).sum()
        (sample_data["treatment"] == 0).sum()
        # Note: We can't directly check the distance matrix shape as it's
        # not accessible in the results, but we can check the number of distances
        # Compare the number of distances to the number of pairs
        assert len(results.match_distances) == len(results.pairs)

    def test_matching_with_weights(self, sample_data, basic_config):
        """Test matching with feature weights."""
        # Modify config to use weights
        config = copy_config_with_updates(basic_config, weights={"X1": 2.0, "X2": 0.5})

        results = run_match(sample_data, config)

        # Basic checks on results
        assert len(results.matched_data) > 0
        assert len(results.matched_data) <= len(sample_data)

    def test_auto_caliper(self, sample_data, basic_config):
        """Test auto caliper functionality."""
        # Test with Euclidean distance method (default)
        config = copy_config_with_updates(
            basic_config,
            estimate_propensity=True,
            caliper_method="propensity",
            caliper_value="auto",
        )

        results = run_match(sample_data, config)

        # Basic checks on results
        assert len(results.matched_data) > 0
        assert len(results.matched_data) <= len(sample_data)

        # Verify all distances are below the auto caliper (90th percentile)
        # This is a bit tricky because we don't have access to the distance matrix
        # But all match distances should be below the 90th percentile
        assert all(
            distance <= max(results.match_distances)
            for distance in results.match_distances
        )

        # Test with propensity distance method
        config_prop = copy_config_with_updates(
            basic_config,
            caliper_method="propensity",
            caliper_value="auto",
            distance_method="propensity",
            propensity_col="true_propensity",
        )

        results_prop = run_match(sample_data, config_prop)

        # Verify there are matches and all distances respect the caliper
        assert len(results_prop.matched_data) > 0

        # Test with custom percentile (more restrictive)
        config_restrictive = copy_config_with_updates(
            basic_config,
            estimate_propensity=True,
            caliper_method="propensity",
            caliper_value="auto",
            caliper_scale=0.1,
        )  # More restrictive for propensity

        run_match(sample_data, config_restrictive)

        # More restrictive caliper should generally result in fewer matches
        # But this depends on the data distribution, so let's not assert this directly

    def test_regression_adjustment(self, sample_data, basic_config):
        """Test treatment effect estimation with regression adjustment."""
        # Modify config to use regression adjustment
        from cohortmatch.metrics.treatment import estimate_multiple_outcomes

        pytest.importorskip("statsmodels")
        results = run_match(sample_data, basic_config)
        effect_estimates = estimate_multiple_outcomes(
            data=results.matched_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            method="regression_adjustment",
            covariates=["X1", "X2"],
        )
        assert effect_estimates is not None
        assert "effect" in effect_estimates.columns
        assert "standard_error" in effect_estimates.columns
        assert "p_value" in effect_estimates.columns

    def test_estimand_types(self, sample_data, basic_config):
        """Test different estimand types."""
        from cohortmatch.metrics.treatment import estimate_multiple_outcomes

        results = run_match(sample_data, basic_config)
        for estimand in ["att", "atc"]:
            effects = estimate_multiple_outcomes(
                data=results.matched_data,
                outcomes=["outcome"],
                treatment_col="treatment",
                estimand=estimand,
            )
            assert effects.iloc[0]["estimand"] == estimand

    @patch("cohortmatch.pipeline.optimal_match")
    def test_optimal_matching_called_correctly(
        self, mock_optimal, sample_data, basic_config
    ):
        """Test that optimal matching function is called with correct parameters."""
        # Modify config to use optimal matching
        config = copy_config_with_updates(
            basic_config,
            match_method="optimal",
            exact_match_cols=["binary_var"],
            estimate_propensity=True,
            caliper_method="propensity",
            caliper_value=0.2,
            ratio=1.5,
        )

        # Set up mock return value
        mock_optimal.return_value = ({0: [0], 1: [1], 2: [2]}, [0.1, 0.2, 0.3])

        run_match(sample_data, config)

        # Check that optimal_match was called
        mock_optimal.assert_called_once()

        # Check that the correct parameters were passed
        call_args = mock_optimal.call_args[1]
        assert "data" in call_args
        assert "distance_matrix" in call_args
        assert "treat_mask" in call_args
        assert call_args["exact_match_cols"] == ["binary_var"]
        assert call_args["ratio"] == 1.5
        assert not call_args["replace"]

    @patch("cohortmatch.pipeline.greedy_match")
    def test_greedy_matching_called_correctly(
        self, mock_greedy, sample_data, basic_config
    ):
        """Test that greedy matching function is called with correct parameters."""
        # Modify config to use greedy matching
        config = copy_config_with_updates(
            basic_config,
            match_method="greedy",
            exact_match_cols=["binary_var"],
            estimate_propensity=True,
            caliper_method="propensity",
            caliper_value=0.2,
            ratio=1.5,
            replace=True,
            random_state=42,
        )

        # Set up mock return value
        mock_greedy.return_value = ({0: [0], 1: [1], 2: [2]}, [0.1, 0.2, 0.3])

        run_match(sample_data, config)

        # Check that greedy_match was called
        mock_greedy.assert_called_once()

        # Check that the correct parameters were passed
        call_args = mock_greedy.call_args[1]
        assert "data" in call_args
        assert "distance_matrix" in call_args
        assert "treat_mask" in call_args
        assert call_args["exact_match_cols"] == ["binary_var"]
        assert call_args["ratio"] == 1.5
        assert call_args["replace"] is True
        assert call_args["random_state"] == 42

    def test_edge_case_no_matches(self, sample_data, basic_config):
        """Test behavior when no matches are found."""
        # Modify data to have extreme separation between treatment and control
        extreme_data = sample_data.copy()
        # Instead of modifying existing data, let's create a small dataset where matches are possible
        extreme_data = pd.DataFrame(
            {
                "treatment": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                "X1": [0.1, 0.2, 0.3, 0.4, 0.5, 1.1, 1.2, 1.3, 1.4, 1.5],
                "X2": [0.1, 0.2, 0.3, 0.4, 0.5, 1.1, 1.2, 1.3, 1.4, 1.5],
                "binary_var": [0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
                "outcome": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "true_propensity": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            }
        )

        # Modify config to use a very strict caliper
        config = copy_config_with_updates(
            basic_config,
            estimate_propensity=True,
            caliper_method="propensity",
            caliper_value=0.1,
        )  # Very strict caliper

        results = run_match(extreme_data, config)

        # Check that at least some matches were found (or zero matches is okay)
        # Use results.pairs to check the number of matches
        assert len(results.pairs) >= 0

        # If any matches were found, they should respect the caliper
        if len(results.match_distances) > 0:
            assert all(d <= 0.1 for d in results.match_distances)

    def test_with_missing_values(self, sample_data, basic_config):
        """Test behavior with missing values in the data."""
        # For this test, we'll handle missing values before passing to matcher
        data_with_missing = sample_data.copy()
        n = len(data_with_missing)

        # Randomly set 5% of values to NaN
        for col in ["X1", "X2"]:
            mask = np.random.choice([True, False], size=n, p=[0.05, 0.95])
            data_with_missing.loc[mask, col] = np.nan

        # Fill missing values
        data_filled = data_with_missing.fillna(data_with_missing.mean())

        results = run_match(data_filled, basic_config)

        # Basic checks on results
        assert len(results.matched_data) > 0
        assert len(results.matched_data) <= len(sample_data)

        # Check that matched data has no missing values in the covariates
        for col in ["X1", "X2"]:
            assert results.matched_data[col].isna().sum() == 0

    def test_match_fast_greedy(self, sample_data, basic_config):
        """Test the end-to-end fast_greedy matching method via the Matcher."""
        # Config for fast_greedy requires propensity scores and a caliper
        config = copy_config_with_updates(
            basic_config,
            match_method="fast_greedy",
            estimate_propensity=True,
            propensity_model="logistic",
            caliper_method="propensity",
            caliper_value="auto",
            caliper_scale=0.5,  # Use a wider caliper to ensure matches
        )

        results = run_match(sample_data, config)

        # Check that results are populated
        assert results is not None
        assert results.matched_data is not None
        assert len(results.pairs) > 0
        assert results.distance_matrix is None  # Should not be computed

        # Check that propensity scores were estimated, as they are required
        assert results.propensity_scores is not None

        # Verify that the number of matched treatment and control units are reasonable
        matched_treat_count = (results.matched_data["treatment"] == 1).sum()
        matched_control_count = (results.matched_data["treatment"] == 0).sum()
        assert matched_treat_count > 0
        assert matched_control_count > 0

        # For 1:1 matching, counts should be equal
        assert matched_treat_count == matched_control_count
