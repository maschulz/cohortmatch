"""Test suite for the memory-efficient fast_greedy matching algorithm."""

import numpy as np
import pandas as pd
import pytest

from cohortmatch.datatypes import MatcherConfig
from cohortmatch.matching.fast_greedy import fast_greedy_match


class TestFastGreedyMatching:
    """Test suite for the fast greedy matching algorithm."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data suitable for testing fast_greedy_match."""
        np.random.seed(42)
        n_samples = 200
        # Create a dataset where propensity scores will have good overlap
        data = pd.DataFrame(
            {
                "treatment": np.random.binomial(1, 0.5, n_samples),
                "x1": np.random.normal(0, 1, n_samples),
                "x2": np.random.normal(5, 2, n_samples),
                "category": np.random.choice(["A", "B"], size=n_samples),
            }
        )
        # Add a propensity score correlated with covariates
        data["propensity"] = 1 / (
            1
            + np.exp(
                -(0.5 * data["x1"] + 0.1 * data["x2"] - 2 * (data["treatment"] - 0.5))
            )
        )
        data["propensity"] = np.clip(data["propensity"], 0.01, 0.99)

        return data

    def test_basic_fast_greedy_matching(self, sample_data):
        """Test that fast greedy matching runs and returns valid matches."""
        treat_mask = (sample_data["treatment"] == 1).values
        propensity_scores = sample_data["propensity"].values
        config = MatcherConfig(
            treatment_col="treatment",
            covariates=["x1", "x2"],
            caliper_method="propensity",
            caliper_value="auto",
            distance_method="euclidean",
        )

        pairs, distances = fast_greedy_match(
            sample_data, treat_mask, propensity_scores, config
        )

        assert isinstance(pairs, dict)
        assert isinstance(distances, list)
        assert len(pairs) > 0  # Should find some matches
        assert len(distances) > 0
        assert len(distances) == sum(len(v) for v in pairs.values())

    def test_candidate_pool_selection_is_correct(self):
        """
        Tests that the internal propensity pre-filtering correctly creates a candidate pool.

        This test creates a scenario where one control unit is much closer on the primary
        distance metric but should be excluded by the propensity pre-filter, while another
        control is further but should be included and therefore selected.
        """
        # T0 (ps=0.5) should be matched with C0 (ps=0.51), not C1 (ps=0.9)
        # The internal logit(ps) pre-filter will exclude C1 because the logit difference
        # is too large, even though C1 might seem closer on other covariates.
        data = pd.DataFrame(
            {
                "treatment": [1, 0, 0],
                "x1": [10, 20, 11],
                "x2": [5, 10, 6],
                "propensity": [0.5, 0.51, 0.9],
            },
            index=["T0", "C0", "C1"],
        )

        treat_mask = (data["treatment"] == 1).values
        propensity_scores = data["propensity"].values

        # Using a small caliper scale to make the pre-filter restrictive
        config = MatcherConfig(
            treatment_col="treatment",
            covariates=["x1", "x2"],
            distance_method="euclidean",  # Primary distance is Euclidean
            caliper_method=None,  # No user-defined caliper for this test
            caliper_value=None,
            caliper_scale=0.1,  # This makes the pre-filter caliper very small
        )

        pairs, _ = fast_greedy_match(data, treat_mask, propensity_scores, config)

        # The logit difference between T0(0.5) and C1(0.9) is ~2.19.
        # The std of logit(ps) across the dataset is high due to C1.
        # The pre-filter caliper will be small enough to exclude C1 but include C0.
        # Therefore, the algorithm's candidate pool for T0 should only contain C0.
        assert 0 in pairs, "Treatment unit T0 was not matched"
        assert len(pairs[0]) == 1, "T0 should have exactly one match"

        # control_indices are original indices [1, 2] corresponding to C0, C1.
        # The positional indices are 0 for C0 and 1 for C1.
        # We expect to match with C0, which has positional index 0.
        matched_control_pos_index = pairs[0][0]
        assert matched_control_pos_index == 0, "Should have matched with C0"

    def test_complex_caliper_application(self):
        """
        Tests that a user-defined caliper is correctly applied on top of the primary distance.

        This scenario has a control unit that is close on the primary distance metric ('euclidean')
        but is outside a stricter user-defined caliper ('propensity'). The algorithm must
        select another control that is further on distance but within the caliper.
        """
        data = pd.DataFrame(
            {
                "treatment": [1, 0, 0],
                "x1": [10, 11, 20],  # C0 is closer to T0 on x1/x2
                "x2": [5, 6, 15],
                "propensity": [
                    0.5,
                    0.7,
                    0.51,
                ],  # C0 has a large PS diff, C1 has a small one
            },
            index=["T0", "C0", "C1"],
        )

        treat_mask = (data["treatment"] == 1).values
        propensity_scores = data["propensity"].values

        # User caliper is on propensity, and is very strict
        config = MatcherConfig(
            treatment_col="treatment",
            covariates=["x1", "x2"],
            distance_method="euclidean",
            caliper_method="propensity",
            caliper_value=0.1,  # Propensity score difference must be < 0.1
            # Use a wide pre-filter so both controls are candidates initially
            fast_prefilter_caliper_scale=0.5,
        )

        pairs, _ = fast_greedy_match(data, treat_mask, propensity_scores, config)

        # T0 (ps=0.5) vs C0 (ps=0.7): euc_dist=1.41, ps_dist=0.2 (OUTSIDE user caliper)
        # T0 (ps=0.5) vs C1 (ps=0.51): euc_dist=14.1, ps_dist=0.01 (INSIDE user caliper)
        # The algorithm must choose C1.
        assert 0 in pairs
        assert len(pairs[0]) == 1

        # control_indices are original indices [1, 2] corresponding to C0, C1.
        # The positional indices are 0 for C0 and 1 for C1.
        # We expect to match with C1, which has positional index 1.
        matched_control_pos_index = pairs[0][0]
        assert matched_control_pos_index == 1, "Should have matched with C1"

    def test_fast_greedy_complex_scenario(self):
        """
        Tests fast_greedy_match with multiple constraints interacting:
        - Ratio matching (1:2)
        - Exact matching
        - Caliper on a different metric than primary distance
        """
        data = pd.DataFrame(
            {
                # T0 (A, ps=0.5), T1 (B, ps=0.8)
                "treatment": [1, 1, 0, 0, 0, 0, 0, 0],
                "x1": [10, 100, 11, 12, 80, 101, 102, 103],
                "x2": [20, 200, 21, 23, 150, 202, 203, 205],
                "category": ["A", "B", "A", "A", "A", "B", "B", "B"],
                "propensity": [0.5, 0.8, 0.51, 0.52, 0.6, 0.81, 0.82, 0.95],
            },
            index=["T0", "T1", "C0", "C1", "C2", "C3", "C4", "C5"],
        )
        # T0 (cat A) candidates: C0, C1, C2.
        #  - vs C0: x_dist is small, ps_dist=0.01 (ok)
        #  - vs C1: x_dist is small, ps_dist=0.02 (ok)
        #  - vs C2: x_dist is large, ps_dist=0.1 (ok)
        # Expected matches for T0: C0, C1

        # T1 (cat B) candidates: C3, C4, C5
        #  - vs C3: x_dist is small, ps_dist=0.01 (ok)
        #  - vs C4: x_dist is small, ps_dist=0.02 (ok)
        #  - vs C5: x_dist is small, ps_dist=0.15 (OUTSIDE user caliper)
        # Expected matches for T1: C3, C4

        treat_mask = (data["treatment"] == 1).values
        propensity_scores = data["propensity"].values

        config = MatcherConfig(
            treatment_col="treatment",
            covariates=["x1", "x2"],
            distance_method="euclidean",
            ratio=2.0,
            replace=False,
            exact_match_cols=["category"],
            caliper_method="propensity",
            caliper_value=0.15,  # Use a reasonable caliper for the raw propensity scale
        )

        pairs, distances = fast_greedy_match(
            data, treat_mask, propensity_scores, config
        )

        # Convert positional indices back to original labels for easier assertion
        treat_labels = data[treat_mask].index
        control_labels = data[~treat_mask].index

        result_pairs = {
            treat_labels[t_pos]: sorted([control_labels[c_pos] for c_pos in c_pos_list])
            for t_pos, c_pos_list in pairs.items()
        }

        # Assertions for T0
        assert "T0" in result_pairs
        assert len(result_pairs["T0"]) == 2
        assert result_pairs["T0"] == ["C0", "C1"]
        for cid in result_pairs["T0"]:
            assert data.loc[cid, "category"] == "A"  # Exact match
            assert (
                abs(data.loc[cid, "propensity"] - data.loc["T0", "propensity"]) <= 0.15
            )  # Caliper

        # Assertions for T1
        assert "T1" in result_pairs
        assert len(result_pairs["T1"]) == 2
        assert result_pairs["T1"] == ["C3", "C4"]
        for cid in result_pairs["T1"]:
            assert data.loc[cid, "category"] == "B"  # Exact match
            assert (
                abs(data.loc[cid, "propensity"] - data.loc["T1", "propensity"]) <= 0.15
            )  # Caliper

        # No replacement check
        all_controls = [item for sublist in result_pairs.values() for item in sublist]
        assert len(all_controls) == len(set(all_controls))

    def test_exact_matching_constraint(self, sample_data):
        """Test that exact matching constraints are respected."""
        treat_mask = (sample_data["treatment"] == 1).values
        propensity_scores = sample_data["propensity"].values

        config = MatcherConfig(
            treatment_col="treatment",
            covariates=["x1", "x2"],
            caliper_method="propensity",
            caliper_value="auto",
            distance_method="euclidean",
            exact_match_cols=["category"],
        )

        pairs, _ = fast_greedy_match(sample_data, treat_mask, propensity_scores, config)

        treat_indices = np.where(treat_mask)[0]
        control_indices = np.where(~treat_mask)[0]

        for t_pos, c_pos_list in pairs.items():
            for c_pos in c_pos_list:
                treat_cat = sample_data.iloc[treat_indices[t_pos]]["category"]
                control_cat = sample_data.iloc[control_indices[c_pos]]["category"]
                assert treat_cat == control_cat

    def test_ratio_matching(self, sample_data):
        """Test matching with a ratio > 1."""
        treat_mask = (sample_data["treatment"] == 1).values
        propensity_scores = sample_data["propensity"].values

        config = MatcherConfig(
            treatment_col="treatment",
            covariates=["x1", "x2"],
            caliper_method="propensity",
            caliper_value=2.0,  # Use a larger caliper appropriate for Euclidean distance
            caliper_scale=1.5,  # Widen the pre-filter search to find more candidates
            distance_method="euclidean",
            ratio=2.0,
        )

        pairs, _ = fast_greedy_match(sample_data, treat_mask, propensity_scores, config)

        # Check that matched treatment units have up to 2 controls
        num_full_matches = 0
        for c_pos_list in pairs.values():
            assert len(c_pos_list) <= 2
            if len(c_pos_list) == 2:
                num_full_matches += 1

        assert num_full_matches > 0  # At least some should get 2 matches

    def test_no_replacement(self, sample_data):
        """Test that without replacement, controls are used only once."""
        treat_mask = (sample_data["treatment"] == 1).values
        propensity_scores = sample_data["propensity"].values

        config = MatcherConfig(
            treatment_col="treatment",
            covariates=["x1", "x2"],
            caliper_method="propensity",
            caliper_value=0.5,
            distance_method="euclidean",
            ratio=2.0,
            replace=False,  # This is the default, but we are explicit
        )

        pairs, _ = fast_greedy_match(sample_data, treat_mask, propensity_scores, config)

        used_controls = []
        for c_pos_list in pairs.values():
            used_controls.extend(c_pos_list)

        assert len(used_controls) == len(set(used_controls))

    def test_with_replacement(self):
        """Test that with replacement, controls can be used multiple times."""
        # C0 is the best match for both T0 and T1. With replacement, both should match to it.
        data = pd.DataFrame(
            {
                "treatment": [1, 1, 0, 0],
                "x1": [10, 11, 10, 20],
                "propensity": [0.5, 0.5, 0.5, 0.8],
            },
            index=["T0", "T1", "C0", "C1"],
        )

        treat_mask = (data["treatment"] == 1).values
        propensity_scores = data["propensity"].values

        config = MatcherConfig(
            treatment_col="treatment",
            covariates=["x1"],
            distance_method="euclidean",
            replace=True,  # Enable replacement
            caliper_method=None,  # Explicitly disable user-defined caliper for clarity
            caliper_value=None,
        )

        pairs, _ = fast_greedy_match(data, treat_mask, propensity_scores, config)

        # We expect both T0 and T1 to match with C0 (positional index 0)
        t0_match = pairs.get(0)  # Positional index of T0
        t1_match = pairs.get(1)  # Positional index of T1

        assert t0_match is not None and t0_match[0] == 0
        assert t1_match is not None and t1_match[0] == 0

    def test_error_on_missing_propensity_scores(self, sample_data):
        """Test that an error is raised if propensity scores are not provided."""
        treat_mask = (sample_data["treatment"] == 1).values
        config = MatcherConfig(
            treatment_col="treatment",
            covariates=["x1", "x2"],
            caliper_method="propensity",
            caliper_value="auto",
        )

        with pytest.raises(ValueError, match="Propensity scores are required"):
            fast_greedy_match(sample_data, treat_mask, None, config)

    def test_runs_without_caliper(self, sample_data):
        """Test that fast_greedy_match runs without error when no caliper is provided."""
        treat_mask = (sample_data["treatment"] == 1).values
        propensity_scores = sample_data["propensity"].values
        config = MatcherConfig(
            treatment_col="treatment",
            covariates=["x1", "x2"],
            caliper_method=None,
            caliper_value=None,
        )
        # This should now run without error, as caliper is optional.
        # We modify the test to assert it runs.
        try:
            fast_greedy_match(sample_data, treat_mask, propensity_scores, config)
        except ValueError:
            pytest.fail("fast_greedy_match should not fail when caliper is None")
