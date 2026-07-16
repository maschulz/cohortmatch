"""
Dedicated test suite for caliper functionality in CohortMatch.

These tests verify that caliper constraints are correctly applied across different
distance methods, matching algorithms, and configurations.
"""

from dataclasses import asdict

import pandas as pd
import pytest

from cohortmatch.datatypes import MatcherConfig
from cohortmatch.pipeline import run_match


def copy_config_with_updates(config, **kwargs):
    """Create a new MatcherConfig with updated values."""
    config_dict = asdict(config)
    config_dict.update(kwargs)
    return MatcherConfig(**config_dict)


@pytest.fixture
def caliper_test_data():
    """
    Creates a tailored dataset for testing caliper functionality.
    - T0: The treatment unit.
    - C_Close_In: A control unit that is close on covariates and *inside* a standard propensity caliper.
    - C_Close_Out: A control unit that is also close on covariates but *outside* a standard propensity caliper.
    - C_Far_In: A control unit that is far on covariates but *inside* the propensity caliper.
    """
    data = pd.DataFrame(
        {
            "treatment": [1, 0, 0, 0],
            "age": [50, 51, 52, 70],  # Covariate 1
            "income": [60000, 61000, 62000, 80000],  # Covariate 2
            "propensity": [0.5, 0.55, 0.8, 0.52],  # Propensity scores
        },
        index=["T0", "C_Close_In", "C_Close_Out", "C_Far_In"],
    )
    return data


@pytest.fixture
def base_config():
    """Provides a basic MatcherConfig instance."""
    return MatcherConfig(
        treatment_col="treatment",
        covariates=["age", "income"],
        propensity_col="propensity",
        random_state=42,
    )


class TestCaliperFunctionality:
    """Test suite for caliper functionality."""

    def test_propensity_caliper_excludes_bad_match(
        self, caliper_test_data, base_config
    ):
        """
        Verify that a propensity caliper correctly excludes a match with a large
        propensity difference, even if it's close on covariates.
        """
        # C_Close_Out is closer on covariates but has a large propensity difference (0.3)
        # C_Close_In is slightly further but has a small propensity difference (0.05)
        config = copy_config_with_updates(
            base_config,
            distance_method="mahalanobis",
            caliper_method="propensity",
            caliper_value=0.1,  # Propensity difference must be < 0.1
        )

        results = run_match(caliper_test_data, config)

        # The matcher must select C_Close_In, not C_Close_Out.
        pairs = results.get_match_pairs()
        assert len(pairs) == 1
        assert pairs.iloc[0]["treatment_id"] == "T0"
        assert pairs.iloc[0]["control_id"] == "C_Close_In"

    def test_logit_caliper_behaves_similarly(self, caliper_test_data, base_config):
        """Verify that a logit-transformed propensity caliper also excludes the correct match."""
        config = copy_config_with_updates(
            base_config,
            distance_method="mahalanobis",
            caliper_method="logit",
            caliper_value=0.5,  # A reasonable caliper on the logit scale
        )

        results = run_match(caliper_test_data, config)

        # The logit difference between T0 (0.5) and C_Close_Out (0.8) is ~1.38.
        # The logit difference between T0 (0.5) and C_Close_In (0.55) is ~0.2.
        # The caliper of 0.5 will exclude C_Close_Out.
        pairs = results.get_match_pairs()
        assert len(pairs) == 1
        assert pairs.iloc[0]["treatment_id"] == "T0"
        assert pairs.iloc[0]["control_id"] == "C_Close_In"

    def test_covariate_specific_caliper(self, caliper_test_data, base_config):
        """Verify that a caliper can be applied to a specific covariate."""
        config = copy_config_with_updates(
            base_config,
            distance_method="mahalanobis",
            caliper_method="age",  # Caliper on the 'age' column
            caliper_value=5,  # Absolute difference in age must be < 5
        )

        with pytest.raises(ValueError, match="Per-covariate calipers"):
            run_match(caliper_test_data, config)

    def test_auto_caliper_with_mahalanobis_rejected(
        self, caliper_test_data, base_config
    ):
        """'auto' calipers are only defined for propensity/logit metrics."""
        config = copy_config_with_updates(
            base_config,
            distance_method="mahalanobis",
            caliper_method="mahalanobis",
            caliper_value="auto",
        )
        with pytest.raises(ValueError, match="only defined for propensity/logit"):
            run_match(caliper_test_data, config)

    def test_strict_caliper_finds_no_matches(self, caliper_test_data, base_config):
        """Verify that a very strict caliper correctly results in no matches."""
        config = copy_config_with_updates(
            base_config,
            distance_method="mahalanobis",
            caliper_method="propensity",
            caliper_value=0.01,  # No pairs have a propensity diff this small
        )

        results = run_match(caliper_test_data, config)

        assert len(results.get_match_pairs()) == 0
        assert len(results.matched_data) == 0

    @pytest.mark.parametrize("match_method", ["greedy", "optimal", "fast_greedy"])
    def test_caliper_works_across_all_match_methods(
        self, caliper_test_data, base_config, match_method
    ):
        """Ensure the caliper logic is correctly applied for all matching algorithms."""
        config = copy_config_with_updates(
            base_config,
            match_method=match_method,
            distance_method="mahalanobis",
            caliper_method="propensity",
            caliper_value=0.1,
        )

        results = run_match(caliper_test_data, config)

        # All three methods should correctly exclude C_Close_Out and select C_Close_In.
        pairs = results.get_match_pairs()
        assert len(pairs) == 1
        assert pairs.iloc[0]["treatment_id"] == "T0"
        assert pairs.iloc[0]["control_id"] == "C_Close_In"

    def test_caliper_on_distance_metric_itself(self, caliper_test_data, base_config):
        """
        Verify that applying a caliper on the same metric as the distance works correctly.
        """
        # The Mahalanobis distance between T0 and C_Far_In is large.
        # A caliper on Mahalanobis distance should exclude it.
        config = copy_config_with_updates(
            base_config,
            distance_method="mahalanobis",
            caliper_method="mahalanobis",
            caliper_value=1.0,  # This value will exclude C_Far_In
        )

        results = run_match(caliper_test_data, config)

        # The match must not be C_Far_In
        pairs = results.get_match_pairs()
        assert len(pairs) == 1
        assert pairs.iloc[0]["control_id"] != "C_Far_In"

        # The closest valid match is C_Close_In
        assert pairs.iloc[0]["control_id"] == "C_Close_In"
        assert results.match_distances[0] < 1.0
