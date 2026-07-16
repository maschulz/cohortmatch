"""Tests for utility functions in the cohortmatch metrics utilities.

This module tests the behavior of utility functions used across
different metrics calculations, with a focus on caliper calculation.
"""

import numpy as np
import pytest
from scipy.special import logit

from cohortmatch.datatypes import MatcherConfig
from cohortmatch.metrics.utils import get_caliper_for_matching


# Mock data for testing
@pytest.fixture
def mock_config():
    """Provides a basic MatcherConfig instance."""
    return MatcherConfig(treatment_col="treatment", covariates=["age", "income"])


class TestCaliperCalculation:
    """Tests for the get_caliper_for_matching utility function."""

    def test_direct_caliper_value(self, mock_config):
        """Test that direct numeric caliper values are passed through."""
        mock_config.caliper_value = 5
        assert get_caliper_for_matching(mock_config) == 5.0

        mock_config.caliper_value = 0.25
        assert get_caliper_for_matching(mock_config) == 0.25

    def test_none_caliper(self, mock_config):
        """Test that None caliper returns None (no caliper)."""
        mock_config.caliper_method = None
        assert get_caliper_for_matching(mock_config) is None

        mock_config.caliper_method = "propensity"
        mock_config.caliper_value = None
        assert get_caliper_for_matching(mock_config) is None

    def test_auto_caliper_propensity(self, mock_config):
        """Test auto caliper calculation for propensity score method."""
        np.random.seed(42)
        propensity_scores = np.random.beta(2, 5, 100)

        mock_config.caliper_method = "propensity"
        mock_config.caliper_value = "auto"
        mock_config.caliper_scale = 0.2

        ps_clipped = np.clip(propensity_scores, 1e-6, 1 - 1e-6)
        logit_ps = logit(ps_clipped)
        expected_caliper = 0.2 * np.std(logit_ps)

        actual_caliper = get_caliper_for_matching(
            config=mock_config, propensity_scores=propensity_scores
        )
        assert np.isclose(actual_caliper, expected_caliper)

    def test_missing_required_data(self, mock_config):
        """Test that appropriate errors are raised when required data is missing."""
        mock_config.caliper_value = "auto"

        mock_config.caliper_method = "propensity"
        with pytest.raises(ValueError, match="Propensity scores are required"):
            get_caliper_for_matching(mock_config)

        # This test is no longer valid as the mahalanobis auto caliper does not require a distance matrix
        # mock_config.caliper_method = "mahalanobis"
        # with pytest.raises(ValueError, match="Distance matrix required"):
        #     get_caliper_for_matching(mock_config)

        mock_config.caliper_method = "age"
        with pytest.raises(ValueError, match="only defined for propensity/logit"):
            get_caliper_for_matching(mock_config)

    def test_invalid_caliper_specification(self, mock_config):
        """Test that invalid caliper specifications raise appropriate errors."""
        mock_config.caliper_value = "invalid_string"
        with pytest.raises(ValueError, match="Invalid caliper_value specification"):
            get_caliper_for_matching(mock_config)
