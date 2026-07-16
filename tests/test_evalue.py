"""Tests for E-values (VanderWeele & Ding 2017)."""

import numpy as np
import pytest

from cohortmatch import e_value


class TestEValue:
    def test_published_example(self):
        # VanderWeele & Ding's breastfeeding example: RR 3.9 -> E-value 7.26
        out = e_value(3.9)
        assert out["e_value"] == pytest.approx(7.26, abs=0.01)

    def test_known_value(self):
        assert e_value(2.0)["e_value"] == pytest.approx(2 + np.sqrt(2), abs=1e-12)

    def test_null_is_one(self):
        assert e_value(1.0)["e_value"] == 1.0

    def test_protective_effects_inverted(self):
        assert e_value(0.5)["e_value"] == pytest.approx(e_value(2.0)["e_value"])

    def test_ci_limit_closer_to_null(self):
        out = e_value(3.9, 1.8, 8.4)
        assert out["e_value_ci"] == pytest.approx(1.8 + np.sqrt(1.8 * 0.8), abs=1e-12)

    def test_ci_containing_null(self):
        assert e_value(1.5, 0.9, 2.5)["e_value_ci"] == 1.0

    def test_protective_ci_uses_upper_limit(self):
        out = e_value(0.5, 0.3, 0.8)
        assert out["e_value_ci"] == pytest.approx(e_value(1 / 0.8)["e_value"])

    def test_common_odds_ratio_sqrt(self):
        assert e_value(4.0, measure="odds_ratio")["e_value"] == pytest.approx(
            e_value(2.0)["e_value"]
        )
        assert e_value(4.0, measure="odds_ratio", rare_outcome=True)[
            "e_value"
        ] == pytest.approx(e_value(4.0)["e_value"])

    def test_hazard_ratio_transform(self):
        # rare outcome: direct; common: the VanderWeele-Ding transform
        assert e_value(2.0, measure="hazard_ratio", rare_outcome=True)[
            "e_value"
        ] == pytest.approx(e_value(2.0)["e_value"])
        common = e_value(2.0, measure="hazard_ratio")["e_value"]
        assert 1.0 < common < e_value(2.0)["e_value"]

    def test_validation(self):
        with pytest.raises(ValueError, match="positive"):
            e_value(0)
        with pytest.raises(ValueError, match="measure"):
            e_value(2.0, measure="beta")
        with pytest.raises(ValueError, match="both"):
            e_value(2.0, ci_lower=1.5)

    def test_end_to_end_with_effects(self):
        import sys

        sys.path.insert(0, "tests")
        from test_api import make_data

        from cohortmatch import match

        rng = np.random.RandomState(5)
        data = make_data(n_treat=150, n_control=450, seed=5)
        logits = -1.0 + 1.0 * data["treatment"]
        data["event"] = rng.binomial(1, 1 / (1 + np.exp(-logits)))
        result = match(
            data, treatment="treatment", covariates=["age", "bmi"], random_state=0
        )
        row = result.estimate_effects("event", family="poisson").iloc[0]
        out = e_value(row["effect"], row["ci_lower"], row["ci_upper"])
        assert out["e_value"] > 1.0
        assert 1.0 <= out["e_value_ci"] <= out["e_value"]
