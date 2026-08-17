"""Tests for how equidistant candidates are resolved.

Ties are unavoidable whenever the matching key is coarse: categorical or
binned covariates, exact matching, a propensity model over a handful of
binary predictors. The default policy ("first") is deterministic and takes
the earlier input row; "random" draws among the tied candidates under
`random_state`, so the matched set stops depending on row order.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from cohortmatch import TieBreakWarning, match


def tie_heavy(n_treat=200, n_control=2000, seed=0):
    """Pool with two binary covariates, sorted by an outcome-relevant site.

    Only four distinct covariate patterns exist among 2000 controls, so
    nearly every match is decided by a tie. Controls are stored site 0
    first, and site shifts the outcome, which is exactly the situation
    where breaking ties by row order biases the estimate.
    """
    rng = np.random.RandomState(seed)
    data = pd.DataFrame(
        {
            "treatment": np.r_[np.ones(n_treat, int), np.zeros(n_control, int)],
            "sex": np.r_[
                rng.binomial(1, 0.5, n_treat), rng.binomial(1, 0.5, n_control)
            ],
            "smoker": np.r_[
                rng.binomial(1, 0.4, n_treat), rng.binomial(1, 0.4, n_control)
            ],
            "site": np.r_[
                rng.binomial(1, 0.5, n_treat), np.repeat([0, 1], n_control // 2)
            ],
        }
    )
    data["y"] = (
        rng.normal(0, 1, len(data)) + 2.0 * data["site"] + 0.5 * data["treatment"]
    )
    return data


def continuous(n=400, seed=3):
    rng = np.random.RandomState(seed)
    data = pd.DataFrame({"age": rng.normal(50, 10, n), "bmi": rng.normal(25, 4, n)})
    data["treatment"] = rng.binomial(1, 1 / (1 + np.exp(-0.05 * (data["age"] - 50))))
    data["ps"] = np.clip(rng.uniform(0.1, 0.9, n), 0.01, 0.99)
    return data


def pair_set(result):
    return set(map(tuple, result.pairs[["treatment_id", "control_id"]].values))


def site_share(result):
    matched = result.matched_data
    return matched.loc[matched["treatment"] == 0, "site"].mean()


def run(data, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return match(
            data, treatment="treatment", covariates=["sex", "smoker"], **kwargs
        )


# engines and methods that select individual controls, each with the
# arguments that route match() down that code path
ENGINES = {
    "dense_greedy": {},
    "optimal": {"method": "optimal"},
    "approximate_window": {"engine": "approximate", "caliper": 0.5},
    "covariate_tree": {"distance": "mahalanobis", "engine": "approximate"},
    "covariate_dense": {"distance": "mahalanobis", "engine": "exact"},
}


class TestTieBreakPolicy:
    def test_default_is_row_order(self):
        """The default stays deterministic and independent of the seed.

        Scores are supplied here so the seed cannot reach the matching
        through propensity cross-fitting.
        """
        data = tie_heavy()
        data["ps"] = 0.3 + 0.2 * data["sex"] + 0.15 * data["smoker"]
        seeded = dict(propensity_scores="ps")
        assert pair_set(run(data, random_state=1, **seeded)) == pair_set(
            run(data, random_state=999, **seeded)
        )
        assert pair_set(run(data, random_state=1, **seeded)) == pair_set(
            run(data, **seeded)
        )

    def test_random_is_reproducible_under_a_seed(self):
        data = tie_heavy()
        first = run(data, tie_break="random", random_state=7)
        again = run(data, tie_break="random", random_state=7)
        assert pair_set(first) == pair_set(again)

    def test_random_seed_changes_the_matched_set(self):
        data = tie_heavy()
        assert pair_set(run(data, tie_break="random", random_state=1)) != pair_set(
            run(data, tie_break="random", random_state=2)
        )

    def test_no_ties_means_no_difference(self):
        """Without ties the policy is inert: identical pairs either way."""
        data = continuous()
        common = dict(
            treatment="treatment",
            covariates=["age", "bmi"],
            propensity_scores="ps",
            random_state=5,
        )
        assert pair_set(match(data, **common)) == pair_set(
            match(data, tie_break="random", **common)
        )

    def test_rejects_unknown_policy(self):
        with pytest.raises(ValueError, match="tie_break must be one of"):
            run(tie_heavy(), tie_break="closest")

    def test_random_without_seed_warns(self):
        with pytest.warns(TieBreakWarning, match="different matched set"):
            match(
                tie_heavy(),
                treatment="treatment",
                covariates=["sex", "smoker"],
                tie_break="random",
            )

    def test_reported_in_config_and_supplement(self):
        result = run(tie_heavy(), tie_break="random", random_state=1)
        assert result.config["tie_break"] == "random"
        assert "tie_break" in result.supplement()


class TestRowOrderBias:
    """Row order must not decide the matched set once ties are broken at random."""

    @pytest.mark.parametrize("engine", sorted(ENGINES))
    def test_first_follows_row_order(self, engine):
        data = tie_heavy()
        forward = site_share(run(data, random_state=1, **ENGINES[engine]))
        reversed_rows = site_share(
            run(data.iloc[::-1], random_state=1, **ENGINES[engine])
        )
        # the pool is half site 1, yet the matched controls come wholesale
        # from whichever end of the file was read first
        assert forward < 0.05
        assert reversed_rows > 0.95

    @pytest.mark.parametrize("engine", sorted(ENGINES))
    def test_random_tracks_the_pool(self, engine):
        data = tie_heavy()
        forward = site_share(
            run(data, tie_break="random", random_state=1, **ENGINES[engine])
        )
        reversed_rows = site_share(
            run(data.iloc[::-1], tie_break="random", random_state=1, **ENGINES[engine])
        )
        # the control pool is 50/50 on site; sampling among tied candidates
        # should land near that regardless of storage order
        assert abs(forward - 0.5) < 0.15
        assert abs(reversed_rows - 0.5) < 0.15

    def test_random_recovers_the_effect(self):
        """The row-order pick biases a naive ATT; random tie-breaking does not."""
        data = tie_heavy()

        def naive_att(result):
            matched = result.matched_data
            treated = matched.loc[matched["treatment"] == 1, "y"].mean()
            control = matched.loc[matched["treatment"] == 0, "y"].mean()
            return treated - control

        # true effect is 0.5, site adds 2.0 to the outcome
        assert naive_att(run(data, random_state=1)) > 1.2
        estimates = [
            naive_att(run(data, tie_break="random", random_state=s)) for s in range(5)
        ]
        assert abs(np.mean(estimates) - 0.5) < 0.3


class TestTieWarning:
    def test_warns_when_the_pool_is_tie_heavy(self):
        with pytest.warns(TieBreakWarning, match="share a matching key"):
            match(
                tie_heavy(),
                treatment="treatment",
                covariates=["sex", "smoker"],
                random_state=1,
            )

    def test_quiet_on_continuous_keys(self):
        data = continuous()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            match(
                data,
                treatment="treatment",
                covariates=["age", "bmi"],
                propensity_scores="ps",
                random_state=1,
            )
        assert not [w for w in caught if issubclass(w.category, TieBreakWarning)]

    def test_quiet_when_ties_are_broken_at_random(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            match(
                tie_heavy(),
                treatment="treatment",
                covariates=["sex", "smoker"],
                tie_break="random",
                random_state=1,
            )
        assert not [w for w in caught if issubclass(w.category, TieBreakWarning)]


class TestSeedAndPropensityScores:
    def test_seed_changes_matches_when_scores_are_cross_fitted(self):
        """random_state moves the estimated scores, hence the matched set.

        Documented behavior, and the reason the seed cannot be described as
        inert: it only leaves matching untouched when the scores are given.
        """
        data = continuous(n=600, seed=11)
        common = dict(treatment="treatment", covariates=["age", "bmi"])
        assert pair_set(match(data, random_state=1, **common)) != pair_set(
            match(data, random_state=2, **common)
        )

    def test_seed_is_inert_with_supplied_scores(self):
        data = continuous(n=600, seed=11)
        common = dict(
            treatment="treatment", covariates=["age", "bmi"], propensity_scores="ps"
        )
        assert pair_set(match(data, random_state=1, **common)) == pair_set(
            match(data, random_state=2, **common)
        )
