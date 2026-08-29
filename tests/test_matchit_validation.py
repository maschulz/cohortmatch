"""Golden-value validation against R's MatchIt/cobalt on the Lalonde data.

`validation/generate_golden.R` fits one full-sample logistic propensity model
and exports its scores; most tests then match on those identical scores, so
they compare the matching and the diagnostics in isolation. A separate test
checks that cohortmatch's own full-sample logistic reproduces R's glm scores,
validating the default estimation machinery end to end.

Comparison tightness varies by design:
- Unadjusted SMDs: exact (1e-6) — same data, same convention, no matching.
- Optimal matching: our exact Hungarian total may be marginally better than
  optmatch's tolerance-bounded solve, never worse.
- Nearest 1:1 and 1:2 (no competition for controls): counts exact and
  post-matching SMDs within 0.05 of MatchIt's.
- Caliper and exact designs (controls are contested, so the matching order
  decides which treated units drop): run under MatchIt's own order they
  reproduce its matched sample unit for unit; under cohortmatch's default
  order a different treated subset is retained, so only the count and the
  balance our default achieves are checked.

Requires validation/golden.json (generated in CI, or locally with R +
MatchIt + cobalt installed); tests skip when it is absent.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cohortmatch import cem, match, subclassify
from cohortmatch.datasets import load_lalonde

GOLDEN_PATH = Path(__file__).parent.parent / "validation" / "golden.json"

pytestmark = pytest.mark.skipif(
    not GOLDEN_PATH.exists(),
    reason="validation/golden.json not present (run validation/generate_golden.R)",
)


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN_PATH.read_text())


@pytest.fixture(scope="module")
def lalonde(golden):
    data = load_lalonde()
    ps = pd.Series(golden["ps"], name="ps")
    assert set(ps.index) == set(data.index)
    data = data.join(ps)
    return data


COVS = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]


def run_design(lalonde, golden, name, **kwargs):
    raw_caliper = None
    if name == "nearest_caliper":
        # MatchIt standardizes calipers by the SD of the raw distance measure;
        # pass the equivalent raw threshold to both implementations.
        raw_caliper = 0.2 * golden["sd_ps"]
        kwargs.update(caliper=raw_caliper, std_caliper=False)
    return match(
        lalonde,
        treatment="treat",
        covariates=COVS,
        propensity_scores="ps",
        estimand="att",
        engine="exact",
        **kwargs,
    )


class TestUnadjustedBalance:
    def test_smd_before_matches_cobalt_exactly(self, lalonde, golden):
        result = run_design(lalonde, golden, "nearest_1to1")
        balance = result.balance().set_index("variable")
        for cov, expected in golden["unadjusted_smd"].items():
            assert balance.loc[cov, "smd_before"] == pytest.approx(
                expected, abs=1e-6
            ), f"unadjusted SMD mismatch for {cov}"


class TestOptimal:
    def test_counts_and_total_distance(self, lalonde, golden):
        g = golden["designs"]["optimal_1to1"]
        result = run_design(lalonde, golden, "optimal_1to1", method="optimal")
        matched = result.matched_data
        assert (matched["treat"] == 1).sum() == g["n_treated"]
        assert (matched["treat"] == 0).sum() == g["n_control"]

        pairs = result.pairs
        ps = lalonde["ps"]
        total = sum(
            abs(ps[row["treatment_id"]] - ps[row["control_id"]])
            for _, row in pairs.iterrows()
        )
        # our Hungarian solve is exact; optmatch stops within a tolerance
        assert total <= g["total_ps_distance"] + 1e-9
        assert g["total_ps_distance"] - total < 0.01


class TestNearestDesigns:
    @pytest.mark.parametrize(
        "name,kwargs",
        [
            ("nearest_1to1", {}),
            ("nearest_1to2", {"ratio": 2}),
        ],
    )
    def test_uncontested_designs_reconcile_tightly(self, lalonde, golden, name, kwargs):
        g = golden["designs"][name]
        result = run_design(lalonde, golden, name, **kwargs)
        matched = result.matched_data
        assert (matched["treat"] == 1).sum() == g["n_treated"]
        assert (matched["treat"] == 0).sum() == g["n_control"]

        balance = result.balance().set_index("variable")
        for cov, expected in g["smd_after"].items():
            assert balance.loc[cov, "smd_after"] == pytest.approx(expected, abs=0.05), (
                f"{name}: post-matching SMD for {cov} deviates from MatchIt"
            )

        effects = result.estimate_effects("re78")
        assert effects["effect"].iloc[0] == pytest.approx(
            g["att"], abs=max(250, 0.5 * g["att_se"])
        )
        assert effects["standard_error"].iloc[0] == pytest.approx(g["att_se"], rel=0.25)

    @pytest.mark.parametrize(
        "name,kwargs,max_swaps,att_abs",
        [
            # NSW74 and NSW35 have identical covariates, so the caliper design
            # has to break a tied propensity score between them; the swap moves
            # the ATT by their re78 difference over 113 pairs and leaves every
            # balance figure untouched
            ("nearest_caliper", {}, 1, 100.0),
            ("nearest_exact_race", {"exact": "race"}, 0, 1e-6),
        ],
    )
    def test_contested_designs_reproduce_matchit_under_its_order(
        self, lalonde, golden, name, kwargs, max_swaps, att_abs
    ):
        # when controls are contested the matching order decides which treated
        # units drop; run MatchIt's order and the matched samples coincide
        g = golden["designs"][name]
        with pytest.warns(UserWarning):
            result = run_design(lalonde, golden, name, m_order="largest", **kwargs)
        matched = result.matched_data

        controls = sorted(str(i) for i in matched.index[matched["treat"] == 0])
        assert controls == sorted(str(i) for i in g["control_ids"])
        assert (matched["treat"] == 1).sum() == g["n_treated"]
        if "treated_ids" in g:
            ours_treated = {str(i) for i in matched.index[matched["treat"] == 1]}
            expected_treated = {str(i) for i in g["treated_ids"]}
            assert len(expected_treated - ours_treated) <= max_swaps

        balance = result.balance().set_index("variable")
        for cov, expected in g["smd_after"].items():
            assert balance.loc[cov, "smd_after"] == pytest.approx(expected, abs=1e-9), (
                f"{name}: post-matching SMD for {cov} deviates from MatchIt"
            )

        effects = result.estimate_effects("re78")
        assert effects["effect"].iloc[0] == pytest.approx(g["att"], abs=att_abs)

    @pytest.mark.parametrize(
        "name,kwargs",
        [
            ("nearest_caliper", {}),
            ("nearest_exact_race", {"exact": "race"}),
        ],
    )
    def test_default_order_matches_a_comparable_sample(
        self, lalonde, golden, name, kwargs
    ):
        # cohortmatch orders treated units by how scarce their eligible
        # controls are, MatchIt by descending propensity score, so the two
        # retain different treated subsets and their balance figures describe
        # different populations. Guard the count and the balance our own
        # default achieves; the reproduction test above owns the comparison.
        g = golden["designs"][name]
        with pytest.warns(UserWarning):
            result = run_design(lalonde, golden, name, **kwargs)
        matched = result.matched_data

        n_treated = (matched["treat"] == 1).sum()
        assert n_treated == pytest.approx(g["n_treated"], abs=5)

        balance = result.balance().set_index("variable")
        ours = balance.loc[list(g["smd_after"]), "smd_after"].abs().mean()
        matchit = np.mean([abs(v) for v in g["smd_after"].values()])
        assert ours <= matchit + 0.05, (
            f"{name}: mean |SMD| {ours:.3f} worse than MatchIt's {matchit:.3f} + 0.05"
        )

    def test_weights_sum_matches(self, lalonde, golden):
        g = golden["designs"]["nearest_1to2"]
        result = run_design(lalonde, golden, "nearest_1to2", ratio=2)
        controls = result.matched_data.index[result.matched_data["treat"] == 0]
        assert result.weights[controls].sum() == pytest.approx(
            g["sum_weights_control"], rel=1e-6
        )


class TestSubclassification:
    def test_subclass_reconciles(self, lalonde, golden):
        if "subclass_6" not in golden["designs"]:
            pytest.skip("golden.json predates the subclass design")
        g = golden["designs"]["subclass_6"]
        result = subclassify(
            lalonde,
            treatment="treat",
            covariates=COVS,
            propensity_scores="ps",
            n_subclasses=6,
            estimand="att",
        )
        matched = result.matched_data
        assert (matched["treat"] == 1).sum() == pytest.approx(g["n_treated"], rel=0.01)
        assert (matched["treat"] == 0).sum() == pytest.approx(g["n_control"], rel=0.01)

        controls = matched.index[matched["treat"] == 0]
        assert result.weights[controls].sum() == pytest.approx(
            g["sum_weights_control"], rel=0.01
        )

        balance = result.balance().set_index("variable")
        for cov, expected in g["smd_after"].items():
            assert balance.loc[cov, "smd_after"] == pytest.approx(expected, abs=0.02), (
                f"subclass: post-stratification SMD for {cov} deviates"
            )

        effects = result.estimate_effects("re78")
        assert effects["effect"].iloc[0] == pytest.approx(
            g["att"], abs=0.5 * g["att_se_hc3"]
        )
        # HC3-robust SE reconciles with R's sandwich::vcovHC(fit, "HC3")
        assert effects["se_type"].iloc[0] == "HC3-robust"
        assert effects["standard_error"].iloc[0] == pytest.approx(
            g["att_se_hc3"], rel=0.02
        )


class TestEstimatorAndDiagnosticGolden:
    """Reconcile the pieces the injected-score golden could not: the default
    estimation machinery, the auto caliper, and Rubin's B/R."""

    def test_full_sample_logistic_matches_r_glm(self, lalonde, golden):
        # cohortmatch's full-sample unregularized logistic reproduces R's glm
        # propensity scores -- validating the default estimation machinery
        # end to end (the shipped default is L2-regularized; penalty=None here
        # isolates the machinery from the regularization choice).
        import warnings

        from sklearn.linear_model import LogisticRegression

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = match(
                lalonde,
                treatment="treat",
                covariates=COVS,
                propensity_model=LogisticRegression(
                    penalty=None, solver="lbfgs", max_iter=5000
                ),
                caliper="auto",
                engine="exact",
            )
        cohort_ps = result.propensity_scores
        r_ps = pd.Series(golden["ps"]).reindex(cohort_ps.index)
        assert np.max(np.abs(cohort_ps.to_numpy() - r_ps.to_numpy())) < 5e-3

    def test_auto_caliper_matches_matchit(self, lalonde, golden):
        # cohortmatch's caliper="auto" equals MatchIt's std.caliper on the
        # logit linear predictor (0.2 x whole-sample SD of the logit PS).
        from cohortmatch.datatypes import MatcherConfig
        from cohortmatch.metrics.utils import get_caliper_for_matching

        config = MatcherConfig(
            treatment_col="treat",
            covariates=COVS,
            caliper_method="propensity",
            caliper_value="auto",
            caliper_scale=0.2,
        )
        cal = get_caliper_for_matching(
            config, propensity_scores=lalonde["ps"].to_numpy()
        )
        assert cal == pytest.approx(golden["auto_caliper"], rel=2e-3)

    def test_rubin_b_r_reconciles(self, lalonde, golden):
        # On the 1:1 design both sides match the same units, so Rubin's B and R
        # on the propensity linear predictor reconcile exactly.
        g = golden["designs"]["nearest_1to1"]
        if "rubin_B" not in g:
            pytest.skip("golden.json predates Rubin B/R")
        result = run_design(lalonde, golden, "nearest_1to1")
        rs = result.rubin_statistics
        assert rs["rubin_B"] == pytest.approx(g["rubin_B"], rel=1e-3)
        assert rs["rubin_R"] == pytest.approx(g["rubin_R"], rel=1e-3)


class TestCovariateAndEffectGolden:
    """Golden reconciliation for covariate-distance, CEM, and GLM effects —
    the features previously validated only internally."""

    def test_mahalanobis_reconciles(self, lalonde, golden):
        if "nearest_mahalanobis" not in golden["designs"]:
            pytest.skip("golden.json predates the mahalanobis design")
        g = golden["designs"]["nearest_mahalanobis"]
        result = match(
            lalonde,
            treatment="treat",
            covariates=COVS,
            distance="mahalanobis",
            estimand="att",
            random_state=0,
        )
        matched = result.matched_data
        assert (matched["treat"] == 1).sum() == g["n_treated"]
        assert (matched["treat"] == 0).sum() == g["n_control"]
        # matching order under contested controls differs; compare aggregate
        # balance quality (both use Mahalanobis, incl. the weak race balance)
        bal = result.balance().set_index("variable")
        ours = bal.loc[COVS, "smd_after"].abs().mean()
        matchit = np.mean([abs(v) for v in g["smd_after"].values()])
        assert ours <= matchit + 0.05

    def test_cem_reconciles(self, lalonde, golden):
        if "cem_fixed" not in golden["designs"]:
            pytest.skip("golden.json predates the cem design")
        g = golden["designs"]["cem_fixed"]
        result = cem(
            lalonde,
            treatment="treat",
            covariates=["age", "educ", "re74", "re75"],
            coarsening={
                "age": [25, 35, 45],
                "educ": [8, 11],
                "re74": [5000, 15000],
                "re75": [5000, 15000],
            },
            estimand="att",
        )
        matched = result.matched_data
        # identical coarsening should retain the same units (edge-inclusivity
        # can differ by a unit or two at bin boundaries)
        assert (matched["treat"] == 1).sum() == pytest.approx(g["n_treated"], abs=3)
        controls = matched.index[matched["treat"] == 0]
        assert result.weights[controls].sum() == pytest.approx(
            g["sum_weights_control"], rel=0.05
        )

    def test_logistic_effect_reconciles(self, lalonde, golden):
        # pure estimator reconciliation: same data, same (unit) weights, no
        # matching in between, so our weighted logistic OR + HC0 sandwich must
        # match R's glm + vcovHC tightly
        if "logistic_effect" not in golden["designs"]:
            pytest.skip("golden.json predates the logistic_effect design")
        from cohortmatch.metrics.treatment import estimate_treatment_effect

        g = golden["designs"]["logistic_effect"]
        d = lalonde.assign(emp=(lalonde["re78"] > 0).astype(int))
        eff = estimate_treatment_effect(
            d,
            "emp",
            "treat",
            family="logistic",
            estimand="att",
        )
        # OR matches R exactly; our GLM SE is HC0 (statsmodels' GLM sandwich is
        # HC0 for any HCx), while the R golden uses vcovHC(HC1), so they differ
        # by the finite-sample factor sqrt(n/(n-k)) ~ 0.16%
        assert eff["effect"] == pytest.approx(g["odds_ratio"], rel=1e-4)
        assert eff["standard_error"] == pytest.approx(g["se"], rel=0.01)
