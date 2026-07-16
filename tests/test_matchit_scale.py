"""At-scale quality validation against MatchIt on a 20k x 480k cohort.

Both implementations match on the identical precomputed propensity column of
a deterministic synthetic cohort (validation/make_scale_cohort.py); MatchIt's
reference values live in validation/golden_scale.json
(validation/validate_scale.R). Matching order differs between
implementations, so pairs are compared on quality — balance achieved, match
tightness, effect recovery — with counts within a small margin.

Slow (regenerates the 500k-row cohort and matches it, ~15 s); runs with
`pytest -m slow` or by default when not deselected.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from cohortmatch import match

GOLDEN_PATH = Path(__file__).parent.parent / "validation" / "golden_scale.json"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not GOLDEN_PATH.exists(),
        reason="validation/golden_scale.json not present (run validation/validate_scale.R)",
    ),
]

COVS = [f"x{i}" for i in range(8)]


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN_PATH.read_text())


@pytest.fixture(scope="module")
def result(golden):
    sys.path.insert(0, str(Path(__file__).parent.parent / "validation"))
    from make_scale_cohort import make_scale_cohort

    data = make_scale_cohort()
    with pytest.warns(UserWarning):  # caliper drops some treated units
        return match(
            data,
            treatment="treatment",
            covariates=COVS,
            propensity_scores="ps",
            caliper=0.2 * golden["sd_ps"],
            std_caliper=False,
            estimand="att",
            engine="approximate",
        ), data


class TestAtScaleQuality:
    def test_unadjusted_smd_matches_cobalt(self, result, golden):
        r, _ = result
        balance = r.balance().set_index("variable")
        for cov, expected in golden["unadjusted_smd"].items():
            assert balance.loc[cov, "smd_before"] == pytest.approx(expected, abs=1e-6)

    def test_matched_counts_comparable(self, result, golden):
        r, _ = result
        n_treated = (r.matched_data["treatment"] == 1).sum()
        assert n_treated == pytest.approx(golden["n_treated"], rel=0.02)

    def test_balance_quality_no_worse_than_matchit(self, result, golden):
        r, _ = result
        balance = r.balance().set_index("variable")
        ours = balance.loc[COVS, "smd_after"].abs()
        matchit = {k: abs(v) for k, v in golden["smd_after"].items()}
        # both must achieve excellent balance at this pool depth
        assert ours.max() < 0.02
        assert ours.mean() <= np.mean(list(matchit.values())) + 0.005

    def test_match_tightness_comparable(self, result, golden):
        r, data = result
        ps = data["ps"]
        pairs = r.pairs
        diffs = np.abs(
            ps.to_numpy()[pairs["treatment_id"].to_numpy()]
            - ps.to_numpy()[pairs["control_id"].to_numpy()]
        )
        assert diffs.mean() <= golden["mean_pair_ps_diff"] * 1.25

    def test_att_recovery(self, result, golden):
        r, _ = result
        effects = r.estimate_effects("y")
        att = effects["effect"].iloc[0]
        # both must recover the simulated effect of 2.0, and agree within
        # a few reference standard errors
        assert att == pytest.approx(2.0, abs=0.1)
        assert att == pytest.approx(golden["att"], abs=3 * golden["att_se"])
