"""Tests for risk-set (incidence-density) matching."""

import numpy as np
import pandas as pd
import pytest

from cohortmatch import match_risk_set


def make_cohort(n=2000, seed=0, exposure_effect=0.0):
    rng = np.random.RandomState(seed)
    age = rng.normal(55, 8, n)
    sex = rng.binomial(1, 0.5, n)
    exposure = rng.binomial(1, 0.3, n)
    hazard = 0.02 * np.exp(exposure_effect * exposure + 0.02 * (age - 55))
    event_time = rng.exponential(1 / hazard)
    censor_time = rng.uniform(0, 40, n)
    time = np.minimum(event_time, censor_time)
    case = (event_time <= censor_time).astype(int)
    return pd.DataFrame(
        {"time": time, "case": case, "age": age, "sex": sex, "exposure": exposure},
        index=[f"u{i}" for i in range(n)],
    )


class TestRiskSetSampling:
    def test_controls_at_risk_at_index_time(self):
        data = make_cohort()
        result = match_risk_set(
            data, event_time="time", event="case", ratio=2, random_state=0
        )
        sets = result.sets
        controls = sets[sets["case"] == 0]
        times = data["time"]
        for _, row in controls.iterrows():
            assert times[row["unit_id"]] > row["index_time"]

    def test_future_cases_serve_as_controls(self):
        data = make_cohort()
        result = match_risk_set(
            data, event_time="time", event="case", ratio=2, random_state=0
        )
        sets = result.sets
        control_ids = set(sets.loc[sets["case"] == 0, "unit_id"])
        case_ids = set(data.index[data["case"] == 1])
        assert control_ids & case_ids, "future cases must be eligible controls"

    def test_exact_and_caliper_respected(self):
        data = make_cohort()
        result = match_risk_set(
            data,
            event_time="time",
            event="case",
            exact="sex",
            covariate_calipers={"age": 3.0},
            covariates=["age"],
            ratio=1,
            random_state=0,
        )
        sets = result.matched_data
        for _, grp in sets.groupby("set_id"):
            case_row = grp[grp["case"] == 1].iloc[0]
            for _, ctrl in grp[grp["case"] == 0].iterrows():
                assert ctrl["sex"] == case_row["sex"]
                assert abs(ctrl["age"] - case_row["age"]) <= 3.0

    def test_no_reuse_when_disabled(self):
        data = make_cohort()
        result = match_risk_set(
            data,
            event_time="time",
            event="case",
            ratio=2,
            replace=False,
            random_state=0,
        )
        controls = result.sets.loc[result.sets["case"] == 0, "unit_id"]
        assert controls.is_unique

    def test_reuse_happens_by_default(self):
        data = make_cohort(n=400)  # small pool forces reuse
        result = match_risk_set(
            data, event_time="time", event="case", ratio=4, random_state=0
        )
        controls = result.sets.loc[result.sets["case"] == 0, "unit_id"]
        assert controls.value_counts().max() > 1

    def test_deterministic_under_seed(self):
        data = make_cohort()
        a = match_risk_set(
            data, event_time="time", event="case", ratio=2, random_state=7
        ).sets
        b = match_risk_set(
            data, event_time="time", event="case", ratio=2, random_state=7
        ).sets
        pd.testing.assert_frame_equal(a, b)

    def test_validation(self):
        data = make_cohort()
        with pytest.raises(ValueError, match="event_time"):
            match_risk_set(data, event_time="nope", event="case")
        with pytest.raises(ValueError, match="binary"):
            match_risk_set(data, event_time="time", event="age")
        with pytest.raises(ValueError, match="ratio"):
            match_risk_set(data, event_time="time", event="case", ratio=0)


class TestOddsRatio:
    def test_null_exposure_or_near_one(self):
        data = make_cohort(exposure_effect=0.0)
        result = match_risk_set(
            data,
            event_time="time",
            event="case",
            ratio=4,
            exact="sex",
            random_state=0,
        )
        or_df = result.estimate_odds_ratio("exposure")
        assert or_df["ci_lower"].iloc[0] < 1.0 < or_df["ci_upper"].iloc[0]

    def test_recovers_hazard_ratio(self):
        # hazard ratio exp(0.7) ~ 2.01; incidence-density OR estimates it
        data = make_cohort(n=6000, exposure_effect=0.7, seed=3)
        result = match_risk_set(
            data, event_time="time", event="case", ratio=4, random_state=0
        )
        or_df = result.estimate_odds_ratio("exposure")
        assert or_df["odds_ratio"].iloc[0] == pytest.approx(np.exp(0.7), rel=0.2)
        assert or_df["ci_lower"].iloc[0] > 1.0


class TestValidationHardening:
    def test_missing_event_rejected(self):
        data = make_cohort()
        data.loc[data.index[0], "case"] = np.nan
        with pytest.raises(ValueError, match="missing"):
            match_risk_set(data, event_time="time", event="case")

    def test_nonfinite_time_rejected(self):
        data = make_cohort()
        data.loc[data.index[0], "time"] = np.nan
        with pytest.raises(ValueError, match="non-finite|missing"):
            match_risk_set(data, event_time="time", event="case")

    def test_duplicate_index_rejected(self):
        data = make_cohort()
        data.index = ["u0"] * len(data)
        with pytest.raises(ValueError, match="unique"):
            match_risk_set(data, event_time="time", event="case")

    def test_missing_covariate_rejected(self):
        data = make_cohort()
        data.loc[data.index[0], "age"] = np.nan
        with pytest.raises(ValueError, match="missing"):
            match_risk_set(data, event_time="time", event="case", covariates=["age"])


class TestOvermatchingWarning:
    def test_nearest_selection_warns(self):
        data = make_cohort()
        with pytest.warns(UserWarning, match="overmatching"):
            match_risk_set(
                data,
                event_time="time",
                event="case",
                covariates=["age"],
                random_state=0,
            )

    def test_random_selection_does_not_warn_about_overmatching(self):
        data = make_cohort()
        import warnings as W

        with W.catch_warnings():
            W.simplefilter("error")
            W.filterwarnings(
                "ignore", category=UserWarning, message=".*received no controls.*"
            )
            match_risk_set(data, event_time="time", event="case", random_state=0)


class TestZeroCases:
    def test_no_cases_clean_error(self):
        # sweep #11: match_risk_set with zero cases must raise clearly
        data = make_cohort()
        data["case"] = 0
        with pytest.raises(ValueError, match="no cases"):
            match_risk_set(data, event_time="time", event="case")


class TestRiskSetReporting:
    def _result(self):
        data = make_cohort(n=4000, seed=5)
        return match_risk_set(
            data,
            event_time="time",
            event="case",
            exact="sex",
            covariate_calipers={"age": 3.0},
            ratio=3,
            random_state=0,
        )

    def test_balance(self):
        r = self._result()
        bal = r.balance()
        assert "smd_after" in bal.columns
        assert set(bal["variable"]) >= {"age"}
        # exact-matched sex is perfectly balanced after
        if "sex" in set(bal["variable"]):
            assert abs(bal.set_index("variable").loc["sex", "smd_after"]) < 0.01
        # age caliper tightens age balance
        age_before = abs(bal.set_index("variable").loc["age", "smd_before"])
        age_after = abs(bal.set_index("variable").loc["age", "smd_after"])
        assert age_after < age_before

    def test_table1(self):
        r = self._result()
        t1 = r.table1()
        assert "smd_after" in t1.columns
        assert "age" in t1["variable"].values

    def test_supplement(self, tmp_path):
        r = self._result()
        text = r.supplement(
            str(tmp_path / "ncc.md"), title="NCC S1", exposures="exposure"
        )
        assert (tmp_path / "ncc.md").read_text() == text
        for marker in [
            "Matched sets",
            "Covariate balance",
            "Odds ratios",
            "E-value",
            "Langholz",
            "Methods text",
            "References",
        ]:
            assert marker in text, marker
        assert "validated against" not in text  # no promotional claims

    def test_supplement_without_exposures(self):
        r = self._result()
        text = r.supplement()
        assert "Covariate balance" in text
        assert "Odds ratios" not in text  # only when exposures given

    def test_balance_explicit_covariates(self):
        r = self._result()
        bal = r.balance(covariates=["age", "exposure"])
        assert set(bal["variable"]) == {"age", "exposure"}
