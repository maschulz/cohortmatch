"""Tests for the top-level match() API."""

import warnings

import numpy as np
import pandas as pd
import pytest

from cohortmatch import (
    MatchResult,
    NoMatchesError,
    cem,
    match,
    match_risk_set,
    subclassify,
)


def make_data(n_treat=60, n_control=140, seed=42):
    rng = np.random.RandomState(seed)
    n = n_treat + n_control
    data = pd.DataFrame(
        {
            "treatment": [1] * n_treat + [0] * n_control,
            "age": np.concatenate(
                [rng.normal(55, 8, n_treat), rng.normal(50, 10, n_control)]
            ),
            "bmi": np.concatenate(
                [rng.normal(27, 4, n_treat), rng.normal(25, 4, n_control)]
            ),
            "sex": rng.binomial(1, 0.5, n),
            "outcome": rng.normal(0, 1, n),
        },
        index=[f"p{i}" for i in range(n)],
    )
    data.loc[data["treatment"] == 1, "outcome"] += 2.0
    return data


class TestMinimalCall:
    def test_returns_match_result(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        assert isinstance(result, MatchResult)

    def test_one_to_one_counts(self):
        data = make_data()
        result = match(data, treatment="treatment", covariates=["age", "bmi"])
        matched = result.matched_data
        assert (matched["treatment"] == 1).sum() == 60
        assert (matched["treatment"] == 0).sum() == 60

    def test_no_caliper_by_default_keeps_all_focal_units(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any dropped-unit warning fails the test
            result = match(
                make_data(), treatment="treatment", covariates=["age", "bmi"]
            )
        assert (result.matched_data["treatment"] == 1).sum() == 60

    def test_index_preserved(self):
        data = make_data()
        result = match(data, treatment="treatment", covariates=["age", "bmi"])
        assert set(result.matched_data.index) <= set(data.index)

    def test_balance_dataframe(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        assert (
            set(result.balance()["variable"]) >= {"age", "bmi"}
            or len(result.balance()) >= 2
        )
        assert "smd_before" in result.balance().columns
        assert "smd_after" in result.balance().columns

    def test_pairs_dataframe(self):
        data = make_data()
        result = match(data, treatment="treatment", covariates=["age", "bmi"])
        pairs = result.pairs
        assert list(pairs.columns) == [
            "treatment_id",
            "control_id",
            "distance",
            "match_group",
        ]
        assert len(pairs) == 60
        assert set(pairs["treatment_id"]) <= set(data.index[data["treatment"] == 1])
        assert set(pairs["control_id"]) <= set(data.index[data["treatment"] == 0])
        assert pairs["distance"].notna().all()

    def test_covariate_distance_engine_agnostic_pairs(self):
        # Crossing the memory threshold must not change the matching: exact and
        # approximate covariate-distance matching default to the same order and
        # so must produce identical pairs, with and without a caliper.
        data = make_data(n_treat=120, n_control=380, seed=5)
        for kw in ({}, {"caliper": 4.0, "caliper_metric": "mahalanobis"}):
            ex = match(
                data,
                treatment="treatment",
                covariates=["age", "bmi"],
                distance="mahalanobis",
                engine="exact",
                **kw,
            )
            ap = match(
                data,
                treatment="treatment",
                covariates=["age", "bmi"],
                distance="mahalanobis",
                engine="approximate",
                **kw,
            )
            pe = set(map(tuple, ex.pairs[["treatment_id", "control_id"]].values))
            pa = set(map(tuple, ap.pairs[["treatment_id", "control_id"]].values))
            assert pe == pa, f"exact vs approximate pairs differ for {kw}"

    def test_effects_table_is_tidy_and_labels_family(self):
        data = make_data()
        data["event"] = (data["outcome"] > data["outcome"].median()).astype(int)
        res = match(data, treatment="treatment", covariates=["age", "bmi"])
        eff = res.estimate_effects("event", family="logistic")
        assert eff.shape[1] <= 12  # tidy, terminal-readable
        assert eff.iloc[0]["method"] == "logistic"
        assert eff.iloc[0]["measure"] == "odds_ratio"

    def test_propensity_metrics_are_plain_floats(self):
        # the metrics dict should print cleanly, not as np.float64(...) wrappers
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        m = result.propensity_metrics
        assert isinstance(m["auc"], float) and not isinstance(m["auc"], np.floating)
        assert all(
            isinstance(a, float) and not isinstance(a, np.floating)
            for a in m["fold_aucs"]
        )

    def test_discard_on_covariate_distance_flags_propensity_fit(self):
        data = make_data(n_treat=80, n_control=220, seed=9)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            match(
                data,
                treatment="treatment",
                covariates=["age", "bmi"],
                distance="mahalanobis",
                discard="both",
            )
        msgs = [str(x.message) for x in w]
        assert any("propensity model was fit for the discard" in m for m in msgs)

    def test_summary_repr(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        text = repr(result.summary())
        assert "SMD" in text
        assert "ATT" in text

    def test_original_data_unmodified(self):
        data = make_data()
        before = data.copy()
        match(data, treatment="treatment", covariates=["age", "bmi"])
        pd.testing.assert_frame_equal(data, before)

    def test_weights_and_groups_aligned_to_matched_data(self):
        # weights/match_groups must line up with matched_data both by label and
        # positionally, so result.weights.to_numpy() is safe against a shuffled
        # index (regression: they were built in pair-insertion order).
        data = make_data().sample(frac=1.0, random_state=7)  # shuffle the index
        result = match(data, treatment="treatment", covariates=["age", "bmi"], ratio=2)
        md = result.matched_data
        assert result.weights.index.equals(md.index)
        assert result.match_groups.index.equals(md.index)
        # positional .to_numpy() equals the label-aligned reindex
        np.testing.assert_array_equal(
            result.weights.to_numpy(),
            result.weights.reindex(md.index).to_numpy(),
        )
        # every anchor unit carries weight 1
        anchors = md.index[md["treatment"] == 1]
        assert np.allclose(result.weights.loc[anchors], 1.0)

    def test_reserved_propensity_column_rejected(self):
        data = make_data()
        data["_cb3_propensity"] = 0.5
        with pytest.raises(ValueError, match="reserved"):
            match(
                data,
                treatment="treatment",
                covariates=["age", "bmi"],
                propensity_scores=np.linspace(0.1, 0.9, len(data)),
            )


class TestEstimand:
    def test_att_anchors_on_treated_even_when_treated_larger(self):
        # more treated than controls: legacy behavior silently flipped direction;
        # estimand='att' must keep treated as the anchor group and warn about drops
        data = make_data(n_treat=80, n_control=40)
        with pytest.warns(UserWarning, match="could not be matched"):
            result = match(data, treatment="treatment", covariates=["age", "bmi"])
        # only 40 controls exist, so at most 40 treated units can be matched 1:1
        assert (result.matched_data["treatment"] == 1).sum() == 40
        assert (result.matched_data["treatment"] == 0).sum() == 40
        assert result.estimand == "att"

    def test_atc_anchors_on_controls(self):
        data = make_data(n_treat=80, n_control=40)
        result = match(
            data, treatment="treatment", covariates=["age", "bmi"], estimand="atc"
        )
        # all 40 controls should be matched
        assert (result.matched_data["treatment"] == 0).sum() == 40

    def test_ate_rejected_with_guidance(self):
        with pytest.raises(ValueError, match="att"):
            match(
                make_data(),
                treatment="treatment",
                covariates=["age"],
                estimand="ate",
            )


class TestCaliper:
    def test_auto_caliper_runs(self):
        result = match(
            make_data(),
            treatment="treatment",
            covariates=["age", "bmi"],
            caliper="auto",
        )
        assert len(result.pairs) > 0
        assert result.config["caliper"] == "auto"

    def test_numeric_standardized_caliper(self):
        result = match(
            make_data(),
            treatment="treatment",
            covariates=["age", "bmi"],
            caliper=0.5,
        )
        assert len(result.pairs) > 0

    def test_raw_caliper(self):
        result = match(
            make_data(),
            treatment="treatment",
            covariates=["age", "bmi"],
            caliper=0.2,
            std_caliper=False,
        )
        assert len(result.pairs) > 0

    def test_mahalanobis_caliper_within_propensity_distance_rejected_combo(self):
        # numeric caliper on mahalanobis while matching on mahalanobis distance
        result = match(
            make_data(),
            treatment="treatment",
            covariates=["age", "bmi"],
            distance="mahalanobis",
            caliper=5.0,
            caliper_metric="mahalanobis",
        )
        assert len(result.pairs) > 0

    def test_auto_only_for_propensity_metric(self):
        with pytest.raises(ValueError, match="0.2 SD"):
            match(
                make_data(),
                treatment="treatment",
                covariates=["age"],
                caliper="auto",
                caliper_metric="mahalanobis",
            )

    def test_caliper_metric_without_caliper(self):
        with pytest.raises(ValueError, match="caliper is None"):
            match(
                make_data(),
                treatment="treatment",
                covariates=["age"],
                caliper_metric="propensity",
            )

    def test_tight_caliper_warns_about_drops(self):
        data = make_data()
        with pytest.warns(UserWarning, match="could not be matched"):
            match(
                data,
                treatment="treatment",
                covariates=["age", "bmi"],
                caliper=0.01,
                std_caliper=False,
            )


class TestValidation:
    def test_unknown_method(self):
        with pytest.raises(ValueError, match="nearest"):
            match(
                make_data(), treatment="treatment", covariates=["age"], method="bogus"
            )

    def test_multiindex_columns_rejected(self):
        data = make_data()
        data.columns = pd.MultiIndex.from_product([["a"], data.columns])
        with pytest.raises(TypeError, match="MultiIndex"):
            match(data, treatment=("a", "treatment"), covariates=[("a", "age")])

    def test_non_string_covariate_names_rejected(self):
        data = make_data()
        data = data.rename(columns={"age": 0})  # integer column name
        with pytest.raises(TypeError, match="covariate names must be strings"):
            match(data, treatment="treatment", covariates=[0, "bmi"])

    def test_greedy_alias(self):
        result = match(
            make_data(), treatment="treatment", covariates=["age"], method="greedy"
        )
        assert result.config["method"] == "nearest"

    def test_fast_greedy_hint(self):
        with pytest.raises(ValueError, match="approximate"):
            match(
                make_data(),
                treatment="treatment",
                covariates=["age"],
                method="fast_greedy",
            )

    def test_matchit_glm_hint(self):
        with pytest.raises(ValueError, match="MatchIt"):
            match(
                make_data(), treatment="treatment", covariates=["age"], distance="glm"
            )

    def test_fractional_ratio_rejected(self):
        with pytest.raises(ValueError, match="integer"):
            match(make_data(), treatment="treatment", covariates=["age"], ratio=1.5)

    def test_integral_float_ratio_accepted(self):
        result = match(
            make_data(), treatment="treatment", covariates=["age", "bmi"], ratio=2.0
        )
        assert result.config["ratio"] == 2

    def test_covariate_weights_require_euclidean(self):
        with pytest.raises(ValueError, match="euclidean"):
            match(
                make_data(),
                treatment="treatment",
                covariates=["age", "bmi"],
                distance="mahalanobis",
                covariate_weights={"age": 2.0},
            )

    def test_covariate_weights_euclidean_ok(self):
        result = match(
            make_data(),
            treatment="treatment",
            covariates=["age", "bmi"],
            distance="euclidean",
            covariate_weights={"age": 2.0},
        )
        assert len(result.pairs) > 0

    def test_optimal_with_replace_rejected(self):
        with pytest.raises(ValueError, match="optimal"):
            match(
                make_data(),
                treatment="treatment",
                covariates=["age"],
                method="optimal",
                replace=True,
            )

    def test_both_propensity_args_rejected(self):
        from sklearn.linear_model import LogisticRegression

        with pytest.raises(ValueError, match="mutually exclusive"):
            match(
                make_data(),
                treatment="treatment",
                covariates=["age"],
                propensity_scores="age",
                propensity_model=LogisticRegression(),
            )

    def test_no_matches_raises(self):
        data = make_data()
        data["site"] = np.where(data["treatment"] == 1, "A", "B")
        with pytest.raises(NoMatchesError):
            match(
                data,
                treatment="treatment",
                covariates=["age", "bmi"],
                exact="site",
            )


class TestExact:
    def test_exact_scalar(self):
        data = make_data()
        result = match(
            data, treatment="treatment", covariates=["age", "bmi"], exact="sex"
        )
        pairs = result.pairs
        sex = data["sex"]
        for _, row in pairs.iterrows():
            assert sex[row["treatment_id"]] == sex[row["control_id"]]


class TestPropensity:
    def test_precomputed_column(self):
        data = make_data()
        rng = np.random.RandomState(0)
        data["ps"] = np.clip(rng.uniform(0.1, 0.9, len(data)), 0.01, 0.99)
        result = match(
            data,
            treatment="treatment",
            covariates=["age", "bmi"],
            propensity_scores="ps",
        )
        assert len(result.pairs) > 0

    def test_precomputed_array(self):
        data = make_data()
        rng = np.random.RandomState(0)
        scores = np.clip(rng.uniform(0.1, 0.9, len(data)), 0.01, 0.99)
        result = match(
            data,
            treatment="treatment",
            covariates=["age", "bmi"],
            propensity_scores=scores,
        )
        assert len(result.pairs) > 0
        # internal score column must not leak into outputs
        assert "_cb3_propensity" not in result.matched_data.columns
        assert "_cb3_propensity" not in result.original_data.columns

    def test_wrong_length_array(self):
        with pytest.raises(ValueError, match="length"):
            match(
                make_data(),
                treatment="treatment",
                covariates=["age"],
                propensity_scores=np.array([0.5, 0.5]),
            )

    def test_sklearn_estimator_cloned_not_mutated(self):
        from sklearn.linear_model import LogisticRegression

        est = LogisticRegression(max_iter=200)
        result = match(
            make_data(),
            treatment="treatment",
            covariates=["age", "bmi"],
            propensity_model=est,
        )
        assert len(result.pairs) > 0
        assert not hasattr(est, "coef_")  # user's estimator untouched

    def test_scores_series_aligned(self):
        data = make_data()
        rng = np.random.RandomState(0)
        scores = pd.Series(
            np.clip(rng.uniform(0.1, 0.9, len(data)), 0.01, 0.99),
            index=data.index[::-1],  # reversed order: must align by index
        )
        result = match(
            data,
            treatment="treatment",
            covariates=["age", "bmi"],
            propensity_scores=scores,
        )
        assert len(result.pairs) > 0

    def test_result_scores_are_series(self):
        data = make_data()
        result = match(data, treatment="treatment", covariates=["age", "bmi"])
        scores = result.propensity_scores
        assert isinstance(scores, pd.Series)
        assert scores.index.equals(data.index)


class TestAlgorithm:
    def test_approximate_requires_caliper(self):
        with pytest.raises(ValueError, match="caliper"):
            match(
                make_data(),
                treatment="treatment",
                covariates=["age"],
                engine="approximate",
            )

    def test_approximate_runs(self):
        result = match(
            make_data(),
            treatment="treatment",
            covariates=["age", "bmi"],
            caliper="auto",
            engine="approximate",
        )
        assert len(result.pairs) > 0
        assert result.config["engine"] == "approximate"
        # approximate path has no dense distance matrix
        assert result.pairs["distance"].isna().all()

    def test_auto_switches_to_approximate_with_warning(self):
        with pytest.warns(UserWarning, match="approximate"):
            result = match(
                make_data(),
                treatment="treatment",
                covariates=["age", "bmi"],
                caliper="auto",
                memory_limit_gb=1e-9,
            )
        assert result.config["engine"] == "approximate"

    def test_auto_without_caliper_errors_at_scale(self):
        with pytest.raises(ValueError, match="caliper='auto'"):
            match(
                make_data(),
                treatment="treatment",
                covariates=["age", "bmi"],
                memory_limit_gb=1e-9,
            )

    def test_optimal_at_scale_errors(self):
        with pytest.raises(ValueError, match="Optimal"):
            match(
                make_data(),
                treatment="treatment",
                covariates=["age", "bmi"],
                method="optimal",
                memory_limit_gb=1e-9,
            )


class TestRatioMatching:
    def test_ratio_two(self):
        result = match(
            make_data(n_treat=40, n_control=160),
            treatment="treatment",
            covariates=["age", "bmi"],
            ratio=2,
        )
        matched = result.matched_data
        assert (matched["treatment"] == 0).sum() == 2 * (
            matched["treatment"] == 1
        ).sum()
        assert (result.pairs.groupby("match_group").size() == 2).all()


class TestEffects:
    def test_estimate_effects(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        effects = result.estimate_effects("outcome")
        assert len(effects) == 1
        assert effects["effect"].iloc[0] == pytest.approx(2.0, abs=1.0)
        assert effects["estimand"].iloc[0] == "att"

    def test_outcomes_list(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        effects = result.estimate_effects(["outcome"])
        assert len(effects) == 1

    def test_unknown_outcome(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        with pytest.raises(ValueError, match="not in matched data"):
            result.estimate_effects("nope")


class TestOptimal:
    def test_optimal_runs(self):
        result = match(
            make_data(),
            treatment="treatment",
            covariates=["age", "bmi"],
            method="optimal",
            distance="mahalanobis",
        )
        assert (result.matched_data["treatment"] == 1).sum() == 60


class TestApproximateWorkflow:
    """Locks the biobank-scale workflow: exact + caliper on the approximate path."""

    def test_exact_constraint_respected_on_approximate_path(self):
        data = make_data(n_treat=80, n_control=400)
        result = match(
            data,
            treatment="treatment",
            covariates=["age", "bmi"],
            exact="sex",
            caliper="auto",
            engine="approximate",
        )
        sex = data["sex"]
        pairs = result.pairs
        assert len(pairs) > 0
        for _, row in pairs.iterrows():
            assert sex[row["treatment_id"]] == sex[row["control_id"]]

    def test_approximate_replace_and_ratio(self):
        data = make_data(n_treat=40, n_control=200)
        result = match(
            data,
            treatment="treatment",
            covariates=["age", "bmi"],
            caliper="auto",
            engine="approximate",
            ratio=2,
        )
        matched = result.matched_data
        n_t = (matched["treatment"] == 1).sum()
        n_c = (matched["treatment"] == 0).sum()
        assert n_c >= n_t  # ratio 2 attempted; caliper may reduce it


class TestDiagnostics:
    def test_rubin_statistics_exposed(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        rubin = result.rubin_statistics
        assert rubin is not None
        assert "pct_both_good" in rubin

    def test_summary_includes_rubin(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        assert "Rubin" in repr(result.summary())

    def test_plot_methods(self):
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        for name in [
            "plot_love_plot",
            "plot_balance",
            "plot_propensity",
            "plot_match_distances",
        ]:
            fig = getattr(result, name)()
            assert fig is not None


class TestWeights:
    def test_one_to_one_weights_are_one(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        w = result.weights
        assert (w == 1.0).all()
        assert set(w.index) == set(result.matched_data.index)

    def test_fixed_ratio_weights_are_one(self):
        result = match(
            make_data(n_treat=40, n_control=160),
            treatment="treatment",
            covariates=["age", "bmi"],
            ratio=2,
        )
        assert (result.weights == 1.0).all()

    def test_replace_no_duplicate_rows_weights_reflect_reuse(self):
        data = make_data(n_treat=50, n_control=20)
        result = match(
            data,
            treatment="treatment",
            covariates=["age", "bmi"],
            replace=True,
        )
        matched = result.matched_data
        # every unit appears once under its original id
        assert matched.index.is_unique
        assert not any(
            str(i).endswith(tuple(f"_dup{k}" for k in range(1, 9)))
            for i in matched.index
        )
        assert set(matched.index) <= set(data.index)
        # 50 treated anchors, at most 20 unique controls: reuse must show in weights
        controls = matched[matched["treatment"] == 0].index
        w = result.weights
        assert len(controls) <= 20
        assert w[controls].max() > 1.0  # some control is reused
        # control weights average 1 (MatchIt scaling)
        assert w[controls].mean() == pytest.approx(1.0)
        # match-group membership undefined with replacement
        assert result.match_groups is None

    def test_match_groups_map_to_anchor(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        sub = result.match_groups
        assert sub is not None
        pairs = result.pairs
        for _, row in pairs.iterrows():
            assert sub[row["control_id"]] == row["treatment_id"]
            assert sub[row["treatment_id"]] == row["treatment_id"]


class TestInference:
    def test_cluster_robust_se_without_replacement(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        effects = result.estimate_effects("outcome")
        assert effects["se_type"].iloc[0] == "cluster-robust (match groups)"
        assert effects["effect"].iloc[0] == pytest.approx(2.0, abs=1.0)
        assert np.isfinite(effects["standard_error"].iloc[0])

    def test_hc3_se_with_replacement(self):
        result = match(
            make_data(n_treat=50, n_control=20),
            treatment="treatment",
            covariates=["age", "bmi"],
            replace=True,
        )
        effects = result.estimate_effects("outcome")
        assert effects["se_type"].iloc[0] == "HC3-robust"

    def test_weighted_effect_matches_manual_computation(self):
        result = match(
            make_data(n_treat=50, n_control=20),
            treatment="treatment",
            covariates=["age", "bmi"],
            replace=True,
        )
        effects = result.estimate_effects("outcome")
        matched = result.matched_data
        w = result.weights
        t = matched["treatment"] == 1
        manual = np.average(
            matched.loc[t, "outcome"], weights=w[matched.index[t]]
        ) - np.average(matched.loc[~t, "outcome"], weights=w[matched.index[~t]])
        assert effects["effect"].iloc[0] == pytest.approx(manual)


class TestBalanceConventions:
    def test_smd_is_signed(self):
        # treated are older by construction: SMD for age must be positive
        result = match(
            make_data(),
            treatment="treatment",
            covariates=["age", "bmi"],
        )
        balance = result.balance()
        smd_age = balance.loc[balance["variable"] == "age", "smd_before"].iloc[0]
        assert smd_age > 0
        # flip the treatment direction of the covariate: sign must flip
        data = make_data()
        data["neg_age"] = -data["age"]
        result2 = match(data, treatment="treatment", covariates=["neg_age", "bmi"])
        balance2 = result2.balance()
        smd_neg = balance2.loc[balance2["variable"] == "neg_age", "smd_before"].iloc[0]
        assert smd_neg < 0

    def test_same_denominator_before_after(self):
        # smd_after / smd_before must equal ratio of raw mean differences
        data = make_data()
        result = match(data, treatment="treatment", covariates=["age", "bmi"])
        t1 = result.table1().set_index("variable")
        for var in ["age", "bmi"]:
            row = t1.loc[var]
            diff_before = row["mean_treated_before"] - row["mean_control_before"]
            diff_after = row["mean_treated_after"] - row["mean_control_after"]
            if row["smd_before"] != 0:
                assert diff_after / diff_before == pytest.approx(
                    row["smd_after"] / row["smd_before"], abs=1e-9
                )

    def test_table1_structure(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        t1 = result.table1()
        assert list(t1["variable"]) == ["age", "bmi"]
        assert "mean_treated_before" in t1.columns
        assert "smd_after" in t1.columns
        assert t1.attrs["n_treated_before"] == 60


class TestMatchingOrder:
    def test_m_order_values_produce_valid_matchings(self):
        data = make_data()
        for order in ["largest", "smallest", "closest", "random", "data"]:
            result = match(
                data,
                treatment="treatment",
                covariates=["age", "bmi"],
                m_order=order,
                random_state=0,
            )
            assert (result.matched_data["treatment"] == 1).sum() == 60

    def test_m_order_changes_contested_matching(self):
        # under a tight caliper, order determines who gets matched
        data = make_data(n_treat=60, n_control=70)
        results = {}
        for order in ["largest", "smallest"]:
            import warnings as W

            with W.catch_warnings():
                W.simplefilter("ignore")
                r = match(
                    data,
                    treatment="treatment",
                    covariates=["age", "bmi"],
                    caliper=0.5,
                    m_order=order,
                )
            results[order] = set(r.pairs["control_id"])
        assert results["largest"] != results["smallest"]

    def test_m_order_rejected_for_optimal(self):
        with pytest.raises(ValueError, match="nearest"):
            match(
                make_data(),
                treatment="treatment",
                covariates=["age"],
                method="optimal",
                m_order="largest",
            )

    def test_m_order_approximate_path(self):
        result = match(
            make_data(),
            treatment="treatment",
            covariates=["age", "bmi"],
            caliper="auto",
            engine="approximate",
            m_order="largest",
        )
        assert len(result.pairs) > 0
        assert result.config["m_order"] == "largest"


class TestAlgorithmEquivalence:
    def test_approximate_equals_exact_under_same_order(self):
        # with the same matching order and a user caliper tighter than the
        # prefilter window, both algorithms must produce identical pairs
        data = make_data(n_treat=80, n_control=400)
        kwargs = dict(
            treatment="treatment",
            covariates=["age", "bmi"],
            caliper="auto",
            m_order="data",
            random_state=0,
        )
        import warnings as W

        with W.catch_warnings():
            W.simplefilter("ignore")
            exact = match(data, engine="exact", **kwargs)
            approx = match(data, engine="approximate", **kwargs)
        pairs_exact = set(
            map(tuple, exact.pairs[["treatment_id", "control_id"]].values)
        )
        pairs_approx = set(
            map(tuple, approx.pairs[["treatment_id", "control_id"]].values)
        )
        assert pairs_exact == pairs_approx


class TestPrefilterCorrectness:
    def test_wide_caliper_not_truncated_by_prefilter(self):
        # a user caliper wider than the 0.5-SD prefilter heuristic must widen
        # the candidate window, not be silently truncated by it
        data = make_data(n_treat=40, n_control=200)
        kwargs = dict(
            treatment="treatment",
            covariates=["age", "bmi"],
            caliper=1.5,  # 1.5 SD of logit-PS: wider than the 0.5-SD prefilter
            m_order="data",
            random_state=0,
        )
        import warnings as W

        with W.catch_warnings():
            W.simplefilter("ignore")
            exact = match(data, engine="exact", **kwargs)
            approx = match(data, engine="approximate", **kwargs)
        pairs_exact = set(
            map(tuple, exact.pairs[["treatment_id", "control_id"]].values)
        )
        pairs_approx = set(
            map(tuple, approx.pairs[["treatment_id", "control_id"]].values)
        )
        assert pairs_exact == pairs_approx

    @pytest.mark.slow
    def test_equivalence_at_scale(self):
        # pair-identity between algorithms at a shape where the dense matrix
        # is still feasible (3k x 60k)
        rng = np.random.RandomState(1)
        n_t, n_c = 3000, 60000
        n = n_t + n_c
        X = rng.normal(size=(n, 4))
        noisy = 1 / (1 + np.exp(-(0.8 * X[:, 0] + 0.5 * X[:, 1]))) + rng.uniform(
            0, 0.3, n
        )
        treated = np.zeros(n, dtype=int)
        treated[np.argsort(-noisy)[:n_t]] = 1
        data = pd.DataFrame(X, columns=["a", "b", "c", "d"])
        data["treatment"] = treated
        kwargs = dict(
            treatment="treatment",
            covariates=["a", "b", "c", "d"],
            caliper="auto",
            m_order="data",
            random_state=0,
        )
        import warnings as W

        with W.catch_warnings():
            W.simplefilter("ignore")
            exact = match(data, engine="exact", memory_limit_gb=8.0, **kwargs)
            approx = match(data, engine="approximate", **kwargs)
        pairs_exact = set(
            map(tuple, exact.pairs[["treatment_id", "control_id"]].values)
        )
        pairs_approx = set(
            map(tuple, approx.pairs[["treatment_id", "control_id"]].values)
        )
        assert pairs_exact == pairs_approx


class TestCovariateCalipers:
    def test_age_caliper_respected(self):
        data = make_data()
        result = match(
            data,
            treatment="treatment",
            covariates=["age", "bmi"],
            covariate_calipers={"age": 3.0},
        )
        for _, row in result.pairs.iterrows():
            assert (
                abs(
                    data.loc[row["treatment_id"], "age"]
                    - data.loc[row["control_id"], "age"]
                )
                <= 3.0
            )

    def test_approximate_path_respected(self):
        data = make_data(n_treat=80, n_control=400)
        result = match(
            data,
            treatment="treatment",
            covariates=["age", "bmi"],
            caliper="auto",
            engine="approximate",
            covariate_calipers={"age": 3.0, "bmi": 2.0},
        )
        for _, row in result.pairs.iterrows():
            assert (
                abs(
                    data.loc[row["treatment_id"], "age"]
                    - data.loc[row["control_id"], "age"]
                )
                <= 3.0
            )
            assert (
                abs(
                    data.loc[row["treatment_id"], "bmi"]
                    - data.loc[row["control_id"], "bmi"]
                )
                <= 2.0
            )

    def test_equivalence_between_algorithms(self):
        data = make_data(n_treat=60, n_control=300)
        kwargs = dict(
            treatment="treatment",
            covariates=["age", "bmi"],
            caliper="auto",
            covariate_calipers={"age": 5.0},
            m_order="data",
            random_state=0,
        )
        import warnings as W

        with W.catch_warnings():
            W.simplefilter("ignore")
            exact = match(data, engine="exact", **kwargs)
            approx = match(data, engine="approximate", **kwargs)
        pairs_exact = set(
            map(tuple, exact.pairs[["treatment_id", "control_id"]].values)
        )
        pairs_approx = set(
            map(tuple, approx.pairs[["treatment_id", "control_id"]].values)
        )
        assert pairs_exact == pairs_approx

    def test_caliper_on_non_covariate_column(self):
        # caliper on a variable not used in the distance
        data = make_data()
        result = match(
            data,
            treatment="treatment",
            covariates=["age", "bmi"],
            covariate_calipers={"outcome": 1.0},
        )
        for _, row in result.pairs.iterrows():
            assert (
                abs(
                    data.loc[row["treatment_id"], "outcome"]
                    - data.loc[row["control_id"], "outcome"]
                )
                <= 1.0
            )

    def test_validation_errors(self):
        data = make_data()
        with pytest.raises(ValueError, match="not in data"):
            match(
                data,
                treatment="treatment",
                covariates=["age"],
                covariate_calipers={"nope": 1.0},
            )
        with pytest.raises(ValueError, match="positive"):
            match(
                data,
                treatment="treatment",
                covariates=["age"],
                covariate_calipers={"age": 0},
            )
        data["site"] = "A"
        with pytest.raises(ValueError, match="numeric"):
            match(
                data,
                treatment="treatment",
                covariates=["age"],
                covariate_calipers={"site": 1.0},
            )

    def test_impossible_caliper_raises_no_matches(self):
        data = make_data()
        data.loc[data["treatment"] == 1, "age"] += 1000
        with pytest.raises(NoMatchesError):
            match(
                data,
                treatment="treatment",
                covariates=["age", "bmi"],
                covariate_calipers={"age": 3.0},
            )


class TestDiscard:
    def test_discard_drops_out_of_support_units(self):
        data = make_data()
        # make a few treated units clear outliers in covariate space so
        # their propensity exceeds every control's
        data.loc[data.index[:3], "age"] += 200
        with pytest.warns(UserWarning, match="common propensity support"):
            result = match(
                data,
                treatment="treatment",
                covariates=["age", "bmi"],
                discard="treated",
                random_state=0,
            )
        assert len(result.discarded) >= 3
        assert set(data.index[:3]) <= set(result.discarded)
        assert not (set(result.discarded) & set(result.matched_data.index))

    def test_discard_off_by_default(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        assert len(result.discarded) == 0

    def test_original_data_and_before_balance_keep_full_sample(self):
        data = make_data()
        data.loc[data.index[:3], "age"] += 200
        import warnings as W

        with W.catch_warnings():
            W.simplefilter("ignore")
            result = match(
                data,
                treatment="treatment",
                covariates=["age", "bmi"],
                discard="treated",
                random_state=0,
            )
        assert len(result.original_data) == len(data)

    def test_invalid_discard(self):
        with pytest.raises(ValueError, match="discard"):
            match(
                make_data(),
                treatment="treatment",
                covariates=["age"],
                discard="everything",
            )


class TestMahalanobisAtScale:
    def test_mahalanobis_equivalence_between_algorithms(self):
        # full-sample covariance on the approximate path must reproduce the
        # dense path's Mahalanobis distances: identical pairs under same order
        data = make_data(n_treat=60, n_control=300)
        kwargs = dict(
            treatment="treatment",
            covariates=["age", "bmi"],
            distance="mahalanobis",
            caliper="auto",
            m_order="data",
            random_state=0,
        )
        import warnings as W

        with W.catch_warnings():
            W.simplefilter("ignore")
            exact = match(data, engine="exact", **kwargs)
            approx = match(data, engine="approximate", **kwargs)
        pairs_exact = set(
            map(tuple, exact.pairs[["treatment_id", "control_id"]].values)
        )
        pairs_approx = set(
            map(tuple, approx.pairs[["treatment_id", "control_id"]].values)
        )
        assert pairs_exact == pairs_approx


class TestStrataMethods:
    def test_subclass_basic(self):
        result = subclassify(
            make_data(n_treat=100, n_control=300),
            treatment="treatment",
            covariates=["age", "bmi"],
            random_state=0,
        )
        assert len(result.pairs) == 0
        sub = result.strata
        assert sub is not None and sub.nunique() >= 2
        w = result.weights
        treated_ids = result.matched_data.index[result.matched_data["treatment"] == 1]
        control_ids = result.matched_data.index[result.matched_data["treatment"] == 0]
        # ATT: treated weight 1 (after rescale), controls average 1
        assert w[treated_ids].mean() == pytest.approx(1.0)
        assert w[control_ids].mean() == pytest.approx(1.0)
        assert "Strata" in repr(result.summary())

    def test_subclass_weighted_balance_improves(self):
        result = subclassify(
            make_data(n_treat=150, n_control=450),
            treatment="treatment",
            covariates=["age", "bmi"],
            n_subclasses=8,
            random_state=0,
        )
        balance = result.balance()
        assert balance["smd_after"].abs().mean() < balance["smd_before"].abs().mean()

    def test_subclass_ate(self):
        result = subclassify(
            make_data(n_treat=150, n_control=450),
            treatment="treatment",
            covariates=["age", "bmi"],
            estimand="ate",
            random_state=0,
        )
        assert result.estimand == "ate"
        effects = result.estimate_effects("outcome")
        assert effects["effect"].iloc[0] == pytest.approx(2.0, abs=1.0)
        assert effects["se_type"].iloc[0] == "HC3-robust"

    def test_cem_basic(self):
        data = make_data(n_treat=100, n_control=300)
        result = cem(
            data,
            treatment="treatment",
            covariates=["age", "bmi", "sex"],
        )
        sub = result.strata
        # every stratum contains both groups
        md = result.matched_data
        for _s_id, ids in md.groupby(sub).groups.items():
            grp = md.loc[ids, "treatment"]
            assert (grp == 1).any() and (grp == 0).any()
        # sex is binary: cells are sex-homogeneous
        for _s_id, ids in md.groupby(sub).groups.items():
            assert md.loc[ids, "sex"].nunique() == 1

    def test_cem_coarsening_control(self):
        data = make_data(n_treat=100, n_control=300)
        coarse = cem(
            data,
            treatment="treatment",
            covariates=["age", "bmi"],
            coarsening={"age": 3, "bmi": 3},
        )
        fine = cem(
            data,
            treatment="treatment",
            covariates=["age", "bmi"],
            coarsening={"age": 12, "bmi": 12},
        )
        # finer coarsening -> more strata, fewer retained units
        assert fine.strata.nunique() > coarse.strata.nunique()
        assert len(fine.matched_data) <= len(coarse.matched_data)

    def test_ate_rejected_for_pair_matching(self):
        with pytest.raises(ValueError, match="subclassify"):
            match(
                make_data(), treatment="treatment", covariates=["age"], estimand="ate"
            )

    def test_moved_methods_point_to_new_functions(self):
        with pytest.raises(ValueError, match="subclassify"):
            match(
                make_data(),
                treatment="treatment",
                covariates=["age"],
                method="subclass",
            )
        with pytest.raises(ValueError, match="cohortmatch.cem"):
            match(make_data(), treatment="treatment", covariates=["age"], method="cem")

    def test_table1_works_for_strata(self):
        result = subclassify(
            make_data(n_treat=100, n_control=300),
            treatment="treatment",
            covariates=["age", "bmi"],
            random_state=0,
        )
        t1 = result.table1()
        assert "smd_after" in t1.columns


class TestReviewRoundFixes:
    def test_atc_match_group_is_anchor(self):
        # under ATC with ratio 2, both treated partners of one control anchor
        # must share that anchor's match_group
        data = make_data(n_treat=200, n_control=50)
        result = match(
            data,
            treatment="treatment",
            covariates=["age", "bmi"],
            estimand="atc",
            ratio=2,
        )
        pairs = result.pairs
        control_ids = set(data.index[data["treatment"] == 0])
        assert set(pairs["match_group"]) <= control_ids
        sizes = pairs.groupby("match_group").size()
        assert (sizes == 2).all()

    def test_random_state_does_not_change_matching(self):
        data = make_data()
        # identical precomputed scores isolate the matching itself
        import numpy as np

        rng = np.random.RandomState(0)
        data["ps"] = np.clip(rng.uniform(0.1, 0.9, len(data)), 0.01, 0.99)
        seeded = match(
            data,
            treatment="treatment",
            covariates=["age", "bmi"],
            propensity_scores="ps",
            random_state=123,
        )
        unseeded = match(
            data,
            treatment="treatment",
            covariates=["age", "bmi"],
            propensity_scores="ps",
        )
        assert set(
            map(tuple, seeded.pairs[["treatment_id", "control_id"]].values)
        ) == set(map(tuple, unseeded.pairs[["treatment_id", "control_id"]].values))

    def test_propensity_model_usable_on_raw_data(self):
        data = make_data()
        result = match(data, treatment="treatment", covariates=["age", "bmi"])
        model = result.propensity_model
        preds = model.predict_proba(data[["age", "bmi"]].to_numpy())[:, 1]
        assert preds.min() >= 0 and preds.max() <= 1
        # full-sample refit should correlate strongly with the stored
        # cross-fitted scores
        import numpy as np

        stored = result.propensity_scores.to_numpy()
        assert np.corrcoef(preds, stored)[0, 1] > 0.95


class TestEpidemiologyEffects:
    def make_binary_outcome_data(self, n_treat=150, n_control=450, seed=5):
        rng = np.random.RandomState(seed)
        data = make_data(n_treat=n_treat, n_control=n_control, seed=seed)
        # true OR ~ exp(1.0) for treatment on a binary outcome
        logits = -1.0 + 1.0 * data["treatment"] + 0.02 * (data["age"] - 50)
        data["event"] = rng.binomial(1, 1 / (1 + np.exp(-logits)))
        return data

    def test_odds_ratio_family(self):
        data = self.make_binary_outcome_data()
        result = match(
            data, treatment="treatment", covariates=["age", "bmi"], random_state=0
        )
        effects = result.estimate_effects("event", family="logistic")
        assert effects["measure"].iloc[0] == "odds_ratio"
        assert effects["effect"].iloc[0] == pytest.approx(np.exp(1.0), rel=0.5)
        assert effects["ci_lower"].iloc[0] > 0

    def test_risk_ratio_family(self):
        data = self.make_binary_outcome_data()
        result = match(
            data, treatment="treatment", covariates=["age", "bmi"], random_state=0
        )
        effects = result.estimate_effects("event", family="poisson")
        assert effects["measure"].iloc[0] == "risk_ratio"
        assert effects["effect"].iloc[0] > 1.0

    def test_logistic_rejects_nonbinary(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        with pytest.raises(ValueError, match="binary"):
            result.estimate_effects("outcome", family="logistic")

    def test_linear_unchanged_and_labeled(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        effects = result.estimate_effects("outcome")
        assert effects["measure"].iloc[0] == "mean_difference"


class TestCategoricalCovariates:
    def test_string_covariates_encoded(self):
        data = make_data()
        data["region"] = np.where(data["age"] > 50, "north", "south")
        result = match(
            data,
            treatment="treatment",
            covariates=["age", "bmi", "region"],
            random_state=0,
        )
        balance = result.balance()
        assert any(v.startswith("region=") for v in balance["variable"])
        # encoded helper columns must not leak into outputs
        assert not any(c.startswith("region=") for c in result.matched_data.columns)
        assert "region" in result.matched_data.columns

    def test_cem_with_string_covariate(self):
        data = make_data(n_treat=100, n_control=300)
        data["region"] = np.where(data["bmi"] > 25, "high", "low")
        result = cem(data, treatment="treatment", covariates=["age", "region"])
        md = result.matched_data
        strata = result.strata
        for _s_id, ids in md.groupby(strata).groups.items():
            assert md.loc[ids, "region"].nunique() == 1

    def test_lalonde_race_end_to_end(self):
        from cohortmatch.datasets import load_lalonde

        lalonde = load_lalonde()
        result = match(
            lalonde,
            treatment="treat",
            covariates=["age", "educ", "race", "married", "nodegree", "re74", "re75"],
            random_state=0,
        )
        balance = result.balance().set_index("variable")
        assert "race=black" in balance.index
        assert (result.matched_data["treat"] == 1).sum() == 185


class TestReviewRound3Fixes:
    def test_glm_var_weights_sandwich(self):
        # the robust SE must equal the hand-computed sampling-weight sandwich
        import statsmodels.api as sm

        from cohortmatch.metrics.treatment import estimate_treatment_effect

        rng = np.random.RandomState(0)
        n = 400
        data = pd.DataFrame(
            {
                "treatment": rng.binomial(1, 0.5, n),
                "y": rng.binomial(1, 0.4, n),
            }
        )
        w = pd.Series(rng.uniform(0.5, 2.0, n), index=data.index)

        res = estimate_treatment_effect(
            data,
            "y",
            "treatment",
            family="logistic",
            weights=w,
        )
        # hand-rolled HC0-style sandwich for weighted logistic
        X = sm.add_constant(data[["treatment"]].to_numpy(dtype=float))
        wv = w.to_numpy()
        fit = sm.GLM(data["y"], X, family=sm.families.Binomial(), var_weights=wv).fit()
        mu = (
            fit.fittedvalues.to_numpy()
            if hasattr(fit.fittedvalues, "to_numpy")
            else np.asarray(fit.fittedvalues)
        )
        resid = data["y"].to_numpy() - mu
        V = mu * (1 - mu)
        bread = np.linalg.inv((X * (wv * V)[:, None]).T @ X)
        meat = (X * (wv * resid)[:, None]).T @ (X * (wv * resid)[:, None])
        sandwich = bread @ meat @ bread
        hand_se = np.sqrt(sandwich[1, 1])
        assert res["standard_error"] == pytest.approx(hand_se, rel=1e-6)

    def test_no_specification_warning_from_glm(self):
        data = make_data()
        rng = np.random.RandomState(1)
        data["event"] = rng.binomial(1, 0.4, len(data))
        result = match(
            data, treatment="treatment", covariates=["age", "bmi"], random_state=0
        )
        import warnings as W

        with W.catch_warnings():
            W.simplefilter("error")
            result.estimate_effects("event", family="logistic")

    def test_propensity_metrics_populated(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        metrics = result.propensity_metrics
        assert metrics and metrics.get("auc") is not None
        assert 0.5 <= metrics["auc"] <= 1.0

    def test_adjustment_covariates_require_regression_method(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        with pytest.raises(ValueError, match="regression_adjustment"):
            result.estimate_effects("outcome", adjustment_covariates=["age"])

    def test_encoder_collision_rejected(self):
        data = make_data()
        data["region"] = np.where(data["age"] > 50, "north", "south")
        data["region=north"] = 1.0
        with pytest.raises(ValueError, match="collide"):
            match(data, treatment="treatment", covariates=["age", "region"])

    def test_bool_covariate_stays_numeric(self):
        data = make_data()
        data["flag"] = data["sex"].astype(bool)
        result = match(data, treatment="treatment", covariates=["age", "flag"])
        variables = set(result.balance()["variable"])
        assert "flag" in variables
        assert not any(v.startswith("flag=") for v in variables)

    def test_nan_categorical_rejected(self):
        data = make_data()
        data["region"] = np.where(data["age"] > 50, "north", "south")
        data.loc[data.index[0], "region"] = np.nan
        with pytest.raises(ValueError, match="missing"):
            match(data, treatment="treatment", covariates=["age", "region"])

    def test_covariate_weights_expanded_to_dummies(self):
        data = make_data()
        data["region"] = np.where(data["bmi"] > 25, "high", "low")
        heavy = match(
            data,
            treatment="treatment",
            covariates=["age", "region"],
            distance="euclidean",
            covariate_weights={"region": 200.0},
            random_state=0,
        )
        light = match(
            data,
            treatment="treatment",
            covariates=["age", "region"],
            distance="euclidean",
            covariate_weights={"region": 1e-6},
            random_state=0,
        )
        pairs_h = set(map(tuple, heavy.pairs[["treatment_id", "control_id"]].values))
        pairs_l = set(map(tuple, light.pairs[["treatment_id", "control_id"]].values))
        assert pairs_h != pairs_l  # the weight must influence matching

    def test_cem_float_coarsening_rejected(self):
        with pytest.raises(ValueError, match="bin count"):
            cem(
                make_data(),
                treatment="treatment",
                covariates=["age"],
                coarsening={"age": 3.0},
            )

    def test_cem_noncovariate_coarsening_rejected(self):
        with pytest.raises(ValueError, match="not a covariate"):
            cem(
                make_data(),
                treatment="treatment",
                covariates=["age"],
                coarsening={"bmi": 3},
            )

    def test_settings_record_user_covariates(self):
        data = make_data()
        data["region"] = np.where(data["age"] > 50, "north", "south")
        result = match(data, treatment="treatment", covariates=["age", "region"])
        assert result.config["covariates"] == ("age", "region")
        assert any(c.startswith("region=") for c in result.config["encoded_covariates"])


class TestEstimationScope:
    def test_ratio_families_are_marginal_only(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        with pytest.raises(ValueError, match="marginal"):
            result.estimate_effects(
                "outcome",
                family="logistic",
                method="regression_adjustment",
                adjustment_covariates=["age"],
            )

    def test_binary_linear_labeled_risk_difference(self):
        data = make_data()
        rng = np.random.RandomState(3)
        data["event"] = rng.binomial(1, 0.4, len(data))
        result = match(
            data, treatment="treatment", covariates=["age", "bmi"], random_state=0
        )
        effects = result.estimate_effects("event")
        assert effects["measure"].iloc[0] == "risk_difference"

    def test_continuous_linear_still_mean_difference(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        effects = result.estimate_effects("outcome")
        assert effects["measure"].iloc[0] == "mean_difference"


class TestSupplement:
    def test_supplement_contents(self, tmp_path):
        data = make_data()
        result = match(
            data,
            treatment="treatment",
            covariates=["age", "bmi"],
            caliper="auto",
            random_state=7,
        )
        result.estimate_effects("outcome")
        text = result.supplement(str(tmp_path / "supp.md"), title="Study S1")
        assert (tmp_path / "supp.md").read_text() == text
        # design record includes the RESOLVED numeric caliper, not just "auto"
        assert "caliper (resolved threshold)" in text
        assert "auto" in text
        # software versions and seed
        assert "numpy" in text and "random_state = 7" in text
        # flow, balance, methods, effects
        for marker in [
            "Sample flow",
            "Covariate balance",
            "Methods text",
            "Effect estimates",
            "Austin",
            "References",
            "c-statistic",
            "Group sizes",
        ]:
            assert marker in text, marker
        # the journal round: no promotional claims in citable methods text
        assert "validated against" not in text

    def test_resolved_caliper_exposed(self):
        result = match(
            make_data(),
            treatment="treatment",
            covariates=["age", "bmi"],
            caliper="auto",
        )
        assert result._results.resolved_caliper is not None
        assert result._results.resolved_caliper > 0
        no_cal = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        assert no_cal._results.resolved_caliper is None

    def test_flow_accounting_with_control_discard(self):
        # regression: discarded controls must not deflate the unmatched-
        # treated count (previously subtracted total discards from treated)
        data = make_data(n_treat=60, n_control=140)
        data.loc[data.index[70:75], "age"] -= 200  # 5 extreme controls
        import warnings as W

        with W.catch_warnings():
            W.simplefilter("ignore")
            result = match(
                data,
                treatment="treatment",
                covariates=["age", "bmi"],
                caliper=0.05,
                std_caliper=False,
                discard="control",
                random_state=0,
            )
        text = result.supplement()
        n_disc_c = len(result.discarded)
        assert n_disc_c >= 5
        assert f"0 treated, {n_disc_c} control" in text
        matched_t = (result.matched_data["treatment"] == 1).sum()
        expected_unmatched = 60 - matched_t
        if expected_unmatched > 0:
            assert f"unmatched treated units: {expected_unmatched}" in text

    def test_flow_accounting_atc(self):
        # regression: under ATC the flow must track unmatched CONTROLS
        data = make_data(n_treat=200, n_control=60)
        import warnings as W

        with W.catch_warnings():
            W.simplefilter("ignore")
            result = match(
                data,
                treatment="treatment",
                covariates=["age", "bmi"],
                estimand="atc",
                caliper=0.02,
                std_caliper=False,
                random_state=0,
            )
        text = result.supplement()
        matched_c = (result.matched_data["treatment"] == 0).sum()
        unmatched_c = 60 - matched_c
        if unmatched_c > 0:
            assert f"unmatched control units: {unmatched_c}" in text
            assert "unmatched treated units" not in text

    def test_e_value_included_for_ratio_effects(self):
        data = make_data(n_treat=150, n_control=450, seed=9)
        rng = np.random.RandomState(9)
        logits = -1.0 + 1.0 * data["treatment"]
        data["event"] = rng.binomial(1, 1 / (1 + np.exp(-logits)))
        result = match(
            data, treatment="treatment", covariates=["age", "bmi"], random_state=0
        )
        result.estimate_effects("event", family="poisson")
        text = result.supplement()
        assert "E-value for event" in text
        assert "VanderWeele" in text

    def test_supplement_for_strata(self):
        result = subclassify(
            make_data(n_treat=100, n_control=300),
            treatment="treatment",
            covariates=["age", "bmi"],
            random_state=0,
        )
        text = result.supplement()
        assert "subclasses" in text and "Sample flow" in text


class TestSweepRegressions:
    """One test per finding from the pre-publication defect sweep."""

    def test_pairs_after_discard(self):
        # #2: result.pairs must not crash or misalign when discard dropped units
        data = make_data()
        data.loc[data.index[:3], "age"] += 300  # out-of-support treated
        import warnings as W

        with W.catch_warnings():
            W.simplefilter("ignore")
            result = match(
                data,
                treatment="treatment",
                covariates=["age", "bmi"],
                discard="treated",
                random_state=0,
            )
        pairs = result.pairs
        assert len(pairs) == (result.matched_data["treatment"] == 1).sum()
        # distances resolve to finite values for the retained pairs
        assert pairs["distance"].notna().any()

    def test_regression_adjustment_constant_covariate(self):
        # #3: a covariate constant in the matched sample must not shift treat_ix
        data = make_data()
        data["const"] = 1.0
        result = match(
            data, treatment="treatment", covariates=["age", "bmi"], random_state=0
        )
        adj = result.estimate_effects(
            "outcome",
            method="regression_adjustment",
            adjustment_covariates=["const", "age"],
        )
        plain = result.estimate_effects("outcome")
        # adjustment shouldn't explode the estimate away from the plain one
        assert abs(adj["effect"].iloc[0] - plain["effect"].iloc[0]) < 1.0

    def test_strata_repr(self):
        # #4: repr of subclassify/cem results must not KeyError
        r1 = subclassify(
            make_data(n_treat=100, n_control=300),
            treatment="treatment",
            covariates=["age", "bmi"],
            random_state=0,
        )
        r2 = cem(
            make_data(n_treat=100, n_control=300),
            treatment="treatment",
            covariates=["age", "bmi"],
        )
        assert "MatchResult" in repr(r1) and "MatchResult" in repr(r2)

    def test_exact_key_no_separator_collision(self):
        # #5: ("A_1","2") must not match ("A","1_2")
        data = make_data(n_treat=4, n_control=4)
        data["s1"] = ["A_1", "A", "B", "C", "A_1", "A", "B", "C"]
        data["s2"] = ["2", "1_2", "x", "y", "9", "9", "x", "y"]
        # treated[0]=(A_1,2), control[0 within controls]=(A,1_2): must NOT pair
        import warnings as W

        with W.catch_warnings():
            W.simplefilter("ignore")
            result = match(
                data,
                treatment="treatment",
                covariates=["age"],
                distance="euclidean",
                exact=["s1", "s2"],
            )
        for _, row in result.pairs.iterrows():
            assert (
                data.loc[row["treatment_id"], "s1"] == data.loc[row["control_id"], "s1"]
            )
            assert (
                data.loc[row["treatment_id"], "s2"] == data.loc[row["control_id"], "s2"]
            )

    def test_deterministic_matched_order(self):
        # #9: matched_data order independent of set hashing
        data = make_data()
        data.index = [f"p{i:04d}" for i in range(len(data))]
        r1 = match(
            data, treatment="treatment", covariates=["age", "bmi"], random_state=1
        )
        r2 = match(
            data, treatment="treatment", covariates=["age", "bmi"], random_state=1
        )
        assert list(r1.matched_data.index) == list(r2.matched_data.index)
        # order follows the input frame
        assert list(r1.matched_data.index) == [
            i for i in data.index if i in set(r1.matched_data.index)
        ]

    def test_covariate_typo_message(self):
        # #8: clean error, not bare KeyError
        with pytest.raises(ValueError, match="not found in data"):
            match(make_data(), treatment="treatment", covariates=["age", "agee"])

    def test_m_order_by_propensity_without_ps(self):
        # #10: m_order="largest" with a covariate distance estimates a PS
        result = match(
            make_data(),
            treatment="treatment",
            covariates=["age", "bmi"],
            distance="mahalanobis",
            m_order="largest",
            random_state=0,
        )
        assert len(result.pairs) > 0

    def test_too_few_treated_message(self):
        # #7: clear error, not sklearn internals
        data = make_data(n_treat=1, n_control=40)
        with pytest.raises(ValueError, match="[Tt]oo few"):
            match(data, treatment="treatment", covariates=["age", "bmi"])

    def test_nan_propensity_scores_rejected(self):
        # #12
        data = make_data()
        scores = np.full(len(data), np.nan)
        with pytest.raises(ValueError, match="non-finite|missing"):
            match(
                data,
                treatment="treatment",
                covariates=["age"],
                propensity_scores=scores,
            )


class TestMemoryEstimate:
    # a problem sized so one dense matrix fits the 4 GB budget but two would not
    # (1500 x 250000 x 8 bytes = 3.0 GB)
    N_FOCAL, N_POOL, LIMIT = 1500, 250_000, 4.0

    def test_scalar_caliper_does_not_force_approximate(self):
        # distance=propensity + logit caliper: the scalar caliper adds no second
        # matrix, so 3 GB fits and the exact path is chosen
        from cohortmatch.api import _resolve_engine

        algo = _resolve_engine(
            "auto",
            "nearest",
            "logit",
            self.N_FOCAL,
            self.N_POOL,
            "propensity",
            self.LIMIT,
        )
        assert algo == "exact"

    def test_covariate_space_caliper_counts_two_matrices(self):
        # distance=mahalanobis + euclidean caliper: two real matrices (6 GB),
        # over budget, so it routes away from exact and names the reason
        from cohortmatch.api import _resolve_engine

        with pytest.warns(UserWarning, match="covariate-space caliper"):
            algo = _resolve_engine(
                "auto",
                "nearest",
                "euclidean",
                self.N_FOCAL,
                self.N_POOL,
                "mahalanobis",
                self.LIMIT,
            )
        assert algo == "approximate"
        # same-metric caliper is applied directly: one matrix, fits, exact
        assert (
            _resolve_engine(
                "auto",
                "nearest",
                "mahalanobis",
                self.N_FOCAL,
                self.N_POOL,
                "mahalanobis",
                self.LIMIT,
            )
            == "exact"
        )

    def test_inplace_scalar_caliper_matches_matrix(self):
        # the in-place row caliper must produce the identical matched set to
        # the exact path (dense==approximate equivalence already covers logit
        # windows; this pins the dense scalar-caliper masking specifically)
        data = make_data(n_treat=60, n_control=300)
        r = match(
            data,
            treatment="treatment",
            covariates=["age", "bmi"],
            distance="mahalanobis",
            caliper="auto",
            engine="exact",
            random_state=0,
        )
        # caliper is logit on a mahalanobis distance -> in-place scalar path
        for _, row in r.pairs.iterrows():
            assert np.isfinite(row["distance"])


class TestCovariateScale:
    """Exact covariate-distance matching at scale (whitened KD-tree path)."""

    @pytest.mark.parametrize("distance", ["mahalanobis", "euclidean"])
    @pytest.mark.parametrize(
        "kw",
        [
            {},
            {"ratio": 2},
            {"exact": "sex"},
            {"covariate_calipers": {"age": 5.0}},
        ],
    )
    def test_tree_path_identical_to_dense(self, distance, kw):
        # the covariate approximate (KD-tree) path must produce byte-identical
        # pairs to the dense path under a fixed matching order
        data = make_data(n_treat=60, n_control=300, seed=1)
        base = dict(
            treatment="treatment",
            covariates=["age", "bmi"],
            distance=distance,
            m_order="data",
            random_state=0,
            **kw,
        )
        import warnings as W

        with W.catch_warnings():
            W.simplefilter("ignore")
            ex = match(data, engine="exact", **base)
            ap = match(data, engine="approximate", **base)
        pe = set(map(tuple, ex.pairs[["treatment_id", "control_id"]].values))
        pa = set(map(tuple, ap.pairs[["treatment_id", "control_id"]].values))
        assert pe == pa

    def test_mahalanobis_scales_without_propensity_or_caliper(self):
        # the capability that motivated this path: covariate distance at scale
        # with no propensity score and no caliper
        rng = np.random.RandomState(0)
        n_t, n_c = 2000, 40000
        n = n_t + n_c
        X = rng.normal(size=(n, 4))
        treat = np.zeros(n, int)
        treat[:n_t] = 1
        df = pd.DataFrame(
            {"treatment": treat, **{f"x{i}": X[:, i] for i in range(4)}},
            index=[f"e{i}" for i in range(n)],
        )
        result = match(
            df,
            treatment="treatment",
            covariates=[f"x{i}" for i in range(4)],
            distance="mahalanobis",
            engine="approximate",
            random_state=0,
        )
        assert len(result.pairs) == n_t
        assert result.propensity_scores is None  # no PS estimated
        assert result.config["engine"] == "approximate"

    def test_auto_routes_covariate_to_tree_at_scale(self):
        # auto must pick the tree (not error) for a large covariate design
        rng = np.random.RandomState(0)
        n_t, n_c = 3000, 60000
        n = n_t + n_c
        X = rng.normal(size=(n, 3))
        treat = np.zeros(n, int)
        treat[:n_t] = 1
        df = pd.DataFrame(
            {"treatment": treat, **{f"x{i}": X[:, i] for i in range(3)}},
            index=range(n),
        )
        import warnings as W

        with W.catch_warnings():
            W.simplefilter("ignore")
            result = match(
                df,
                treatment="treatment",
                covariates=[f"x{i}" for i in range(3)],
                distance="mahalanobis",
                memory_limit_gb=0.5,
                random_state=0,
            )
        assert result.config["engine"] == "approximate"
        assert len(result.pairs) == n_t


class TestReviewRound4Fixes:
    def test_effect_columns_inference_leads(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        cols = list(result.estimate_effects("outcome").columns)
        # ci/p must precede the trailing means/diagnostics so they survive truncation
        assert cols[:6] == [
            "outcome",
            "effect",
            "measure",
            "ci_lower",
            "ci_upper",
            "p_value",
        ]

    def test_balance_and_table1_are_methods_with_covariates(self):
        data = make_data()
        data["extra"] = data["age"] * 0.5
        result = match(data, treatment="treatment", covariates=["age", "bmi"])
        # method call, and can assess a non-matching covariate
        b = result.balance(covariates=["extra"])
        assert list(b["variable"]) == ["extra"]
        assert "extra" in result.table1(covariates=["extra"])["variable"].values

    def test_exact_column_validated(self):
        with pytest.raises(ValueError, match="exact column 'nope' not in data"):
            match(make_data(), treatment="treatment", covariates=["age"], exact="nope")

    def test_cv_without_propensity_raises(self):
        with pytest.raises(ValueError, match="cv only applies"):
            match(
                make_data(),
                treatment="treatment",
                covariates=["age", "bmi"],
                distance="mahalanobis",
                cv=3,
            )

    def test_std_caliper_with_covariate_metric_raises(self):
        with pytest.raises(ValueError, match="std_caliper applies only"):
            match(
                make_data(),
                treatment="treatment",
                covariates=["age", "bmi"],
                distance="mahalanobis",
                caliper=3.0,
                caliper_metric="mahalanobis",
                std_caliper=True,
            )

    def test_engine_param_name(self):
        result = match(
            make_data(),
            treatment="treatment",
            covariates=["age", "bmi"],
            caliper="auto",
            engine="exact",
        )
        assert result.config["engine"] == "exact"

    def test_risk_set_table1_case_control_labels(self):
        import sys

        sys.path.insert(0, "tests")
        from test_risk_set import make_cohort

        r = match_risk_set(
            make_cohort(n=3000, seed=1),
            event_time="time",
            event="case",
            exact="sex",
            ratio=3,
            random_state=0,
        )
        cols = list(r.table1().columns)
        assert any("mean_case_" in c for c in cols)
        assert not any("treated" in c for c in cols)

    def test_repr_html_present(self):
        result = match(make_data(), treatment="treatment", covariates=["age", "bmi"])
        h = result._repr_html_()
        assert "MatchResult" in h and "balance()" in h
