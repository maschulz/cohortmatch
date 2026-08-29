"""The default analysis is deterministic; cross-fitting is opt-in via cv."""

import numpy as np

from cohortmatch import match
from cohortmatch.datasets import load_lalonde

COVS = ["age", "educ", "race", "married", "re74", "re75"]


def _pairset(result):
    return set(map(tuple, result.pairs[["treatment_id", "control_id"]].to_numpy()))


def test_default_matching_is_reproducible():
    """Two unseeded default runs produce identical scores, pairs, and effect."""
    lal = load_lalonde()
    r1 = match(lal, treatment="treat", covariates=COVS)
    r2 = match(lal, treatment="treat", covariates=COVS)

    np.testing.assert_array_equal(
        r1.propensity_scores.to_numpy(), r2.propensity_scores.to_numpy()
    )
    assert _pairset(r1) == _pairset(r2)
    assert (
        r1.estimate_effects("re78").iloc[0]["effect"]
        == r2.estimate_effects("re78").iloc[0]["effect"]
    )


def test_cross_fitting_is_opt_in():
    """cv=k cross-fits (out-of-fold scores differ from the full-sample fit) and
    is itself reproducible given a seed.
    """
    lal = load_lalonde()
    full = match(lal, treatment="treat", covariates=COVS)
    cf1 = match(lal, treatment="treat", covariates=COVS, cv=5, random_state=0)
    cf2 = match(lal, treatment="treat", covariates=COVS, cv=5, random_state=0)

    np.testing.assert_array_equal(
        cf1.propensity_scores.to_numpy(), cf2.propensity_scores.to_numpy()
    )
    assert not np.allclose(
        full.propensity_scores.to_numpy(), cf1.propensity_scores.to_numpy()
    )


def test_few_clusters_warns():
    """Cluster-robust inference on very few match groups warns."""
    import pandas as pd
    import pytest

    df = pd.DataFrame(
        {
            "t": [1, 1, 1, 0, 0, 0, 0, 0],
            "x": [0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.9, 0.95],
        },
        index=[f"u{i}" for i in range(8)],
    )
    result = match(df, treatment="t", covariates=["x"], distance="euclidean")
    with pytest.warns(UserWarning, match="few"):
        result.estimate_effects("x")


def test_supplement_labels_in_sample_auc(tmp_path):
    """The default full-sample c-statistic is labeled in-sample, not
    cross-validated, and no mean/std AUC (which would imply CV) is exposed.
    """
    lal = load_lalonde()
    r = match(lal, treatment="treat", covariates=COVS, caliper="auto")
    metrics = r.propensity_metrics or {}
    assert "mean_auc" not in metrics and "std_auc" not in metrics
    out = tmp_path / "supp.md"
    r.supplement(str(out))
    text = out.read_text()
    assert "c-statistic (in-sample)" in text
