"""Correctness and edge-case tests for the matching algorithms.

Covers optimal matching on degenerate distances and ratio optimality,
consistency between the exact and approximate engines under replacement,
covariate-distance reporting, and covariate_weights validation.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linear_sum_assignment

from cohortmatch import match
from cohortmatch.matching.optimal import optimal_match


def test_covariate_replace_honored_on_approximate_engine():
    """With replacement, the covariate KD-tree path reuses controls like the
    exact path: three treated on top of one control all match it.
    """
    df = pd.DataFrame(
        {"t": [1, 1, 1, 0, 0], "x": [0.0, 0.0, 0.0, 0.0, 100.0]},
        index=["t0", "t1", "t2", "c0", "c1"],
    )
    exact = match(
        df,
        treatment="t",
        covariates=["x"],
        distance="euclidean",
        replace=True,
        engine="exact",
    )
    approx = match(
        df,
        treatment="t",
        covariates=["x"],
        distance="euclidean",
        replace=True,
        engine="approximate",
    )

    assert approx.pairs["treatment_id"].nunique() == 3
    assert approx.pairs["control_id"].value_counts().max() >= 2
    assert (
        exact.pairs["treatment_id"].nunique() == approx.pairs["treatment_id"].nunique()
    )


def test_optimal_finds_offdiagonal_exact_match_with_zero_distances():
    """Optimal matching finds a valid off-diagonal exact match even when every
    permitted distance is zero.
    """
    data = pd.DataFrame(
        {"treatment": [1, 1, 0, 0], "grp": ["A", "B", "B", "A"]},
    )
    treat_mask = data["treatment"] == 1
    dist = np.zeros((2, 2))

    pairs, _ = optimal_match(data, dist, treat_mask, exact_match_cols=["grp"])
    matched = sum(len(v) for v in pairs.values())
    assert matched == 2  # t0(A)->c1(A), t1(B)->c0(B)


def _global_kopt_total(dist: np.ndarray, k: int) -> float:
    """Reference minimum-cost 1:k without replacement (duplicate each treated k times)."""
    dk = np.repeat(dist, k, axis=0)
    ri, ci = linear_sum_assignment(dk)
    return float(dist[ri // k, ci].sum())


def test_optimal_ratio_matching_minimizes_total_distance():
    """1:k optimal matching attains the global minimum total distance."""
    rng = np.random.RandomState(0)
    data = pd.DataFrame({"treatment": [1, 1, 1, 0, 0, 0, 0, 0, 0]})
    treat_mask = data["treatment"] == 1
    for _ in range(30):
        dist = np.round(rng.rand(3, 6) * 3, 3)
        _, distances = optimal_match(data, dist, treat_mask, ratio=2.0)
        got = float(sum(distances))
        opt = _global_kopt_total(dist, 2)
        assert got == pytest.approx(opt, abs=1e-9), (got, opt)


def test_negative_covariate_weights_rejected():
    """Negative covariate_weights raise instead of yielding NaN distances."""
    df = pd.DataFrame(
        {"t": [1, 1, 0, 0], "x": [0.0, 1.0, 0.2, 0.9], "y": [0.0, 1.0, 0.1, 0.8]},
        index=["t0", "t1", "c0", "c1"],
    )
    with pytest.raises(ValueError, match="covariate_weights"):
        match(
            df,
            treatment="t",
            covariates=["x", "y"],
            distance="euclidean",
            covariate_weights={"x": -1.0, "y": 1.0},
        )


def test_covariate_approximate_reports_real_distances():
    """The covariate approximate path reports its exact pair distances, equal
    to the exact engine's for the same pairs.
    """
    rng = np.random.RandomState(3)
    n_t, n_c = 6, 15
    df = pd.DataFrame(
        {
            "t": [1] * n_t + [0] * n_c,
            "a": rng.rand(n_t + n_c),
            "b": rng.rand(n_t + n_c),
        },
        index=[f"u{i}" for i in range(n_t + n_c)],
    )
    common = dict(
        treatment="t", covariates=["a", "b"], distance="euclidean", m_order="data"
    )
    exact = match(df, engine="exact", **common)
    approx = match(df, engine="approximate", **common)

    assert approx.pairs["distance"].notna().all()
    e = exact.pairs.set_index(["treatment_id", "control_id"])["distance"]
    a = approx.pairs.set_index(["treatment_id", "control_id"])["distance"]
    np.testing.assert_allclose(
        a.reindex(e.index).to_numpy(), e.to_numpy(), rtol=1e-6, atol=1e-9
    )
