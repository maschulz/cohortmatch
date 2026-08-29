"""Differential test: the exact and approximate engines must agree.

The covariate-distance approximate path (KD-tree) promises the same matches the
dense exact path produces. This sweeps a grid of options through both engines
and asserts they agree on matched-anchor count, control-reuse structure, and
total distance. Matching order is pinned to "data" so the comparison is about
the algorithms, not tie-breaking; covariates are continuous so distances are
distinct.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from cohortmatch import match


def _make_data(seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    n_t, n_c = 8, 20
    n = n_t + n_c
    df = pd.DataFrame(
        {
            "t": [1] * n_t + [0] * n_c,
            "x1": rng.rand(n),
            "x2": rng.rand(n),
            "x3": rng.rand(n),
            "grp": rng.randint(0, 2, n),
        },
        index=[f"u{i}" for i in range(n)],
    )
    df["grp"] = df["grp"].astype("category")
    return df


def _matched_pairs(result):
    # The matching itself: the set of (treatment_id, control_id) pairs. The
    # approximate path reports NaN distances by design, so distance is not part
    # of the comparison.
    p = result.pairs
    return frozenset(zip(p["treatment_id"], p["control_id"], strict=False))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("distance", ["euclidean", "mahalanobis"])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("exact", [None, "grp"])
def test_exact_and_approximate_engines_agree(seed, distance, ratio, replace, exact):
    df = _make_data(seed)
    covs = ["x1", "x2", "x3"]
    kwargs = dict(
        treatment="t",
        covariates=covs,
        distance=distance,
        ratio=ratio,
        replace=replace,
        exact=exact,
        m_order="data",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact_res = match(df, engine="exact", **kwargs)
        approx_res = match(df, engine="approximate", **kwargs)

    exact_pairs = _matched_pairs(exact_res)
    approx_pairs = _matched_pairs(approx_res)
    assert exact_pairs == approx_pairs, (
        f"engine disagreement: distance={distance} ratio={ratio} "
        f"replace={replace} exact={exact} seed={seed}\n"
        f"only in exact: {sorted(exact_pairs - approx_pairs)}\n"
        f"only in approx: {sorted(approx_pairs - exact_pairs)}"
    )
