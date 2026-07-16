#!/usr/bin/env python
"""Deterministic large synthetic cohort for at-scale MatchIt validation.

Writes a 20k treated x 480k control cohort with a full-sample logistic
propensity score column ("ps") and an outcome with a true treatment effect
of 2.0. Both cohortmatch and MatchIt match on the identical "ps" column, so
the comparison isolates matching quality from propensity estimation.

Usage: python validation/make_scale_cohort.py <out.csv>
"""

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

N_TREAT, N_CONTROL, N_COVS, SEED = 20_000, 480_000, 8, 0
TRUE_EFFECT = 2.0


def make_scale_cohort() -> pd.DataFrame:
    rng = np.random.RandomState(SEED)
    n = N_TREAT + N_CONTROL
    X = rng.normal(size=(n, N_COVS))
    logits = 0.8 * X[:, 0] + 0.5 * X[:, 1] - 0.4 * X[:, 2]
    noisy = 1 / (1 + np.exp(-logits)) + rng.uniform(0, 0.3, n)
    treated = np.zeros(n, dtype=int)
    treated[np.argsort(-noisy)[:N_TREAT]] = 1

    data = pd.DataFrame(X, columns=[f"x{i}" for i in range(N_COVS)])
    data["treatment"] = treated
    data["y"] = (
        1.5 * X[:, 0]
        - 1.0 * X[:, 2]
        + 0.5 * X[:, 4]
        + TRUE_EFFECT * treated
        + rng.normal(0, 1, n)
    )

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X, treated)
    data["ps"] = model.predict_proba(X)[:, 1]
    data.index.name = "id"
    return data


if __name__ == "__main__":
    out = sys.argv[1]
    data = make_scale_cohort()
    data.to_csv(out)
    print(f"wrote {out}: {data.shape}, sd(ps)={data['ps'].std(ddof=1):.6f}")
