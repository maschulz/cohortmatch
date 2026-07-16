#!/usr/bin/env python
"""Basic CohortMatch example: match, check balance, estimate an effect.

Self-contained; run directly. Saves a love plot if the viz extras are
installed.
"""

import numpy as np
import pandas as pd

from cohortmatch import match


def generate_synthetic_data(n=1000, random_state=42):
    """Confounded treatment assignment with a known effect of 2.0."""
    rng = np.random.RandomState(random_state)
    data = pd.DataFrame(
        {
            "age": rng.normal(50, 10, n),
            "bmi": rng.normal(25, 5, n),
            "bp": rng.normal(120, 15, n),
            "sex": rng.binomial(1, 0.5, n),
        }
    )
    p = 1 / (1 + np.exp(-(data["age"] / 10 + data["bmi"] / 5 - 10)))
    data["treatment"] = rng.binomial(1, p)
    data["outcome"] = (
        5
        + 0.1 * data["age"]
        + 0.2 * data["bmi"]
        + 0.1 * data["bp"]
        + 2.0 * data["treatment"]
        + rng.normal(0, 1, n)
    )
    return data


def main():
    data = generate_synthetic_data()

    result = match(
        data,
        treatment="treatment",
        covariates=["age", "bmi", "bp", "sex"],
        distance="mahalanobis",
        caliper="auto",  # Mahalanobis matching within a propensity caliper
        exact="sex",
        random_state=42,
    )

    print(result.summary())
    print()
    print(result.balance().round(3))

    effects = result.estimate_effects("outcome")
    print()
    print(effects[["outcome", "effect", "ci_lower", "ci_upper", "p_value"]].round(3))

    try:
        result.plot_love_plot().savefig("love_plot.png", bbox_inches="tight")
        print("\nWrote love_plot.png")
    except ImportError:
        print("\nInstall the viz extras for plots: pip install 'cohortmatch[viz]'")


if __name__ == "__main__":
    main()
