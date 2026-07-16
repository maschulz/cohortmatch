"""Tests for the bundled Lalonde dataset and an end-to-end run on it."""

import numpy as np
import pytest

from cohortmatch import match
from cohortmatch.datasets import load_lalonde


class TestLoadLalonde:
    def test_structure(self):
        data = load_lalonde()
        assert len(data) == 614
        assert (data["treat"] == 1).sum() == 185
        assert (data["treat"] == 0).sum() == 429
        for col in [
            "age",
            "educ",
            "race",
            "married",
            "nodegree",
            "re74",
            "re75",
            "re78",
            "black",
            "hispan",
        ]:
            assert col in data.columns
        assert data.index.is_unique
        assert str(data.index[0]).startswith("NSW")

    def test_known_values(self):
        data = load_lalonde()
        # canonical facts about this dataset
        assert data["re78"].mean() == pytest.approx(6792.834, abs=0.01)
        assert (data["race"] == "black").sum() == 243

    def test_end_to_end_match(self):
        data = load_lalonde()
        result = match(
            data,
            treatment="treat",
            covariates=[
                "age",
                "educ",
                "black",
                "hispan",
                "married",
                "nodegree",
                "re74",
                "re75",
            ],
            random_state=42,
        )
        matched = result.matched_data
        assert (matched["treat"] == 1).sum() == 185
        assert (matched["treat"] == 0).sum() == 185
        # matching must reduce the worst covariate imbalance substantially
        balance = result.balance()
        assert balance["smd_after"].abs().max() < balance["smd_before"].abs().max()
        effects = result.estimate_effects("re78")
        assert np.isfinite(effects["effect"].iloc[0])
