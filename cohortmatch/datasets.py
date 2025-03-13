"""Bundled example datasets."""

from importlib import resources

import pandas as pd


def load_lalonde() -> pd.DataFrame:
    """Load the Lalonde (1986) / Dehejia-Wahba (1999) job-training data.

    The canonical matching benchmark: 185 treated men from the NSW program
    and 429 non-experimental comparison units from the PSID, as shipped with
    R's MatchIt package.

    Returns:
        DataFrame indexed by unit id (NSW1..., PSID1...) with columns
        treat, age, educ, race, married, nodegree, re74, re75, re78, plus
        binary indicator columns black and hispan derived from race.
    """
    with resources.files("cohortmatch.data").joinpath("lalonde.csv").open() as f:
        data = pd.read_csv(f, index_col=0)
    data.index.name = "id"
    data["black"] = (data["race"] == "black").astype(int)
    data["hispan"] = (data["race"] == "hispan").astype(int)
    return data
