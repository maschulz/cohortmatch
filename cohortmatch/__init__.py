"""CohortMatch: statistical matching for cohort studies.

Match treatment and control units on covariates, assess balance, and
estimate treatment effects, from small samples to biobank scale.
"""

__version__ = "0.1.0"

import logging

# Libraries are quiet by default; call configure_logging() for progress output.
logging.getLogger(__name__).addHandler(logging.NullHandler())

from cohortmatch import datasets
from cohortmatch.api import MatchResult, MatchSummary, cem, match, subclassify
from cohortmatch.evalue import e_value
from cohortmatch.exceptions import (
    ApproximateMatchWarning,
    CommonSupportWarning,
    IncompleteMatchWarning,
    MatchingError,
    NoMatchesError,
)
from cohortmatch.risk_set import RiskSetResult, RiskSetSummary, match_risk_set
from cohortmatch.utils.logging import configure_logging

__all__ = [
    "ApproximateMatchWarning",
    "CommonSupportWarning",
    "IncompleteMatchWarning",
    "MatchResult",
    "MatchSummary",
    "MatchingError",
    "NoMatchesError",
    "RiskSetResult",
    "RiskSetSummary",
    "cem",
    "configure_logging",
    "datasets",
    "e_value",
    "match",
    "match_risk_set",
    "subclassify",
]
