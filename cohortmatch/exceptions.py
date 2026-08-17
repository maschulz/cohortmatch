"""Exceptions for CohortMatch."""


class MatchingError(Exception):
    """Base class for matching-related errors."""


class NoMatchesError(MatchingError):
    """Raised when a matching run produces zero pairs.

    This typically means the caliper is too strict, exact-matching columns
    admit no cross-group agreement, or the groups do not overlap.
    """


class IncompleteMatchWarning(UserWarning):
    """Anchor units or requested matches were lost (caliper, exact, pool)."""


class CommonSupportWarning(UserWarning):
    """Units outside the common propensity support were discarded."""


class ApproximateMatchWarning(UserWarning):
    """The memory-efficient approximate algorithm was selected automatically."""


class TieBreakWarning(UserWarning):
    """Exact distance ties are common and are being broken by input row order."""
