"""Internal utility functions for matching algorithms."""

import warnings
from typing import Any

import numpy as np
import pandas as pd

from cohortmatch.exceptions import TieBreakWarning
from cohortmatch.utils.logging import get_logger

logger = get_logger(__name__)

TIE_BREAKS = ("first", "random")

# Warn once the pool carries this share of duplicate matching keys: duplicates
# guarantee exact ties, and under tie_break="first" the winner is then decided
# by input row order alone.
_TIE_WARN_SHARE = 0.05

# Cap on rows scanned for duplicate multi-column keys (a single-column key is
# cheap enough to check whole)
_TIE_CHECK_MAX_ROWS = 50_000


def resolve_tie_break(tie_break: str | None) -> str:
    """Validate a tie_break setting, defaulting to "first"."""
    if tie_break is None:
        return "first"
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"tie_break must be one of {TIE_BREAKS}, got {tie_break!r}")
    return tie_break


def argmin_with_tie_break(distances: np.ndarray, tie_break: str, rng: Any) -> int:
    """Index of the smallest distance, with an explicit tie policy.

    "first" keeps numpy's argmin (the earliest row among tied candidates);
    "random" draws uniformly among all candidates that attain the minimum,
    so the matched set no longer depends on the order of the input rows.
    """
    j = int(np.argmin(distances))
    if tie_break != "random":
        return j
    best = distances[j]
    if np.isinf(best):  # nothing selectable; caller checks and breaks out
        return j
    tied = np.flatnonzero(distances == best)
    if tied.size == 1:
        return j
    return int(rng.choice(tied))


def shuffle_tied_neighbors(
    distances: np.ndarray, indices: np.ndarray, rng: Any
) -> np.ndarray:
    """Permute distance-sorted neighbours within runs of equal distance.

    Spatial-tree queries return neighbours sorted by distance, and equal
    distances come back in tree order, which traces the input rows. Shuffling
    each tied run makes the pick uniform among tied candidates. A tie group
    split across the k-neighbour boundary is only partially shuffled; k grows
    geometrically, so that affects the tail of a batch, not the argmin.
    """
    out = np.asarray(indices).copy()
    start = 0
    n = len(distances)
    for i in range(1, n + 1):
        if i == n or distances[i] != distances[start]:
            if i - start > 1:
                out[start:i] = rng.permutation(out[start:i])
            start = i
    return out


def warn_if_pool_keys_tie(keys: np.ndarray | None, tie_break: str) -> None:
    """Warn when the pool has duplicate matching keys and ties go by row order.

    Duplicate keys (categorical or coarsened covariates, a propensity model
    over a handful of binary predictors) mean many candidates sit at exactly
    the same distance from an anchor. Under tie_break="first" the earlier
    input row always wins those, so anything that correlates with row order —
    site, batch, enrollment date — leaks into the matched set silently.

    Multi-column keys are checked on a strided subsample so the scan stays
    negligible against the match itself. That understates duplication in a
    large pool where duplicates are rare, and those are precisely the pools
    where tie-breaking barely moves the result; a coarse key still reads as
    near-fully duplicated at any sample size.
    """
    if tie_break != "first" or keys is None:
        return
    keys = np.asarray(keys)
    n = keys.shape[0]
    if n == 0:
        return
    keys = keys.reshape(n, -1)
    if keys.shape[1] > 1 and n > _TIE_CHECK_MAX_ROWS:
        keys = keys[:: int(np.ceil(n / _TIE_CHECK_MAX_ROWS))]
    # pandas handles the mixed and non-numeric dtypes that reach exact-matching
    # designs, where np.unique(axis=0) would not
    duplicate_share = float(pd.DataFrame(keys).duplicated().mean())
    if duplicate_share < _TIE_WARN_SHARE:
        return
    warnings.warn(
        f"{duplicate_share:.0%} of pool units share a matching key with another "
        "pool unit, so many candidates are exactly tied. With tie_break='first' "
        "(the default) the earlier input row wins every tie, which biases the "
        "matched set whenever row order carries information (site, batch, "
        "enrollment date). Pass tie_break='random' with a random_state to draw "
        "among tied candidates instead, and vary the seed to check how much the "
        "result depends on tie-breaking.",
        TieBreakWarning,
        stacklevel=3,
    )


def _apply_exact_matching(
    data: pd.DataFrame,
    treat_indices: np.ndarray,
    control_indices: np.ndarray,
    distances: np.ndarray,
    exact_match_cols: list[str],
) -> np.ndarray:
    """Apply exact matching constraints efficiently using pandas operations."""
    logger.debug(f"Applying exact matching on {len(exact_match_cols)} columns")

    # Extract matching columns
    treat_data = data.iloc[treat_indices][exact_match_cols]
    control_data = data.iloc[control_indices][exact_match_cols]

    # Factorize the tuple of exact-match values (collision-free, unlike a
    # string join which conflates ("A_1","2") with ("A","1_2"))
    import pandas as pd

    n_treat = len(treat_data)
    combined = pd.MultiIndex.from_frame(
        pd.concat([treat_data, control_data], ignore_index=True)
    ).to_flat_index()
    codes = pd.factorize(combined)[0]
    treat_keys = pd.Series(codes[:n_treat])
    control_keys = pd.Series(codes[n_treat:])

    # Create match matrix (n_treat x n_control)
    match_matrix = np.zeros((len(treat_indices), len(control_indices)), dtype=bool)

    # Vectorized exact matching
    unique_treat_keys = set(treat_keys)
    logger.debug(
        f"Found {len(unique_treat_keys)} unique combinations in treatment group"
    )

    for i, t_key in enumerate(treat_keys):
        match_matrix[i] = control_keys == t_key

    # Set distances to infinity where exact matches don't exist
    n_before = np.sum(~np.isinf(distances))
    distances[~match_matrix] = np.inf
    n_after = np.sum(~np.isinf(distances))

    logger.debug(f"Exact matching removed {n_before - n_after} potential matches")
    logger.debug(
        f"Treatment units with at least one match: {np.sum(np.any(~np.isinf(distances), axis=1))}"
    )

    return distances
