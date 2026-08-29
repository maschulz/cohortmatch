"""Optimal matching via minimum-cost bipartite assignment.

The matching that minimizes total distance is found with a sparse min-cost
full matching (``scipy.sparse.csgraph.min_weight_full_bipartite_matching``).
Forbidden pairs (exact-stratum or caliper violations) are simply absent from
the graph, so there is no large-sentinel arithmetic to misfire when the
permitted distances are small or zero. Ratio (1:k) matching duplicates each
treated node k times and solves once, giving the true global 1:k optimum
rather than a sequence of greedy 1:1 rounds. A per-slot dummy "no-match"
column at a dominating cost lets anchors that cannot be matched drop out, so
the solver maximizes the number of real matches and then minimizes total
distance.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from cohortmatch.matching._utils import _apply_exact_matching
from cohortmatch.utils.logging import get_logger

logger = get_logger(__name__)


def optimal_match(
    data: pd.DataFrame,
    distance_matrix: np.ndarray,
    treat_mask: np.ndarray,
    exact_match_cols: list[str] | None = None,
    ratio: float = 1.0,
    replace: bool = False,
) -> tuple[dict[int, list[int]], list[float]]:
    """Optimal matching that minimizes total distance.

    Indices in the returned dictionary are positions in the arrays of treatment
    and control units, not original dataframe indices; the pipeline translates
    these to participant IDs.

    Without replacement (default) the result is a true 1:k minimum-cost matching
    with each control used at most once. With replacement each treated unit
    independently takes its k nearest permitted controls, and controls may be
    reused across treated units.

    Args:
        data: DataFrame containing the data (used only for exact matching).
        distance_matrix: Pre-computed, pre-calipered distance matrix
            (n_treatment x n_control); forbidden pairs are +inf.
        treat_mask: Boolean mask indicating treatment units.
        exact_match_cols: Columns to match exactly on.
        ratio: Matching ratio (e.g. 2 means 1:2 matching).
        replace: Whether controls may be reused across treated units.

    Returns:
        Tuple of (match_pairs, match_distances).
    """
    logger.info(f"Starting optimal matching (replace={replace}, ratio={ratio})")
    n_treat, n_control = distance_matrix.shape

    # Working copy; exact matching marks forbidden pairs as +inf
    distances = distance_matrix.copy()
    if exact_match_cols:
        logger.debug(f"Applying exact matching on columns: {exact_match_cols}")
        treat_indices = np.where(treat_mask)[0]
        control_indices = np.where(~treat_mask)[0]
        distances = _apply_exact_matching(
            data, treat_indices, control_indices, distances, exact_match_cols
        )

    permitted = np.isfinite(distances)
    if not permitted.any():
        logger.warning(
            "No finite distances available for matching. No pairs will be found."
        )
        return {}, []

    k = max(1, int(ratio))
    match_pairs: dict[int, list[int]] = {i: [] for i in range(n_treat)}
    match_distances: list[float] = []

    if replace:
        # With replacement the problem separates across treated units: each one
        # independently takes its k nearest permitted controls.
        for t in range(n_treat):
            cand = np.where(permitted[t])[0]
            if cand.size == 0:
                continue
            order = cand[np.argsort(distances[t, cand], kind="stable")][:k]
            for c in order:
                match_pairs[t].append(int(c))
                match_distances.append(float(distances[t, c]))
        logger.info(f"Optimal matching complete: {len(match_distances)} total matches")
        return match_pairs, match_distances

    # --- without replacement: true 1:k minimum-cost matching ----------------
    # Duplicate each treated unit into k row-slots; columns are the real
    # controls plus one dedicated dummy no-match column per row-slot.
    n_rows = n_treat * k
    max_finite = float(distances[permitted].max())
    # Shift real weights strictly positive: the sparse matcher treats an
    # explicit 0 as "no edge", and a uniform shift never changes which
    # assignment is optimal.
    eps = 1.0
    # Dummy "no-match" cost, tiered by slot so a focal's first match is filled
    # before any focal's second, and so on: leaving slot j unmatched costs
    # strictly more than every later slot of every focal plus all real
    # distances combined. `base` dominates any all-real assignment; the tier
    # factor dominates the focal count. Real matches fill the highest-cost
    # (earliest) slots first, so 1:k matching never sacrifices focal coverage
    # (and the ATT population) to lower the total distance.
    base = n_rows * (max_finite + eps) + 1.0
    tier_factor = n_treat + 1

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for t in range(n_treat):
        cand = np.where(permitted[t])[0]
        for slot in range(k):
            r = t * k + slot
            for c in cand:
                rows.append(r)
                cols.append(int(c))
                vals.append(float(distances[t, c]) + eps)
            rows.append(r)
            cols.append(n_control + r)  # this slot's dedicated dummy column
            vals.append(base * tier_factor ** (k - 1 - slot))

    graph = csr_matrix((vals, (rows, cols)), shape=(n_rows, n_control + n_rows))
    row_ind, col_ind = min_weight_full_bipartite_matching(graph)

    for r, c in zip(row_ind, col_ind, strict=False):
        if c >= n_control:
            continue  # matched to its dummy -> this slot stays unmatched
        t = r // k
        match_pairs[t].append(int(c))
        match_distances.append(float(distances[t, c]))

    logger.info(f"Optimal matching complete: {len(match_distances)} total matches")
    if match_distances:
        logger.debug(
            f"Match distances - min: {min(match_distances):.4f}, "
            f"mean: {np.mean(match_distances):.4f}, "
            f"max: {max(match_distances):.4f}"
        )
    return match_pairs, match_distances
