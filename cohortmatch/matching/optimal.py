"""Optimal matching algorithm implementation using the Hungarian algorithm.

This module provides an implementation of the optimal matching algorithm,
which finds the matching that minimizes the total distance across all pairs.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from cohortmatch.matching._utils import _apply_exact_matching, resolve_tie_break
from cohortmatch.utils.logging import get_logger

# Create a logger for this module
logger = get_logger(__name__)


def optimal_match(
    data: pd.DataFrame,
    distance_matrix: np.ndarray,
    treat_mask: np.ndarray,
    exact_match_cols: list[str] | None = None,
    ratio: float = 1.0,
    replace: bool = False,
    tie_break: str = "first",
    random_state: int | None = None,
) -> tuple[dict[int, list[int]], list[float]]:
    """Implement optimal matching algorithm using the Hungarian algorithm.
    The algorithm takes a distance matrix between treatment and control units and
    finds the optimal matching that minimizes the total distance. Indices in the
    returned dictionary are positions in the arrays of treatment and control units,
    not original dataframe indices. The pipeline translates these to
    participant IDs.

    Ratio matching is implemented using one of two strategies:
    - With replacement (`replace=True`): The control unit matrix is tiled `ratio` times,
      allowing controls to be matched multiple times across the larger matrix.
    - Without replacement (`replace=False`): The matching algorithm is run iteratively,
      removing used controls from the pool in each iteration until the desired
      ratio is achieved or no more matches can be found.

    Args:
        data: DataFrame containing the data
        distance_matrix: Pre-computed and pre-calipered distance matrix (n_treatment x n_control)
        treat_mask: Boolean mask indicating treatment units
        exact_match_cols: Columns to match exactly on
        ratio: Matching ratio (e.g., 2 means 1:2 matching)
        replace: Whether to allow replacement in matching
        tie_break: How equally good assignments are resolved. The total
            distance is unique but the assignment achieving it need not be;
            the solver settles those degenerate optima by column order, which
            is input row order. "random" shuffles the control columns first,
            so the choice among equally optimal solutions is uniform.
        random_state: Seed for tie_break="random"
    Returns:
        Tuple of (match_pairs, match_distances)
    """
    logger.info(f"Starting optimal matching (replace={replace}, ratio={ratio})")
    n_treat, n_control = distance_matrix.shape
    treat_indices = np.where(treat_mask)[0]

    # Create working copy of distance matrix and apply exact matching
    distances = distance_matrix.copy()
    if exact_match_cols:
        logger.debug(f"Applying exact matching on columns: {exact_match_cols}")
        control_indices = np.where(~treat_mask)[0]
        distances = _apply_exact_matching(
            data, treat_indices, control_indices, distances, exact_match_cols
        )

    # Degenerate optima are settled by column order; permuting the columns
    # makes that choice random instead of a function of the input row order.
    # Everything below works in permuted column space; only the control index
    # recorded for a match is mapped back.
    tie_break = resolve_tie_break(tie_break)
    col_map = np.arange(n_control)
    if tie_break == "random":
        col_map = np.random.RandomState(random_state).permutation(n_control)
        distances = distances[:, col_map]

    # Initialize match storage
    match_pairs: dict[int, list[int]] = {i: [] for i in range(n_treat)}
    match_distances: list[float] = []

    # Replace inf with a large finite value for the solver
    finite_distances = distances[~np.isinf(distances)]
    if finite_distances.size == 0:
        logger.warning(
            "No finite distances available for matching. No pairs will be found."
        )
        return {}, []  # Return empty matches

    max_finite = np.nanmax(finite_distances)

    if replace:
        # --- WITH REPLACEMENT: Use matrix tiling ---
        n_copies = int(ratio)
        if n_copies > 1:
            logger.debug(
                f"Implementing {ratio}:1 matching by tiling control matrix {n_copies} times."
            )
            solver_distances = np.tile(distances, (1, n_copies))
        else:
            solver_distances = distances.copy()

        solver_distances[np.isinf(solver_distances) | np.isnan(solver_distances)] = (
            max_finite * 1e6
        )

        row_ind, col_ind = linear_sum_assignment(solver_distances)

        # Process matches, mapping tiled columns back to original
        for r, c in zip(row_ind, col_ind, strict=False):
            if np.isinf(distances[r, c % n_control]):
                continue
            solver_c_idx = c % n_control
            match_pairs[r].append(int(col_map[solver_c_idx]))
            match_distances.append(distances[r, solver_c_idx])

    else:
        # --- WITHOUT REPLACEMENT: Use iterative solving ---
        solver_distances = distances.copy()
        solver_distances[np.isinf(solver_distances) | np.isnan(solver_distances)] = (
            max_finite * 1e6
        )
        used_controls = np.zeros(n_control, dtype=bool)

        for i in range(int(ratio)):
            logger.debug(f"Matching iteration {i + 1}/{int(ratio)}")
            if np.any(used_controls):
                solver_distances[:, used_controls] = max_finite * 1e6

            row_ind, col_ind = linear_sum_assignment(solver_distances)

            new_matches_found = 0
            for r, c in zip(row_ind, col_ind, strict=False):
                if np.isinf(distances[r, c]) or used_controls[c]:
                    continue

                match_pairs[r].append(int(col_map[c]))
                match_distances.append(distances[r, c])
                used_controls[c] = True
                new_matches_found += 1

            if new_matches_found == 0:
                logger.info(f"No further matches found on iteration {i + 1}. Stopping.")
                break

    total_matches = sum(len(v) for v in match_pairs.values())
    logger.info(f"Optimal matching complete: {total_matches} total matches found")
    if match_distances:
        logger.debug(
            f"Match distances - min: {min(match_distances):.4f}, "
            f"mean: {np.mean(match_distances):.4f}, "
            f"max: {max(match_distances):.4f}"
        )

    return match_pairs, match_distances
