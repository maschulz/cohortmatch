"""Greedy matching algorithm implementation using numpy's efficient operations."""

import logging

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from cohortmatch.matching._utils import _apply_exact_matching
from cohortmatch.utils.logging import get_logger

logger = get_logger(__name__)


def _matching_order(
    distances: np.ndarray,
    m_order: str | None,
    order_scores: np.ndarray | None,
    rng,
    random_state: int | None,
) -> np.ndarray:
    """Order in which treated units pick their matches."""
    n_treat = distances.shape[0]
    if m_order is None:
        # fewest potential matches first (helps under exact/caliper constraints)
        return np.argsort(np.sum(~np.isinf(distances), axis=1), kind="stable")
    if m_order == "data":
        return np.arange(n_treat)
    if m_order == "random":
        return rng.permutation(n_treat)
    if m_order == "closest":
        return np.argsort(np.min(distances, axis=1), kind="stable")
    if m_order in ("largest", "smallest"):
        if order_scores is None:
            raise ValueError(f"m_order='{m_order}' requires order scores (propensity)")
        # stable, and negate rather than reverse, so tied scores keep data
        # order the way R's order() does: an unstable sort breaks the ties
        # differently across numpy builds and CPU architectures
        if m_order == "largest":
            return np.argsort(-np.asarray(order_scores), kind="stable")
        return np.argsort(order_scores, kind="stable")
    raise ValueError(f"Unknown m_order: {m_order}")


def greedy_match(
    data: pd.DataFrame,
    distance_matrix: np.ndarray,
    treat_mask: np.ndarray,
    exact_match_cols: list[str] | None = None,
    replace: bool = False,
    ratio: float = 1.0,
    random_state: int | None = None,
    m_order: str | None = None,
    order_scores: np.ndarray | None = None,
) -> tuple[dict[int, list[int]], list[float]]:
    """Implement greedy matching algorithm.

    The algorithm takes a distance matrix between treatment and control units and
    greedily matches units based on the smallest distance. Indices in the returned
    dictionary are positions in the arrays of treatment and control units, not
    original dataframe indices. The pipeline translates these to
    participant IDs.

    Args:
        data: DataFrame containing the data
        distance_matrix: Pre-computed and pre-calipered distance matrix (n_treatment x n_control)
        treat_mask: Boolean mask indicating treatment units
        exact_match_cols: Columns to match exactly on
        replace: Whether to allow replacement in matching
        ratio: Matching ratio (e.g., 2 means 1:2 matching)
        random_state: Seed for m_order="random"
        m_order: Matching order: None (fewest potential matches first),
            "closest" (best available distance first), "largest"/"smallest"
            (by order_scores, e.g. the propensity score), "random", or "data"
        order_scores: Per-treated-unit scores for "largest"/"smallest"

    Returns:
        Tuple of (match_pairs, match_distances)

    """
    logger.debug(f"GREEDY MATCHING: Ratio = {ratio}")

    logger.info("Starting greedy matching")
    logger.debug(f"Distance matrix shape: {distance_matrix.shape}")
    logger.debug(
        f"Treatment units: {np.sum(treat_mask)}, Control units: {np.sum(~treat_mask)}"
    )
    logger.debug(f"Matching with replacement: {replace}, ratio: {ratio}")

    # Get treatment and control indices
    treat_indices = np.where(treat_mask)[0]
    control_indices = np.where(~treat_mask)[0]

    logger.debug(
        f"Treatment indices count = {len(treat_indices)}, control indices count = {len(control_indices)}"
    )

    # Create working copy of distance matrix
    distances = distance_matrix.copy()

    # Apply exact matching constraints if needed
    if exact_match_cols:
        logger.debug(f"Applying exact matching on columns: {exact_match_cols}")
        distances = _apply_exact_matching(
            data, treat_indices, control_indices, distances, exact_match_cols
        )
        logger.debug(
            f"After exact matching, {np.sum(~np.isinf(distances))} potential matches remain"
        )

    # Initialize match storage
    n_treat = len(treat_indices)
    matches_per_unit = max(1, int(ratio))
    logger.debug(f"Attempting to find {matches_per_unit} matches per treatment unit")

    match_pairs: dict[int, list[int]] = {i: [] for i in range(n_treat)}
    match_distances: list[float] = []

    # Set random state if provided
    if random_state is not None:
        logger.debug(f"Using random state: {random_state}")
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random

    # Track available control units if not replacing
    if not replace:
        available_mask = np.ones(len(control_indices), dtype=bool)
        logger.debug(f"Initially {np.sum(available_mask)} control units available")

    treat_order = _matching_order(distances, m_order, order_scores, rng, random_state)

    # Main matching loop
    n_matched_units = 0
    n_total_matches = 0

    logger.debug("Starting main matching loop")
    for t_pos in tqdm(
        treat_order,
        desc="Greedy Matching",
        disable=not logger.isEnabledFor(logging.INFO),
    ):
        # Get distances for this treatment unit
        t_distances = distances[t_pos].copy()

        # Skip if no valid matches
        if np.all(np.isinf(t_distances)):
            continue

        # Find matches for this treatment unit
        matches_found = 0
        for match_idx in range(matches_per_unit):
            if not replace:
                # Mask out unavailable controls
                t_distances[~available_mask] = np.inf

            # Find best remaining match
            if np.all(np.isinf(t_distances)):
                logger.debug(
                    f"No more valid matches for treatment unit {t_pos} at match_idx {match_idx}"
                )
                break

            c_pos = np.argmin(t_distances)
            match_dist = t_distances[c_pos]

            # Store match
            match_pairs[t_pos].append(c_pos)
            match_distances.append(match_dist)
            matches_found += 1

            if not replace:
                # Mark control as used
                available_mask[c_pos] = False
                logger.debug(
                    f"Marked control unit {c_pos} as used, {np.sum(available_mask)} remaining"
                )

            # Mark this control as used for this iteration
            t_distances[c_pos] = np.inf

        if matches_found > 0:
            n_matched_units += 1
            n_total_matches += matches_found
            logger.debug(f"Found {matches_found} matches for treatment unit {t_pos}")

    logger.debug(
        f"Final matches: {n_matched_units}/{n_treat} treatment units matched with {n_total_matches} total matches"
    )
    logger.debug(
        f"Average matches per unit: {n_total_matches / max(1, n_matched_units):.2f}"
    )

    # Count how many treatment units got the full ratio of matches
    full_ratio_count = sum(
        1 for controls in match_pairs.values() if len(controls) == matches_per_unit
    )
    logger.debug(
        f"Treatment units with full {matches_per_unit} matches: {full_ratio_count}/{n_matched_units}"
    )

    # Log match counts distribution
    match_counts = {
        t_idx: len(controls)
        for t_idx, controls in match_pairs.items()
        if len(controls) > 0
    }
    logger.debug(f"Match counts distribution: {match_counts}")

    logger.info(
        f"Greedy matching complete: {n_matched_units}/{n_treat} treatment units matched"
    )
    logger.info(
        f"Total matches: {n_total_matches}, average: {n_total_matches / max(1, n_matched_units):.2f} per matched unit"
    )

    if match_distances:
        logger.debug(
            f"Match distances - min: {min(match_distances):.4f}, "
            f"mean: {np.mean(match_distances):.4f}, "
            f"max: {max(match_distances):.4f}"
        )

    return match_pairs, match_distances
