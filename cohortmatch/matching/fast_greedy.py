"""Fast, memory-efficient greedy matching for large datasets.

Controls are sorted by logit propensity score once; each anchor unit's
candidate pool is then a contiguous window found by binary search, so the
per-unit cost scales with the pool size instead of the number of controls.
Propensity/logit distances and calipers are computed inline on the window;
covariate-space distances fall back to the general distance machinery on the
candidate pool only.
"""

import logging
import warnings as _warnings

import numpy as np
import pandas as pd
from scipy.special import logit
from tqdm.auto import tqdm

from cohortmatch.datatypes import MatcherConfig
from cohortmatch.exceptions import ApproximateMatchWarning
from cohortmatch.matching._utils import argmin_with_tie_break, resolve_tie_break
from cohortmatch.matching.distances import calculate_distance_matrix
from cohortmatch.utils.logging import get_logger

logger = get_logger(__name__)

_PS_METRICS = ("propensity", "logit")


def fast_greedy_match(
    data: pd.DataFrame,
    treat_mask: np.ndarray,
    propensity_scores: np.ndarray,
    config: "MatcherConfig",
) -> tuple[dict[int, list[int]], list[float]]:
    """Memory-efficient greedy matching via a sorted propensity window.

    The candidate pool per anchor unit is bounded by a prefilter caliper of
    `config.fast_prefilter_caliper_scale` standard deviations of the logit
    propensity score. The user-defined caliper (`config.caliper_method`,
    numeric `config.caliper_value`) is applied within the pool, as are exact
    constraints. Returns positional (treated -> [control]) matches and their
    distances, like `greedy_match`.
    """
    logger.info("Starting fast greedy matching (memory-efficient)")
    if propensity_scores is None:
        raise ValueError("Propensity scores are required for fast_greedy_match.")

    tie_break = resolve_tie_break(getattr(config, "tie_break", "first"))
    rng = np.random.RandomState(config.random_state)

    treat_indices = np.where(treat_mask)[0]
    n_treat = len(treat_indices)
    n_control = int((~treat_mask).sum())

    logger.debug(f"Treatment units: {n_treat}, Control units: {n_control}")

    ps_clipped = np.clip(propensity_scores, 1e-6, 1 - 1e-6)
    search_scores = logit(ps_clipped)
    pre_filter_caliper = np.std(search_scores) * config.fast_prefilter_caliper_scale

    treat_logit = search_scores[treat_mask]
    treat_raw = propensity_scores[treat_mask]

    # Sort controls by logit propensity once; candidate pools become windows.
    control_logit = search_scores[~treat_mask]
    control_raw = propensity_scores[~treat_mask]
    sort_idx = np.argsort(control_logit, kind="stable")
    sorted_logit = control_logit[sort_idx]
    sorted_raw = control_raw[sort_idx]

    # Per-scale views for distances and calipers
    scale_treat = {"propensity": treat_raw, "logit": treat_logit}
    scale_sorted_control = {"propensity": sorted_raw, "logit": sorted_logit}

    ps_distance = config.distance_method in _PS_METRICS
    caliper_metric = config.caliper_method
    caliper_value = (
        float(config.caliper_value)
        if caliper_metric is not None and isinstance(config.caliper_value, (int, float))
        else None
    )
    ps_caliper = caliper_metric in _PS_METRICS and caliper_value is not None

    # Window strategy: a propensity/logit caliper defines the candidate
    # window exactly (nothing outside it can match), so the heuristic
    # prefilter is only used when the caliper lives in covariate space,
    # there the window is an approximation, which we log.
    if caliper_value is not None and caliper_metric == "logit":
        window_mode = "logit"
        logger.info(f"Candidate windows from the logit caliper ({caliper_value:.4f}).")
    elif caliper_value is not None and caliper_metric == "propensity":
        window_mode = "propensity"
        logger.info(
            f"Candidate windows from the propensity caliper ({caliper_value:.4f})."
        )
    else:
        window_mode = "prefilter"
        logger.info(
            f"Using pre-filtering caliper value: {pre_filter_caliper:.4f} on "
            "'logit' scale."
        )
        if caliper_value is not None:
            _warnings.warn(
                "Covariate-space caliper with the approximate algorithm: candidate "
                "pools are limited to a propensity prefilter window of "
                f"{config.fast_prefilter_caliper_scale} SD of the logit propensity "
                "score; matches outside it are not considered.",
                ApproximateMatchWarning,
                stacklevel=2,
            )

    # Covariate data (general lane and covariate-space calipers).
    # Standardization and the Mahalanobis covariance use full-sample
    # statistics computed once, so window distances equal dense-path
    # distances instead of drifting with each candidate pool.
    X_covariates = (
        data[config.covariates].to_numpy(dtype=float) if config.covariates else None
    )
    needs_covariate_space = not ps_distance or config.caliper_method in (
        "mahalanobis",
        "euclidean",
    )
    global_cov = None
    if X_covariates is not None and needs_covariate_space:
        if config.standardize:
            mean = X_covariates.mean(axis=0)
            std = X_covariates.std(axis=0)
            std[std < 1e-10] = 1.0
            X_covariates = (X_covariates - mean) / std
        if (
            config.distance_method == "mahalanobis"
            or config.caliper_method == "mahalanobis"
        ):
            global_cov = np.cov(X_covariates, rowvar=False)
    X_treat_covariates = X_covariates[treat_mask] if X_covariates is not None else None
    X_control_covariates_sorted = (
        X_covariates[~treat_mask][sort_idx] if X_covariates is not None else None
    )

    cov_calipers = getattr(config, "covariate_calipers", None) or {}
    cov_caliper_treat = {
        col: data[col].to_numpy(dtype=float)[treat_mask] for col in cov_calipers
    }
    cov_caliper_control_sorted = {
        col: data[col].to_numpy(dtype=float)[~treat_mask][sort_idx]
        for col in cov_calipers
    }

    exact_data = (
        data[config.exact_match_cols].values if config.exact_match_cols else None
    )
    treat_exact = exact_data[treat_mask] if exact_data is not None else None
    control_exact_sorted = (
        exact_data[~treat_mask][sort_idx] if exact_data is not None else None
    )

    matches_per_unit = max(1, int(config.ratio))
    match_pairs: dict[int, list[int]] = {i: [] for i in range(n_treat)}
    match_distances: list[float] = []

    available = np.ones(n_control, dtype=bool)  # in sorted-control space

    # Matching order; hardest-to-match units first by default ("largest":
    # extreme propensity scores have the fewest candidate neighbors, and
    # benchmarks show this closes the recall gap vs dense matching)
    m_order = getattr(config, "m_order", None)
    if m_order is None:
        m_order = "largest"
    if m_order == "data":
        treat_order = np.arange(n_treat)
    elif m_order == "random":
        treat_order = rng.permutation(n_treat)
    elif m_order in ("largest", "smallest"):
        order = np.argsort(treat_raw, kind="stable")
        treat_order = order[::-1] if m_order == "largest" else order
    else:
        raise ValueError(
            f"m_order='{m_order}' is not supported by the approximate algorithm; "
            "use 'data', 'random', 'largest', or 'smallest'."
        )

    n_matched_units = 0
    n_total_matches = 0

    for t_pos in tqdm(
        treat_order,
        desc="Fast Greedy Matching",
        disable=not logger.isEnabledFor(logging.INFO),
    ):
        # 1. Candidate window on the sorted logit scale
        center = treat_logit[t_pos]
        if window_mode == "logit":
            lo_bound, hi_bound = center - caliper_value, center + caliper_value
        elif window_mode == "propensity":
            ps_t = treat_raw[t_pos]
            lo_bound = logit(np.clip(ps_t - caliper_value, 1e-6, 1 - 1e-6))
            hi_bound = logit(np.clip(ps_t + caliper_value, 1e-6, 1 - 1e-6))
        else:
            lo_bound, hi_bound = (
                center - pre_filter_caliper,
                center + pre_filter_caliper,
            )
        lo = np.searchsorted(sorted_logit, lo_bound, side="left")
        hi = np.searchsorted(sorted_logit, hi_bound, side="right")
        if lo >= hi:
            continue

        window_available = available[lo:hi]
        if not config.replace and not window_available.any():
            continue

        # 2. Distances on the window
        if ps_distance:
            dist = np.abs(
                scale_sorted_control[config.distance_method][lo:hi]
                - scale_treat[config.distance_method][t_pos]
            )
        else:
            dist = calculate_distance_matrix(
                X_treat=X_treat_covariates[t_pos].reshape(1, -1),
                X_control=X_control_covariates_sorted[lo:hi],
                method=config.distance_method,
                standardize=False,  # standardized globally above
                cov_matrix=global_cov,
                weights=np.array(
                    [config.weights.get(c, 1.0) for c in config.covariates]
                )
                if config.weights
                else None,
            ).ravel()
        dist = dist.astype(float, copy=True)

        # 3. User caliper on the window
        if caliper_value is not None:
            if ps_caliper:
                caliper_dist = np.abs(
                    scale_sorted_control[caliper_metric][lo:hi]
                    - scale_treat[caliper_metric][t_pos]
                )
            elif caliper_metric == config.distance_method:
                caliper_dist = dist
            else:  # mahalanobis / euclidean caliper on covariates
                caliper_dist = calculate_distance_matrix(
                    X_treat=X_treat_covariates[t_pos].reshape(1, -1),
                    X_control=X_control_covariates_sorted[lo:hi],
                    method=caliper_metric,
                    standardize=False,  # standardized globally above
                    cov_matrix=global_cov,
                ).ravel()
            dist[caliper_dist > caliper_value] = np.inf

        # 4. Per-covariate calipers on the window
        for col, threshold in cov_calipers.items():
            gap = np.abs(
                cov_caliper_control_sorted[col][lo:hi] - cov_caliper_treat[col][t_pos]
            )
            dist[gap > threshold] = np.inf

        # 4b. Exact constraints on the window
        if treat_exact is not None:
            same = (control_exact_sorted[lo:hi] == treat_exact[t_pos]).all(axis=1)
            dist[~same] = np.inf

        if not config.replace:
            dist[~window_available] = np.inf

        # 5. Greedy selection within the window
        matches_found = 0
        while matches_found < matches_per_unit:
            j = argmin_with_tie_break(dist, tie_break, rng)
            d = dist[j]
            if np.isinf(d):
                break
            match_pairs[t_pos].append(int(sort_idx[lo + j]))
            match_distances.append(float(d))
            matches_found += 1
            dist[j] = np.inf
            if not config.replace:
                available[lo + j] = False

        if matches_found > 0:
            n_matched_units += 1
            n_total_matches += matches_found

    logger.info(
        f"Fast greedy matching complete: {n_matched_units}/{n_treat} treatment units matched"
    )
    logger.info(
        f"Total matches: {n_total_matches}, "
        f"average: {n_total_matches / max(1, n_matched_units):.2f} per matched unit"
    )

    return match_pairs, match_distances
