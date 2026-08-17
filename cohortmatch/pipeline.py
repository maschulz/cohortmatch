"""Matching pipeline: propensity scores -> direction -> matching -> balance.

`run_match(data, config)` is the single internal entry point; `match()` in
`api.py` translates the public keyword surface into a MatcherConfig and calls
it. The pipeline never mutates the caller's data or config.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit

from cohortmatch.datatypes import MatcherConfig, MatchResults
from cohortmatch.matching._utils import warn_if_pool_keys_tie
from cohortmatch.matching.covariate_nn import covariate_nn_match
from cohortmatch.matching.distances import calculate_distance_matrix
from cohortmatch.matching.fast_greedy import fast_greedy_match
from cohortmatch.matching.greedy import greedy_match
from cohortmatch.matching.optimal import optimal_match
from cohortmatch.matching.strata import cem_strata, stratum_weights, subclassify
from cohortmatch.metrics.balance import (
    calculate_balance_index,
    calculate_balance_stats,
    calculate_rubin_rules,
)
from cohortmatch.metrics.propensity import (
    assess_propensity_overlap,
    estimate_propensity_scores,
)
from cohortmatch.metrics.utils import get_caliper_for_matching
from cohortmatch.utils.logging import get_logger
from cohortmatch.validation import validate_data, validate_matcher_config

logger = get_logger(__name__)


def run_match(data: pd.DataFrame, config: MatcherConfig) -> MatchResults:
    """Run the full matching pipeline and return the results container."""
    data = data.copy()

    validate_matcher_config(config)
    validate_data(
        data=data,
        treatment_col=config.treatment_col,
        covariates=config.covariates,
        propensity_col=config.propensity_col,
        exact_match_cols=config.exact_match_cols,
    )
    logger.info(f"Matching {len(data)} observations with method: {config.match_method}")

    # Step 1: propensity scores, when the configuration needs them
    need_propensity = (
        config.estimate_propensity
        or bool(config.propensity_col)
        or config.distance_method in ("propensity", "logit")
        or (
            config.caliper_method in ("propensity", "logit")
            and config.caliper_value is not None
        )
        or (
            # the approximate path needs a score only for its propensity-window
            # variant; the covariate-space tree variant does not
            config.match_method == "fast_greedy"
            and config.distance_method in ("propensity", "logit")
        )
    )

    propensity_scores = None
    propensity_model = None
    propensity_metrics = None
    if need_propensity:
        ps_result = _estimate_propensity(data, config)
        propensity_scores = ps_result["propensity_scores"]
        propensity_model = ps_result["model"]
        propensity_metrics = ps_result["metrics"]

    # Step 1b: common-support discard
    work, work_ps, discarded = _apply_discard(data, config, propensity_scores)

    # Stratum methods take a separate path: strata + weights, no pairs
    if config.match_method in ("subclass", "cem"):
        return _run_strata(
            data,
            work,
            work_ps,
            config,
            discarded,
            propensity_scores,
            propensity_model,
            propensity_metrics,
        )

    # Step 2: matching direction
    flipped = _determine_matching_direction(work, config)
    if flipped:
        logger.info("Matching from the control side (control units are the anchors)")
    treatment_mask = _get_treatment_mask(work, config, flipped)

    # Step 3: matching
    match_results = _perform_matching(work, config, work_ps, treatment_mask, flipped)
    matched_data = match_results["matched_data"]
    logger.info(f"Matched dataset contains {len(matched_data)} observations")

    # Step 4: matching weights and subclass membership. Order both to match
    # matched_data's rows so result.weights.to_numpy() lines up positionally
    # with matched_data, not only by label (they are built in pair-insertion
    # order, which differs from the original-data order matched_data uses).
    weights, subclass = _compute_weights_and_subclass(
        match_results["pairs"], flipped=flipped, replace=config.replace
    )
    if len(matched_data):
        weights = weights.reindex(matched_data.index)
        if subclass is not None:
            subclass = subclass.reindex(matched_data.index)
    anchor = "control" if flipped else "treatment"

    # Step 5: balance diagnostics (weighted, anchored); "before" refers to
    # the full input sample, including any discarded units
    balance = _calculate_balance(data, matched_data, config, weights, anchor)

    return MatchResults(
        original_data=data,
        matched_data=matched_data,
        pairs=match_results["pairs"],
        match_groups=match_results["match_groups"],
        match_distances=match_results["match_distances"],
        distance_matrix=match_results["distance_matrix"],
        dm_anchor_ids=match_results["dm_anchor_ids"],
        dm_pool_ids=match_results["dm_pool_ids"],
        resolved_caliper=match_results["resolved_caliper"],
        propensity_scores=propensity_scores,
        propensity_model=propensity_model,
        propensity_metrics=propensity_metrics,
        balance_statistics=balance["balance_statistics"],
        rubin_statistics=balance["rubin_statistics"],
        balance_index=balance["balance_index"],
        weights=weights,
        subclass=subclass,
        anchor=anchor,
        discarded=discarded,
        config=config,
    )


def _run_strata(
    data: pd.DataFrame,
    work: pd.DataFrame,
    work_ps: np.ndarray | None,
    config: MatcherConfig,
    discarded: pd.Index,
    propensity_scores: np.ndarray | None,
    propensity_model: Any | None,
    propensity_metrics: dict | None,
) -> MatchResults:
    """Subclassification / CEM: stratum labels -> weights -> diagnostics."""
    estimand = getattr(config, "estimand", "att")
    treated = (work[config.treatment_col] == 1).to_numpy()

    if config.match_method == "subclass":
        if work_ps is None:
            raise ValueError("Subclassification requires propensity scores")
        labels = subclassify(work_ps, treated, config.n_subclasses, estimand)
    else:
        labels = cem_strata(
            work,
            config.covariates,
            treated,
            coarsening=config.cem_coarsening,
            exact_cols=config.exact_match_cols,
        )

    weights_arr = stratum_weights(labels, treated, estimand)
    kept = labels >= 0
    matched_data = work.loc[kept].copy()
    weights = pd.Series(weights_arr[kept], index=work.index[kept], name="weights")
    subclass = pd.Series(labels[kept], index=work.index[kept], name="subclass")

    anchor = {"att": "treatment", "atc": "control", "ate": "pooled"}[estimand]
    balance = _calculate_balance(data, matched_data, config, weights, anchor)

    logger.info(
        f"{config.match_method}: {int(kept.sum())} units in {subclass.nunique()} strata"
    )
    return MatchResults(
        original_data=data,
        matched_data=matched_data,
        pairs=[],
        match_groups={},
        match_distances=[],
        distance_matrix=None,
        propensity_scores=propensity_scores,
        propensity_model=propensity_model,
        propensity_metrics=propensity_metrics,
        balance_statistics=balance["balance_statistics"],
        rubin_statistics=balance["rubin_statistics"],
        balance_index=balance["balance_index"],
        weights=weights,
        subclass=subclass,
        anchor=anchor,
        discarded=discarded,
        config=config,
    )


def _apply_discard(
    data: pd.DataFrame,
    config: MatcherConfig,
    propensity_scores: np.ndarray | None,
) -> tuple[pd.DataFrame, np.ndarray | None, pd.Index]:
    """Drop units outside the common propensity support before matching.

    The support region is the intersection of the two groups' score ranges;
    `config.discard` selects which group(s) to drop from.
    """
    discard = getattr(config, "discard", None)
    if not discard:
        return data, propensity_scores, data.index[:0]
    if propensity_scores is None:
        raise ValueError("discard requires propensity scores")

    treated = (data[config.treatment_col] == 1).to_numpy()
    lo = max(propensity_scores[treated].min(), propensity_scores[~treated].min())
    hi = min(propensity_scores[treated].max(), propensity_scores[~treated].max())
    outside = (propensity_scores < lo) | (propensity_scores > hi)

    if discard == "treated":
        drop = outside & treated
    elif discard == "control":
        drop = outside & ~treated
    else:  # "both"
        drop = outside

    discarded = data.index[drop]
    if len(discarded):
        logger.info(
            f"Discarding {len(discarded)} unit(s) outside common support "
            f"[{lo:.4f}, {hi:.4f}]"
        )
    return data.loc[~drop], propensity_scores[~drop], discarded


def _estimate_propensity(data: pd.DataFrame, config: MatcherConfig) -> dict[str, Any]:
    """Use a provided propensity column or estimate scores by cross-fitting."""
    if config.propensity_col and config.propensity_col in data.columns:
        metrics = assess_propensity_overlap(
            data=data,
            propensity_col=config.propensity_col,
            treatment_col=config.treatment_col,
        )
        return {
            "propensity_scores": data[config.propensity_col].values,
            "model": None,
            "metrics": metrics,
        }

    logger.info("Estimating propensity scores")
    result = estimate_propensity_scores(
        data=data,
        treatment_col=config.treatment_col,
        covariates=config.covariates,
        model_type=config.propensity_model,
        model_params=config.model_params,
        cv=config.cv_folds,
        random_state=config.random_state,
    )
    # plain floats so the dict prints cleanly (not np.float64(...) wrappers)
    cv = result.get("cv_results", {})
    metrics: dict[str, Any] = {}
    if result.get("auc") is not None:
        metrics["auc"] = float(result["auc"])
    if "fold_aucs" in cv:
        metrics["fold_aucs"] = [float(a) for a in cv["fold_aucs"]]
    if "mean_auc" in cv:
        metrics["mean_auc"] = float(cv["mean_auc"])
    if "std_auc" in cv:
        metrics["std_auc"] = float(cv["std_auc"])
    return {
        "propensity_scores": result["propensity_scores"],
        "model": result["model"],
        "metrics": metrics,
    }


def _determine_matching_direction(data: pd.DataFrame, config: MatcherConfig) -> bool:
    """True if matching should anchor on the control group.

    Direction is always explicit ("treatment" or "control", set by the
    estimand). The v3 size-based auto-flip is gone on purpose.
    """
    return config.matching_direction == "control"


def _get_treatment_mask(
    data: pd.DataFrame, config: MatcherConfig, flipped: bool
) -> np.ndarray:
    """Boolean mask of the anchor ("from") group."""
    if flipped:
        mask = data[config.treatment_col] == 0
    else:
        mask = data[config.treatment_col] == 1
    return np.array(mask, dtype=bool)


def _perform_matching(
    data: pd.DataFrame,
    config: MatcherConfig,
    propensity_scores: np.ndarray | None,
    treatment_mask: np.ndarray,
    flipped: bool,
) -> dict[str, Any]:
    """Compute distances, apply calipers, and run the matching algorithm."""
    algorithm_treatment_indices = data.index[treatment_mask].tolist()
    algorithm_control_indices = data.index[~treatment_mask].tolist()

    # Duplicate matching keys in the pool guarantee exact ties, which the
    # default tie policy resolves by input row order (see matching/_utils.py).
    if config.distance_method in ("propensity", "logit"):
        pool_keys = (
            propensity_scores[~treatment_mask]
            if propensity_scores is not None
            else None
        )
    elif config.covariates:
        pool_keys = data[config.covariates].to_numpy()[~treatment_mask]
    else:
        pool_keys = None
    warn_if_pool_keys_tie(pool_keys, config.tie_break)

    distance_matrix_for_results = None
    dm_anchor_ids = None
    dm_pool_ids = None

    resolved_caliper = None

    # --- memory-efficient path -------------------------------------------
    if config.match_method == "fast_greedy":
        covariate_distance = config.distance_method in ("mahalanobis", "euclidean")
        ps_caliper = config.caliper_method in ("propensity", "logit")

        if covariate_distance and not ps_caliper:
            # exact covariate-space matching via a spatial tree (no propensity
            # score needed); the caliper, if any, is on the distance metric
            cal = None
            if config.caliper_method is not None and config.caliper_value is not None:
                cal = get_caliper_for_matching(
                    config=config,
                    propensity_scores=propensity_scores,
                    data=data,
                    treat_mask=treatment_mask,
                )
            resolved_caliper = cal
            algorithm_match_pairs, match_distances = covariate_nn_match(
                data=data,
                treat_mask=treatment_mask,
                config=config,
                caliper_value=cal,
            )
        else:
            if propensity_scores is None:
                raise ValueError(
                    "Propensity scores are required for 'fast_greedy' matching."
                )
            final_caliper_value = get_caliper_for_matching(
                config=config,
                propensity_scores=propensity_scores,
                data=data,
                treat_mask=treatment_mask,
            )
            resolved_caliper = final_caliper_value
            temp_config = config.__class__(**config.__dict__)
            temp_config.caliper_value = final_caliper_value

            algorithm_match_pairs, match_distances = fast_greedy_match(
                data=data,
                treat_mask=treatment_mask,
                propensity_scores=propensity_scores,
                config=temp_config,
            )

    # --- dense-matrix path -----------------------------------------------
    else:
        logger.info(
            f"Calculating distance matrix with method: {config.distance_method}"
        )
        if config.distance_method in ("propensity", "logit"):
            if propensity_scores is None:
                raise ValueError(
                    "Propensity scores are required for this distance method."
                )
            X_treat_primary = propensity_scores[treatment_mask].reshape(-1, 1)
            X_control_primary = propensity_scores[~treatment_mask].reshape(-1, 1)
        else:
            X_treat_primary = data[config.covariates][treatment_mask].values
            X_control_primary = data[config.covariates][~treatment_mask].values

        primary_distances = calculate_distance_matrix(
            X_treat=X_treat_primary,
            X_control=X_control_primary,
            method=config.distance_method,
            standardize=config.standardize,
            weights=np.array([config.weights.get(c, 1.0) for c in config.covariates])
            if config.weights
            else None,
        )

        if config.caliper_method is not None and config.caliper_value is not None:
            final_caliper_value = get_caliper_for_matching(
                config=config,
                propensity_scores=propensity_scores,
                distance_matrix=primary_distances,
                data=data,
                treat_mask=treatment_mask,
            )

            resolved_caliper = final_caliper_value
            if config.caliper_method == config.distance_method:
                logger.info(
                    f"Applying '{config.caliper_method}' caliper directly to distance matrix."
                )
                primary_distances[primary_distances > final_caliper_value] = np.inf
            else:
                logger.info(
                    f"Applying '{config.caliper_method}' caliper as a mask on "
                    f"'{config.distance_method}' distances."
                )
                if config.caliper_method in ("propensity", "logit"):
                    if propensity_scores is None:
                        raise ValueError(
                            "Propensity scores are required for this caliper method."
                        )
                    # A scalar-score caliper is |s_i - s_j|; apply it row by row
                    # so no second n x m matrix is materialized (the estimate in
                    # _resolve_algorithm counts on this).
                    scores = propensity_scores
                    if config.caliper_method == "logit":
                        scores = logit(np.clip(scores, 1e-6, 1 - 1e-6))
                    treat_scores = scores[treatment_mask]
                    control_scores = scores[~treatment_mask]
                    for i in range(treat_scores.shape[0]):
                        over = (
                            np.abs(control_scores - treat_scores[i])
                            > final_caliper_value
                        )
                        primary_distances[i, over] = np.inf
                else:  # mahalanobis / euclidean caliper: a genuine second matrix
                    caliper_matrix = calculate_distance_matrix(
                        X_treat=data[config.covariates][treatment_mask].values,
                        X_control=data[config.covariates][~treatment_mask].values,
                        method=config.caliper_method,
                        standardize=config.standardize,
                    )
                    primary_distances[caliper_matrix > final_caliper_value] = np.inf

        cov_calipers = getattr(config, "covariate_calipers", None)
        if cov_calipers:
            for col, threshold in cov_calipers.items():
                col_vals = data[col].to_numpy(dtype=float)
                t_vals = col_vals[treatment_mask]
                c_vals = col_vals[~treatment_mask]
                primary_distances[
                    np.abs(t_vals[:, None] - c_vals[None, :]) > threshold
                ] = np.inf

        if config.match_method == "optimal":
            algorithm_match_pairs, match_distances = optimal_match(
                data=data,
                distance_matrix=primary_distances,
                treat_mask=treatment_mask,
                exact_match_cols=config.exact_match_cols,
                ratio=config.ratio,
                replace=config.replace,
                tie_break=config.tie_break,
                random_state=config.random_state,
            )
        else:
            # Covariate distances scale via a spatial tree (covariate_nn), which
            # can only order by data/random. Default the dense path to data
            # order too, so the same request gives the same pairs whether it
            # runs exact or approximate (crossing memory_limit_gb must not
            # silently change the matching order).
            m_order = getattr(config, "m_order", None)
            if m_order is None and config.distance_method in (
                "mahalanobis",
                "euclidean",
            ):
                m_order = "data"
            algorithm_match_pairs, match_distances = greedy_match(
                data=data,
                distance_matrix=primary_distances,
                treat_mask=treatment_mask,
                exact_match_cols=config.exact_match_cols,
                replace=config.replace,
                ratio=config.ratio,
                random_state=config.random_state,
                m_order=m_order,
                order_scores=propensity_scores[treatment_mask]
                if propensity_scores is not None
                else None,
                tie_break=config.tie_break,
            )

        distance_matrix_for_results = primary_distances
        dm_anchor_ids = algorithm_treatment_indices
        dm_pool_ids = algorithm_control_indices

    # --- translate positions to participant IDs ---------------------------
    pairs: list[tuple[Any, Any]] = []
    match_groups: dict[Any, list[Any]] = {}
    matched_ids = set()

    for t_pos, c_pos_list in algorithm_match_pairs.items():
        if len(c_pos_list) == 0:
            continue
        t_idx = algorithm_treatment_indices[t_pos]
        c_indices = [algorithm_control_indices[c_pos] for c_pos in c_pos_list]

        if flipped:
            # t_idx is actually a control unit; c_indices are treated units
            for c_idx in c_indices:
                pairs.append((c_idx, t_idx))
                match_groups.setdefault(c_idx, []).append(t_idx)
                matched_ids.add(c_idx)
                matched_ids.add(t_idx)
        else:
            for c_idx in c_indices:
                pairs.append((t_idx, c_idx))
                match_groups.setdefault(t_idx, []).append(c_idx)
                matched_ids.add(t_idx)
                matched_ids.add(c_idx)

    # Each matched unit appears exactly once; reuse is expressed through the
    # matching weights, not duplicated rows. Preserve original data order so
    # outputs are byte-identical across interpreter runs (matched_ids is a set).
    ordered_ids = [idx for idx in data.index if idx in matched_ids]
    matched_data = data.loc[ordered_ids].copy()

    n_treatment = (matched_data[config.treatment_col] == 1).sum()
    n_control = (matched_data[config.treatment_col] == 0).sum()
    logger.debug(
        f"Matched dataset: {n_treatment} treatment and {n_control} control units"
    )

    if config.ratio == 1.0 and not config.replace and n_treatment != n_control:
        logger.warning(
            f"Expected equal group sizes for 1:1 matching, got "
            f"{n_treatment} treatment and {n_control} control units"
        )

    if not pairs:
        logger.warning("No matching pairs found. Returning empty matched dataset.")
        matched_data = pd.DataFrame(columns=data.columns)

    return {
        "pairs": pairs,
        "match_groups": match_groups,
        "matched_data": matched_data,
        "match_distances": match_distances,
        "distance_matrix": distance_matrix_for_results,
        "dm_anchor_ids": dm_anchor_ids,
        "dm_pool_ids": dm_pool_ids,
        "resolved_caliper": resolved_caliper,
    }


def _compute_weights_and_subclass(
    pairs: list[tuple[Any, Any]], flipped: bool, replace: bool
) -> tuple[pd.Series, pd.Series | None]:
    """Matching weights (MatchIt convention) and match-group membership.

    Anchor units get weight 1. Each partner accumulates 1/k per match group it
    serves in (k = group size); partner weights are then rescaled to sum to
    the number of unique partners. `subclass` maps each matched unit to its
    match group (the anchor's id) and is only defined without replacement,
    where membership is unique.
    """
    groups: dict[Any, list[Any]] = {}
    for t_id, c_id in pairs:
        a_id, p_id = (c_id, t_id) if flipped else (t_id, c_id)
        groups.setdefault(a_id, []).append(p_id)

    weights: dict[Any, float] = {}
    subclass: dict[Any, Any] = {}
    for a_id, partners in groups.items():
        weights[a_id] = 1.0
        subclass[a_id] = a_id
        k = len(partners)
        for p_id in partners:
            weights[p_id] = weights.get(p_id, 0.0) + 1.0 / k
            subclass[p_id] = a_id

    partner_ids = {p for partners in groups.values() for p in partners}
    total = sum(weights[p] for p in partner_ids)
    if total > 0:
        scale = len(partner_ids) / total
        for p_id in partner_ids:
            weights[p_id] *= scale

    weights_series = pd.Series(weights, name="weights", dtype=float)
    subclass_series = None if replace else pd.Series(subclass, name="subclass")
    return weights_series, subclass_series


def _calculate_balance(
    data: pd.DataFrame,
    matched_data: pd.DataFrame,
    config: MatcherConfig,
    weights: pd.Series | None = None,
    anchor: str = "treatment",
) -> dict[str, Any]:
    """Balance statistics before/after matching."""
    if matched_data.empty:
        logger.warning("No matches found. Balance statistics cannot be calculated.")
        return {
            "balance_statistics": None,
            "rubin_statistics": None,
            "balance_index": None,
        }

    balance_statistics = calculate_balance_stats(
        data=data,
        matched_data=matched_data,
        covariates=config.covariates,
        treatment_col=config.treatment_col,
        weights=weights,
        anchor=anchor,
    )
    rubin_statistics = calculate_rubin_rules(balance_statistics)
    balance_index = calculate_balance_index(balance_statistics)

    if balance_index:
        logger.info(
            f"Mean SMD before: {balance_index.get('mean_smd_before', float('nan')):.3f}, "
            f"after: {balance_index.get('mean_smd_after', float('nan')):.3f}"
        )

    return {
        "balance_statistics": balance_statistics,
        "rubin_statistics": rubin_statistics,
        "balance_index": balance_index,
    }
