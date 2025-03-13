"""Exact covariate-distance nearest-neighbor matching at scale.

The windowed approximate path (fast_greedy) organises candidates by the
propensity score, which is the wrong neighbourhood for a covariate distance.
This module finds candidates in covariate space directly with a spatial tree,
so Mahalanobis/Euclidean matching scales without a propensity score and
returns the same pairs the dense path would. Mahalanobis distance equals
Euclidean distance after whitening the covariates by the Cholesky factor of
the inverse covariance, so a Euclidean KD-tree on whitened coordinates gives
exact Mahalanobis nearest neighbours.

Uses sklearn, already a core dependency.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

from cohortmatch.datatypes import MatcherConfig
from cohortmatch.utils.logging import get_logger

logger = get_logger(__name__)


def _whiten(config: "MatcherConfig", X_all: np.ndarray) -> np.ndarray:
    """Transform covariates so Euclidean distance equals the chosen metric.

    Matches distances.calculate_distance_matrix exactly: standardize on the
    full sample if requested, then (for Mahalanobis) rotate by the Cholesky
    factor of the regularized inverse covariance. Euclidean weights, if any,
    scale the standardized columns.
    """
    X = X_all.astype(float)
    if config.standardize:
        X = StandardScaler().fit_transform(X)

    if config.distance_method == "euclidean":
        if config.weights:
            w = np.sqrt(
                np.array([config.weights.get(c, 1.0) for c in config.covariates])
            )
            X = X * w
        return X

    # mahalanobis: cov on the (standardized) full sample, + 1e-6 ridge as in
    # distances.py, then whiten by L where inv(cov) = L L^T
    cov = np.cov(X, rowvar=False)
    cov = np.atleast_2d(cov) + 1e-6 * np.eye(X.shape[1])
    cov_inv = np.linalg.inv(cov)
    L = np.linalg.cholesky(cov_inv)
    return X @ L


def covariate_nn_match(
    data: pd.DataFrame,
    treat_mask: np.ndarray,
    config: "MatcherConfig",
    caliper_value: float | None,
) -> tuple[dict[int, list[int]], list[float]]:
    """Greedy nearest-neighbour matching in covariate space via a KD-tree.

    Returns (match_pairs, match_distances) with the same positional-index
    contract as greedy_match / fast_greedy_match: keys are anchor positions
    within the treated array, values are lists of control positions within
    the control array.

    Exact vs the dense path: each anchor takes its nearest *available* control
    (within the caliper, honouring exact strata and per-covariate calipers),
    which is the same argmin the dense greedy computes.
    """
    treat_indices = np.where(treat_mask)[0]
    control_indices = np.where(~treat_mask)[0]
    n_treat = len(treat_indices)
    n_control = len(control_indices)
    matches_per_unit = max(1, int(config.ratio))

    Xw = _whiten(config, data[config.covariates].to_numpy())
    Xw_treat = Xw[treat_mask]
    Xw_control = Xw[~treat_mask]

    # exact strata: one tree per (treated-present) stratum keeps queries cheap
    # and makes the exact constraint free instead of a post-filter
    if config.exact_match_cols:
        exact_vals = pd.MultiIndex.from_frame(
            data[config.exact_match_cols]
        ).to_flat_index()
        strata = pd.factorize(exact_vals)[0]
    else:
        strata = np.zeros(len(data), dtype=int)
    strata_treat = strata[treat_mask]
    strata_control = strata[~treat_mask]

    # per-covariate calipers, in the same raw units as the dense path
    cov_calipers = getattr(config, "covariate_calipers", None) or {}
    cc_treat = {c: data[c].to_numpy(dtype=float)[treat_mask] for c in cov_calipers}
    cc_control = {c: data[c].to_numpy(dtype=float)[~treat_mask] for c in cov_calipers}

    # build a tree over the control points of each stratum
    trees: dict[int, tuple[KDTree, np.ndarray]] = {}
    for s in np.unique(strata_control):
        pos = np.where(strata_control == s)[0]
        if len(pos):
            trees[s] = (KDTree(Xw_control[pos]), pos)

    # matching order (covariate path supports data / random)
    m_order = getattr(config, "m_order", None)
    if m_order in (None, "data"):
        order = np.arange(n_treat)
    elif m_order == "random":
        order = np.random.RandomState(config.random_state).permutation(n_treat)
    else:
        raise ValueError(
            f"m_order={m_order!r} is not supported for covariate-distance "
            "matching at scale; use 'data' or 'random'."
        )

    available = np.ones(n_control, dtype=bool)
    match_pairs: dict[int, list[int]] = {i: [] for i in range(n_treat)}
    match_distances: list[float] = []

    for t_pos in order:
        s = strata_treat[t_pos]
        entry = trees.get(s)
        if entry is None:
            continue
        tree, local_to_global = entry
        q = Xw_treat[t_pos].reshape(1, -1)

        # pull nearest neighbours in expanding batches, skipping used controls
        # and those failing the per-covariate calipers, until `ratio` are found
        # or the stratum is exhausted (exact, and bounded when a caliper holds)
        found = 0
        k = min(len(local_to_global), max(matches_per_unit * 4, 10))
        seen: set[int] = set()
        while found < matches_per_unit:
            dist, idx = tree.query(q, k=k)
            dist, idx = dist[0], idx[0]
            progressed = False
            for d, local in zip(dist, idx, strict=False):
                local = int(local)
                if local in seen:
                    continue
                seen.add(local)
                progressed = True
                if caliper_value is not None and d > caliper_value:
                    # neighbours are distance-sorted: nothing further qualifies
                    found = matches_per_unit
                    break
                c_pos = int(local_to_global[local])
                if not available[c_pos]:
                    continue
                if any(
                    abs(cc_control[c][c_pos] - cc_treat[c][t_pos]) > thr
                    for c, thr in cov_calipers.items()
                ):
                    continue
                match_pairs[t_pos].append(c_pos)
                match_distances.append(float(d))
                available[c_pos] = False
                found += 1
                if found >= matches_per_unit:
                    break
            if found >= matches_per_unit:
                break
            if k >= len(local_to_global):
                break  # stratum exhausted
            if not progressed:
                break
            k = min(len(local_to_global), k * 4)

    n_matched = sum(1 for v in match_pairs.values() if v)
    logger.info(
        f"Covariate-distance matching complete: {n_matched}/{n_treat} anchors matched"
    )
    return match_pairs, match_distances
