"""Stratum-based methods: propensity subclassification and coarsened exact matching.

Both assign units to strata and express the design through stratum weights
instead of pairs; balance and effect estimation then reuse the standard
weighted machinery.
"""

import warnings as _warnings

import numpy as np
import pandas as pd

from cohortmatch.utils.logging import get_logger

logger = get_logger(__name__)


def subclassify(
    propensity_scores: np.ndarray,
    treated: np.ndarray,
    n_subclasses: int,
    estimand: str,
) -> np.ndarray:
    """Assign units to propensity-score strata.

    Stratum edges are quantiles of the target group's scores (treated for
    ATT, control for ATC, everyone for ATE), following MatchIt. Returns an
    integer stratum label per unit; -1 marks units in strata that lack one
    of the groups (they are excluded from the design).
    """
    if estimand == "att":
        reference = propensity_scores[treated]
    elif estimand == "atc":
        reference = propensity_scores[~treated]
    else:
        reference = propensity_scores

    edges = np.quantile(reference, np.linspace(0, 1, n_subclasses + 1))
    edges = np.unique(edges)
    if len(edges) - 1 < n_subclasses:
        _warnings.warn(
            f"Tied propensity quantiles reduce the number of subclasses to "
            f"{len(edges) - 1}.",
            UserWarning,
            stacklevel=3,
        )
    edges[0], edges[-1] = -np.inf, np.inf

    labels = np.digitize(propensity_scores, edges[1:-1], right=True)

    # drop strata that lack either group
    for s in np.unique(labels):
        in_s = labels == s
        if not (treated[in_s].any() and (~treated[in_s]).any()):
            logger.warning(
                f"Subclass {s} lacks one group ({int(in_s.sum())} units); excluded."
            )
            labels[in_s] = -1
    return labels


def cem_strata(
    data: pd.DataFrame,
    covariates: list[str],
    treated: np.ndarray,
    coarsening: dict | None = None,
    exact_cols: list[str] | None = None,
) -> np.ndarray:
    """Assign units to coarsened-exact cells.

    Continuous covariates are cut into bins (Sturges' count by default, or
    per-variable via `coarsening`: an int bin count or explicit bin edges);
    binary covariates and `exact_cols` enter uncoarsened. Returns an integer
    cell label per unit; -1 marks cells lacking one of the groups.
    """
    coarsening = coarsening or {}
    keys = []
    for cov in covariates:
        vals = data[cov].to_numpy()
        spec = coarsening.get(cov)
        unique = np.unique(vals[~pd.isna(vals)])
        if spec is None and set(unique) <= {0, 1}:
            keys.append(vals.astype(int))
            continue
        if isinstance(spec, (list, np.ndarray)):
            bins = np.asarray(spec, dtype=float)
        else:
            n_bins = (
                spec if isinstance(spec, int) else int(np.ceil(np.log2(len(data)) + 1))
            )
            bins = np.linspace(np.nanmin(vals), np.nanmax(vals), n_bins + 1)[1:-1]
        keys.append(np.digitize(vals.astype(float), bins, right=True))
    for col in exact_cols or []:
        keys.append(data[col].to_numpy())

    cells = pd.MultiIndex.from_arrays(keys).to_flat_index()
    labels = pd.factorize(cells)[0]

    for s in np.unique(labels):
        in_s = labels == s
        if not (treated[in_s].any() and (~treated[in_s]).any()):
            labels[in_s] = -1
    n_dropped = int((labels == -1).sum())
    if n_dropped:
        logger.info(f"CEM: {n_dropped} unit(s) in cells lacking one group; excluded.")
    return labels


def stratum_weights(
    labels: np.ndarray, treated: np.ndarray, estimand: str
) -> np.ndarray:
    """Marginal-mean weights from stratum membership (MatchIt convention).

    ATT: treated weight 1, controls reweighted to the treated distribution
    across strata; ATC symmetric; ATE: both groups reweighted to the full
    sample's stratum distribution. Weights are rescaled to average 1 within
    each group; excluded units (label -1) get weight 0.
    """
    weights = np.zeros(len(labels), dtype=float)
    kept = labels >= 0

    for s in np.unique(labels[kept]):
        in_s = labels == s
        n_t = int((in_s & treated).sum())
        n_c = int((in_s & ~treated).sum())
        n_s = n_t + n_c
        if estimand == "att":
            w_t, w_c = 1.0, n_t / n_c
        elif estimand == "atc":
            w_t, w_c = n_c / n_t, 1.0
        else:  # ate
            w_t, w_c = n_s / n_t, n_s / n_c
        weights[in_s & treated] = w_t
        weights[in_s & ~treated] = w_c

    # rescale to mean 1 within each retained group
    for group_mask in (treated, ~treated):
        sel = kept & group_mask
        total = weights[sel].sum()
        if total > 0:
            weights[sel] *= sel.sum() / total
    return weights
