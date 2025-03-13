"""Balance assessment for CohortMatch.

Conventions follow cobalt/Stuart (2010):
- SMDs are signed: (treated mean - control mean) / denominator.
- The denominator is the standard deviation of the anchor group (treated for
  ATT, control for ATC) in the ORIGINAL sample, and the same denominator is
  used before and after matching, so the two numbers are comparable.
- Post-matching statistics use matching weights.
- Variance ratios are reported raw (treated variance / control variance),
  not folded to >= 1.
"""

import numpy as np
import pandas as pd

from cohortmatch.utils.logging import get_logger
from cohortmatch.validation import validate_data

logger = get_logger(__name__)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if weights.sum() == 0:
        return np.nan
    return float(np.average(values, weights=weights))


def _weighted_var(values: np.ndarray, weights: np.ndarray) -> float:
    """Bessel-corrected variance under reliability weights.

    Reduces to the ddof=1 sample variance when all weights are 1.
    """
    w_sum = weights.sum()
    w2_sum = (weights**2).sum()
    denom = w_sum - w2_sum / w_sum if w_sum > 0 else 0.0
    if denom <= 0:
        return np.nan
    mean = np.average(values, weights=weights)
    return float(np.sum(weights * (values - mean) ** 2) / denom)


def _group_stats(
    data: pd.DataFrame,
    var_name: str,
    treatment_col: str,
    weights: pd.Series | None,
) -> dict[str, float]:
    out = {}
    for label, group in (("treated", 1), ("control", 0)):
        mask = data[treatment_col] == group
        vals = data.loc[mask, var_name].to_numpy(dtype=float)
        if weights is not None:
            w = weights.reindex(data.index[mask]).fillna(0).to_numpy(dtype=float)
        else:
            w = np.ones(len(vals))
        out[f"mean_{label}"] = _weighted_mean(vals, w)
        out[f"var_{label}"] = _weighted_var(vals, w)
    return out


def standardized_mean_difference(
    data: pd.DataFrame,
    var_name: str,
    treatment_col: str,
    weights: pd.Series | None = None,
    denominator_std: float | None = None,
) -> float:
    """Signed standardized mean difference for one variable.

    Args:
        data: DataFrame with the units to compare.
        var_name: Variable to assess.
        treatment_col: Binary treatment column.
        weights: Optional per-unit weights (indexed like `data`).
        denominator_std: Fixed standardizing SD. If None, the pooled SD
            sqrt((var_t + var_c)/2) of `data` is used.

    Returns:
        (treated mean - control mean) / SD. Signed; may be negative.
    """
    stats = _group_stats(data, var_name, treatment_col, weights)
    diff = stats["mean_treated"] - stats["mean_control"]

    if denominator_std is None:
        denominator_std = np.sqrt((stats["var_treated"] + stats["var_control"]) / 2)

    if not np.isfinite(denominator_std) or denominator_std == 0:
        if diff == 0:
            return 0.0
        return np.inf if diff > 0 else -np.inf
    return diff / denominator_std


def variance_ratio(
    data: pd.DataFrame,
    var_name: str,
    treatment_col: str,
    weights: pd.Series | None = None,
) -> float:
    """Variance ratio (treated variance / control variance), unfolded."""
    stats = _group_stats(data, var_name, treatment_col, weights)
    var_t, var_c = stats["var_treated"], stats["var_control"]

    if var_t == 0 and var_c == 0:
        return 1.0
    if var_c == 0:
        return np.inf
    return var_t / var_c


def anchor_std(
    data: pd.DataFrame, var_name: str, treatment_col: str, anchor: str
) -> float:
    """SD of the anchor group in the original sample (the SMD denominator).

    Binary variables are standardized by sqrt(p(1-p)) and continuous ones by
    the ddof=1 sample SD, matching cobalt's conventions exactly. For ATE
    designs `anchor="pooled"` uses sqrt((var_t + var_c) / 2).
    """

    def _one(group: int) -> float:
        vals = data.loc[data[treatment_col] == group, var_name].to_numpy(dtype=float)
        unique = np.unique(vals[~np.isnan(vals)])
        if set(unique) <= {0.0, 1.0}:
            p = np.nanmean(vals)
            return float(p * (1 - p))  # variance
        return float(np.nanvar(vals, ddof=1))

    if anchor == "pooled":
        return float(np.sqrt((_one(1) + _one(0)) / 2))
    return float(np.sqrt(_one(1 if anchor == "treatment" else 0)))


def calculate_balance_stats(
    data: pd.DataFrame,
    matched_data: pd.DataFrame,
    covariates: list[str],
    treatment_col: str,
    weights: pd.Series | None = None,
    anchor: str = "treatment",
) -> pd.DataFrame:
    """Balance statistics before and after matching.

    Args:
        data: Original sample.
        matched_data: Matched sample.
        covariates: Covariates to assess.
        treatment_col: Binary treatment column.
        weights: Matching weights for the matched sample (indexed by unit).
        anchor: Which group's original-sample SD standardizes the SMDs
            ("treatment" for ATT designs, "control" for ATC).

    Returns:
        DataFrame with columns variable, smd_before, smd_after,
        var_ratio_before, var_ratio_after. SMDs are signed and share one
        denominator per covariate.
    """
    validate_data(data=data, treatment_col=treatment_col, covariates=covariates)
    matched_has_control = (matched_data[treatment_col] == 0).any()
    validate_data(
        data=matched_data,
        treatment_col=treatment_col,
        covariates=covariates,
        require_both_groups=False,
    )
    if not matched_has_control:
        logger.warning(
            "Matched data has no control units. Post-matching balance set to NaN."
        )

    results = []
    for cov in covariates:
        denom = anchor_std(data, cov, treatment_col, anchor)
        smd_before = standardized_mean_difference(
            data, cov, treatment_col, denominator_std=denom
        )
        vr_before = variance_ratio(data, cov, treatment_col)

        if matched_has_control:
            smd_after = standardized_mean_difference(
                matched_data,
                cov,
                treatment_col,
                weights=weights,
                denominator_std=denom,
            )
            vr_after = variance_ratio(matched_data, cov, treatment_col, weights=weights)
        else:
            smd_after = np.nan
            vr_after = np.nan

        results.append(
            {
                "variable": cov,
                "smd_before": smd_before,
                "smd_after": smd_after,
                "var_ratio_before": vr_before,
                "var_ratio_after": vr_after,
            }
        )

    return pd.DataFrame(results)


def create_table1(
    data: pd.DataFrame,
    matched_data: pd.DataFrame,
    covariates: list[str],
    treatment_col: str,
    weights: pd.Series | None = None,
    anchor: str = "treatment",
    group_labels: tuple[str, str] = ("treated", "control"),
) -> pd.DataFrame:
    """Cohort characteristics table (Table 1): group means and SDs with SMDs.

    `group_labels` names the two groups in the column headers (e.g.
    ("case", "control") for a nested case-control design).

    Returns:
        DataFrame with one row per variable (its name in the `variable`
        column, like `balance()`) and columns mean_<hi>_before, sd_<hi>_before,
        mean_<lo>_before, sd_<lo>_before, smd_before, and the same for after.
        Post-matching statistics use the matching weights.
    """
    hi, lo = group_labels
    rows = []
    for cov in covariates:
        denom = anchor_std(data, cov, treatment_col, anchor)
        before = _group_stats(data, cov, treatment_col, weights=None)
        after = _group_stats(matched_data, cov, treatment_col, weights=weights)
        rows.append(
            {
                "variable": cov,
                f"mean_{hi}_before": before["mean_treated"],
                f"sd_{hi}_before": np.sqrt(before["var_treated"]),
                f"mean_{lo}_before": before["mean_control"],
                f"sd_{lo}_before": np.sqrt(before["var_control"]),
                "smd_before": standardized_mean_difference(
                    data, cov, treatment_col, denominator_std=denom
                ),
                f"mean_{hi}_after": after["mean_treated"],
                f"sd_{hi}_after": np.sqrt(after["var_treated"]),
                f"mean_{lo}_after": after["mean_control"],
                f"sd_{lo}_after": np.sqrt(after["var_control"]),
                "smd_after": standardized_mean_difference(
                    matched_data,
                    cov,
                    treatment_col,
                    weights=weights,
                    denominator_std=denom,
                ),
            }
        )
    table = pd.DataFrame(rows)
    table.attrs["n_treated_before"] = int((data[treatment_col] == 1).sum())
    table.attrs["n_control_before"] = int((data[treatment_col] == 0).sum())
    table.attrs["n_treated_after"] = int((matched_data[treatment_col] == 1).sum())
    table.attrs["n_control_after"] = int((matched_data[treatment_col] == 0).sum())
    return table


def calculate_rubin_rules(
    balance_df: pd.DataFrame,
    smd_threshold: float = 0.25,
    var_ratio_threshold: float = 2.0,
    smd_col: str = "smd_after",
    var_ratio_col: str = "var_ratio_after",
) -> dict[str, float]:
    """Share of covariates satisfying Rubin's rules (|SMD|, variance ratio)."""
    if balance_df.empty:
        return {
            "n_variables_total": 0,
            "n_smd_small": 0,
            "pct_smd_small": 0.0,
            "n_var_ratio_good": 0,
            "pct_var_ratio_good": 0.0,
            "n_both_good": 0,
            "pct_both_good": 0.0,
        }

    n_variables_total = len(balance_df)
    abs_smd = balance_df[smd_col].abs()
    vr = balance_df[var_ratio_col]

    smd_ok = abs_smd < smd_threshold
    vr_ok = (vr >= 1 / var_ratio_threshold) & (vr <= var_ratio_threshold)

    n_smd_small = int(smd_ok.sum())
    n_var_ratio_good = int(vr_ok.sum())
    n_both_good = int((smd_ok & vr_ok).sum())

    return {
        "n_variables_total": n_variables_total,
        "n_smd_small": n_smd_small,
        "pct_smd_small": 100 * n_smd_small / n_variables_total,
        "n_var_ratio_good": n_var_ratio_good,
        "pct_var_ratio_good": 100 * n_var_ratio_good / n_variables_total,
        "n_both_good": n_both_good,
        "pct_both_good": 100 * n_both_good / n_variables_total,
    }


def calculate_balance_index(balance_df: pd.DataFrame) -> dict[str, float]:
    """Summary of balance improvement based on |SMD| before vs after."""
    abs_before = balance_df["smd_before"].abs()
    abs_after = balance_df["smd_after"].abs()

    mean_smd_before = abs_before.mean()
    mean_smd_after = abs_after.mean()
    max_smd_before = abs_before.max()
    max_smd_after = abs_after.max()

    mean_balance_ratio = (
        mean_smd_before / mean_smd_after if mean_smd_after > 0 else np.inf
    )
    max_balance_ratio = max_smd_before / max_smd_after if max_smd_after > 0 else np.inf

    n_improved = int((abs_after < abs_before).sum())
    pct_improved = 100 * n_improved / len(balance_df) if len(balance_df) > 0 else np.nan

    if (
        np.isfinite(mean_smd_before)
        and np.isfinite(mean_smd_after)
        and mean_smd_before > 0
    ):
        mean_smd_reduction_pct = (
            100 * (mean_smd_before - mean_smd_after) / mean_smd_before
        )
    else:
        mean_smd_reduction_pct = 0.0
    balance_index = (
        (0.7 * mean_smd_reduction_pct + 0.3 * pct_improved)
        if not np.isnan(pct_improved)
        else mean_smd_reduction_pct
    )
    balance_index = max(0, min(100, balance_index))

    return {
        "mean_smd_before": mean_smd_before,
        "mean_smd_after": mean_smd_after,
        "max_smd_before": max_smd_before,
        "max_smd_after": max_smd_after,
        "mean_balance_ratio": mean_balance_ratio,
        "max_balance_ratio": max_balance_ratio,
        "n_variables_improved": n_improved,
        "pct_variables_improved": pct_improved,
        "balance_index": balance_index,
    }


def calculate_overall_balance(
    balance_df: pd.DataFrame, threshold: float = 0.1
) -> dict[str, float]:
    """Overall balance metrics on |SMD|."""
    df = balance_df.copy()
    df["abs_before"] = df["smd_before"].abs()
    df["abs_after"] = df["smd_after"].abs()
    has_after_stats = not df["abs_after"].isna().all()

    results = {
        "mean_smd_before": df["abs_before"].mean(),
        "max_smd_before": df["abs_before"].max(),
        "prop_balanced_before": (df["abs_before"] < threshold).mean(),
    }

    if has_after_stats:
        valid = df[~df["abs_after"].isna()].copy()
        results.update(
            {
                "mean_smd_after": valid["abs_after"].mean(),
                "max_smd_after": valid["abs_after"].max(),
                "prop_balanced_after": (valid["abs_after"] < threshold).mean(),
            }
        )
        valid["smd_reduction"] = valid["abs_before"] - valid["abs_after"]
        valid["smd_reduction_percent"] = np.where(
            valid["abs_before"] > 0,
            100 * valid["smd_reduction"] / valid["abs_before"],
            0.0,
        )
        results.update(
            {
                "mean_reduction": valid["smd_reduction"].mean(),
                "mean_reduction_percent": valid["smd_reduction_percent"].mean(),
                "percent_balanced_improved": 100
                * (valid["abs_after"] < valid["abs_before"]).mean(),
            }
        )
    else:
        results.update(
            {
                "mean_smd_after": np.nan,
                "max_smd_after": np.nan,
                "prop_balanced_after": np.nan,
                "mean_reduction": np.nan,
                "mean_reduction_percent": np.nan,
                "percent_balanced_improved": np.nan,
            }
        )
    return results
