"""Treatment effect estimation on matched samples.

Effects are estimated by weighted least squares on the matched sample using
the matching weights. Standard errors account for the matched design:
cluster-robust on match groups when subclass membership is available
(matching without replacement), heteroskedasticity-robust (HC1) otherwise.
This mirrors the workflow recommended in the MatchIt documentation
(weighted outcome regression + cluster-robust variance).
"""

from typing import Any

import numpy as np
import pandas as pd

from cohortmatch.utils.logging import get_logger
from cohortmatch.validation import validate_data

logger = get_logger(__name__)

VALID_METHODS = {"mean_difference", "regression_adjustment"}
VALID_FAMILIES = {"linear", "logistic", "poisson"}


def estimate_treatment_effect(
    data: pd.DataFrame,
    outcome: str,
    treatment_col: str,
    method: str = "mean_difference",
    covariates: list[str] | None = None,
    estimand: str = "att",
    family: str = "linear",
    weights: pd.Series | None = None,
    subclass: pd.Series | None = None,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Estimate the treatment effect for one outcome on a matched sample.

    Args:
        data: Matched data (must contain both groups).
        outcome: Outcome column.
        treatment_col: Binary treatment column.
        method: "mean_difference" (treatment only in the model) or
            "regression_adjustment" (adds covariates to the outcome model).
        covariates: Adjustment covariates (regression_adjustment only).
        estimand: Label inherited from the matching design ("att", "atc",
            or "ate" for stratum designs).
        family: Outcome model family. "linear" (weighted least squares; the
            effect is a mean difference, labeled risk_difference for binary
            outcomes), "logistic" (marginal odds ratio; binary outcome), or
            "poisson" (marginal risk/rate ratio; robust errors). The
            nonlinear families fit the treatment-only model and cannot be
            combined with regression adjustment.
        weights: Matching weights indexed by unit (defaults to 1).
        subclass: Match-group membership indexed by unit. When given,
            standard errors are cluster-robust on these groups; otherwise HC1.
        confidence_level: Level for the confidence interval.

    Returns:
        Dictionary with the effect (difference or exponentiated ratio, see
        the "measure" key), standard error, statistic, p-value, CI, the
        standard-error type, and sample sizes.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown estimation method: {method}. Must be one of: {', '.join(sorted(VALID_METHODS))}"
        )
    if family not in VALID_FAMILIES:
        raise ValueError(
            f"Unknown family: {family}. Must be one of: {', '.join(sorted(VALID_FAMILIES))}"
        )
    if estimand not in ("att", "atc", "ate"):
        raise ValueError(
            f"Unknown estimand: {estimand}. Must be 'att', 'atc', or 'ate'"
        )
    if not 0 < confidence_level < 1:
        raise ValueError(
            f"Confidence level must be between 0 and 1, got {confidence_level}"
        )
    if method == "regression_adjustment" and covariates is None:
        raise ValueError("Covariates must be provided for regression adjustment")
    if method == "mean_difference" and covariates:
        raise ValueError("adjustment covariates require method='regression_adjustment'")
    if family != "linear" and method == "regression_adjustment":
        raise ValueError(
            "odds and risk ratios are estimated from the treatment-only model "
            "(a marginal effect); regression adjustment is linear-only. "
            "Covariate-conditional ratios differ from marginal ones "
            "(non-collapsibility) and are out of scope."
        )

    validate_data(
        data=data,
        treatment_col=treatment_col,
        covariates=[] if method != "regression_adjustment" else covariates or [],
        outcomes=[outcome],
    )

    model_covariates = covariates if method == "regression_adjustment" else []
    used_cols = [outcome, treatment_col] + list(model_covariates)
    model_data = data[used_cols].dropna()

    n_treat = int((model_data[treatment_col] == 1).sum())
    n_control = int((model_data[treatment_col] == 0).sum())
    if n_treat == 0 or n_control == 0:
        raise ValueError(
            f"Effect estimation needs both groups; got {n_treat} treated and "
            f"{n_control} control units."
        )

    w = (
        weights.reindex(model_data.index).fillna(0).to_numpy(dtype=float)
        if weights is not None
        else np.ones(len(model_data))
    )
    groups = (
        subclass.reindex(model_data.index).to_numpy() if subclass is not None else None
    )

    if family == "linear":
        result = _wls_effect(
            model_data,
            outcome,
            treatment_col,
            model_covariates,
            w,
            groups,
            confidence_level,
        )
    else:
        result = _glm_effect(
            model_data,
            outcome,
            treatment_col,
            model_covariates,
            w,
            groups,
            confidence_level,
            family,
        )
    # report the estimator that produced the effect: the GLM family for a
    # ratio measure, else the point-estimate method (mean_difference vs
    # regression_adjustment). "mean_difference" beside an odds ratio was
    # confusing.
    reported_method = family if family in ("logistic", "poisson") else method
    result.update(
        {
            "method": reported_method,
            "estimand": estimand,
            "confidence_level": confidence_level,
            "n_treatment": n_treat,
            "n_control": n_control,
            "n_total": len(model_data),
        }
    )
    if method == "mean_difference":
        treat_mask = model_data[treatment_col] == 1
        y = model_data[outcome].to_numpy(dtype=float)
        result["treat_mean"] = float(np.average(y[treat_mask], weights=w[treat_mask]))
        result["control_mean"] = float(
            np.average(y[~treat_mask], weights=w[~treat_mask])
        )
    return result


def _wls_effect(
    model_data: pd.DataFrame,
    outcome: str,
    treatment_col: str,
    covariates: list[str],
    w: np.ndarray,
    groups: np.ndarray | None,
    confidence_level: float,
) -> dict[str, float]:
    """Weighted outcome regression; effect = treatment coefficient."""
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError(
            "Statsmodels is required for effect estimation. "
            "Install it with 'pip install statsmodels'"
        ) from None

    y = model_data[outcome].to_numpy(dtype=float)
    X = sm.add_constant(
        model_data[[treatment_col] + covariates].to_numpy(dtype=float),
        has_constant="add",
    )

    model = sm.WLS(y, X, weights=w)
    if groups is not None:
        # cluster-robust on match groups; statsmodels uses n_groups-based df
        valid = pd.Series(groups).notna().to_numpy()
        if not valid.all():
            logger.warning(
                "Units without subclass membership present; dropping them from "
                "cluster-robust inference."
            )
            model = sm.WLS(y[valid], X[valid], weights=w[valid])
            groups = groups[valid]
        codes = pd.factorize(groups)[0]
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": codes}, use_t=True)
        se_type = "cluster-robust (match groups)"
    else:
        fit = model.fit(cov_type="HC1", use_t=True)
        se_type = "HC1-robust"

    treat_ix = 1  # column order: const, treatment, covariates...
    alpha = 1 - confidence_level
    ci = fit.conf_int(alpha=alpha)[treat_ix]

    binary_outcome = set(np.unique(y)) <= {0.0, 1.0}
    result = {
        "effect": float(fit.params[treat_ix]),
        "measure": "risk_difference" if binary_outcome else "mean_difference",
        "standard_error": float(fit.bse[treat_ix]),
        "t_statistic": float(fit.tvalues[treat_ix]),
        "p_value": float(fit.pvalues[treat_ix]),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "se_type": se_type,
    }
    if covariates:
        result["r_squared"] = float(fit.rsquared)
        result["adj_r_squared"] = float(fit.rsquared_adj)
    return result


def _glm_effect(
    model_data: pd.DataFrame,
    outcome: str,
    treatment_col: str,
    covariates: list[str],
    w: np.ndarray,
    groups: np.ndarray | None,
    confidence_level: float,
    family: str,
) -> dict[str, float]:
    """Weighted GLM; the exponentiated treatment coefficient is the effect."""
    import statsmodels.api as sm

    y = model_data[outcome].to_numpy(dtype=float)
    if family == "logistic":
        if not set(np.unique(y)) <= {0.0, 1.0}:
            raise ValueError(
                f"family='logistic' needs a binary outcome; '{outcome}' is not 0/1"
            )
        glm_family = sm.families.Binomial()
        measure = "odds_ratio"
    else:
        if (y < 0).any():
            raise ValueError(
                f"family='poisson' needs a non-negative outcome; '{outcome}' has "
                "negative values"
            )
        glm_family = sm.families.Poisson()
        measure = "risk_ratio"

    X = sm.add_constant(model_data[[treatment_col] + covariates].to_numpy(dtype=float))
    # matching weights are analytic/sampling weights: var_weights gives
    # the correct robust sandwich (freq_weights treats them as counts and
    # understates HC variances)
    model = sm.GLM(y, X, family=glm_family, var_weights=w)
    if groups is not None:
        valid = pd.Series(groups).notna().to_numpy()
        if not valid.all():
            model = sm.GLM(y[valid], X[valid], family=glm_family, var_weights=w[valid])
            groups = groups[valid]
        codes = pd.factorize(groups)[0]
        fit = _fit_glm_robust(model, cov_type="cluster", cov_kwds={"groups": codes})
        se_type = "cluster-robust (match groups)"
    else:
        fit = _fit_glm_robust(model, cov_type="HC1")
        se_type = "HC1-robust"

    treat_ix = 1
    alpha = 1 - confidence_level
    ci = fit.conf_int(alpha=alpha)[treat_ix]
    return {
        "effect": float(np.exp(fit.params[treat_ix])),
        "measure": measure,
        "log_effect": float(fit.params[treat_ix]),
        "standard_error": float(fit.bse[treat_ix]),  # on the log scale
        "t_statistic": float(fit.tvalues[treat_ix]),  # z for GLM families
        "p_value": float(fit.pvalues[treat_ix]),
        "ci_lower": float(np.exp(ci[0])),
        "ci_upper": float(np.exp(ci[1])),
        "se_type": se_type,
    }


def _fit_glm_robust(model, **fit_kwargs):
    """Fit a var_weights GLM with a robust covariance.

    statsmodels emits a blanket SpecificationWarning for any robust cov_type
    with var_weights; the var_weights sandwich is verified correct for
    sampling weights (see tests/test_api.py::test_glm_var_weights_sandwich),
    so the warning is suppressed here.
    """
    import warnings as _warnings

    from statsmodels.tools.sm_exceptions import SpecificationWarning

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", SpecificationWarning)
        return model.fit(**fit_kwargs)


def estimate_multiple_outcomes(
    data: pd.DataFrame,
    outcomes: list[str],
    treatment_col: str,
    method: str = "mean_difference",
    covariates: list[str] | None = None,
    estimand: str = "att",
    family: str = "linear",
    weights: pd.Series | None = None,
    subclass: pd.Series | None = None,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Estimate treatment effects for several outcomes.

    Returns:
        DataFrame with one row per outcome. Raises on the first outcome that
        cannot be estimated rather than returning partial results.
    """
    if not outcomes:
        raise ValueError("outcomes must be a non-empty list")

    results = []
    for outcome in outcomes:
        result = estimate_treatment_effect(
            data=data,
            outcome=outcome,
            treatment_col=treatment_col,
            method=method,
            covariates=covariates,
            estimand=estimand,
            family=family,
            weights=weights,
            subclass=subclass,
            confidence_level=confidence_level,
        )
        result["outcome"] = outcome
        results.append(result)

    results_df = pd.DataFrame(results)
    # A tidy, inference-led frame that stays readable in a terminal. The
    # diagnostics dropped here (log_effect, t_statistic, group means, r^2,
    # n_total) remain on the per-outcome dict from estimate_treatment_effect.
    tidy_columns = [
        "outcome",
        "effect",
        "measure",
        "ci_lower",
        "ci_upper",
        "p_value",
        "standard_error",
        "se_type",
        "estimand",
        "method",
        "n_treatment",
        "n_control",
    ]
    keep = [c for c in tidy_columns if c in results_df.columns]
    return results_df[keep]
