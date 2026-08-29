"""Top-level matching API.

This module provides `match()`, the primary entry point of cohortmatch,
and `MatchResult`, the object it returns.
"""

from __future__ import annotations

import warnings
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from cohortmatch.datatypes import MatcherConfig, MatchResults
from cohortmatch.exceptions import (
    ApproximateMatchWarning,
    CommonSupportWarning,
    IncompleteMatchWarning,
    NoMatchesError,
)
from cohortmatch.metrics.treatment import estimate_multiple_outcomes
from cohortmatch.pipeline import run_match

_METHODS = {"nearest", "optimal"}
_DISTANCES = {"propensity", "logit", "mahalanobis", "euclidean"}
_PAIR_ESTIMANDS = {"att", "atc"}
_STRATUM_ESTIMANDS = {"att", "atc", "ate"}
_ENGINES = {"auto", "exact", "approximate"}
_PS_METRICS = {"propensity", "logit"}

# Internal column name used when propensity scores are passed as an array
_PS_COL = "_cb3_propensity"


def match(
    data: pd.DataFrame,
    *,
    treatment: str,
    covariates: list[str],
    method: str = "nearest",
    distance: str = "propensity",
    estimand: str = "att",
    caliper: float | str | None = None,
    caliper_metric: str | None = None,
    std_caliper: bool | None = None,
    covariate_calipers: dict[str, float] | None = None,
    ratio: int = 1,
    replace: bool = False,
    exact: str | list[str] | None = None,
    propensity_scores: str | pd.Series | np.ndarray | None = None,
    propensity_model: Any | None = None,
    cv: int | None = None,
    engine: str = "auto",
    m_order: str | None = None,
    discard: str | None = None,
    covariate_weights: dict[str, float] | None = None,
    standardize: bool = True,
    random_state: int | None = None,
    memory_limit_gb: float = 4.0,
) -> MatchResult:
    """Match treatment and control units on covariates.

    Args:
        data: DataFrame with one row per unit. The index identifies units and
            is preserved in all outputs; it must be unique and have string or
            integer labels.
        treatment: Name of the binary treatment column (1=treated, 0=control).
        covariates: Covariate columns to balance.
        method: "nearest" (nearest-neighbor, a.k.a. greedy) or "optimal"
            (Hungarian algorithm, minimizes total distance). For stratum
            designs see `subclassify()` and `cem()`.
        distance: Distance metric: "propensity" (absolute difference of
            propensity scores), "logit" (difference on the logit scale),
            "mahalanobis", or "euclidean".
        estimand: Which group anchors the matching. "att" matches controls to
            treated units (every treated unit is retained if possible);
            "atc" matches treated units to controls. For estimand="ate" use
            the stratum designs `subclassify()` or `cem()`.
        caliper: Maximum allowed distance between matched units. None (default)
            imposes no caliper. "auto" applies the standard propensity caliper
            of 0.2 standard deviations of the logit propensity score
            (Austin, 2011). A float sets the threshold explicitly, interpreted
            according to `caliper_metric` and `std_caliper`.
        caliper_metric: Metric the caliper applies to: "propensity", "logit",
            "mahalanobis", or "euclidean". Defaults to "propensity". May differ
            from `distance` (e.g. Mahalanobis matching within a propensity
            caliper).
        covariate_calipers: Per-variable caliper: maximum absolute difference
            allowed between matched units, in the variable's raw units (e.g.
            {"age": 3.0} for "within 3 years"). Applies on top of `caliper`
            and `exact`. Note: MatchIt's named calipers default to SD units;
            these are raw by design.
        std_caliper: If True (default), a numeric propensity/logit caliper is
            interpreted in standard deviations of the logit propensity score;
            if False, in raw units (probability-scale difference for
            "propensity", logit-scale difference for "logit"). Ignored for
            covariate-space metrics, which are always in raw distance units.
        ratio: Number of controls to match to each anchor unit (integer).
        replace: Whether controls may be reused across matches ("nearest" only).
        exact: Column name or list of columns that must match exactly.
        propensity_scores: Precomputed propensity scores: a column name in
            `data`, a Series aligned to `data.index`, or an array of
            length len(data). Mutually exclusive with `propensity_model`.
        propensity_model: An sklearn-compatible classifier used to estimate
            propensity scores. The estimator is cloned and, by default, fit on
            the full sample; pass `cv` to estimate scores out-of-fold instead.
            If neither this nor `propensity_scores` is given and scores are
            needed, logistic regression is used.
        cv: Cross-fitting folds for propensity estimation. None (default) fits
            the score on the full sample (deterministic, MatchIt's convention);
            an integer >= 2 opts into k-fold cross-fitting, where each unit is
            scored by a model that did not see it. Only meaningful when scores
            are estimated.
        engine: Compute strategy (distinct from `method`, which picks the
            matching algorithm). "exact" builds the full distance matrix;
            "approximate" is memory-bounded ("nearest" only), a
            propensity-sorted window for propensity/logit distances (needs a
            caliper), or a whitened KD-tree for Mahalanobis/Euclidean
            distances (no caliper needed). "auto" (default) picks "exact"
            when the distance matrix fits within `memory_limit_gb`, else
            "approximate".
        discard: Drop units outside the common propensity support prior to
            matching, "treated", "control", or "both". Dropped ids are on
            `result.discarded` and a warning reports the count. Requires
            propensity scores (estimated automatically if absent).
        m_order: Order in which anchor units pick their matches ("nearest"
            only). Default: fewest potential matches first (exact path) or
            largest propensity score first (approximate path), both match
            the hardest units while the pool is fullest. Also
            "largest"/"smallest" by propensity score, "closest" by best
            available distance (exact path only), "random", and "data".
        covariate_weights: Per-covariate weights for the distance calculation.
            Only valid with distance="euclidean".
        standardize: Standardize covariates before computing covariate-space
            distances.
        random_state: Seed for tie-breaking, m_order="random", and
            propensity cross-fitting. Never changes the matching order by
            itself.
        memory_limit_gb: Budget (GB) for the dense distance matrix under
            engine="auto". The dense path's peak is ~2x this during
            matching (a working copy), so set it near a third of available
            RAM; the approximate path holds no matrix.

    Returns:
        MatchResult with matched data, pairs, and balance diagnostics.

    Raises:
        ValueError: On invalid or inconsistent arguments.
        NoMatchesError: If no pairs satisfy the constraints.

    Example:
        >>> from cohortmatch.datasets import load_lalonde
        >>> data = load_lalonde()
        >>> result = match(
        ...     data, treatment="treat", covariates=["age", "educ", "re74"], caliper="auto"
        ... )
        >>> result.balance()  # doctest: +SKIP
    """
    # --- validate string choices -------------------------------------------
    method = _check_choice(
        "method",
        method,
        _METHODS,
        aliases={"greedy": "nearest"},
        hints={
            "fast_greedy": "use engine='approximate' instead",
            "subclass": "use cohortmatch.subclassify()",
            "cem": "use cohortmatch.cem()",
        },
    )
    distance = _check_choice(
        "distance",
        distance,
        _DISTANCES,
        hints={
            "glm": "MatchIt's distance='glm' estimates a propensity score; "
            "use distance='propensity' (optionally with propensity_model=...)",
            "ps": "use distance='propensity'",
        },
    )
    estimand = _check_choice(
        "estimand",
        estimand,
        _PAIR_ESTIMANDS,
        hints={
            "ate": "pair matching targets 'att' or 'atc'; use subclassify() "
            "or cem() for the ATE"
        },
    )
    engine = _check_choice("engine", engine, _ENGINES)

    std_caliper_given = std_caliper is not None
    cv_given = cv is not None
    if std_caliper_given and caliper is None:
        raise ValueError(
            "std_caliper was given but caliper is None; set caliper as well"
        )
    std_caliper = True if std_caliper is None else std_caliper
    if cv is not None and propensity_scores is not None:
        raise ValueError(
            "cv only applies when propensity scores are estimated; remove it "
            "or remove propensity_scores"
        )
    # cv is None -> fit the propensity score on the full sample (deterministic,
    # MatchIt's convention). cv=k -> opt into k-fold cross-fitting.
    if m_order == "closest" and engine == "approximate":
        raise ValueError(
            "m_order='closest' requires the dense distance matrix; use "
            "engine='exact' or another order"
        )

    # --- validate simple parameters ----------------------------------------
    if not isinstance(covariates, (list, tuple)) or not covariates:
        raise ValueError("covariates must be a non-empty list of column names")
    covariates = list(covariates)

    if isinstance(ratio, float) and not ratio.is_integer():
        raise ValueError(
            f"ratio must be an integer, got {ratio}. Fractional ratios are not "
            "supported; the previous API silently truncated them."
        )
    ratio = int(ratio)
    if ratio < 1:
        raise ValueError(f"ratio must be >= 1, got {ratio}")

    if isinstance(exact, str):
        exact = [exact]
    for col in exact or []:
        if col not in data.columns:
            raise ValueError(f"exact column '{col}' not in data")

    if cv is not None and cv < 2:
        raise ValueError(f"cv must be >= 2, got {cv}")

    if propensity_scores is not None and propensity_model is not None:
        raise ValueError(
            "propensity_scores and propensity_model are mutually exclusive: "
            "pass precomputed scores or a model to estimate them, not both."
        )

    if covariate_weights is not None:
        if distance != "euclidean":
            raise ValueError(
                "covariate_weights only affect euclidean distances; they are "
                f"ignored by distance='{distance}'. Remove the argument or use "
                "distance='euclidean'."
            )
        unknown = set(covariate_weights) - set(covariates)
        if unknown:
            raise ValueError(
                f"covariate_weights refer to non-covariates: {sorted(unknown)}"
            )

    if covariate_calipers is not None:
        for col, value in covariate_calipers.items():
            if col not in data.columns:
                raise ValueError(f"covariate_calipers column '{col}' not in data")
            if not np.issubdtype(np.asarray(data[col]).dtype, np.number):
                raise ValueError(f"covariate_calipers column '{col}' must be numeric")
            if not value > 0:
                raise ValueError(
                    f"covariate_calipers['{col}'] must be positive, got {value}"
                )

    if discard is not None and discard not in ("treated", "control", "both"):
        raise ValueError(
            f"discard must be 'treated', 'control', or 'both', got {discard!r}"
        )

    if m_order is not None:
        valid_orders = {"largest", "smallest", "closest", "random", "data"}
        if m_order not in valid_orders:
            raise ValueError(
                f"m_order must be one of {sorted(valid_orders)}, got {m_order!r}"
            )
        if method != "nearest":
            raise ValueError("m_order only applies to method='nearest'")

    if method == "optimal" and replace:
        raise ValueError(
            "replace=True is not supported with method='optimal'; the optimal "
            "assignment already uses each control at most once. Use "
            "method='nearest' for matching with replacement."
        )

    if covariate_weights is not None:
        for col, value in covariate_weights.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"covariate_weights['{col}'] must be non-negative and finite, "
                    f"got {value}"
                )

    # --- caliper translation ------------------------------------------------
    caliper_method, caliper_value, caliper_scale = _resolve_caliper(
        caliper, caliper_metric, std_caliper
    )
    if std_caliper_given and caliper_method in ("mahalanobis", "euclidean"):
        raise ValueError(
            "std_caliper applies only to propensity/logit calipers; "
            f"caliper_metric='{caliper_method}' calipers are always in raw "
            "distance units. Remove std_caliper."
        )

    # --- categorical covariates ---------------------------------------------
    user_covariates = list(covariates)
    work_data, covariates, injected_cols = _encode_covariates(data, covariates)
    if covariate_weights and injected_cols:
        expanded_weights = {}
        for name, value in covariate_weights.items():
            levels = [c for c in injected_cols if c.startswith(f"{name}=")]
            if levels:
                for level_col in levels:
                    expanded_weights[level_col] = value
            else:
                expanded_weights[name] = value
        covariate_weights = expanded_weights

    # --- propensity plumbing ------------------------------------------------
    (work_data, propensity_col, model_type, model_params, injected_ps_col) = (
        _resolve_propensity_input(data, work_data, propensity_scores, propensity_model)
    )

    needs_ps = (
        distance in _PS_METRICS
        or caliper_method in _PS_METRICS
        or propensity_col is not None
        or propensity_model is not None
        or discard is not None
        or m_order in ("largest", "smallest")  # order by propensity score
    )
    if cv_given and not needs_ps:
        raise ValueError(
            "cv only applies when propensity scores are estimated; this design "
            "estimates none (covariate distance, no propensity caliper/discard). "
            "Remove cv."
        )

    # --- group sizes and algorithm selection --------------------------------
    if treatment not in data.columns:
        raise ValueError(f"treatment column '{treatment}' not in data")
    n_treated = int((data[treatment] == 1).sum())
    n_control = int((data[treatment] == 0).sum())
    n_focal, n_pool = (
        (n_treated, n_control) if estimand == "att" else (n_control, n_treated)
    )

    engine = _resolve_engine(
        engine,
        method,
        caliper_method,
        n_focal,
        n_pool,
        distance,
        memory_limit_gb,
    )
    match_method = {"nearest": "greedy", "optimal": "optimal"}[method]
    if engine == "approximate":
        match_method = "fast_greedy"
        # the covariate-space tree path needs no propensity score; only the
        # propensity-window path does
        if distance in _PS_METRICS or caliper_method in _PS_METRICS:
            needs_ps = True

    # --- run ---------------------------------------------------------------
    config = MatcherConfig(
        treatment_col=treatment,
        covariates=covariates,
        match_method=match_method,
        distance_method=distance,
        exact_match_cols=exact or [],
        standardize=standardize,
        caliper_method=caliper_method,
        caliper_value=caliper_value,
        caliper_scale=caliper_scale,
        replace=replace,
        ratio=float(ratio),
        random_state=random_state,
        weights=covariate_weights,
        m_order=m_order,
        covariate_calipers=covariate_calipers,
        discard=discard,
        estimand=estimand,
        matching_direction="control" if estimand == "atc" else "treatment",
        estimate_propensity=needs_ps and propensity_col is None,
        propensity_col=propensity_col,
        propensity_model=model_type,
        model_params=model_params,
        cv_folds=cv,
    )

    results = run_match(work_data, config)

    if not results.pairs or results.matched_data.empty:
        raise NoMatchesError(
            "No matches satisfied the constraints "
            f"(caliper={caliper!r}, exact={exact!r}, method='{method}'). "
            "Relax the caliper, drop exact-matching columns, or check group overlap."
        )

    settings = {
        "treatment": treatment,
        "covariates": tuple(user_covariates),
        "encoded_covariates": tuple(covariates),
        "method": method,
        "distance": distance,
        "estimand": estimand,
        "caliper": caliper,
        "covariate_calipers": covariate_calipers,
        "caliper_metric": caliper_method,
        "std_caliper": std_caliper,
        "ratio": ratio,
        "replace": replace,
        "exact": tuple(exact) if exact else None,
        "engine": engine,
        "m_order": m_order,
        "discard": discard,
        "standardize": standardize,
        "cv": cv,
        "random_state": random_state,
    }
    result = MatchResult(
        results,
        estimand=estimand,
        settings=settings,
        injected_ps_col=injected_ps_col,
        injected_cols=injected_cols,
    )

    if result._results.discarded is not None and len(result._results.discarded):
        note = ""
        if distance in ("mahalanobis", "euclidean"):
            note = (
                " Common support is defined on a propensity score, so a "
                "propensity model was fit for the discard even though matching "
                "uses the covariate distance; pass discard=None to skip it."
            )
        warnings.warn(
            f"{len(result._results.discarded)} unit(s) outside the common "
            "propensity support were discarded before matching "
            f"(ids on result.discarded).{note}",
            CommonSupportWarning,
            stacklevel=2,
        )
        focal_value = 1 if estimand == "att" else 0
        n_focal -= int(
            (data.loc[result._results.discarded, treatment] == focal_value).sum()
        )
    _warn_on_dropped_units(result, n_focal, ratio)
    return result


def _run_strata_design(
    data: pd.DataFrame,
    *,
    method: str,
    treatment: str,
    covariates: list[str],
    estimand: str,
    settings: dict[str, Any],
    config_kwargs: dict[str, Any],
    injected_ps_col: bool = False,
    injected_cols: list[str] | None = None,
    work_data: pd.DataFrame | None = None,
) -> MatchResult:
    config = MatcherConfig(
        treatment_col=treatment,
        covariates=list(covariates),
        match_method=method,
        estimand=estimand,
        matching_direction="control" if estimand == "atc" else "treatment",
        caliper_method=None,
        caliper_value=None,
        **config_kwargs,
    )
    results = run_match(work_data if work_data is not None else data, config)
    if results.matched_data.empty:
        raise NoMatchesError(
            f"No stratum contained both groups (method='{method}'). "
            "Coarsen less aggressively or check group overlap."
        )
    result = MatchResult(
        results,
        estimand=estimand,
        settings=settings,
        injected_ps_col=injected_ps_col,
        injected_cols=injected_cols,
    )
    if results.discarded is not None and len(results.discarded):
        warnings.warn(
            f"{len(results.discarded)} unit(s) outside the common propensity "
            "support were discarded (ids on result.discarded).",
            CommonSupportWarning,
            stacklevel=3,
        )
    n_excluded = (
        len(results.original_data)
        - len(results.matched_data)
        - len(results.discarded if results.discarded is not None else [])
    )
    if n_excluded > 0:
        warnings.warn(
            f"{n_excluded} unit(s) fell in strata lacking one group and were "
            "excluded from the design.",
            IncompleteMatchWarning,
            stacklevel=3,
        )
    return result


def subclassify(
    data: pd.DataFrame,
    *,
    treatment: str,
    covariates: list[str],
    n_subclasses: int = 6,
    estimand: str = "att",
    propensity_scores: str | pd.Series | np.ndarray | None = None,
    propensity_model: Any | None = None,
    cv: int | None = None,
    discard: str | None = None,
    random_state: int | None = None,
) -> MatchResult:
    """Propensity-score subclassification.

    Units are stratified on the propensity score (quantiles of the estimand's
    target group); the design is expressed through stratum weights on the
    result (`result.strata`, `result.weights`) instead of pairs. Supports
    estimand="ate". Effects use HC-robust errors, a handful of strata are
    too few clusters for cluster-robust inference.

    Args:
        data: DataFrame with one row per unit; the index identifies units.
        treatment: Binary treatment column (1=treated, 0=control).
        covariates: Covariate columns to balance.
        n_subclasses: Number of propensity strata.
        estimand: "att", "atc", or "ate".
        propensity_scores: Precomputed scores (column name, Series, or
            array); mutually exclusive with `propensity_model`.
        propensity_model: sklearn classifier to estimate scores (cloned,
            cross-fitted). Defaults to logistic regression.
        cv: Cross-fitting folds (default 5); only when scores are estimated.
        discard: Common-support discard ("treated", "control", "both").
        random_state: Seed for propensity cross-fitting.

    Returns:
        MatchResult with strata, weights, and balance diagnostics.

    Example:
        >>> result = subclassify(
        ...     data, treatment="treat", covariates=["age", "educ"], estimand="ate"
        ... )  # doctest: +SKIP
    """
    estimand = _check_choice("estimand", estimand, _STRATUM_ESTIMANDS)
    if n_subclasses < 2:
        raise ValueError(f"n_subclasses must be >= 2, got {n_subclasses}")
    if propensity_scores is not None and propensity_model is not None:
        raise ValueError(
            "propensity_scores and propensity_model are mutually exclusive"
        )
    if cv is not None and propensity_scores is not None:
        raise ValueError("cv only applies when propensity scores are estimated")
    if discard is not None and discard not in ("treated", "control", "both"):
        raise ValueError(
            f"discard must be 'treated', 'control', or 'both', got {discard!r}"
        )

    work_data, covariates, injected_cols = _encode_covariates(data, covariates)
    (work_data, propensity_col, model_type, model_params, injected) = (
        _resolve_propensity_input(data, work_data, propensity_scores, propensity_model)
    )

    settings = {
        "treatment": treatment,
        "covariates": tuple(covariates),
        "method": "subclass",
        "estimand": estimand,
        "n_subclasses": n_subclasses,
        "discard": discard,
        "cv": 5 if cv is None else cv,
        "random_state": random_state,
    }
    return _run_strata_design(
        data,
        method="subclass",
        treatment=treatment,
        covariates=covariates,
        estimand=estimand,
        settings=settings,
        injected_ps_col=injected,
        injected_cols=injected_cols,
        work_data=work_data,
        config_kwargs=dict(
            estimate_propensity=propensity_col is None,
            propensity_col=propensity_col,
            propensity_model=model_type,
            model_params=model_params,
            cv_folds=5 if cv is None else cv,
            n_subclasses=n_subclasses,
            discard=discard,
            random_state=random_state,
        ),
    )


def cem(
    data: pd.DataFrame,
    *,
    treatment: str,
    covariates: list[str],
    coarsening: dict | None = None,
    exact: str | list[str] | None = None,
    estimand: str = "att",
) -> MatchResult:
    """Coarsened exact matching.

    Continuous covariates are cut into bins (Sturges' count by default, or
    per-variable via `coarsening`: an int bin count or explicit edges); the
    coarsened covariates plus `exact` columns define cells, and cells
    containing both groups form the strata. The design is expressed through
    stratum weights (`result.strata`, `result.weights`). Supports
    estimand="ate". Scales to biobank-size data (a groupby, no distances).

    Args:
        data: DataFrame with one row per unit; the index identifies units.
        treatment: Binary treatment column (1=treated, 0=control).
        covariates: Covariate columns to coarsen and match on.
        coarsening: Per-covariate bin count or explicit bin edges. Binary
            covariates always enter uncoarsened.
        exact: Column(s) entering the cells uncoarsened.
        estimand: "att", "atc", or "ate".

    Returns:
        MatchResult with strata, weights, and balance diagnostics.

    Example:
        >>> result = cem(
        ...     data, treatment="treat", covariates=["age", "educ"], coarsening={"age": 5}
        ... )  # doctest: +SKIP
    """
    estimand = _check_choice("estimand", estimand, _STRATUM_ESTIMANDS)
    if isinstance(exact, str):
        exact = [exact]
    for col, spec in (coarsening or {}).items():
        if col not in covariates:
            raise ValueError(
                f"coarsening key '{col}' is not a covariate; coarsening only "
                "applies to columns in `covariates`"
            )
        if not (
            isinstance(spec, int) and not isinstance(spec, bool)
        ) and not isinstance(spec, (list, tuple, np.ndarray)):
            raise ValueError(
                f"coarsening['{col}'] must be an int bin count or a list of bin "
                f"edges, got {type(spec).__name__}"
            )

    work_data, covariates, injected_cols = _encode_covariates(data, covariates)
    settings = {
        "treatment": treatment,
        "covariates": tuple(covariates),
        "method": "cem",
        "estimand": estimand,
        "coarsening": coarsening,
        "exact": tuple(exact) if exact else None,
    }
    return _run_strata_design(
        data,
        method="cem",
        treatment=treatment,
        covariates=covariates,
        estimand=estimand,
        settings=settings,
        injected_cols=injected_cols,
        work_data=work_data,
        config_kwargs=dict(
            exact_match_cols=exact or [],
            cem_coarsening=coarsening,
        ),
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _check_choice(
    name: str,
    value: str,
    valid: set[str],
    aliases: dict[str, str] | None = None,
    hints: dict[str, str] | None = None,
) -> str:
    if aliases and value in aliases:
        return aliases[value]
    if value in valid:
        return value
    message = f"{name} must be one of {sorted(valid)}, got {value!r}."
    if hints and value in hints:
        message += f" Hint: {hints[value]}."
    raise ValueError(message)


def _resolve_caliper(
    caliper: float | str | None,
    caliper_metric: str | None,
    std_caliper: bool,
) -> tuple[str | None, float | str | None, float]:
    """Translate the public caliper parameters into internal config fields.

    Standardized propensity calipers are computed and applied on the logit
    scale (internal metric "logit"), so the threshold and the distances it is
    compared against share units.
    """
    if caliper is None:
        if caliper_metric is not None:
            raise ValueError(
                "caliper_metric was given but caliper is None; set caliper as well"
            )
        return None, None, 0.2

    metric = caliper_metric or "propensity"
    valid_metrics = _PS_METRICS | {"mahalanobis", "euclidean"}
    if metric not in valid_metrics:
        raise ValueError(
            f"caliper_metric must be one of {sorted(valid_metrics)}, got {metric!r}. "
            "Per-covariate calipers are not supported yet."
        )

    if isinstance(caliper, str):
        if caliper != "auto":
            raise ValueError(
                f"caliper must be a number, 'auto', or None, got {caliper!r}"
            )
        if metric not in _PS_METRICS:
            raise ValueError(
                f"caliper='auto' is only defined for propensity calipers (the 0.2 SD "
                f"rule); pass a numeric caliper for caliper_metric='{metric}'."
            )
        return "logit", "auto", 0.2

    caliper = float(caliper)
    if caliper <= 0:
        raise ValueError(f"caliper must be positive, got {caliper}")

    if metric in _PS_METRICS:
        if std_caliper:
            # interpreted as multiples of SD(logit(ps)), applied on the logit scale
            return "logit", "auto", caliper
        return metric, caliper, 0.2
    return metric, caliper, 0.2


def _encode_covariates(
    data: pd.DataFrame, covariates: list[str]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """One-hot encode non-numeric covariates ("var=level" columns).

    Returns (possibly-copied data, expanded covariate list, added columns).
    """
    if isinstance(data.columns, pd.MultiIndex):
        raise TypeError(
            "MultiIndex columns are not supported; flatten them before "
            "matching, e.g. "
            "data.columns = ['_'.join(map(str, c)) for c in data.columns]."
        )
    non_str = [c for c in covariates if not isinstance(c, str)]
    if non_str:
        raise TypeError(
            f"covariate names must be strings, got {non_str}; the balance "
            "tables and one-hot encoding label columns by name. Rename them "
            "(data.columns = data.columns.map(str)) before matching."
        )
    missing = [c for c in covariates if c not in data.columns]
    if missing:
        raise ValueError(f"covariates not found in data: {missing}")
    cat_cols = [c for c in covariates if not pd.api.types.is_numeric_dtype(data[c])]
    if not cat_cols:
        return data, list(covariates), []
    for col in cat_cols:
        if data[col].isna().any():
            raise ValueError(
                f"categorical covariate '{col}' contains missing values; "
                "handle them before matching"
            )
    dummies = pd.get_dummies(
        data[cat_cols].astype("category"), prefix=cat_cols, prefix_sep="="
    ).astype(float)
    collisions = [c for c in dummies.columns if c in data.columns]
    if collisions:
        raise ValueError(
            f"columns {collisions} already exist in the data and collide with "
            "the encoded categorical levels; rename them"
        )
    work = data.copy()
    for col in dummies.columns:
        work[col] = dummies[col]
    expanded = [c for c in covariates if c not in cat_cols] + list(dummies.columns)
    return work, expanded, list(dummies.columns)


def _resolve_propensity_input(
    data: pd.DataFrame,
    work_data: pd.DataFrame,
    propensity_scores,
    propensity_model,
) -> tuple:
    """Translate the propensity arguments into internal config inputs.

    Shared by match() and subclassify(). Returns
    (work_data, propensity_col, model_type, model_params, injected): the
    (possibly copied) working frame, the score column name if any, the
    propensity model type and params, and whether a score column was injected.
    """
    propensity_col = None
    injected = False
    if propensity_scores is not None:
        if isinstance(propensity_scores, str):
            if propensity_scores not in data.columns:
                raise ValueError(
                    f"propensity_scores column '{propensity_scores}' not in data"
                )
            propensity_col = propensity_scores
        else:
            if _PS_COL in data.columns:
                raise ValueError(
                    f"column '{_PS_COL}' is reserved for internal use; "
                    "rename it before matching"
                )
            values = _align_scores(propensity_scores, data)
            if work_data is data:
                work_data = data.copy()
            work_data[_PS_COL] = values
            propensity_col = _PS_COL
            injected = True

    model_type, model_params = "logistic", {}
    if propensity_model is not None:
        from sklearn.base import clone

        model_type = "custom"
        model_params = {"model": clone(propensity_model)}

    return work_data, propensity_col, model_type, model_params, injected


def _align_scores(scores: pd.Series | np.ndarray, data: pd.DataFrame) -> np.ndarray:
    if isinstance(scores, pd.Series):
        aligned = scores.reindex(data.index)
        if aligned.isna().any():
            raise ValueError(
                "propensity_scores are missing for some rows (incomplete index "
                "coverage or NaN values)"
            )
        return aligned.to_numpy(dtype=float)
    values = np.asarray(scores, dtype=float).ravel()
    if len(values) != len(data):
        raise ValueError(
            f"propensity_scores has length {len(values)}, expected {len(data)}"
        )
    if not np.isfinite(values).all():
        raise ValueError("propensity_scores contain missing or non-finite values")
    return values


def _resolve_engine(
    engine: str,
    method: str,
    caliper_method: str | None,
    n_focal: int,
    n_pool: int,
    distance: str,
    memory_limit_gb: float,
) -> str:
    # A second full matrix is materialized only for a covariate-space caliper
    # on a different distance metric; propensity/logit calipers are applied
    # in place (see pipeline._perform_matching), so they cost no extra matrix.
    covariate_space_caliper = caliper_method in ("mahalanobis", "euclidean")
    n_matrices = 2 if (covariate_space_caliper and caliper_method != distance) else 1
    dense_gb = n_focal * n_pool * 8 * n_matrices / 1e9
    matrix_note = (
        " (distance plus a covariate-space caliper matrix)" if n_matrices == 2 else ""
    )
    # covariate distances scale via a spatial tree (no propensity score, no
    # caliper required); propensity/logit distances need the propensity window
    covariate_distance = distance in ("mahalanobis", "euclidean")

    if engine == "approximate":
        if method != "nearest":
            raise ValueError("engine='approximate' requires method='nearest'")
        if caliper_method is None and not covariate_distance:
            raise ValueError(
                "engine='approximate' requires a caliper for candidate selection; "
                "set caliper='auto' (recommended) or a numeric caliper."
            )
        return "approximate"

    if engine == "exact":
        return "exact"

    # engine == "auto"
    if dense_gb <= memory_limit_gb:
        return "exact"
    if method == "optimal":
        raise ValueError(
            f"Optimal matching on {n_focal:,} x {n_pool:,} units needs ~{dense_gb:.1f} GB "
            f"for the distance matrix (limit {memory_limit_gb} GB). Use method='nearest' "
            "with engine='approximate', or raise memory_limit_gb."
        )
    if caliper_method is None and not covariate_distance:
        raise ValueError(
            f"Matching {n_focal:,} x {n_pool:,} units needs ~{dense_gb:.1f} GB for the "
            f"dense distance matrix{matrix_note} (limit {memory_limit_gb} GB). Set "
            "caliper='auto' to enable the memory-efficient approximate algorithm, or "
            "raise memory_limit_gb to force the dense computation."
        )
    if covariate_distance:
        # the whitened KD-tree returns the same nearest neighbors the dense
        # path would; no propensity score, no matrix, no approximation
        warnings.warn(
            f"Distance matrix would need ~{dense_gb:.1f} GB{matrix_note}; matching "
            "in covariate space with a spatial tree instead (exact nearest "
            "neighbors, no distance matrix, no propensity score). Pass "
            "engine='approximate' to make this explicit and silence this warning.",
            ApproximateMatchWarning,
            stacklevel=3,
        )
    else:
        warnings.warn(
            f"Distance matrix would need ~{dense_gb:.1f} GB{matrix_note}; using the "
            "memory-efficient approximate algorithm (propensity-score candidate "
            "prefilter). Results may differ slightly from exact nearest-neighbor "
            "matching. Pass engine='approximate' to make this explicit and silence "
            "this warning.",
            ApproximateMatchWarning,
            stacklevel=3,
        )
    return "approximate"


def _warn_on_dropped_units(result: MatchResult, n_focal: int, ratio: int) -> None:
    estimand = result.estimand
    treatment_col = result._results.config.treatment_col
    matched = result._results.matched_data
    focal_value = 1 if estimand == "att" else 0
    n_focal_matched = int((matched[treatment_col] == focal_value).sum())

    if n_focal_matched < n_focal:
        dropped = n_focal - n_focal_matched
        group = "treated" if estimand == "att" else "control"
        warnings.warn(
            f"{dropped} of {n_focal} {group} units could not be matched and were "
            f"dropped. The estimand is the effect on the *matched* {group} units, "
            f"which may differ from the {estimand.upper()}. To retain more units, "
            "widen or drop the caliper, loosen exact/covariate constraints, or "
            "check overlap with result.plot_propensity().",
            IncompleteMatchWarning,
            stacklevel=3,
        )

    n_pairs = len(result._results.pairs)
    if ratio > 1 and n_focal_matched > 0 and n_pairs / n_focal_matched < ratio - 0.2:
        warnings.warn(
            f"Requested {ratio} matches per unit but achieved "
            f"{n_pairs / n_focal_matched:.2f} on average; the pool or caliper "
            "does not support the requested ratio.",
            IncompleteMatchWarning,
            stacklevel=3,
        )


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


class MatchResult:
    """Result of a `match()` call.

    Attributes are plain pandas objects; unit identity is the DataFrame index
    of the input data throughout.
    """

    def __init__(
        self,
        results: MatchResults,
        *,
        estimand: str,
        settings: dict[str, Any],
        injected_ps_col: bool = False,
        injected_cols: list[str] | None = None,
    ):
        self._results = results
        self._estimand = estimand
        self._settings = settings
        self._injected_ps_col = injected_ps_col
        self._injected_cols = list(injected_cols or [])

    # -- data ---------------------------------------------------------------

    def _strip_injected(self, df: pd.DataFrame) -> pd.DataFrame:
        drop = [c for c in self._injected_cols if c in df.columns]
        if self._injected_ps_col and _PS_COL in df.columns:
            drop.append(_PS_COL)
        return df.drop(columns=drop) if drop else df

    @property
    def matched_data(self) -> pd.DataFrame:
        """Matched units (all columns of the input data)."""
        return self._strip_injected(self._results.matched_data)

    @property
    def original_data(self) -> pd.DataFrame:
        """The input data before matching."""
        return self._strip_injected(self._results.original_data)

    @property
    def pairs(self) -> pd.DataFrame:
        """Matched pairs: treatment_id, control_id, distance, match_group.

        `match_group` groups the rows belonging to one anchor unit (relevant
        for ratio > 1). `distance` is NaN only on the propensity approximate
        path, which never forms a dense distance matrix; the covariate
        approximate (KD-tree) path reports its exact distances.
        """
        rows = []
        distance_lookup = self._distance_lookup()
        control_anchored = self._results.anchor == "control"
        # The covariate approximate path has no dense matrix but does compute
        # exact distances, aligned index-for-index with the pairs list.
        md = self._results.match_distances
        use_md = (
            self._results.distance_matrix is None
            and md is not None
            and len(md) == len(self._results.pairs)
            and self._results.config.distance_method in ("mahalanobis", "euclidean")
        )
        for i, (t_id, c_id) in enumerate(self._results.pairs):
            rows.append(
                {
                    "treatment_id": t_id,
                    "control_id": c_id,
                    "distance": md[i] if use_md else distance_lookup(t_id, c_id),
                    "match_group": c_id if control_anchored else t_id,
                }
            )
        return pd.DataFrame(
            rows, columns=["treatment_id", "control_id", "distance", "match_group"]
        )

    def _distance_lookup(self):
        matrix = self._results.distance_matrix
        anchor_ids = self._results.dm_anchor_ids
        pool_ids = self._results.dm_pool_ids
        if matrix is None or anchor_ids is None:
            return lambda t_id, c_id: np.nan
        # matrix rows index the anchor group, columns the pool, in the order
        # they entered matching (post-discard), never the full original frame
        anchor_pos = {idx: i for i, idx in enumerate(anchor_ids)}
        pool_pos = {idx: i for i, idx in enumerate(pool_ids)}
        control_anchored = self._results.anchor == "control"

        def lookup(t_id, c_id):
            a_id, p_id = (c_id, t_id) if control_anchored else (t_id, c_id)
            try:
                return float(matrix[anchor_pos[a_id], pool_pos[p_id]])
            except KeyError:
                return np.nan

        return lookup

    # -- diagnostics --------------------------------------------------------

    def balance(self, covariates: list[str] | None = None) -> pd.DataFrame:
        """Balance per covariate: signed SMDs and variance ratios, before/after.

        SMDs are (treated - control) / SD of the anchor group in the original
        sample, with the same denominator before and after matching (cobalt
        convention). Post-matching statistics use the matching weights.

        Args:
            covariates: Assess these covariates instead of the matched ones
                (e.g. to check balance on variables not matched on). Defaults
                to the matching covariates.

        Example:
            >>> result.balance()
            >>> result.balance(covariates=["age", "region"])
        """
        if covariates is None:
            return self._results.balance_statistics
        from cohortmatch.metrics.balance import calculate_balance_stats

        return calculate_balance_stats(
            data=self._results.original_data,
            matched_data=self._results.matched_data,
            covariates=list(covariates),
            treatment_col=self._results.config.treatment_col,
            weights=self._results.weights,
            anchor=self._results.anchor,
        )

    @property
    def propensity_scores(self) -> pd.Series | None:
        """Propensity scores aligned to the input data index, or None."""
        scores = self._results.propensity_scores
        if scores is None:
            return None
        return pd.Series(
            np.asarray(scores),
            index=self._results.original_data.index,
            name="propensity_score",
        )

    @property
    def propensity_model(self) -> Any | None:
        """The fitted propensity model, or None."""
        return self._results.propensity_model

    @property
    def discarded(self) -> pd.Index:
        """Units dropped by common-support discard (empty when discard off)."""
        d = self._results.discarded
        return d if d is not None else self._results.original_data.index[:0]

    @property
    def weights(self) -> pd.Series:
        """Matching weights, indexed by unit id.

        Anchor units have weight 1; partners accumulate 1/k per match group
        they serve in, rescaled to average 1 (MatchIt convention). Use these
        for any analysis of the matched sample, with `replace=True` or
        ratio matching, unweighted analysis of `matched_data` is biased.
        """
        return self._results.weights

    @property
    def match_groups(self) -> pd.Series | None:
        """Match-group membership (anchor id) per unit, for pair designs.

        None with replacement (membership is not unique) and for stratum
        designs (see `strata`).
        """
        if self._settings.get("method") in ("subclass", "cem"):
            return None
        return self._results.subclass

    @property
    def strata(self) -> pd.Series | None:
        """Stratum label per unit, for subclassify()/cem() results."""
        if self._settings.get("method") in ("subclass", "cem"):
            return self._results.subclass
        return None

    def table1(self, covariates: list[str] | None = None) -> pd.DataFrame:
        """Cohort characteristics table: group means/SDs and SMDs, before and after.

        Post-matching statistics use the matching weights. Group sizes are in
        `DataFrame.attrs` (n_treated_before, n_control_before, ...).

        Args:
            covariates: Rows to include; defaults to the matching covariates.

        Example:
            >>> result.table1()
        """
        from cohortmatch.metrics.balance import create_table1

        return create_table1(
            data=self._results.original_data,
            matched_data=self._results.matched_data,
            covariates=list(covariates or self._results.config.covariates),
            treatment_col=self._results.config.treatment_col,
            weights=self._results.weights,
            anchor=self._results.anchor,
        )

    @property
    def propensity_metrics(self) -> dict | None:
        """Diagnostics of the propensity model (AUC, overlap), or None."""
        return self._results.propensity_metrics

    @property
    def rubin_statistics(self) -> dict | None:
        """Rubin's B and R on the propensity-score linear predictor, plus a
        per-covariate balance-threshold summary.

        `rubin_B` is the standardized difference in mean linear propensity
        between groups (target |B| < 25); `rubin_R` is the ratio of their
        variances (target 0.5 < R < 2). Both are absent when no propensity
        score is available (covariate-distance matching). The `pct_*`/`n_*`
        fields count covariates within the |SMD| < 0.25 and variance-ratio
        thresholds -- a separate diagnostic, not Rubin's B/R.
        """
        base = dict(self._results.rubin_statistics or {})
        ps = self.propensity_scores
        md = self.matched_data
        if ps is not None and not md.empty:
            from cohortmatch.metrics.balance import rubin_b_r

            treated = (md[self._results.config.treatment_col] == 1).to_numpy()
            ps_matched = ps.reindex(md.index).to_numpy()
            w = self.weights
            w_matched = w.reindex(md.index).to_numpy() if w is not None else None
            base.update(rubin_b_r(ps_matched, treated, w_matched))
        return base or None

    @property
    def estimand(self) -> str:
        """The estimand the design targets ('att', 'atc', or 'ate' for strata)."""
        return self._estimand

    @property
    def config(self) -> MappingProxyType:
        """The resolved settings of the match() call (read-only)."""
        return MappingProxyType(self._settings)

    def summary(self) -> MatchSummary:
        """Summary of sample sizes and balance."""
        return MatchSummary(self)

    # -- verbs --------------------------------------------------------------

    def estimate_effects(
        self,
        outcomes: str | list[str],
        *,
        method: str = "mean_difference",
        family: str = "linear",
        adjustment_covariates: list[str] | None = None,
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        """Estimate treatment effects on the matched sample.

        Estimation is a weighted outcome model with the matching weights.
        Standard errors are cluster-robust on match groups (matching without
        replacement) or heteroskedasticity-robust otherwise (HC3 for the
        weighted linear model, HC0 for the weighted GLM); the `se_type` column
        records which was used.

        Args:
            outcomes: Outcome column(s) in the matched data.
            method: "mean_difference" (treatment only) or
                "regression_adjustment" (adds covariates to the model). The
                latter reports the treatment coefficient, which equals the
                target estimand only if the effect does not vary with the
                covariates; prefer "mean_difference" when unsure.
            family: "linear" (mean difference; risk difference for binary
                outcomes), "logistic" (marginal odds ratio), or "poisson"
                (marginal risk ratio). Nonlinear families fit the
                treatment-only model; regression adjustment is linear-only.
                The `measure` column records what the effect is.
            adjustment_covariates: Covariates for regression adjustment.
            confidence_level: Confidence level for the intervals.

        Returns:
            DataFrame with one row per outcome (effect, measure, CI,
            p-value, se_type). For time-to-event outcomes see the Cox
            recipe in the README (Effects on the matched sample).

        Example:
            >>> result.estimate_effects("re78")  # doctest: +SKIP
            >>> result.estimate_effects("event", family="logistic")  # odds ratio  # doctest: +SKIP
        """
        if isinstance(outcomes, str):
            outcomes = [outcomes]
        matched = self.matched_data
        missing = [o for o in outcomes if o not in matched.columns]
        if missing:
            raise ValueError(f"outcome columns not in matched data: {missing}")

        # cluster on match groups for pair designs; stratum designs have far
        # too few strata for valid cluster inference, so they use HC-robust
        is_strata = self._settings.get("method") in ("subclass", "cem")
        effects = estimate_multiple_outcomes(
            data=matched,
            outcomes=outcomes,
            treatment_col=self._results.config.treatment_col,
            method=method,
            covariates=adjustment_covariates,
            estimand=self._estimand,
            family=family,
            weights=self._results.weights,
            subclass=None if is_strata else self._results.subclass,
            confidence_level=confidence_level,
        )
        # keep the latest estimates available to supplement()
        self._results.effect_estimates = effects
        return effects

    # -- plots (require the viz extras) -------------------------------------

    def plot_love_plot(self, **kwargs: Any):
        """Love plot of standardized mean differences before/after matching."""
        from cohortmatch.visualization import plot_love_plot

        return plot_love_plot(self._results, **kwargs)

    def plot_balance(self, **kwargs: Any):
        """Bar plot of covariate balance before and after matching."""
        from cohortmatch.visualization import plot_balance

        return plot_balance(self._results, **kwargs)

    def plot_propensity(self, **kwargs: Any):
        """Propensity score distributions before and after matching."""
        from cohortmatch.visualization import plot_propensity_comparison

        return plot_propensity_comparison(self._results, **kwargs)

    def plot_match_distances(self, **kwargs: Any):
        """Histogram of matched-pair distances."""
        from cohortmatch.visualization import plot_matched_pairs_distance

        return plot_matched_pairs_distance(self._results, **kwargs)

    def supplement(self, path: str | None = None, *, title: str | None = None) -> str:
        """Markdown supplement for a paper: design, versions, flow, balance.

        A self-contained record of what was done, the resolved design
        specification (including the numeric caliper actually applied),
        software versions, sample flow, the balance table, effect estimates
        when already computed, and a citable methods paragraph.

        Args:
            path: Optional file to write the Markdown to.
            title: Heading for the document.

        Returns:
            The Markdown text.

        Example:
            >>> result.supplement("supplement_S1.md", title="Study S1")  # doctest: +SKIP
        """
        from cohortmatch.supplement import build_supplement

        text = build_supplement(self, title=title)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text

    def __repr__(self) -> str:
        s = self._settings
        treatment_col = self._results.config.treatment_col
        matched = self._results.matched_data
        n_t = (matched[treatment_col] == 1).sum()
        n_c = (matched[treatment_col] == 0).sum()
        detail = f"distance='{s['distance']}', " if "distance" in s else ""
        return (
            f"MatchResult(estimand='{self._estimand}', method='{s['method']}', "
            f"{detail}n_treated={n_t}, n_control={n_c})"
        )

    def _repr_html_(self) -> str:
        # notebooks: show the informative summary, not the one-line repr
        import html

        return (
            f"<pre style='font-family:monospace;font-size:12px'>"
            f"{html.escape(repr(self.summary()))}</pre>"
            "<div style='font-size:11px;color:#666'>"
            "methods: balance(), table1(), estimate_effects(), supplement(), "
            "plot_love_plot()</div>"
        )


class MatchSummary:
    """Printable summary of a MatchResult."""

    def __init__(self, result: MatchResult):
        self._result = result
        legacy = result._results
        self.counts = legacy.get_match_summary()
        self.balance_index = legacy.balance_index or {}
        balance = legacy.balance_statistics
        if balance is not None and "smd_after" in balance.columns:
            self.max_smd_before = float(balance["smd_before"].abs().max())
            self.max_smd_after = float(balance["smd_after"].abs().max())
            self.n_imbalanced = int((balance["smd_after"].abs() > 0.1).sum())
            self.n_covariates = len(balance)
        else:
            self.max_smd_before = self.max_smd_after = float("nan")
            self.n_imbalanced = self.n_covariates = 0

    def __repr__(self) -> str:
        r = self._result
        s = r.config
        c = self.counts
        focal = "treated" if r.estimand == "att" else "control"
        is_strata = s["method"] in ("subclass", "cem")
        header = (
            f"MatchResult ({r.estimand.upper()}, method={s['method']})"
            if is_strata
            else f"MatchResult ({r.estimand.upper()}, method={s['method']}, "
            f"distance={s['distance']}, engine={s['engine']})"
        )
        lines = [
            header,
            f"  Treated:  {c['n_treatment_matched']} matched of {c['n_treatment_orig']}",
            f"  Controls: {c['n_control_matched']} matched of {c['n_control_orig']}",
        ]
        if is_strata:
            sub = r._results.subclass
            n_strata = sub.nunique() if sub is not None else 0
            lines.append(f"  Strata: {n_strata} (weights on result.weights)")
        else:
            lines.append(
                f"  Pairs: {c['n_pairs']} (anchor group: {focal}, 1:{s['ratio']}"
                + (", with replacement)" if s["replace"] else ")")
            )
        if self.n_covariates:
            mean_before = self.balance_index.get("mean_smd_before", float("nan"))
            mean_after = self.balance_index.get("mean_smd_after", float("nan"))
            lines += [
                f"  Balance ({self.n_covariates} covariates): "
                f"mean |SMD| {mean_before:.3f} -> {mean_after:.3f}, "
                f"max |SMD| {self.max_smd_before:.3f} -> {self.max_smd_after:.3f}",
                f"  Covariates with |SMD| > 0.1 after matching: "
                f"{self.n_imbalanced} of {self.n_covariates}",
            ]
        rubin = r.rubin_statistics
        if rubin and "rubin_B" in rubin and np.isfinite(rubin["rubin_B"]):
            lines.append(
                f"  Rubin's B={rubin['rubin_B']:.0f} (target <25), "
                f"R={rubin['rubin_R']:.2f} (target 0.5-2)"
            )
        if rubin and rubin.get("n_variables_total"):
            lines.append(
                f"  Covariates within thresholds: "
                f"{rubin.get('pct_both_good', float('nan')):.0f}% "
                f"(|SMD| < 0.25 and variance ratio in [0.5, 2])"
            )
        return "\n".join(lines)
