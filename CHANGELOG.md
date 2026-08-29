# Changelog

## [0.1.0] - 2026-08-29

First release of CohortMatch, the successor to cohortbalancer3. The matching
internals carry over; the public API is new.

### Fixed

-   A user caliper wider than the approximate path's prefilter window is no
    longer silently truncated: propensity/logit calipers now define the
    candidate window exactly. Covariate-space calipers still use the
    prefilter window and warn about the approximation.
-   The dense and approximate paths clip propensity scores identically
    before logit transforms (1e-6; the dense path previously clipped at
    0.001), so caliper decisions at the tails agree between algorithms.

### Added

-   `m_order` on `match()`: the order in which anchor units pick matches
    ("largest"/"smallest" by propensity, "closest", "random", "data").
-   Scale: the approximate path finds candidate pools by binary search over
    propensity-sorted controls and computes propensity distances inline:
    20k x 480k matches in ~2 s in ~0.4 GB (was 43 s). Hardest-first
    matching order is its default and recovers the same match count as
    dense matching. `benchmarks/` has the harness and results.
-   Binary covariates are standardized by sqrt(p(1-p)) in SMDs, matching
    cobalt exactly; continuous covariates use the ddof=1 sample SD.
-   At-scale validation: on a 20k x 480k cohort with shared propensity
    scores and caliper, cohortmatch and MatchIt produce identical matched
    counts, balance, match tightness, and effect estimates
    (tests/test_matchit_scale.py; numbers in BENCHMARKS.md).
-   MatchIt validation harness: `validation/generate_golden.R` produces
    golden values (MatchIt + cobalt on Lalonde). `tests/test_matchit_validation.py`
    reconciles against R, across nearest/optimal/caliper/ratio/exact propensity
    designs, subclassification, Mahalanobis matching, coarsened exact
    matching, and the weighted logistic (odds-ratio) estimator: counts,
    balance, weights, optimal distance, ATT, and OR. Runs in CI with an R
    toolchain.
-   `cohortmatch.datasets.load_lalonde()`; the dataset ships with the
    package.
-   Matching weights (`result.weights`, MatchIt convention) and match-group
    membership (`result.match_groups`). With `replace=True`, `matched_data` now
    contains each unit once under its original id; reuse is expressed through
    the weights instead of duplicated `_dup` rows.
-   Effect estimation by weighted outcome models with the matching weights:
    cluster-robust standard errors on match groups, else heteroskedasticity-
    robust (HC3 for the linear model, HC0 for the GLM); the `se_type` column
    records which, and a warning fires when too few match groups remain for
    reliable cluster-robust inference.
-   `result.table1()`: group means/SDs with SMDs, before and after matching.
-   Balance follows cobalt conventions: signed SMDs standardized by the
    anchor group's SD in the original sample (same denominator before and
    after), weighted post-matching statistics, unfolded variance ratios
    (treated variance / control variance).
-   Plot methods on the result: `plot_love_plot()`, `plot_balance()`,
    `plot_propensity()`, `plot_match_distances()`.
-   `result.rubin_statistics` reports Rubin's B and R on the propensity linear
    predictor, alongside a per-covariate balance-threshold summary; both appear
    in `summary()`.
-   `match()` as the single entry point, returning a `MatchResult` with
    `matched_data`, `pairs`, `balance`, `propensity_scores`, `summary()`,
    `estimate_effects()`, and `supplement()`.
-   `estimand="att"|"atc"` on `match()` selects which group anchors the
    matching. Replaces the previous behavior of silently matching from
    whichever group was smaller.
-   Propensity scores are fit on the full sample by default, so an unseeded run
    is deterministic; pass `cv=k` to cross-fit over k folds, with feature
    scaling fit inside each fold. `propensity_model` accepts any sklearn
    classifier (cloned); `propensity_scores` accepts a column name, Series, or
    array.
-   `method="optimal"` returns the minimum-total-distance matching, including
    true 1:k ratio matching, via sparse min-cost bipartite assignment; exact
    and caliper constraints are honored as absent edges.
-   `covariate_weights` are validated as non-negative and finite.
-   `engine="auto"` selects between the dense distance matrix and a
    memory-efficient prefiltered path based on `memory_limit_gb`, so large
    cohorts never hit an out-of-memory crash.
-   `NoMatchesError` raised when no pairs satisfy the constraints, instead of
    returning empty results.
-   `UserWarning` when anchor units are dropped (caliper/exact constraints) or
    a requested ratio cannot be met.
-   Analytic t-based confidence intervals when the effect bootstrap is
    disabled (`n_resamples=0`); Welch standard errors for mean differences.

### Changed

-   No caliper is applied by default (was: 0.2 SD auto-caliper). `caliper="auto"`
    keeps the 0.2 SD rule as an explicit choice.
-   Standardized propensity calipers are computed and applied on the logit
    scale consistently. Previously a logit-scale threshold was compared against
    probability-scale distances.
-   Propensity scores are no longer isotonic-calibrated by default.
-   `ratio` must be an integer; fractional values raised an error instead of
    being silently truncated.
-   `replace=True` with optimal matching is rejected (the solver never reused
    controls anyway).
-   Importing the package no longer requires matplotlib; reporting imports are
    lazy.
-   The package is silent by default (`logging.NullHandler`); call
    `configure_logging()` for progress output.

### Removed

-   The `Matcher` class; the internals are a functional pipeline
    (`cohortmatch.pipeline.run_match`). `MatcherConfig`/`MatchResults` remain
    as internal carriers only.
-   The row-level bootstrap for effect confidence intervals (statistically
    invalid for matched samples); intervals are analytic t-based until
    cluster-robust inference lands.
-   The propensity model string registry ("random_forest", "xgboost",
    "logisticcv") and the optional xgboost dependency; pass any sklearn
    classifier as `propensity_model` instead.
-   Isotonic calibration machinery; pass `CalibratedClassifierCV` as
    `propensity_model` if you want calibrated scores.
-   The chi-squared "auto" caliper rule for Mahalanobis/Euclidean distances
    from the public surface.
-   Common-support trimming from the public surface (its threshold parameter
    was ignored by the implementation; it will return as an explicit
    preprocessing step).
