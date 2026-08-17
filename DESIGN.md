# Design notes

Reference for the statistical conventions and the non-obvious implementation
decisions. Each convention names its literature anchor and the test that pins
it. Changing a convention should break a golden test.

## Module map

```
api.py            public surface: match(), subclassify(), cem(); MatchResult
risk_set.py       match_risk_set() + RiskSetResult (self-contained)
pipeline.py       run_match(): propensity, discard, direction, match,
                  weights/groups, balance
matching/
  greedy.py       dense nearest-neighbor (argmin loop; default order is
                  fewest-potential-matches-first, plus largest/smallest/
                  closest/random/data)
  fast_greedy.py  memory-bounded nearest via a propensity-sorted window
  covariate_nn.py memory-bounded nearest via a whitened KD-tree (exact
                  Mahalanobis/Euclidean at scale, no propensity score)
  optimal.py      Hungarian assignment (scipy), iterative for ratio>1
  strata.py       subclass/CEM labels and stratum weights
  distances.py    distance matrices, standardization, Mahalanobis inverse
  _utils.py       exact-match masking shared by greedy/optimal
metrics/
  propensity.py   cross-fitted score estimation (sklearn), overlap metrics
  balance.py      SMDs, variance ratios, table1, Rubin's rules
  treatment.py    WLS and GLM effect estimators with robust covariances
  utils.py        caliper resolution
visualization.py  the four paper plots (matplotlib)
supplement.py     Markdown records for pair/strata (build_supplement) and
                  nested case-control (build_risk_set_supplement); no deps
evalue.py         E-value sensitivity (VanderWeele & Ding); no deps
exceptions.py     typed warnings and errors the API raises
datasets.py       load_lalonde() + data/lalonde.csv
datatypes.py      internal MatcherConfig/MatchResults carriers
validation.py     input checks (not the validation/ directory below)
utils/logging.py  configure_logging()
validation/       R golden generators (MatchIt/cobalt) + pixi env
benchmarks/       competitor measurers + live speed/memory report
```

`match()` validates and translates the public arguments into an internal
`MatcherConfig`, calls `pipeline.run_match()`, and wraps the internal
`MatchResults` in the public `MatchResult`. The internal carriers in
`datatypes.py` are not exported.

## Statistical conventions and their anchors

**Estimand and matching direction.** `estimand="att"` anchors on treated
units, `"atc"` on controls. There is no size-based auto-flip; the v3 behavior
was removed and `matching_direction` is always explicit. ATE is available only
for the stratum designs. When anchor units cannot be matched the population
changes, so the API raises `IncompleteMatchWarning` with counts.

**Calipers.** `caliper="auto"` is 0.2 standard deviations of the logit
propensity score (Austin 2011, Pharm Stat 10:150). Numeric propensity calipers
are standardized the same way unless `std_caliper=False`. Standardized PS
calipers are computed and applied on the logit scale (`caliper_method="logit"`)
so the threshold and the distances share units; v3 compared logit-scale
thresholds against probability-scale distances. Mahalanobis and Euclidean
calipers are in raw distance units and have no "auto" (the v3 chi-squared rule
is not a field convention and was removed). MatchIt standardizes calipers by
the SD of the raw distance measure rather than the logit, so the golden caliper
design passes a raw threshold to both implementations. Per-variable
`covariate_calipers` are in raw units (MatchIt's are SD units), because "age
within 3 years" is what analysts specify.

**Propensity scores.** Logistic regression, L2-regularized (sklearn default),
5-fold cross-fitted, so each unit is scored by a model that did not see it.
Folds reduce to the minority-class count when treatment is rare
(propensity.py, `n_splits = max(2, min(cv, min_class_count))`). The scaler is
fit on the full sample before cross-fitting; the leakage is negligible because
the model itself is out-of-fold. No calibration is applied. The returned model
is a Pipeline carrying its scaler, so it accepts raw covariates. Logit
transforms clip at 1e-6 everywhere; the dense and approximate paths must clip
identically or tail units get different caliper decisions (see
tests/test_api.py::TestPrefilterCorrectness).

**Matching weights** (pipeline.`_compute_weights_and_subclass`): anchors get 1;
each partner accumulates 1/k per match group it serves in, and partner weights
then rescale to a global mean of 1 across the non-anchor units. This is
MatchIt's convention, and point estimates are invariant to the rescaling.
`replace=True` does not duplicate rows; reuse is expressed in the weights, and
`match_groups` is defined only without replacement.

**Stratum weights** (strata.py): ATT uses w_t=1, w_c=n_ts/n_cs; ATC is
symmetric; ATE uses w=n_s/n_gs. These are the marginal-mean subclass weights,
verified against ATE = sum_s (n_s/N)(ybar_ts - ybar_cs). Subclass edges are
quantiles of the estimand's target group (MatchIt).

**Balance** (balance.py): SMDs are signed and standardized by the anchor
group's SD in the original sample, using the same denominator before and after
(Stuart 2010; cobalt's s.d.denom="treated"), with a pooled denominator for ATE.
Binary variables are standardized by sqrt(p(1-p)), cobalt's convention, which
is why unadjusted SMDs reconcile with cobalt to 1e-6 in the golden suite.
Variance ratios are raw treated/control, not folded. Weighted post-matching
statistics use the Bessel-corrected reliability-weight variance, which reduces
to the ddof=1 sample variance at unit weights.

**Effects** (treatment.py): weighted outcome models on the matched sample. The
linear family is WLS (mean or risk difference); logistic and Poisson are GLMs
with `var_weights`, because matching weights are sampling weights. `freq_weights`
would treat a reused control as k independent observations and understate the
robust SE (about 15% on the sandwich test's data). The var_weights sandwich is
pinned to a hand computation in
tests/test_api.py::test_glm_var_weights_sandwich. statsmodels emits a
SpecificationWarning for var_weights with a robust covariance; it is verified
safe here and suppressed. statsmodels' GLM `cov_type="HC1"` returns HC0 (no
n/(n-k) factor), so GLM robust SEs are HC0-equivalent, about 0.16% smaller than
R's vcovHC(HC1) at n=614; the point estimate matches R exactly (golden
test_logistic_effect_reconciles). Standard errors are cluster-robust on match
groups without replacement (Abadie & Spiess 2022 justifies pair clustering) and
HC1 with replacement and for stratum designs, where a handful of strata are too
few clusters. The row bootstrap is invalid for matched samples (Abadie & Imbens
2008) and was removed. Nonlinear families fit the treatment-only model, so
every reported ratio is marginal and the non-collapsibility trap cannot occur.
Binary outcomes under the linear family are labeled risk_difference. The E-value
(VanderWeele & Ding 2017) is the only in-package sensitivity tool: a closed-form
formula on a ratio estimate, with no model, data, or dependency. Cox models,
Rosenbaum bounds, and doubly-robust estimators are README recipes.

**Risk-set matching** (risk_set.py): the risk set at case time t is the set of
units with event_time strictly greater than t; future cases are eligible
controls; controls may serve several sets (Langholz & Goldstein 1996, which is
what lets the conditional-logistic OR estimate the hazard ratio). The valid
design restricts eligibility (exact, calipers) and samples at random.
Nearest-neighbor selection has unknown selection probabilities and can attenuate
the OR (overmatching), so it warns. Same-time units are excluded from the risk
set; measure time finely.

**Categoricals**: one-hot encoded at the API layer as `var=level` columns
(collisions with existing columns are rejected). The dummies feed the PS model
and the distances and appear as per-level balance rows; CEM cells treat them as
exact categories; encoded columns do not leak into outputs. Bool and nullable
dtypes stay numeric. Dummies in Mahalanobis or Euclidean distances give rare
levels large leverage, so prefer `exact=` for strict categorical control.

## Scale architecture

The public API never constructs an O(n_treat x n_control) object. `engine="auto"`
computes the dense matrix when it fits `memory_limit_gb`, otherwise it uses a
memory-bounded candidate finder. There are two, chosen by the distance metric.

Propensity and logit distances use a propensity window (fast_greedy.py):
controls are sorted by logit-PS once, and each anchor's candidates come from a
binary search. The window needs a caliper to bound it.

Mahalanobis and Euclidean distances use a whitened KD-tree (covariate_nn.py).
Mahalanobis distance equals Euclidean distance after whitening the covariates by
the Cholesky factor of the inverse covariance, matching distances.py, so a
Euclidean KD-tree on whitened coordinates returns the true nearest neighbors.
No propensity score and no caliper are required. Exact constraints build one
tree per stratum. Each anchor takes its nearest available control (query k, skip
used, expand k, or a radius query under a caliper), which is the argmin the
dense path computes, so the pairs are identical to the dense path across
ratio/exact/caliper designs on both metrics. A 20k by 480k Mahalanobis match
runs in about 1 second and 0.4 GB, against about 70 seconds for MatchIt.

The windowed-covariate approximation (a covariate distance inside a 0.5-SD
propensity prefilter window) now applies only when a covariate distance is
combined with a propensity caliper. A propensity or logit caliper defines the
window exactly; only covariate-space calipers fall back to the prefilter
(`fast_prefilter_caliper_scale` = 0.5 SD, warned). On the windowed path anchors
match hardest-first ("largest" PS) by default, because extreme scores have the
fewest neighbors and data order left about 7% unmatched in benchmarks. The dense
path's default is fewest-potential-matches-first, which generalizes the same
idea under exact and caliper constraints. Under identical order with a caliper
inside the window, the dense and approximate paths produce identical pairs
(tested at 3k by 60k). Covariate-space distances use full-sample standardization
and covariance, so the windows reproduce the dense results. A 20k by 480k
propensity match runs in about 2 seconds and 0.4 GB (BENCHMARKS.md). MatchIt's
C++ nearest-neighbor also handles this scale; the claim is MatchIt-identical
results in Python at about a third of the memory.

`random_state` seeds cross-fitting, `m_order="random"`, and
`tie_break="random"`. It does not change the matching order on its own; a seed
that silently changes results would be a trap.

Tie-breaking is therefore an explicit choice rather than a seed side effect.
Every selection path resolves the argmin by input row order under the default
`tie_break="first"`, which is deterministic and right for continuous
distances, where exact ties are measure-zero. It is wrong the moment the key
is coarse — categorical covariates, exact matching, a propensity model over a
few binary predictors — because then row order alone decides the matched set,
and row order carries site, batch, or enrollment date often enough to bias the
estimate. `tie_break="random"` draws uniformly among the tied candidates under
the seed, so sensitivity to tie-breaking becomes measurable by varying it, and
a `TieBreakWarning` fires when the pool has enough duplicate keys to make the
choice consequential. The mechanism differs by path but the semantics do not:
the dense and windowed paths draw among the tied minima, the covariate tree
inserts stratum points in random order (a tie group can exceed any single
k-neighbor batch), and the optimal solver permutes control columns so that
degenerate optima are not settled by column order.

## Validation layers

1. Unit and property tests, run on every push.
2. Algorithm equivalence: dense and approximate paths produce the same pairs
   under a fixed order.
3. Golden reconciliation against MatchIt/cobalt on Lalonde (pair designs,
   subclassification, Mahalanobis, CEM, the logistic OR), with shared
   propensity scores so matching is isolated from estimation. The committed
   `validation/golden.json` keeps the comparison pure-Python.
4. At-scale golden on 20k by 480k: the snapshot in VALIDATION.md is identical to
   MatchIt to four decimals, while the automated assertions use tolerances
   (counts within 2%, ATT within 3 SE, balance no worse) to absorb environment
   drift.
5. Hand-derived guard: the GLM var_weights sandwich test.
6. Weekly CI regenerates the goldens against the latest R/MatchIt to detect
   external drift; it does not run on push, so its failures are not tied to a
   given commit.

`validation/` and `benchmarks/` share a structure: reference generators (R
scripts, competitor scripts) produce a committed JSON of reference values
(golden.json, competitors.json), and `report.py` turns that into a root Markdown
artifact (VALIDATION.md, BENCHMARKS.md). Both reports regenerate from the
committed data without an external toolchain.

Regenerate the golden values locally with
`pixi run --manifest-path validation/pixi.toml Rscript validation/generate_golden.R`.

Nothing statistical exists in this package without an executable external
referent: R, a hand derivation in a test, or a property test. Add the referent
before the statistic.

## Decisions log

- Function-first API, no fit/transform estimator. Matching is one-shot on a
  fixed sample, so sklearn estimator semantics would not fit.
- Stratum designs are separate entry points. They share almost no arguments
  with pair matching, and one combined signature produced silently-ignored
  parameters (flagged in two review rounds).
- Full matching is deferred. It needs a network-flow solver and validation
  against optmatch, does not scale, and quickmatch already covers large-N full
  matching.
- Spatial designs, competing risks, AIPW, g-computation, and Rosenbaum bounds
  are out of scope. The pairs and weights exports make each a short script
  against statsmodels or lifelines.
- HTML reports and the full plot gallery were removed for maintainability
  (matplotlib churn, a third of the codebase, no statistical content).
  `supplement.py` is a plain-text methods and reproducibility record (resolved
  parameters, versions, sample flow, balance) that also serves as the audit
  manifest requested in the RWE review.
- Effects are kept minimal. MatchIt estimates nothing; this package estimates
  the measures whose correctness can be pinned with a test and hands off the
  rest.
