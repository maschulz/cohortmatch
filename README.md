# CohortMatch

[![tests](https://github.com/maschulz/cohortmatch/actions/workflows/tests.yml/badge.svg)](https://github.com/maschulz/cohortmatch/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/cohortmatch)](https://pypi.org/project/cohortmatch/)
[![Validated against MatchIt](https://img.shields.io/badge/validated%20against-MatchIt-success)](VALIDATION.md)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)](https://github.com/maschulz/cohortmatch/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/maschulz/cohortmatch/blob/main/LICENSE)

Statistical matching for cohort studies: nearest-neighbor and optimal matching
on propensity scores or covariate distances, propensity subclassification,
coarsened exact matching, and risk-set matching, with calipers, exact
constraints, balance diagnostics, and treatment effect estimation. Validated
against R's MatchIt, and handles biobank-scale cohorts (500k rows). I first
wrote this for in-house use by students in the lab and to support my own research at
that scale; it grew over time and now seems stable enough to release.

**Scope.** cohortmatch constructs and diagnoses matched
samples at any scale, and estimates the standard effect measures on them
(risk difference, odds ratio, risk ratio). Everything beyond that (survival
models, sensitivity analysis, doubly-robust estimators) is a documented
handoff to statsmodels/lifelines with the matching weights and groups
attached (see "Effects on the matched sample").

## Installation

```bash
pip install "cohortmatch[viz] @ git+https://github.com/maschulz/cohortmatch.git"
```

Not yet on PyPI. The `viz` extra adds plotting.

## Quick start

Runnable as-is; `load_lalonde()` ships with the package. Match, then check the
balance:

```python
from cohortmatch import match
from cohortmatch.datasets import load_lalonde

data = load_lalonde()   # 614 units; treatment column "treat", outcome "re78"

result = match(data, treatment="treat",
               covariates=["age", "educ", "race", "married", "re74", "re75"])

print(result.summary())   # counts, balance, Rubin's rules
result.matched_data       # the matched cohort, original index preserved
result.balance()          # signed SMD per covariate, before and after
result.pairs              # treatment_id, control_id, distance, match_group
```

The matched cohort is the output. cohortmatch includes the standard effect
estimators, or you can take it to statsmodels/lifelines (see "Effects on the
matched sample"):

```python
result.estimate_effects("re78")   # weighted effect, cluster-robust SE
result.supplement("supp.md")      # methods and results record for a paper
```

## Which function do I use?

| Design | Function | Estimand | Result shape |
|---|---|---|---|
| Nearest / optimal pair matching | `match()` | ATT / ATC | pairs + weights |
| Propensity subclassification | `subclassify()` | ATT / ATC / **ATE** | strata + weights |
| Coarsened exact matching | `cem()` | ATT / ATC / **ATE** | strata + weights |
| Nested case-control (incident disease) | `match_risk_set()` | rate/hazard ratio | matched sets |

The **estimand** is which average effect you get: ATT (on the treated units),
ATC (on the controls), or ATE (on the whole population). A **propensity
score** is each unit's estimated probability of being treated given its
covariates; matching units with similar scores makes those covariates
comparable between the groups.

For `match()`, `distance="propensity"` (default) matches on the propensity
score, the default for confounder control. Use `distance="mahalanobis"`
to match directly in covariate space (no propensity model; scales via a
KD-tree). All designs scale to biobank size.

By default this estimates propensity scores with cross-fitted logistic
regression (each unit is scored by a model that never saw it, over 5 folds),
matches each treated unit to its nearest control (ATT), applies no caliper,
and computes balance statistics.

**Data contract:** the treatment column is 0/1; the DataFrame index
identifies units, must be unique, and has string or integer labels; column
names are strings; covariates must be complete (no NaN).

> **Categoricals:** string covariates are one-hot encoded automatically and
> appear in balance tables as `var=level` rows. A categorical coded as
> numbers (e.g. smoking 0/1/2) is otherwise treated as *continuous*. Cast
> it to `category` dtype first (`df["smoking"] = df["smoking"].astype("category")`).

> **Missing data:** cohortmatch does not impute. Handle NaN before matching:
> complete-case (`df.dropna(subset=covariates)`) or your own imputation. Note
> that imputing then matching propagates imputation uncertainty into the
> matched set; multiple imputation with matching inside each imputation is
> the rigorous route.

## What matching assumes

Matching adjusts only for what you match on. The causal reading of any
effect below requires: (1) no unmeasured confounding, every variable that
influences both treatment and outcome is in `covariates`; (2) covariates
measured **before** treatment (matching on post-treatment variables biases
the estimate, and nothing in the data can reveal this); (3) overlap between
the groups. Good balance is evidence the *measured* covariates are
comparable, never evidence for (1) or (2). For sensitivity to unmeasured
confounding, export `result.pairs` to R's `rbounds`/`sensemakr` (Rosenbaum
bounds are planned).

## The estimand is set by the matching

`estimand="att"` (default) anchors matching on the treated units: every treated
unit is kept if possible, and the result estimates the effect on the treated.
`estimand="atc"` anchors on the controls. If anchor units cannot be matched
(caliper, exact constraints, pool exhausted), you get a warning with the count,
because dropping anchor units changes the population your estimate refers to.

There is no silent fallback: with more treated than controls, `estimand="att"`
still matches from the treated side and warns about the shortfall.

## Calipers

A **caliper** is the largest distance two units may be apart and still be
matched; a pair farther apart is left unmatched. No caliper is applied unless
you ask for one, and on lopsided pools that default can be a bias trap: on the
classic Lalonde data, 1:1 matching
without a caliper retains all 185 treated but leaves a maximum |SMD| of
1.03 and halves the effect estimate, silently. Check `summary()` before
believing any effect; `caliper="auto"` is the standard remedy.

```python
# the standard choice: 0.2 SD of the logit propensity score (Austin 2011)
match(data, treatment="treated", covariates=covs, caliper="auto")

# same rule, different width
match(data, treatment="treated", covariates=covs, caliper=0.1)

# raw units instead of standardized (here: max difference in probability)
match(data, treatment="treated", covariates=covs, caliper=0.05, std_caliper=False)

# Mahalanobis matching within a propensity caliper (Rubin & Thomas)
match(data, treatment="treated", covariates=covs,
      distance="mahalanobis", caliper="auto")

# caliper on the matching distance itself
match(data, treatment="treated", covariates=covs,
      distance="mahalanobis", caliper=4.0, caliper_metric="mahalanobis")

# per-variable calipers, raw units: age within 3 years, BMI within 2
match(data, treatment="treated", covariates=covs,
      caliper="auto", covariate_calipers={"age": 3.0, "bmi": 2.0})
```

Numeric propensity calipers are standardized (multiples of the SD of the logit
propensity score) unless `std_caliper=False`; Mahalanobis and Euclidean
calipers are always in raw distance units.

## Propensity scores

When scores are needed and none are supplied, cohortmatch fits logistic
regression with 5-fold cross-fitting, so each unit's score comes from a model
that did not see it. No calibration is applied.

```python
# any sklearn classifier; it is cloned, your object is not touched
from sklearn.ensemble import GradientBoostingClassifier
match(data, treatment="treated", covariates=covs,
      propensity_model=GradientBoostingClassifier())

# precomputed scores: a column name, Series, or array
match(data, treatment="treated", covariates=covs, propensity_scores="ps")
```

`result.propensity_scores` returns the scores as a Series aligned to your
data's index; `result.propensity_model` a fitted pipeline usable on raw
covariates; `result.propensity_metrics` the cross-validated AUC and overlap
diagnostics.

## Large datasets

`match()` refuses to walk into an out-of-memory crash. With `engine="auto"`
(default) it computes the dense distance matrix when it fits into
`memory_limit_gb` (default 4 GB); beyond that it switches to a memory-bounded
algorithm that draws candidates from a propensity-score window, announced
with a warning. **The memory-bounded path needs a propensity caliper** to
define its windows: at biobank scale, plans built only on `exact` and
`covariate_calipers` will raise with the exact argument to add
(`caliper="auto"`).

```python
# 20k cases against 480k controls: ~2 seconds, <0.5 GB
result = match(biobank, treatment="case", covariates=covs, caliper="auto")

# pin it explicitly (reproducible across data sizes, silences the warning)
result = match(biobank, treatment="case", covariates=covs,
               caliper="auto", engine="approximate")
```

Candidate pools come from binary search over propensity-sorted controls, and
anchor units match hardest-first. On a 20k x 480k cohort with shared
propensity scores, cohortmatch and R's MatchIt produce identical matched
counts, balance, and effect estimates, at a third of the memory (see
`BENCHMARKS.md`).

Covariate distances scale too: Mahalanobis and Euclidean matching use a
whitened KD-tree (no propensity score, no caliper required) and return the
*same* pairs as the exact path. A 20k x 480k Mahalanobis match runs in
~1 s in ~0.4 GB, where R's MatchIt takes ~70 s.

`method="optimal"` has no approximate variant; at that scale use
`method="nearest"`.

## Common support

```python
result = match(data, treatment="treated", covariates=covs, discard="treated")
result.discarded                 # ids dropped before matching, with a warning
```

Drops units whose propensity score falls outside the other group's range
before matching ("treated", "control", or "both"). `result.original_data`
and the pre-matching balance always describe the full input sample.

## Stratum designs: subclassify() and cem()

Stratum designs are their own entry points: they express the design through
weights instead of pairs, accept different arguments than pair matching, and
support `estimand="ate"`:

```python
from cohortmatch import subclassify, cem

# propensity-score subclassification
result = subclassify(data, treatment="treated", covariates=covs,
                     n_subclasses=6, estimand="ate")

# coarsened exact matching: bin, cross, keep cells with both groups
result = cem(data, treatment="treated", covariates=covs,
             coarsening={"age": 5}, exact="sex")

result.strata                    # stratum per unit
result.weights                   # stratum weights: each group reweighted to
                                 # the target population's stratum distribution
```

Balance, `table1()`, and `estimate_effects()` use the weights automatically,
with HC-robust rather than cluster-robust errors (a handful of strata are too
few clusters). Subclassification is validated against MatchIt; CEM's default
binning is Sturges' rule per continuous covariate. Note CEM is a different
design, not a drop-in sensitivity swap for pair matching: there is no ratio
or caliper; closeness is expressed through the coarsening.

## Other constraints

```python
match(data, treatment="treated", covariates=covs,
      method="optimal",          # global optimum instead of nearest-neighbor
      distance="mahalanobis",
      ratio=2,                   # 1:2 matching (two controls per anchor)
      exact="sex",               # or a list of columns
      random_state=42)
```

`replace=True` allows controls to be reused across matches (`"nearest"` only).

## Balance

```python
result.balance()                   # signed SMDs and variance ratios, before/after
result.table1()                  # group means/SDs with SMDs, the cohort table
result.rubin_statistics          # Rubin's rules: share of covariates with
                                 # |SMD| < 0.25 and variance ratio in [0.5, 2]
print(result.summary())          # counts, mean/max |SMD|, Rubin's rules
```

A **standardized mean difference (SMD)** is the gap in a covariate's mean
between the groups measured in standard-deviation units, so it is comparable
across covariates; |SMD| < 0.1 is the usual target for good balance.
cohortmatch's SMDs are signed and standardized by the anchor group's SD in the
original sample, with the same denominator before and after matching, so the
two numbers are directly comparable (cobalt's convention). Post-matching
statistics use the matching weights.

Notes on encoded categoricals and the default propensity model: one-hot
dummies enter Euclidean/Mahalanobis distances, where a k-level categorical
contributes k columns and rare levels get large standardized leverage;
prefer `exact=` for categoricals you want strictly controlled. The default
propensity model is L2-regularized logistic regression (scores are shrunk
relative to an unpenalized GLM); pass your own
`propensity_model=LogisticRegression(penalty=None)` for MLE scores.

With the `viz` extra, the standard diagnostics are one call each:

```python
result.plot_love_plot()          # SMDs before/after, the cobalt-style plot
result.plot_balance()
result.plot_propensity()         # score overlap before/after
result.plot_match_distances()
```

`match()` does not flag balance quality; `summary()` reports the SMDs and the
judgment is yours.

## Matching weights

```python
result.weights                   # Series indexed by unit; anchors get 1
result.match_groups              # anchor id per unit (None with replacement)
```

Every unit appears once in `matched_data`; reuse under `replace=True` and
ratio matching are expressed through the weights, never duplicated rows. Any
analysis of the matched sample should use them, for example
`sm.WLS(y, X, weights=result.weights)`.

## Treatment effects

```python
effects = result.estimate_effects(
    ["outcome1", "outcome2"],
    method="mean_difference",    # or "regression_adjustment"
)
```

```python
result.estimate_effects("event", family="logistic")   # odds ratio
result.estimate_effects("event", family="poisson")    # risk ratio
```

Effects are weighted outcome models with the matching weights: `family=`
selects a mean/risk difference ("linear", default; for a binary outcome
this is an absolute difference in probabilities, not a relative effect), an
odds ratio ("logistic"), or a risk ratio ("poisson"); hazard ratios are a
five-line recipe (see "Effects on the matched sample"). The `measure`
column records what the effect is. Standard errors are cluster-robust on match groups (matching
without replacement) or HC1-robust (with replacement); the `se_type` column
records which. One caveat: all standard errors assume errors independent across
match groups; spatially or network-correlated outcomes need external
correction. The estimand is inherited from the matching design; there is no
way to relabel an ATT matched sample as ATE after the fact.

## Effects on the matched sample: the handoff

Anything beyond the built-in estimators is a few lines with the weights and
match groups the result carries:

```python
# hazard ratio: weighted Cox with robust errors clustered on match groups
from lifelines import CoxPHFitter
df = result.matched_data[["follow_up", "event", "treated"]].copy()
df["w"] = result.weights
df["g"] = result.match_groups
CoxPHFitter().fit(df, "follow_up", "event",
                  weights_col="w", cluster_col="g", robust=True)

# anything statsmodels: weighted design, cluster-robust covariance.
# Align weights and groups to the matched_data row order first, statsmodels
# consumes them positionally, so pass numpy arrays in the right order.
import statsmodels.formula.api as smf
md = result.matched_data
fit = smf.wls("outcome ~ treated + age", data=md,
              weights=result.weights.reindex(md.index).to_numpy()).fit(
    cov_type="cluster",
    cov_kwds={"groups": result.match_groups.reindex(md.index).to_numpy()})
```

For sensitivity to unmeasured confounding, cohortmatch includes the E-value
(VanderWeele & Ding 2017), the minimum confounder strength on the risk-ratio
scale needed to explain the effect away:

```python
from cohortmatch import e_value
row = result.estimate_effects("event", family="poisson").iloc[0]
e_value(row["effect"], row["ci_lower"], row["ci_upper"], measure="risk_ratio")
# {"e_value": ..., "e_value_ci": ...}
```

Odds and hazard ratios are converted via the standard approximations
(`rare_outcome=True` uses them directly). For Rosenbaum bounds, export
`result.pairs` to R's `rbounds`.

cohortmatch is silent by default. `cohortmatch.configure_logging()` turns on
progress output, including progress bars for long matching runs.

## Risk-set matching (nested case-control)

```python
from cohortmatch import match_risk_set

result = match_risk_set(
    cohort, event_time="follow_up_years", event="diagnosed",
    ratio=4, exact="sex", covariate_calipers={"age": 3.0},
)
result.sets                       # set_id, unit_id, case, index_time
result.balance()                  # cases vs matched controls (SMDs)
result.table1()                   # case/control means and SDs
result.estimate_odds_ratio(
    "exposure", adjustment_covariates=["smoking"]
)                                 # conditional logistic; OR estimates the hazard ratio
result.supplement("ncc_S1.md", exposures="exposure")   # paper-ready record
```

Controls are drawn from each case's risk set, units still at risk at the
case's event time (strictly later event times; measure time finely to avoid
ties), including future cases (incidence-density sampling). Control
confounders by *restricting* eligibility (`exact`, `covariate_calipers`)
and sampling at random; that is the design under which the odds ratio
estimates the hazard ratio. Nearest-neighbor selection (`covariates=`)
departs from random sampling and can bias the odds ratio toward the null
(overmatching); a warning says so, and any selection covariates should also
be adjusted in `estimate_odds_ratio`. Neither MatchIt nor any Python
package offers this design.

## Supplementary material

```python
result.supplement("supplement_S1.md", title="Study S1 matching supplement")
```

One call writes a self-contained Markdown record for a paper's
supplementary material: the resolved design specification (including the
numeric caliper actually applied, not just "auto"), software versions and
seed, the sample flow, the balance table, effect estimates, and a citable
methods paragraph with references. Plain text, no extra dependencies;
convert with pandoc if the journal wants PDF or Word.

## How to cite

If you use cohortmatch in published work, please cite it (see
`CITATION.cff`):

> Schulz, M.-A. (2026). *cohortmatch: statistical matching for cohort
> studies at scale.* https://github.com/maschulz/cohortmatch

## Validation against MatchIt

**[VALIDATION.md](VALIDATION.md)** is a generated report reconciling every
design and effect estimator against R, row by row. **[BENCHMARKS.md](BENCHMARKS.md)**
is the generated speed/memory report. Both regenerate from the harness
(`python validation/report.py`, `python benchmarks/report.py`).

The balance conventions and matching designs are validated against R's
MatchIt/cobalt on the Lalonde data: identical propensity scores go into both
implementations and the outputs are reconciled, unadjusted SMDs to 1e-6,
optimal matching by total distance, nearest designs by counts, balance, and
effect estimates. Runs in CI and locally
(`pixi run --manifest-path validation/pixi.toml Rscript validation/generate_golden.R`,
then `pytest tests/test_matchit_validation.py`). The benchmark dataset ships
with the package:

```python
from cohortmatch.datasets import load_lalonde
lalonde = load_lalonde()
```

## No matches?

`match()` raises `NoMatchesError` instead of returning an empty result. Relax
the caliper, drop exact constraints, or check that the groups overlap.

## match() reference

| Parameter | Default | Applies to | Meaning |
|---|---|---|---|
| `data` | required | all | DataFrame, one row per unit; the index identifies units |
| `treatment` | required | all | binary treatment column (1/0) |
| `covariates` | required | all | columns to balance (numeric, no NaN) |
| `method` | `"nearest"` | all | `"nearest"` or `"optimal"` |
| `distance` | `"propensity"` | all | `"propensity"`, `"logit"`, `"mahalanobis"`, `"euclidean"` |
| `estimand` | `"att"` | all | `"att"` or `"atc"`, which group anchors the matching |
| `caliper` | `None` | all | `None`, `"auto"` (0.2 SD logit-PS), or a number |
| `caliper_metric` | `"propensity"` | all | metric the caliper applies to |
| `std_caliper` | `True` | with `caliper` | numeric PS calipers in SD-of-logit-PS units |
| `covariate_calipers` | `None` | all | per-variable max difference, raw units |
| `ratio` | `1` | all | controls per anchor unit (integer, 1:k) |
| `replace` | `False` | nearest | reuse controls across matches |
| `exact` | `None` | all | column(s) that must match exactly |
| `propensity_scores` | `None` | all | precomputed scores (column, Series, or array) |
| `propensity_model` | `None` | all | sklearn classifier to estimate scores |
| `cv` | `5` | estimated scores | cross-fitting folds |
| `discard` | `None` | all | common-support discard before matching |
| `algorithm` | `"auto"` | nearest | `"exact"`, `"approximate"`, or size-dependent |
| `m_order` | hardest-first | nearest | matching order (`"largest"`, `"smallest"`, `"closest"`, `"random"`, `"data"`) |
| `covariate_weights` | `None` | euclidean | distance weights |
| `standardize` | `True` | covariate distances | standardize before distance computation |
| `tie_break` | `"first"` | nearest, optimal | equidistant candidates: `"first"` (input row order) or `"random"` |
| `random_state` | `None` | all | seed for `tie_break="random"`, `m_order="random"`, cross-fitting |
| `memory_limit_gb` | `4.0` | `engine="auto"` | dense-matrix budget |

`subclassify()` and `cem()` have their own, smaller signatures, see their
docstrings. Warnings are typed (`IncompleteMatchWarning`,
`CommonSupportWarning`, `ApproximateMatchWarning`, `TieBreakWarning`), so they
can be filtered individually.

### Ties

With continuous distances, two controls are almost never exactly equidistant
from an anchor, and matching is deterministic. Coarse keys change that:
categorical or binned covariates, exact matching, or a propensity model over a
few binary predictors leave many candidates at the same distance. The default
`tie_break="first"` awards those to the earlier input row, which is
reproducible but makes the matched set a function of storage order — and
anything that correlates with it, such as site, batch, or enrollment date,
then leaks into the comparison group. `match()` warns (`TieBreakWarning`) when
the pool carries enough duplicate keys for this to matter.

```python
result = match(data, treatment="treated", covariates=["sex", "smoker"],
               tie_break="random", random_state=42)
```

`tie_break="random"` draws uniformly among the tied candidates under
`random_state`; re-running across seeds shows how much the estimate depends on
tie-breaking. It applies to every selection path (dense and approximate
nearest-neighbor, the covariate-space tree, and the degenerate optima of
`method="optimal"`). Where no ties exist it changes nothing.
