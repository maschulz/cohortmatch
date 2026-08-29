# Validation report

CohortMatch reconciled against R's **MatchIt 4.7.2** and **cobalt 4.6.3** on the Lalonde data (R version 4.5.3). Generated from committed golden values by `validation/report.py`; regenerate with `python validation/report.py`. Every row is checked in CI (`tests/test_matchit_validation.py`).

**✓** marks agreement within the tolerance verified in CI: unadjusted SMDs to 1e-6, effect estimates within a few percent, matched counts exact unless footnoted.

## Balance conventions (unadjusted SMD vs cobalt)

Signed SMDs standardized by the treated-group SD, cobalt's convention.

| covariate | cohortmatch | cobalt | agree |
|---|---|---|---|
| age | -0.309445 | -0.309445 | ✓ |
| educ | +0.054965 | +0.054965 | ✓ |
| black | +1.761542 | +1.761542 | ✓ |
| hispan | -0.349843 | -0.349843 | ✓ |
| married | -0.826309 | -0.826309 | ✓ |
| nodegree | +0.244970 | +0.244970 | ✓ |
| re74 | -0.721084 | -0.721084 | ✓ |
| re75 | -0.290263 | -0.290263 | ✓ |

## Matching designs vs MatchIt

Matched counts, and mean |SMD| after matching (lower is better). Every design here keeps all 185 treated units, so both implementations balance the same population. Under contested controls the matching *order* differs from MatchIt, so pairs are not always identical and ✓ on a balance row means within 0.05.

| design | metric | cohortmatch | MatchIt | agree |
|---|---|---|---|---|
| nearest 1:1 (propensity) | matched treated | 185 | 185 | ✓ |
| | mean \|SMD\| after | 0.2719 | 0.2719 | ✓ |
| optimal 1:1 (propensity) | matched treated | 185 | 185 | ✓ |
| | mean \|SMD\| after | 0.2719 | 0.2689 | ✓ |
| nearest 1:2 (propensity) | matched treated | 185 | 185 | ✓ |
| | mean \|SMD\| after | 0.4704 | 0.4704 | ✓ |
| nearest, Mahalanobis | matched treated | 185 | 185 | ✓ |
| | mean \|SMD\| after | 0.2334 | 0.2312 | ✓ |
| coarsened exact matching † | matched treated | 180 | 178 | ✓ |

† CEM counts differ by a unit or two: numpy's `digitize` and R's `cut()` place values that fall *exactly* on a bin boundary in adjacent cells. Balance within cells is identical; only a couple of boundary units are assigned differently.

## Designs where treated units must drop

A caliper or an exact constraint leaves some treated units without an eligible control, and the matching *order* decides which ones go. Run under MatchIt's own order (`m_order="largest"`), cohortmatch reproduces its matched sample unit for unit.

| design | metric | cohortmatch | MatchIt | agree |
|---|---|---|---|---|
| nearest, caliper 0.2 SD | matched treated | 113 | 113 | ✓ |
| | treated units shared | 113 | 113 | ✓ |
| | control units shared | 113 | 113 | ✓ |
| | mean \|SMD\| after | 0.0679 | 0.0679 | ✓ |
| | ATT (re78) | 1571.7 | 1571.7 | ✓ |
| nearest, exact race | matched treated | 116 | 116 | ✓ |
| | treated units shared | 116 | 116 | ✓ |
| | control units shared | 116 | 116 | ✓ |
| | mean \|SMD\| after | 0.2190 | 0.2190 | ✓ |
| | ATT (re78) | 1472.7 | 1472.7 | ✓ |

### Under cohortmatch's default order

cohortmatch orders treated units by how scarce their eligible controls are, MatchIt by descending propensity score. The two then keep different treated subsets, so the balance below is achieved on a different matched-treated population and does not compare with MatchIt's figures above.

| design | matched treated | shared with MatchIt | mean \|SMD\| after |
|---|---|---|---|
| nearest, caliper 0.2 SD | 111 | 83 of 113 | 0.0973 |
| nearest, exact race | 116 | 82 of 116 | 0.1881 |

## Treatment effect vs MatchIt

Weighted effect on `re78` with cluster-robust SEs on match groups.

| design | quantity | cohortmatch | MatchIt | agree |
|---|---|---|---|---|
| nearest 1:1 | ATT (re78) | 894.4 | 894.4 | ✓ |
| | cluster-robust SE | 686.3 | 705.2 | ✓ |
| full cohort | odds ratio (glm) | 0.921114 | 0.921114 | ✓ |
| | robust SE | 0.2064 | 0.2067 | ✓ |

## Estimator, caliper, and diagnostics vs MatchIt

The default estimation machinery, the auto caliper, and Rubin's B/R -- reconciled against R directly rather than through injected scores.

| quantity | cohortmatch | MatchIt/R | agree |
|---|---|---|---|
| full-sample logistic PS, max abs diff vs R glm | 1.7e-04 | 0 | ✓ |
| auto caliper (0.2 x SD logit PS) | 0.3606 | 0.3609 | ✓ |
| Rubin's B (1:1) | 86.01 | 86.01 | ✓ |
| Rubin's R (1:1) | 0.771 | 0.771 | ✓ |
| subclass HC3 SE vs R vcovHC(HC3) | 840.4 | 841.6 | ✓ |

## At biobank scale (20,000 × 480,000)

A synthetic cohort matched on a shared propensity score with the same caliper in both implementations.

Same design, both implementations, identical results.

| quantity | cohortmatch | MatchIt |
|---|---|---|
| matched treated | 18,426 | 18,426 |
| ATT | 2.017 | 2.017 |
| cluster-robust SE | 0.015 | 0.015 |

For speed and memory at scale, see the benchmark report.

## Validated by other means

- **Euclidean distance, per-covariate calipers, `replace`, `m_order`**: the approximate/tree path produces byte-identical pairs to the exact path (`tests/test_api.py`), and the exact path is validated above.
- **Odds/risk/hazard ratios beyond the OR shown**: the weighted sandwich is checked against a hand computation (`test_glm_var_weights_sandwich`).
- **E-values**: reproduce the published VanderWeele & Ding (2017) worked example.
- **Risk-set (incidence-density) matching**: no reference implementation exists; validated by property tests and recovery of a simulated hazard ratio.

