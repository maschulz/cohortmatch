# Generates golden validation values from MatchIt/cobalt on the Lalonde data.
# Output: validation/golden.json, consumed by tests/test_matchit_validation.py.
#
# The propensity score is fit once here (full-sample logistic regression) and
# exported, so both sides match on identical scores and the comparison
# isolates the matching itself.

library(MatchIt)
library(cobalt)
library(sandwich)
library(lmtest)
library(jsonlite)

data("lalonde", package = "MatchIt")
lalonde$id <- rownames(lalonde)
lalonde$black <- as.integer(lalonde$race == "black")
lalonde$hispan <- as.integer(lalonde$race == "hispan")

covs <- c("age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75")
form <- reformulate(covs, response = "treat")

ps_model <- glm(form, data = lalonde, family = binomial)
lalonde$ps <- fitted(ps_model)

smd_from_baltab <- function(bal, col) {
  smd <- bal$Balance[[col]]
  names(smd) <- rownames(bal$Balance)
  as.list(smd[covs])
}

design_block <- function(m.out) {
  md <- match.data(m.out, data = lalonde)
  bal <- bal.tab(m.out, un = TRUE, binary = "std", s.d.denom = "treated")
  fit <- lm(re78 ~ treat, data = md, weights = weights)
  vc <- vcovCL(fit, cluster = ~subclass, data = md)
  ct <- coeftest(fit, vcov. = vc)
  mm <- m.out$match.matrix
  pair_dist <- 0
  if (!is.null(mm)) {
    for (t_id in rownames(mm)) {
      for (c_id in mm[t_id, ]) {
        if (!is.na(c_id)) {
          pair_dist <- pair_dist + abs(lalonde[t_id, "ps"] - lalonde[c_id, "ps"])
        }
      }
    }
  }
  list(
    n_treated = sum(md$treat == 1),
    n_control = sum(md$treat == 0),
    sum_weights_control = sum(md$weights[md$treat == 0]),
    control_ids = sort(md$id[md$treat == 0]),
    smd_after = smd_from_baltab(bal, "Diff.Adj"),
    att = unname(ct["treat", "Estimate"]),
    att_se = unname(ct["treat", "Std. Error"]),
    total_ps_distance = pair_dist
  )
}

golden <- list(
  ps = setNames(as.list(lalonde$ps), lalonde$id),
  sd_ps = sd(lalonde$ps),
  covariates = covs,
  unadjusted_smd = smd_from_baltab(
    bal.tab(form, data = lalonde, binary = "std", s.d.denom = "treated"),
    "Diff.Un"
  ),
  designs = list(
    optimal_1to1 = design_block(
      matchit(form, data = lalonde, method = "optimal",
              distance = lalonde$ps, estimand = "ATT")
    ),
    nearest_1to1 = design_block(
      matchit(form, data = lalonde, method = "nearest",
              distance = lalonde$ps, estimand = "ATT")
    ),
    nearest_caliper = design_block(
      matchit(form, data = lalonde, method = "nearest",
              distance = lalonde$ps, caliper = 0.2, estimand = "ATT")
    ),
    nearest_1to2 = design_block(
      matchit(form, data = lalonde, method = "nearest",
              distance = lalonde$ps, ratio = 2, estimand = "ATT")
    ),
    nearest_exact_race = design_block(
      matchit(form, data = lalonde, method = "nearest",
              distance = lalonde$ps, exact = ~race, estimand = "ATT")
    ),
    subclass_6 = local({
      m <- matchit(form, data = lalonde, method = "subclass",
                   distance = lalonde$ps, subclass = 6, estimand = "ATT")
      md <- match.data(m, data = lalonde)
      bal <- bal.tab(m, un = TRUE, binary = "std", s.d.denom = "treated")
      fit <- lm(re78 ~ treat, data = md, weights = weights)
      ct <- coeftest(fit, vcov. = vcovHC(fit, type = "HC1"))
      list(
        n_treated = sum(md$treat == 1),
        n_control = sum(md$treat == 0),
        sum_weights_control = sum(md$weights[md$treat == 0]),
        smd_after = smd_from_baltab(bal, "Diff.Adj"),
        att = unname(ct["treat", "Estimate"]),
        att_se = unname(ct["treat", "Std. Error"])
      )
    }),
    nearest_mahalanobis = local({
      m <- matchit(form, data = lalonde, method = "nearest",
                   distance = "mahalanobis", estimand = "ATT")
      md <- match.data(m, data = lalonde)
      bal <- bal.tab(m, un = TRUE, binary = "std", s.d.denom = "treated")
      list(
        n_treated = sum(md$treat == 1),
        n_control = sum(md$treat == 0),
        smd_after = smd_from_baltab(bal, "Diff.Adj")
      )
    }),
    cem_fixed = local({
      # explicit cutpoints so both implementations coarsen identically
      cut <- list(age = c(25, 35, 45),
                  educ = c(8, 11),
                  re74 = c(5000, 15000),
                  re75 = c(5000, 15000))
      m <- matchit(treat ~ age + educ + re74 + re75, data = lalonde,
                   method = "cem", cutpoints = cut, estimand = "ATT")
      md <- match.data(m, data = lalonde)
      bal <- bal.tab(m, un = TRUE, binary = "std", s.d.denom = "treated")
      list(
        n_treated = sum(md$treat == 1),
        n_control = sum(md$treat == 0),
        sum_weights_control = sum(md$weights[md$treat == 0]),
        smd_after = as.list(setNames(bal$Balance[["Diff.Adj"]][c("age","educ","re74","re75")],
                                     c("age","educ","re74","re75")))
      )
    }),
    logistic_effect = local({
      # pure estimator reconciliation on the full cohort (no matching): a
      # weighted logistic OR with an HC1-robust SE, isolating our GLM machinery
      d <- lalonde
      d$emp <- as.integer(d$re78 > 0)
      fit <- glm(emp ~ treat, data = d, family = quasibinomial)
      ct <- coeftest(fit, vcov. = vcovHC(fit, type = "HC1"))
      list(
        odds_ratio = unname(exp(ct["treat", "Estimate"])),
        log_or = unname(ct["treat", "Estimate"]),
        se = unname(ct["treat", "Std. Error"])
      )
    })
  ),
  versions = list(
    matchit = as.character(packageVersion("MatchIt")),
    cobalt = as.character(packageVersion("cobalt")),
    r = R.version.string
  )
)

write_json(golden, "validation/golden.json", auto_unbox = TRUE, digits = 12)
cat("Wrote validation/golden.json\n")
