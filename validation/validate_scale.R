# At-scale reference values from MatchIt/cobalt on the deterministic large
# cohort (see make_scale_cohort.py). Matches on the precomputed "ps" column.
#
# Usage: Rscript validation/validate_scale.R <cohort.csv> <out.json>

library(MatchIt)
library(cobalt)
library(sandwich)
library(lmtest)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
data <- read.csv(args[1])
covs <- paste0("x", 0:7)
form <- reformulate(covs, response = "treatment")

sd_ps <- sd(data$ps)

t0 <- Sys.time()
m <- matchit(form, data = data, method = "nearest",
             distance = data$ps, caliper = 0.2, estimand = "ATT")
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

md <- match.data(m, data = data)
bal <- bal.tab(m, un = TRUE, binary = "std", s.d.denom = "treated")
smd <- function(col) {
  v <- bal$Balance[[col]]
  names(v) <- rownames(bal$Balance)
  as.list(v[covs])
}

fit <- lm(y ~ treatment, data = md, weights = weights)
ct <- coeftest(fit, vcov. = vcovCL(fit, cluster = ~subclass, data = md))

mm <- m$match.matrix
pair_diffs <- abs(data[rownames(mm), "ps"] - data[mm[, 1], "ps"])
pair_diffs <- pair_diffs[!is.na(pair_diffs)]

golden <- list(
  sd_ps = sd_ps,
  elapsed_seconds = elapsed,
  n_treated = sum(md$treatment == 1),
  n_control = sum(md$treatment == 0),
  unadjusted_smd = smd("Diff.Un"),
  smd_after = smd("Diff.Adj"),
  att = unname(ct["treatment", "Estimate"]),
  att_se = unname(ct["treatment", "Std. Error"]),
  mean_pair_ps_diff = mean(pair_diffs),
  versions = list(matchit = as.character(packageVersion("MatchIt")))
)
write_json(golden, args[2], auto_unbox = TRUE, digits = 12)
cat("wrote", args[2], "- matchit", round(elapsed, 1), "s, matched treated:",
    golden$n_treated, "\n")
