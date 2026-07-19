library(MatchIt)
# Usage: Rscript benchmark_matchit.R <cohort.csv>
# (export a cohort with: python -c "import sys; sys.path.insert(0,'benchmarks');
#  from benchmark_scale import make_cohort; make_cohort(20000,480000).to_csv('cohort.csv')")
args <- commandArgs(trailingOnly = TRUE)
data <- read.csv(args[1])
cat("rows:", nrow(data), "\n")
t0 <- Sys.time()
m <- matchit(treatment ~ x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7,
             data = data, method = "nearest", distance = "glm",
             caliper = 0.2, estimand = "ATT")
t1 <- Sys.time()
md <- match.data(m)
cat("matchit nearest+caliper:", round(as.numeric(difftime(t1, t0, units = "secs")), 1),
    "s, matched treated:", sum(md$treatment == 1), "\n")
