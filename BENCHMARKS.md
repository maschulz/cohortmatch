# Benchmark report

cohortmatch matching time and peak memory, measured on macOS / Python 3.12.13 / numpy 2.5.1. Each case is a full `match()` call (propensity estimation included where used), run in its own process for a clean peak-memory reading. Shapes are round total cohort sizes (50k, 100k, 500k) at ~4% treated; 500k is UK Biobank scale. Regenerate with `python benchmarks/report.py`.

## cohortmatch across cohort shapes

The dense (exact) path peaks at roughly twice the distance matrix during matching (a working copy); `memory_limit_gb` bounds the matrix, so set it near a third of available RAM if you rely on the exact path. The approximate path holds no matrix; its memory stays flat and sub-gigabyte at 20k × 480k.

| shape (treated × controls) | design | time | pairs | peak memory | path |
|---|---|---|---|---|---|
| 2k × 48k | propensity, caliper=auto | 0.7 s | 1,968 | 1.80 GB | exact |
| 2k × 48k | Mahalanobis (no PS, no caliper) | 2.5 s | 2,000 | 1.80 GB | exact |
| 2k × 48k | propensity + exact on sex | 1.0 s | 1,961 | 1.90 GB | exact |
| 4k × 96k | propensity, caliper=auto | 2.5 s | 3,906 | 6.73 GB | exact |
| 4k × 96k | Mahalanobis (no PS, no caliper) | 10.0 s | 4,000 | 6.73 GB | exact |
| 4k × 96k | propensity + exact on sex | 3.8 s | 3,898 | 7.11 GB | exact |
| 20k × 480k | propensity, caliper=auto | 1.2 s | 19,622 | 0.44 GB | approximate |
| 20k × 480k | Mahalanobis (no PS, no caliper) | 2.7 s | 20,000 | 0.38 GB | approximate |
| 20k × 480k | propensity + exact on sex | 1.6 s | 19,604 | 0.44 GB | approximate |

## Same 20k × 480k cohort, other packages

Competitor packages, measured 2026-07-19 on Apple Silicon (M-series), macOS. Compare against the cohortmatch rows in the table above.

| package | design | time | peak memory | notes |
|---|---|---|---|---|
| R MatchIt 4.7.2 | propensity NN + caliper | 1.6 s | 1.4 GB | C++ nearest-neighbor |
| R MatchIt 4.7.2 | Mahalanobis NN | 70 s | 0.58 GB | brute-force per unit |
| psmpy 0.3.13 | propensity kNN | n/a | n/a | OOM-killed at this scale (57 s at 2k×38k) |

