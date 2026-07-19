# Contributing

## Setup

Development uses [pixi](https://pixi.sh):

```bash
git clone https://github.com/maschulz/cohortmatch
cd cohortmatch
pixi install
pixi run test        # or: pixi run lint, pixi run fmt, pixi run typecheck
```

`pixi run -e py310 test` (also `py311`, `py313`) runs the suite on other
Python versions; CI runs all of them.

## Tests

The suite runs without R. `pixi run test` covers the API, matching, balance,
effects, and the reconciliation against committed golden values from MatchIt.

The `test_matchit_validation.py` and `test_matchit_scale.py` tests compare
against `validation/golden.json` and `validation/golden_scale.json`, which are
committed. Regenerating them uses the pinned R toolchain (MatchIt and cobalt
from conda; optmatch, for the optimal designs, is compiled from CRAN into the
same env):

```bash
pixi run --manifest-path validation/pixi.toml Rscript \
  -e 'install.packages("optmatch", repos = "https://cloud.r-project.org")'
pixi run --manifest-path validation/pixi.toml Rscript validation/generate_golden.R
```

CI regenerates the golden values weekly against the latest MatchIt/cobalt to
catch upstream drift; the per-push suite only compares against the committed
files.

## Changing a statistical convention

Balance denominators, caliper scales, weight formulas, and effect estimators
are pinned by golden tests and documented in `DESIGN.md`, each with its
literature anchor. A change there should update `DESIGN.md`, regenerate the
golden values, and explain in the PR why the new number is correct.

## Style

`ruff` config is in `pyproject.toml`. Run `pixi run lint` and `pixi run fmt`
before opening a PR. Prose in docs, docstrings, and commit messages is plain
and declarative; no em-dashes.

## Scope

cohortmatch constructs and diagnoses matched samples and estimates the standard
effect measures. Survival models, sensitivity analysis, and doubly-robust
estimators are out of scope and handed off to statsmodels/lifelines. New
matching designs and scale improvements are in scope.
