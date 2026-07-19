#!/usr/bin/env python
"""Generate BENCHMARKS.md by running cohortmatch across shapes and designs.

Each case runs in a forked subprocess so peak memory (ru_maxrss) is clean.
cohortmatch rows are measured live on this machine; competitor rows are
measured references (see competitors.json) with their provenance stated.

    python benchmarks/report.py
"""

import json
import multiprocessing as mp
import platform
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _peak_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9


def _cohort(n_treat, n_control, n_covs=8, seed=0):
    rng = np.random.RandomState(seed)
    n = n_treat + n_control
    X = rng.normal(size=(n, n_covs))
    noisy = 1 / (1 + np.exp(-(0.8 * X[:, 0] + 0.5 * X[:, 1]))) + rng.uniform(0, 0.3, n)
    treated = np.zeros(n, dtype=int)
    treated[np.argsort(-noisy)[:n_treat]] = 1
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_covs)])
    df["treatment"] = treated
    df["sex"] = rng.binomial(1, 0.5, n)
    return df


def _child(conn, n_treat, n_control, kwargs):
    from cohortmatch import match

    df = _cohort(n_treat, n_control)
    covs = [c for c in df.columns if c.startswith("x")]
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = match(df, treatment="treatment", covariates=covs, **kwargs)
    conn.send((time.perf_counter() - t0, len(r.pairs), _peak_gb(), r.config["engine"]))
    conn.close()


def _bench(n_treat, n_control, **kwargs):
    parent, child = mp.Pipe()
    proc = mp.get_context("fork").Process(
        target=_child, args=(child, n_treat, n_control, kwargs)
    )
    proc.start()
    elapsed, pairs, gb, algo = parent.recv()
    proc.join()
    return elapsed, pairs, gb, algo


def main() -> None:
    # round total cohort sizes at a realistic ~4% treated fraction;
    # 500k is UK Biobank scale
    quick = "--quick" in sys.argv
    shapes = [(2_000, 48_000), (4_000, 96_000)]  # 50k, 100k total
    if not quick:
        shapes.append((20_000, 480_000))  # 500k total (UK Biobank)

    designs = [
        ("propensity, caliper=auto", dict(caliper="auto")),
        ("Mahalanobis (no PS, no caliper)", dict(distance="mahalanobis")),
        ("propensity + exact on sex", dict(caliper="auto", exact="sex")),
    ]

    out: list[str] = []
    w = out.append
    w("# Benchmark report\n")
    w(
        f"cohortmatch matching time and peak memory, measured on "
        f"{platform.platform().split('-')[0]} / Python "
        f"{sys.version.split()[0]} / numpy {np.__version__}. Each case is a "
        "full `match()` call (propensity estimation included where used), run "
        "in its own process for a clean peak-memory reading. Shapes are round "
        "total cohort sizes (50k, 100k, 500k) at ~4% treated; 500k is UK "
        "Biobank scale. Regenerate with `python benchmarks/report.py`.\n"
    )

    w("## cohortmatch across cohort shapes\n")
    w(
        "The dense (exact) path peaks at roughly twice the distance matrix "
        "during matching (a working copy); `memory_limit_gb` bounds the "
        "matrix, so set it near a third of available RAM if you rely on the "
        "exact path. The approximate path holds no matrix; its memory stays "
        "flat and sub-gigabyte at 20k × 480k.\n"
    )
    w("| shape (treated × controls) | design | time | pairs | peak memory | path |")
    w("|---|---|---|---|---|---|")
    for n_t, n_c in shapes:
        label = f"{n_t // 1000}k × {n_c // 1000}k"
        for dlabel, kw in designs:
            elapsed, pairs, gb, algo = _bench(n_t, n_c, **kw)
            w(
                f"| {label} | {dlabel} | {elapsed:.1f} s | {pairs:,} | "
                f"{gb:.2f} GB | {algo} |"
            )
    w("")

    # --- competitor comparison (measured references) ------------------------
    comp_path = ROOT / "benchmarks" / "competitors.json"
    if comp_path.exists():
        c = json.loads(comp_path.read_text())
        w("## Same 20k × 480k cohort, other packages\n")
        w(
            f"Competitor packages, measured {c['measured']} on "
            f"{c['machine']}. Compare against the cohortmatch rows in the "
            "table above.\n"
        )
        w("| package | design | time | peak memory | notes |")
        w("|---|---|---|---|---|")
        for row in c["rows"]:
            w(
                f"| {row['package']} | {row['design']} | {row['time']} | "
                f"{row['memory']} | {row['notes']} |"
            )
        w("")

    (ROOT / "BENCHMARKS.md").write_text("\n".join(out) + "\n")
    print("wrote BENCHMARKS.md")


if __name__ == "__main__":
    main()
