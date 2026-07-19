#!/usr/bin/env python
"""psmpy comparison on the same synthetic cohorts (pip install psmpy)."""

import resource
import signal
import sys
import time

sys.path.insert(0, "benchmarks")
from benchmark_scale import make_cohort


def alarm(sig, frame):
    raise TimeoutError


signal.signal(signal.SIGALRM, alarm)

for n_t, n_c, cap in [(2_000, 38_000, 600), (20_000, 480_000, 600)]:
    data = make_cohort(n_t, n_c).reset_index(drop=True)
    data["id"] = data.index
    print(f"--- psmpy at {n_t // 1000}k x {n_c // 1000}k ---", flush=True)
    signal.alarm(cap)
    try:
        from psmpy import PsmPy

        t0 = time.perf_counter()
        psm = PsmPy(data, treatment="treatment", indx="id", exclude=["sex"])
        psm.logistic_ps(balance=False)
        t1 = time.perf_counter()
        psm.kdtree_matched(matcher="propensity_logit", replacement=False, caliper=None)
        t2 = time.perf_counter()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
        print(
            f"ps: {t1 - t0:.1f}s  match: {t2 - t1:.1f}s  peak_rss: {rss:.1f} GB",
            flush=True,
        )
    except TimeoutError:
        print(f"TIMEOUT after {cap}s", flush=True)
    finally:
        signal.alarm(0)
