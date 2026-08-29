"""Risk-set (incidence-density) matching for nested case-control designs.

For each case, controls are drawn from the units still at risk at the case's
event time, including units that later become cases themselves, which is
what makes odds ratios from this design estimate the hazard ratio (Langholz
& Goldstein, 1996). Conventions:

- The risk set at case time t contains units with event_time strictly
  greater than t (the case itself and same-time failures are excluded).
- A unit may serve as control for several cases (`replace=True`,
  the classic design) or at most once (`replace=False`).
- Within a risk set, controls are the nearest by covariate distance when
  `covariates` are given, or sampled at random otherwise; `exact` columns
  and `covariate_calipers` restrict eligibility either way.
- Matched controls inherit the case's index time; the result is a
  long-format table of matched sets for conditional logistic regression.
"""

import warnings
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from cohortmatch.exceptions import IncompleteMatchWarning
from cohortmatch.utils.logging import get_logger

logger = get_logger(__name__)


def match_risk_set(
    data: pd.DataFrame,
    *,
    event_time: str,
    event: str,
    covariates: list[str] | None = None,
    ratio: int = 1,
    exact: str | list[str] | None = None,
    covariate_calipers: dict[str, float] | None = None,
    replace: bool = True,
    random_state: int | None = None,
) -> "RiskSetResult":
    """Match controls to cases from each case's risk set.

    Args:
        data: One row per cohort member; the index identifies units.
        event_time: Column with time of event or censoring.
        event: Binary column, 1 for cases.
        covariates: Columns for nearest-neighbor selection within the risk
            set (standardized Euclidean distance). None selects at random.
        ratio: Controls per case (fewer when the eligible pool is smaller).
        exact: Column(s) that must match the case exactly.
        covariate_calipers: Per-variable maximum absolute difference from
            the case, raw units (e.g. {"age": 3.0}).
        replace: Whether a control may serve in several matched sets
            (sampling with replacement, the classic incidence-density
            design; default True). Same concept as `replace` in `match()`.
        random_state: Seed for random selection and tie-breaking.

    Returns:
        RiskSetResult with the matched sets.

    Raises:
        ValueError: On invalid arguments.

    Example:
        >>> result = match_risk_set(
        ...     cohort, event_time="follow_up", event="diagnosed", exact="sex", ratio=4
        ... )  # doctest: +SKIP
        >>> result.estimate_odds_ratio("exposure")  # doctest: +SKIP
    """
    if event_time not in data.columns:
        raise ValueError(f"event_time column '{event_time}' not in data")
    if event not in data.columns:
        raise ValueError(f"event column '{event}' not in data")
    if not data.index.is_unique:
        raise ValueError("DataFrame index must be unique")
    if data[event].isna().any():
        raise ValueError(f"event column '{event}' contains missing values")
    values = set(pd.unique(data[event]))
    if not values <= {0, 1}:
        raise ValueError(
            f"event column must be binary 0/1, got values {sorted(values)}"
        )
    time_vals = data[event_time].to_numpy(dtype=float)
    if not np.isfinite(time_vals).all():
        raise ValueError(
            f"event_time column '{event_time}' contains missing or non-finite values"
        )
    if isinstance(ratio, float) and not ratio.is_integer():
        raise ValueError(f"ratio must be an integer, got {ratio}")
    ratio = int(ratio)
    if ratio < 1:
        raise ValueError(f"ratio must be >= 1, got {ratio}")
    if isinstance(exact, str):
        exact = [exact]
    for col in exact or []:
        if col not in data.columns:
            raise ValueError(f"exact column '{col}' not in data")
    for col, value in (covariate_calipers or {}).items():
        if col not in data.columns:
            raise ValueError(f"covariate_calipers column '{col}' not in data")
        if not np.issubdtype(np.asarray(data[col]).dtype, np.number):
            raise ValueError(f"covariate_calipers column '{col}' must be numeric")
        if not value > 0:
            raise ValueError(f"covariate_calipers['{col}'] must be positive")
    for col in covariates or []:
        if col not in data.columns:
            raise ValueError(f"covariate column '{col}' not in data")
        if data[col].isna().any():
            raise ValueError(
                f"covariate column '{col}' contains missing values; "
                "handle them before matching"
            )

    if covariates:
        warnings.warn(
            "Nearest-neighbor selection within risk sets is not simple random "
            "sampling: selection probabilities are unknown, which can bias the "
            "odds ratio toward the null (overmatching) when the selection "
            "covariates correlate with the exposure. Prefer restricting "
            "eligibility via exact/covariate_calipers and sampling at random "
            "(covariates=None), and adjust for any selection covariates in "
            "estimate_odds_ratio(adjustment_covariates=...).",
            UserWarning,
            stacklevel=2,
        )

    if int((data[event] == 1).sum()) == 0:
        raise ValueError(f"no cases (event == 1) in '{event}'; nothing to match")

    rng = np.random.RandomState(random_state)

    times = data[event_time].to_numpy(dtype=float)
    is_case = data[event].to_numpy() == 1
    ids = data.index.to_numpy()
    n = len(data)

    # covariate matrix, standardized once on the full sample
    X = None
    if covariates:
        X = data[covariates].to_numpy(dtype=float)
        mean, std = X.mean(axis=0), X.std(axis=0)
        std[std < 1e-10] = 1.0
        X = (X - mean) / std

    exact_keys = None
    if exact:
        exact_keys = pd.factorize(
            pd.MultiIndex.from_frame(data[exact]).to_flat_index()
        )[0]

    caliper_cols = {
        col: data[col].to_numpy(dtype=float) for col in (covariate_calipers or {})
    }

    # process cases in ascending event time; risk set = strictly later times
    order = np.argsort(times, kind="stable")
    sorted_times = times[order]

    case_positions = np.where(is_case)[0]
    case_order = case_positions[np.argsort(times[case_positions], kind="stable")]

    used = np.zeros(n, dtype=bool)  # only consulted when replace=False
    rows: list[dict[str, Any]] = []
    n_short = 0

    for case_pos in case_order:
        t = times[case_pos]
        start = np.searchsorted(sorted_times, t, side="right")
        pool = order[start:]  # positions with event_time > t
        if len(pool) == 0:
            n_short += 1
            continue

        eligible = np.ones(len(pool), dtype=bool)
        if exact_keys is not None:
            eligible &= exact_keys[pool] == exact_keys[case_pos]
        for col, threshold in (covariate_calipers or {}).items():
            eligible &= (
                np.abs(caliper_cols[col][pool] - caliper_cols[col][case_pos])
                <= threshold
            )
        if not replace:
            eligible &= ~used[pool]

        candidates = pool[eligible]
        if len(candidates) == 0:
            n_short += 1
            continue

        k = min(ratio, len(candidates))
        if X is not None:
            dist = np.sqrt(((X[candidates] - X[case_pos]) ** 2).sum(axis=1))
            # random tie-breaking, then k nearest
            jitter = rng.permutation(len(candidates))
            chosen = candidates[jitter[np.argsort(dist[jitter], kind="stable")][:k]]
        else:
            chosen = rng.choice(candidates, size=k, replace=False)

        if k < ratio:
            n_short += 1
        if not replace:
            used[chosen] = True

        rows.append(
            {
                "set_id": ids[case_pos],
                "unit_id": ids[case_pos],
                "case": 1,
                "index_time": t,
            }
        )
        for c in chosen:
            rows.append(
                {"set_id": ids[case_pos], "unit_id": ids[c], "case": 0, "index_time": t}
            )

    sets = pd.DataFrame(rows, columns=["set_id", "unit_id", "case", "index_time"])
    n_cases = int(is_case.sum())
    n_cases_matched = int((sets["case"] == 1).sum()) if len(sets) else 0
    if n_cases_matched < n_cases or n_short:
        warnings.warn(
            f"{n_cases - n_cases_matched} of {n_cases} case(s) received no "
            f"controls and {n_short} case(s) had a short or empty eligible "
            "pool; the matched sets cover a subset of the case series.",
            IncompleteMatchWarning,
            stacklevel=2,
        )

    settings = {
        "event_time": event_time,
        "event": event,
        "covariates": tuple(covariates) if covariates else None,
        "ratio": ratio,
        "exact": tuple(exact) if exact else None,
        "covariate_calipers": covariate_calipers,
        "replace": replace,
        "random_state": random_state,
    }
    return RiskSetResult(
        sets=sets,
        data=data,
        n_cases=n_cases,
        n_short=n_short,
        event_time=event_time,
        event=event,
        settings=settings,
    )


class RiskSetSummary:
    """Printable summary of a RiskSetResult."""

    def __init__(self, result: "RiskSetResult"):
        sets = result.sets
        self.n_cases = result.n_cases
        self.n_sets = int(sets["set_id"].nunique()) if len(sets) else 0
        self.n_controls = int((sets["case"] == 0).sum()) if len(sets) else 0
        self.n_short = result.n_short
        if self.n_controls:
            counts = sets.loc[sets["case"] == 0, "unit_id"].value_counts()
            self.n_reused = int((counts > 1).sum())
        else:
            self.n_reused = 0

    def __repr__(self) -> str:
        return (
            f"RiskSetResult: {self.n_sets} matched sets of {self.n_cases} cases; "
            f"{self.n_controls} control selections "
            f"({self.n_reused} units serve multiple sets); "
            f"{self.n_short} case(s) with short pools"
        )

    def _repr_html_(self) -> str:
        import html

        return f"<pre style='font-family:monospace;font-size:12px'>{html.escape(repr(self))}</pre>"


class RiskSetResult:
    """Matched sets from risk-set sampling."""

    def __init__(
        self,
        sets: pd.DataFrame,
        data: pd.DataFrame,
        n_cases: int,
        n_short: int,
        event_time: str,
        event: str,
        settings: dict[str, Any] | None = None,
    ):
        self.sets = sets
        self._data = data
        self.n_cases = n_cases
        self.n_short = n_short
        self._event_time = event_time
        self._event = event
        self._settings = settings or {}

    @property
    def config(self) -> MappingProxyType:
        """The resolved settings of the match_risk_set() call (read-only)."""
        return MappingProxyType(self._settings)

    @property
    def original_data(self) -> pd.DataFrame:
        """The input cohort."""
        return self._data

    @property
    def matched_data(self) -> pd.DataFrame:
        """Long format: one row per set membership, joined with the cohort data.

        A unit appears once per matched set it belongs to; `set_id`, `case`,
        and `index_time` describe its role. Cohort columns whose names clash
        with these design columns get a "_cohort" suffix. Suitable for
        conditional logistic regression grouped by `set_id`.
        """
        return self.sets.merge(
            self._data,
            left_on="unit_id",
            right_index=True,
            how="left",
            suffixes=("", "_cohort"),
        )

    def estimate_odds_ratio(
        self,
        exposures: str | list[str],
        *,
        adjustment_covariates: list[str] | None = None,
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        """Conditional logistic regression of case status on exposure(s).

        Under incidence-density sampling with random control selection the odds
        ratio estimates the hazard ratio of the exposure. This does not hold
        cleanly when controls are chosen non-randomly (nearest-neighbor
        selection or continuous-caliper restriction); adjust for any selection
        covariates via `adjustment_covariates` in that case. One model is fit
        per exposure.

        Args:
            exposures: Exposure column(s); must be numeric.
            adjustment_covariates: Additional numeric covariates entering
                each model (confounders beyond the matching factors).
            confidence_level: Level for the confidence intervals.

        Returns:
            DataFrame with one row per exposure (odds ratio, CI, p-value).
        """
        try:
            from statsmodels.discrete.conditional_models import ConditionalLogit
        except ImportError:
            raise ImportError(
                "statsmodels is required for estimate_odds_ratio"
            ) from None

        if isinstance(exposures, str):
            exposures = [exposures]
        long = self.matched_data
        adjustment_covariates = list(adjustment_covariates or [])
        for col in exposures + adjustment_covariates:
            if col not in long.columns:
                raise ValueError(f"column '{col}' not in data")

        alpha = 1 - confidence_level
        rows = []
        for exposure in exposures:
            cols = [exposure] + adjustment_covariates
            model = ConditionalLogit(
                long["case"].to_numpy(),
                long[cols].to_numpy(dtype=float),
                groups=long["set_id"].to_numpy(),
            )
            fit = model.fit(disp=False)
            ci = fit.conf_int(alpha=alpha)
            rows.append(
                {
                    "exposure": exposure,
                    "odds_ratio": float(np.exp(fit.params[0])),
                    "ci_lower": float(np.exp(ci[0][0])),
                    "ci_upper": float(np.exp(ci[0][1])),
                    "log_or": float(fit.params[0]),
                    "standard_error": float(fit.bse[0]),
                    "p_value": float(fit.pvalues[0]),
                }
            )
        return pd.DataFrame(rows)

    def _assessment_covariates(self, covariates: list[str] | None) -> list[str]:
        """Numeric covariates to assess balance on (matching factors by default)."""
        if covariates is not None:
            return list(covariates)
        s = self._settings
        cols: list[str] = []
        for group in (s.get("covariates"), s.get("covariate_calipers"), s.get("exact")):
            for c in group or []:
                if c not in cols:
                    cols.append(c)
        return [
            c
            for c in cols
            if c in self._data.columns and pd.api.types.is_numeric_dtype(self._data[c])
        ]

    def _before_after(self, covariates: list[str] | None):
        """The full cohort (before) and the matched sets (after), case-labelled."""
        covs = self._assessment_covariates(covariates)
        if not covs:
            raise ValueError(
                "No numeric matching covariates to assess; pass covariates=[...]."
            )
        before = self._data.copy()
        before["case"] = (before[self._event] == 1).astype(int)
        after = self.matched_data
        return before, after, covs

    def balance(self, covariates: list[str] | None = None) -> pd.DataFrame:
        """Balance of cases vs matched controls (signed SMDs, cobalt conventions).

        "before" compares cases to the full non-case cohort; "after" compares
        cases to their sampled controls within the matched sets. Covariates
        default to the numeric matching factors.
        """
        from cohortmatch.metrics.balance import calculate_balance_stats

        before, after, covs = self._before_after(covariates)
        return calculate_balance_stats(
            data=before,
            matched_data=after,
            covariates=covs,
            treatment_col="case",
            anchor="treatment",
        )

    def table1(self, covariates: list[str] | None = None) -> pd.DataFrame:
        """Table 1 for the matched sets: case/control means and SDs with SMDs."""
        from cohortmatch.metrics.balance import create_table1

        before, after, covs = self._before_after(covariates)
        return create_table1(
            data=before,
            matched_data=after,
            covariates=covs,
            treatment_col="case",
            anchor="treatment",
            group_labels=("case", "control"),
        )

    def supplement(
        self,
        path: str | None = None,
        *,
        title: str | None = None,
        exposures: str | list[str] | None = None,
        adjustment_covariates: list[str] | None = None,
    ) -> str:
        """Markdown supplement for a nested case-control study.

        Records the design, set counts, case/control balance, and, when
        `exposures` are given, the conditional-logistic odds ratios with
        E-values. Writes to `path` if provided; returns the Markdown.
        """
        from cohortmatch.supplement import build_risk_set_supplement

        text = build_risk_set_supplement(
            self,
            title=title,
            exposures=exposures,
            adjustment_covariates=adjustment_covariates,
        )
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text

    def summary(self) -> "RiskSetSummary":
        """Summary of set counts and pool shortfalls."""
        return RiskSetSummary(self)

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"RiskSetResult(n_sets={s.n_sets}, n_cases={self.n_cases}, "
            f"n_controls={s.n_controls})"
        )

    def _repr_html_(self) -> str:
        import html

        return (
            f"<pre style='font-family:monospace;font-size:12px'>"
            f"{html.escape(repr(self.summary()))}</pre>"
            "<div style='font-size:11px;color:#666'>"
            "methods: balance(), table1(), estimate_odds_ratio(), supplement()"
            "</div>"
        )
