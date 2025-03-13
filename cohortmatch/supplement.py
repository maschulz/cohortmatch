"""Supplementary-material report: a self-contained Markdown record of a match.

Plain text, no dependencies: the resolved design specification, software
versions, sample flow, balance table, and a citable methods paragraph,
what a journal's supplementary material and a reproducibility audit both
need. Convert with pandoc if a PDF/Word supplement is required.
"""

import sys
from datetime import datetime, timezone
from importlib import metadata

import pandas as pd

_CORE_PACKAGES = [
    "cohortmatch",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
]


def _versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for pkg in _CORE_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def _md_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return float_fmt.format(v)
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _fmt_setting(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, tuple):
        return ", ".join(str(v) for v in value)
    return str(value)


def _user_covariates(settings: dict) -> list[str]:
    """Covariates as the user passed them (pre-encoding), from the settings."""
    return list(settings.get("covariates") or ())


def build_supplement(result, title: str | None = None) -> str:
    """Assemble the Markdown supplement for a MatchResult."""
    s = dict(result.config)
    counts = result.summary().counts
    internal = result._results
    versions = _versions()

    method = s.get("method", "?")
    is_strata = method in ("subclass", "cem")

    lines: list[str] = []
    lines.append(f"# {title or 'Matching supplement'}")
    lines.append("")
    lines.append(
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"by cohortmatch {versions['cohortmatch']}."
    )

    # --- software ----------------------------------------------------------
    lines.append("\n## Software\n")
    lines.append(
        "; ".join(f"{k} {v}" for k, v in versions.items())
        + f"; random_state = {_fmt_setting(s.get('random_state'))}."
    )

    # --- design ------------------------------------------------------------
    lines.append("\n## Design specification\n")
    spec = {k: v for k, v in s.items() if k != "encoded_covariates" and v is not None}
    if internal.resolved_caliper is not None:
        spec["caliper (resolved threshold)"] = f"{internal.resolved_caliper:.4g}"
    spec_df = pd.DataFrame(
        [(k, _fmt_setting(v)) for k, v in spec.items()],
        columns=["parameter", "value"],
    )
    lines.append(_md_table(spec_df))

    # --- sample flow (per group; the focal group drives the estimand) ------
    lines.append("\n## Sample flow\n")
    treatment_col = internal.config.treatment_col
    original = result.original_data
    disc = result.discarded
    n_disc_t = int((original.loc[disc, treatment_col] == 1).sum()) if len(disc) else 0
    n_disc_c = int((original.loc[disc, treatment_col] == 0).sum()) if len(disc) else 0
    elig_t = counts["n_treatment_orig"] - n_disc_t
    elig_c = counts["n_control_orig"] - n_disc_c

    flow = [
        f"- Input: {counts['n_treatment_orig']} treated, "
        f"{counts['n_control_orig']} control units.",
    ]
    if len(disc):
        flow.append(
            f"- Discarded outside common propensity support: {n_disc_t} treated, "
            f"{n_disc_c} control (ids recorded on the result)."
        )
        flow.append(f"- Eligible for matching: {elig_t} treated, {elig_c} control.")
    flow.append(
        f"- Matched: {counts['n_treatment_matched']} treated, "
        f"{counts['n_control_matched']} control units"
        + ("" if is_strata else f"; {counts['n_pairs']} matched pairs")
        + "."
    )
    focal_is_treated = result.estimand != "ATC".lower() and result.estimand != "atc"
    focal_label = "treated" if focal_is_treated else "control"
    elig_focal = elig_t if focal_is_treated else elig_c
    matched_focal = (
        counts["n_treatment_matched"]
        if focal_is_treated
        else counts["n_control_matched"]
    )
    unmatched_focal = elig_focal - matched_focal
    if unmatched_focal > 0:
        flow.append(
            f"- Eligible but unmatched {focal_label} units: {unmatched_focal}, "
            "the estimand refers to the matched population."
        )
    lines.extend(flow)

    # matched-vs-unmatched focal characteristics, when units were lost
    if unmatched_focal > 0 and not is_strata:
        focal_value = 1 if focal_is_treated else 0
        focal_mask = original[treatment_col] == focal_value
        if len(disc):
            focal_mask &= ~original.index.isin(disc)
        focal_ids = original.index[focal_mask]
        matched_ids = result.matched_data.index[
            result.matched_data[treatment_col] == focal_value
        ]
        unmatched_ids = focal_ids.difference(matched_ids)
        numeric_covs = [
            c
            for c in _user_covariates(s)
            if c in original.columns and pd.api.types.is_numeric_dtype(original[c])
        ]
        if numeric_covs and len(unmatched_ids):
            comp = pd.DataFrame(
                {
                    "covariate": numeric_covs,
                    f"matched {focal_label} (mean)": [
                        float(original.loc[matched_ids, c].mean()) for c in numeric_covs
                    ],
                    f"unmatched {focal_label} (mean)": [
                        float(original.loc[unmatched_ids, c].mean())
                        for c in numeric_covs
                    ],
                }
            )
            lines.append(
                f"\nCharacteristics of matched vs eligible-but-unmatched "
                f"{focal_label} units:\n"
            )
            lines.append(_md_table(comp))

    # --- balance -----------------------------------------------------------
    lines.append("\n## Covariate balance\n")
    t1 = result.table1()
    attrs = dict(t1.attrs)
    vr = result.balance()[["variable", "var_ratio_before", "var_ratio_after"]]
    t1 = t1.merge(vr, on="variable")
    lines.append(
        f"Group sizes: before matching {attrs.get('n_treated_before')} treated / "
        f"{attrs.get('n_control_before')} control; after matching "
        f"{attrs.get('n_treated_after')} treated / "
        f"{attrs.get('n_control_after')} control (weighted statistics).\n"
    )

    # binary variables: render means as n (%)
    def _binary_vars():
        out = set()
        for var in t1["variable"]:
            if var in original.columns:
                vals = original[var].dropna().unique()
                if set(vals) <= {0, 1}:
                    out.add(var)
            elif "=" in str(var):
                out.add(var)  # one-hot level indicator
        return out

    binary_vars = _binary_vars()
    group_n = {
        ("treated", "before"): attrs.get("n_treated_before"),
        ("control", "before"): attrs.get("n_control_before"),
        ("treated", "after"): attrs.get("n_treated_after"),
        ("control", "after"): attrs.get("n_control_after"),
    }
    for col in list(t1.columns):
        m = None
        for grp in ("treated", "control"):
            for when in ("before", "after"):
                if col == f"mean_{grp}_{when}":
                    m = (grp, when)
        if m is None:
            continue
        n_total = group_n[m]

        def render(row, col=col, n_total=n_total):
            v = row[col]
            if row["variable"] in binary_vars and n_total:
                return f"{round(v * n_total)} ({100 * v:.1f}%)"
            return f"{v:.3f}"

        t1[col] = t1.apply(render, axis=1)
        sd_col = col.replace("mean_", "sd_")
        if sd_col in t1.columns:
            t1[sd_col] = t1.apply(
                lambda r, c=sd_col: (
                    "-" if r["variable"] in binary_vars else f"{r[c]:.3f}"
                ),
                axis=1,
            )
    lines.append(_md_table(t1))
    bal_idx = internal.balance_index or {}
    rubin = result.rubin_statistics or {}
    lines.append(
        f"\nMean |SMD| {bal_idx.get('mean_smd_before', float('nan')):.3f} before "
        f"vs {bal_idx.get('mean_smd_after', float('nan')):.3f} after matching; "
        f"{rubin.get('pct_both_good', float('nan')):.0f}% of covariates satisfy "
        "Rubin's rules (|SMD| < 0.25, variance ratio in [0.5, 2]). SMDs are "
        "signed and standardized by the anchor group's standard deviation in "
        "the original sample (identical denominator before and after). Binary "
        "variables are shown as n (%)."
    )
    metrics = result.propensity_metrics or {}
    if metrics.get("auc") is not None:
        lines.append(
            f"\nPropensity model c-statistic (cross-validated): "
            f"{metrics['auc']:.3f}. See `plot_propensity()` for the overlap of "
            "score distributions."
        )

    # --- effects, if estimated --------------------------------------------
    e_value_lines = []
    if internal.effect_estimates is not None:
        eff = internal.effect_estimates
        level = (
            f"{100 * eff['confidence_level'].iloc[0]:.0f}%"
            if "confidence_level" in eff.columns
            else "95%"
        )
        lines.append(f"\n## Effect estimates ({level} confidence intervals)\n")
        eff_cols = [
            c
            for c in [
                "outcome",
                "effect",
                "measure",
                "standard_error",
                "ci_lower",
                "ci_upper",
                "p_value",
                "se_type",
            ]
            if c in eff.columns
        ]
        table = eff[eff_cols].copy()
        if "p_value" in table.columns:
            table["p_value"] = table["p_value"].map(
                lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
            )
        lines.append(_md_table(table))

        from cohortmatch.evalue import e_value as _e_value

        for _, row in eff.iterrows():
            if row.get("measure") in ("odds_ratio", "risk_ratio"):
                ev = _e_value(
                    row["effect"],
                    row["ci_lower"],
                    row["ci_upper"],
                    measure=row["measure"],
                )
                e_value_lines.append(
                    f"E-value for {row['outcome']} ({row['measure']}): "
                    f"{ev['e_value']:.2f} (confidence limit: "
                    f"{ev['e_value_ci']:.2f})."
                )
        if e_value_lines:
            lines.append(
                "\nSensitivity to unmeasured confounding (VanderWeele & Ding "
                "2017): " + " ".join(e_value_lines)
            )

    # --- methods paragraph -------------------------------------------------
    lines.append("\n## Methods text\n")
    lines.append(
        _methods_paragraph(
            result, s, counts, internal, unmatched_focal, n_disc_t, n_disc_c
        )
    )
    refs = [
        "Austin PC (2011). Optimal caliper widths for propensity-score "
        "matching. Pharmaceutical Statistics 10(2):150-161.",
        "Stuart EA (2010). Matching methods for causal inference: a review. "
        "Statistical Science 25(1):1-21.",
    ]
    if e_value_lines:
        refs.append(
            "VanderWeele TJ, Ding P (2017). Sensitivity analysis in "
            "observational research: introducing the E-value. Annals of "
            "Internal Medicine 167(4):268-274."
        )
    refs.append(
        "cohortmatch "
        + versions["cohortmatch"]
        + " (Schulz MA). https://github.com/maschulz/cohortmatch"
    )
    lines.append("\n### References\n")
    lines.extend(f"- {r}" for r in refs)
    return "\n".join(lines) + "\n"


def _methods_paragraph(
    result, s, counts, internal, unmatched_focal, n_disc_t, n_disc_c
) -> str:
    method = s.get("method")
    estimand = result.estimand.upper()
    covs = ", ".join(_user_covariates(s))

    if method == "subclass":
        design = (
            f"Units were stratified into {s.get('n_subclasses')} propensity-score "
            f"subclasses (quantiles of the {estimand} target group) and weighted "
            "by the marginal-mean convention"
        )
    elif method == "cem":
        design = (
            "Units were matched by coarsened exact matching (cells from binned "
            "covariates; cells lacking either group excluded) and weighted by "
            "the marginal-mean convention"
        )
    else:
        kind = "optimal" if method == "optimal" else "nearest-neighbor"
        replacement = "with" if s.get("replace") else "without"
        design = (
            f"Treated and control units were matched 1:{s.get('ratio', 1)} "
            f"({replacement} replacement) by {kind} matching on the "
            f"{s.get('distance')} distance over {covs}"
        )
        if internal.resolved_caliper is not None:
            design += (
                f" within a caliper of {internal.resolved_caliper:.4g} "
                f"({_caliper_description(s)})"
            )
        if s.get("exact"):
            design += f", with exact agreement on {_fmt_setting(s['exact'])}"
        if s.get("covariate_calipers"):
            cc = ", ".join(
                f"{k} within {v}" for k, v in s["covariate_calipers"].items()
            )
            design += f", requiring {cc}"

    ps_sentence = ""
    if result.propensity_scores is not None:
        ps_sentence = (
            " Propensity scores were estimated by L2-regularized logistic "
            f"regression (scikit-learn defaults) with {s.get('cv', 5)}-fold "
            "cross-fitting, so each unit was scored by a model not fitted "
            "to it."
        )

    discard_sentence = ""
    if n_disc_t or n_disc_c:
        discard_sentence = (
            f" Before matching, {n_disc_t} treated and {n_disc_c} control "
            "unit(s) outside the region of common propensity support were "
            "excluded; the estimand therefore refers to the population within "
            "common support."
        )

    focal_label = "treated" if result.estimand != "atc" else "control"
    population_sentence = (
        f" Of the eligible {focal_label} units, "
        f"{counts['n_treatment_matched'] if focal_label == 'treated' else counts['n_control_matched']} "
        "were matched"
        + (
            f" and {unmatched_focal} could not be matched under the specified "
            "constraints; estimates refer to the matched population."
            if unmatched_focal > 0
            else "."
        )
    )

    return (
        f"{design} (estimand: {estimand}).{ps_sentence}{discard_sentence}"
        f"{population_sentence} Covariates were required to be complete; "
        "no missing-data imputation was performed by the matching software. "
        "Covariate balance was assessed with signed standardized mean "
        "differences (denominator: anchor-group standard deviation in the "
        "original sample). Analyses of the matched sample used the matching "
        "weights with cluster-robust standard errors on match groups. "
        "Matching was performed with cohortmatch "
        f"{_versions()['cohortmatch']} (see References)."
    )


def _caliper_description(s) -> str:
    if s.get("caliper") == "auto":
        return "0.2 SD of the logit propensity score, Austin 2011"
    if s.get("std_caliper") in (None, True) and s.get("caliper_metric") in (
        "logit",
        "propensity",
        None,
    ):
        return f"{s.get('caliper')} SD of the logit propensity score"
    return f"raw units on the {s.get('caliper_metric')} metric"


def build_risk_set_supplement(
    result,
    title: str | None = None,
    exposures=None,
    adjustment_covariates: list[str] | None = None,
) -> str:
    """Markdown supplement for a nested case-control (risk-set) study."""
    from datetime import datetime, timezone

    s = dict(result.config)
    summ = result.summary()
    versions = _versions()

    lines: list[str] = [f"# {title or 'Nested case-control supplement'}", ""]
    lines.append(
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"by cohortmatch {versions['cohortmatch']}."
    )

    lines.append("\n## Software\n")
    lines.append(
        "; ".join(f"{k} {v}" for k, v in versions.items())
        + f"; random_state = {_fmt_setting(s.get('random_state'))}."
    )

    lines.append("\n## Design specification\n")
    spec = {k: v for k, v in s.items() if v is not None}
    spec_df = pd.DataFrame(
        [(k, _fmt_setting(v)) for k, v in spec.items()],
        columns=["parameter", "value"],
    )
    lines.append(_md_table(spec_df))

    lines.append("\n## Matched sets\n")
    lines.append(
        f"- {summ.n_cases} case(s); {summ.n_sets} matched set(s) formed.\n"
        f"- {summ.n_controls} control selection(s); {summ.n_reused} control(s) "
        f"served more than one set.\n"
        f"- {summ.n_short} case(s) had a short or empty eligible pool."
    )

    lines.append("\n## Covariate balance (cases vs matched controls)\n")
    try:
        bal = result.balance()
        lines.append(_md_table(bal))
        lines.append(
            "\nSigned SMDs standardized by the case group's SD in the original "
            "cohort. Exact-matched factors are balanced by construction."
        )
    except ValueError as e:
        lines.append(f"*(not available: {e})*")

    e_lines = []
    if exposures is not None:
        lines.append("\n## Odds ratios (conditional logistic)\n")
        orr = result.estimate_odds_ratio(
            exposures, adjustment_covariates=adjustment_covariates
        )
        show = orr[["exposure", "odds_ratio", "ci_lower", "ci_upper", "p_value"]].copy()
        show["p_value"] = show["p_value"].map(
            lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
        )
        lines.append(_md_table(show))
        from cohortmatch.evalue import e_value as _e_value

        for _, row in orr.iterrows():
            ev = _e_value(
                row["odds_ratio"],
                row["ci_lower"],
                row["ci_upper"],
                measure="odds_ratio",
            )
            e_lines.append(
                f"E-value for {row['exposure']}: {ev['e_value']:.2f} "
                f"(confidence limit: {ev['e_value_ci']:.2f})."
            )
        lines.append(
            "\nUnder incidence-density sampling the odds ratio estimates the "
            "hazard ratio. Sensitivity to unmeasured confounding (VanderWeele & "
            "Ding 2017): " + " ".join(e_lines)
        )

    lines.append("\n## Methods text\n")
    ratio = s.get("ratio", 1)
    reuse = s.get("replace", True)
    selection = (
        "at random" if not s.get("covariates") else "by nearest covariate distance"
    )
    factors = (
        ", ".join(
            list(s.get("exact") or [])
            + list((s.get("covariate_calipers") or {}).keys())
        )
        or "none"
    )
    lines.append(
        f"Incident cases were matched to {ratio} control(s) each by risk-set "
        f"(incidence-density) sampling: for every case, controls were drawn "
        f"{selection} from cohort members still at risk at the case's event "
        f"time, matched on {factors}"
        + (", with controls eligible for reuse across sets" if reuse else "")
        + ". Future cases were eligible as controls, so the conditional-"
        "logistic odds ratio estimates the hazard ratio (Langholz & Goldstein "
        "1996). Covariates were required complete; no imputation was performed "
        "by the matching software. Matching used cohortmatch "
        f"{versions['cohortmatch']} (see References)."
    )
    refs = [
        "Langholz B, Goldstein L (1996). Risk set sampling in epidemiologic "
        "cohort studies. Statistical Science 11(1):35-53.",
    ]
    if exposures is not None:
        refs.append(
            "VanderWeele TJ, Ding P (2017). Sensitivity analysis in "
            "observational research: introducing the E-value. Annals of "
            "Internal Medicine 167(4):268-274."
        )
    refs.append(
        "cohortmatch "
        + versions["cohortmatch"]
        + " (Schulz MA). https://github.com/maschulz/cohortmatch"
    )
    lines.append("\n### References\n")
    lines.extend(f"- {r}" for r in refs)
    return "\n".join(lines) + "\n"
