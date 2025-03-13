"""E-values: sensitivity of an effect to unmeasured confounding.

The E-value is the minimum strength of association, on the risk-ratio
scale, that an unmeasured confounder would need with both treatment and
outcome to fully explain away an observed effect (VanderWeele & Ding 2017,
Ann Intern Med 167:268). It is a closed-form transformation of the
estimate, no model, no data.
"""

import numpy as np


def _to_risk_ratio(value: float, measure: str, rare_outcome: bool) -> float:
    if measure == "risk_ratio":
        return value
    if measure == "odds_ratio":
        # for common outcomes the OR overstates the RR; sqrt(OR) is the
        # standard approximation (VanderWeele & Ding)
        return value if rare_outcome else float(np.sqrt(value))
    if measure == "hazard_ratio":
        if rare_outcome:
            return value
        return float((1 - 0.5 ** np.sqrt(value)) / (1 - 0.5 ** np.sqrt(1 / value)))
    raise ValueError(
        f"measure must be 'risk_ratio', 'odds_ratio', or 'hazard_ratio', "
        f"got {measure!r}"
    )


def _e_value_of_rr(rr: float) -> float:
    if rr < 1:
        rr = 1 / rr
    if rr == 1:
        return 1.0
    return float(rr + np.sqrt(rr * (rr - 1)))


def e_value(
    estimate: float,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    *,
    measure: str = "risk_ratio",
    rare_outcome: bool = False,
) -> dict[str, float]:
    """E-value for an effect estimate and (optionally) its confidence limit.

    Args:
        estimate: The effect on a ratio scale (> 0).
        ci_lower: Lower confidence limit (optional).
        ci_upper: Upper confidence limit (optional).
        measure: "risk_ratio" (used directly), "odds_ratio", or
            "hazard_ratio" (both converted to the risk-ratio scale via the
            standard approximations unless `rare_outcome=True`).
        rare_outcome: If True, odds and hazard ratios are treated as risk
            ratios directly (valid when the outcome is rare).

    Returns:
        Dict with "e_value" for the point estimate and, when a confidence
        interval is given, "e_value_ci" for the limit closer to the null
        (1.0 when the interval contains the null).
    """
    if not estimate > 0:
        raise ValueError(f"estimate must be positive, got {estimate}")

    rr = _to_risk_ratio(float(estimate), measure, rare_outcome)
    out = {"e_value": _e_value_of_rr(rr)}

    if ci_lower is not None or ci_upper is not None:
        if ci_lower is None or ci_upper is None:
            raise ValueError("provide both ci_lower and ci_upper, or neither")
        if not 0 < ci_lower <= ci_upper:
            raise ValueError(f"invalid confidence interval [{ci_lower}, {ci_upper}]")
        if ci_lower <= 1 <= ci_upper:
            out["e_value_ci"] = 1.0
        else:
            limit = ci_lower if rr >= 1 else ci_upper
            out["e_value_ci"] = _e_value_of_rr(
                _to_risk_ratio(float(limit), measure, rare_outcome)
            )
    return out
