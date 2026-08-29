"""Utility functions shared across metrics calculations."""

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logit

from cohortmatch.utils.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

    from cohortmatch.datatypes import MatcherConfig

logger = get_logger(__name__)


def get_caliper_for_matching(
    config: "MatcherConfig",
    propensity_scores: np.ndarray | None = None,
    distance_matrix: np.ndarray | None = None,
    data: "pd.DataFrame | None" = None,
    treat_mask: np.ndarray | None = None,
) -> float | None:
    """Resolve the caliper threshold from the configuration.

    Numeric values pass through unchanged. "auto" is only defined for
    propensity/logit calipers: `caliper_scale` (default 0.2) times the SD of
    the logit propensity score over the full sample -- MatchIt's standardized-
    caliper convention on the logit scale. Note this differs from Austin
    (2011), whose rule uses the pooled within-group SD; the two coincide only
    when the groups are not separated on the score. The threshold is in logit
    units and must be applied to logit-scale distances.

    Returns:
        Caliper value, or None if no caliper is configured.
    """
    caliper_method = config.caliper_method
    caliper_value = config.caliper_value

    if caliper_method is None or caliper_value is None:
        return None

    if isinstance(caliper_value, (int, float)):
        return float(caliper_value)

    if isinstance(caliper_value, str) and caliper_value.lower() == "auto":
        if caliper_method not in ("propensity", "logit"):
            raise ValueError(
                "'auto' calipers are only defined for propensity/logit metrics, "
                f"got caliper_method={caliper_method!r}"
            )
        if propensity_scores is None:
            raise ValueError(
                "Propensity scores are required for 'auto' propensity caliper."
            )

        ps_clipped = np.clip(propensity_scores, 1e-6, 1 - 1e-6)
        logit_ps_sd = np.std(logit(ps_clipped))
        auto_caliper = config.caliper_scale * logit_ps_sd
        logger.info(
            f"Auto caliper for '{caliper_method}': {auto_caliper:.4f} "
            f"({config.caliper_scale} * SD of logit propensity = {logit_ps_sd:.4f})"
        )
        return auto_caliper

    raise ValueError(
        f"Invalid caliper_value specification: {caliper_value}. "
        "Must be a numeric value, 'auto', or None."
    )
