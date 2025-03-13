"""Datatypes for CohortMatch.

This module defines the data structures used by the CohortMatch package,
including the configuration settings and result container.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MatcherConfig:
    """Unified configuration for CohortMatcher with flattened parameters."""

    # Core parameters
    treatment_col: str
    covariates: list[str]

    # Matching parameters
    match_method: str = "greedy"  # "greedy", "optimal", "fast_greedy"
    distance_method: str = (
        "propensity"  # "euclidean", "mahalanobis", "propensity", "logit"
    )
    exact_match_cols: list[str] = field(default_factory=list)
    standardize: bool = True
    caliper_method: str | None = (
        "propensity"  # Metric for the caliper ('propensity', 'logit', 'mahalanobis', a covariate name, or None).
    )
    caliper_value: float | str | None = (
        "auto"  # Threshold for the caliper. Can be a numeric value, 'auto', or None.
    )
    caliper_scale: float = (
        0.2  # SD-of-logit(ps) multiplier for 'auto' propensity/logit calipers
    )
    fast_prefilter_caliper_scale: float = 0.5  # For fast_greedy, scales the SD of logit(propensity) to create the initial candidate search caliper.
    replace: bool = False  # Whether to allow replacement. Applies to 'greedy', 'optimal', and 'fast_greedy' methods.
    ratio: float = 1.0
    random_state: int | None = None
    weights: dict[str, float] | None = None
    matching_direction: str = "treatment"  # anchor group: "treatment" or "control"
    m_order: str | None = (
        None  # None (scarcity-first dense / data-order fast), "largest", "smallest", "closest", "random", "data"
    )
    covariate_calipers: dict[str, float] | None = (
        None  # per-variable max absolute difference, raw units
    )
    discard: str | None = (
        None  # drop units outside common propensity support: "treated", "control", "both"
    )
    estimand: str = "att"  # "att", "atc"; "ate" for stratum methods
    n_subclasses: int = 6  # strata for match_method="subclass"
    cem_coarsening: dict | None = (
        None  # per-covariate bin count or edges for match_method="cem"
    )

    # Propensity parameters
    estimate_propensity: bool = False
    propensity_col: str | None = None
    propensity_model: str = (
        "logistic"  # "logistic", "random_forest", "xgboost", "gbm", "custom"
    )
    model_params: dict[str, Any] = field(default_factory=dict)
    cv_folds: int = 5


@dataclass
class MatchResults:
    """Container for all matching results.

    This class stores the results of a matching operation, including the original data,
    matched data, and matching pairs. It provides methods for retrieving matching information
    and summarizing the results.

    Attributes:
        original_data: The original DataFrame before matching
        matched_data: DataFrame containing only the matched units
        pairs: List of tuples (treatment_id, control_id) representing matched pairs
        match_groups: Dictionary mapping treatment IDs to lists of control IDs
        match_distances: List of distances for each matched pair

    """

    # Original and matched data
    original_data: pd.DataFrame
    matched_data: pd.DataFrame

    # Matching results as pairs of participant IDs
    # Each tuple is (treatment_id, control_id)
    pairs: list[tuple[Any, Any]]

    # Dictionary mapping treatment IDs to lists of control IDs
    # This is particularly useful for ratio matching and efficient lookups
    match_groups: dict[Any, list[Any]]

    # Distances for each matched pair, in the same order as pairs
    match_distances: list[float]

    # Optional distance matrix for debugging/visualization, with the anchor
    # (row) and pool (column) unit ids in matrix order (post-discard)
    distance_matrix: np.ndarray | None = None
    dm_anchor_ids: list | None = None
    dm_pool_ids: list | None = None

    # Propensity score results
    propensity_scores: np.ndarray | None = None
    propensity_model: Any | None = None
    propensity_metrics: dict[str, float] | None = None

    # Balance assessment results
    balance_statistics: pd.DataFrame | None = None
    rubin_statistics: dict[str, float] | None = None
    balance_index: dict[str, float] | None = None

    # Matching weights (anchors 1; partners scaled 1/k accumulations),
    # subclass membership (anchor id per unit; None with replacement),
    # and which group anchored the matching ("treatment" or "control")
    weights: pd.Series | None = None
    subclass: pd.Series | None = None
    anchor: str = "treatment"

    # Units dropped by common-support discard (empty Index when discard off)
    discarded: pd.Index | None = None

    # The numeric caliper threshold actually applied (None when no caliper)
    resolved_caliper: float | None = None

    # Treatment effect results
    effect_estimates: pd.DataFrame | None = None

    # Configuration used
    config: MatcherConfig = None

    def get_match_summary(self) -> dict[str, int | float]:
        """Get summary statistics about the matching.

        Returns:
            Dictionary with match summary statistics

        """
        treatment_col = self.config.treatment_col
        result = {
            "n_treatment_orig": (self.original_data[treatment_col] == 1).sum(),
            "n_control_orig": (self.original_data[treatment_col] == 0).sum(),
            "n_treatment_matched": (self.matched_data[treatment_col] == 1).sum(),
            "n_control_matched": (self.matched_data[treatment_col] == 0).sum(),
            "n_pairs": len(self.pairs),
            "n_match_groups": len(self.match_groups),
        }

        # Calculate match ratio
        if result["n_treatment_matched"] > 0:
            result["match_ratio"] = (
                result["n_control_matched"] / result["n_treatment_matched"]
            )
        else:
            result["match_ratio"] = 0

        return result

    def get_match_pairs(self) -> pd.DataFrame:
        """Get detailed matching information as a DataFrame.

        This method converts the internal pairs representation into a DataFrame
        with treatment_id and control_id columns, suitable for analysis and export.

        Returns:
            DataFrame with columns 'treatment_id' and 'control_id'

        """
        if not self.pairs:
            return pd.DataFrame(columns=["treatment_id", "control_id"])

        rows = []
        for t_id, c_id in self.pairs:
            rows.append({"treatment_id": t_id, "control_id": c_id})

        return pd.DataFrame(rows)
