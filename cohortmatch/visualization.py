"""Diagnostic plots for matching results.

The four plots that appear in papers: covariate balance (bar and love plot),
propensity-score overlap before/after matching, and the matched-pair
distance distribution. Requires the viz extra (matplotlib).
"""

from typing import TYPE_CHECKING

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as _err:
    raise ImportError(
        "Plotting requires the viz extras: pip install 'cohortmatch[viz]'"
    ) from _err

from cohortmatch.utils.logging import get_logger

if TYPE_CHECKING:
    from cohortmatch.datatypes import MatchResults

logger = get_logger(__name__)

# color-blind friendly palette (Okabe-Ito)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]

__all__ = [
    "plot_balance",
    "plot_love_plot",
    "plot_matched_pairs_distance",
    "plot_propensity_comparison",
]


def plot_balance(
    results: "MatchResults", max_vars: int = 20, figsize: tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Plot covariate balance before and after matching.

    Creates a bar chart showing standardized mean differences (SMD) before and after matching
    for covariates. Includes reference lines at 0.1 and 0.2 thresholds. Variables are sorted
    by pre-matching SMD with the largest imbalances at the top.

    Args:
        results: MatchResults object containing matching results
        max_vars: Maximum number of variables to plot (default: 20)
        figsize: Figure size (width, height) in inches

    Returns:
        Matplotlib figure object

    """
    logger.debug(f"Creating balance plot with up to {max_vars} variables")

    # Extract balance statistics
    balance_stats = results.balance_statistics
    if balance_stats is None:
        raise ValueError("No balance statistics available to plot")
    balance_stats = balance_stats.assign(
        smd_before=balance_stats["smd_before"].abs(),
        smd_after=balance_stats["smd_after"].abs(),
    )
    if balance_stats is None:
        logger.warning("No balance statistics available in MatchResults object")
        # Create empty figure with message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No balance statistics available",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.set_axis_off()
        return fig

    # Focus on SMD columns and get variable names
    smd_cols = ["smd_before", "smd_after"]
    if "smd_after" not in balance_stats.columns:
        smd_cols = ["smd_before"]
        logger.debug("Only pre-matching SMD values found")

    # Sort by pre-matching SMD and take top variables
    sorted_stats = balance_stats.sort_values("smd_before", ascending=False)

    # Use standard column for variable names
    var_col = "variable"

    # Take top N variables
    top_stats = sorted_stats.head(max_vars)
    logger.debug(f"Plotting balance for {len(top_stats)} variables")

    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)

    # For multiple series (before and after)
    if len(smd_cols) > 1:
        palette = PALETTE

        # Plot bars
        bar_width = 0.4
        positions1 = np.arange(len(top_stats))
        positions2 = positions1 + bar_width

        # Pre-matching bars
        ax.barh(
            positions1,
            top_stats["smd_before"],
            height=bar_width,
            label="Before matching",
            color=palette[0],
            alpha=0.7,
        )

        # Post-matching bars
        ax.barh(
            positions2,
            top_stats["smd_after"],
            height=bar_width,
            label="After matching",
            color=palette[1],
            alpha=0.7,
        )

        # Set y-tick positions and labels
        ax.set_yticks(positions1 + bar_width / 2)
        ax.set_yticklabels(top_stats[var_col])
    else:
        # Single series (only before matching)
        ax.barh(
            top_stats[var_col], top_stats["smd_before"], color="steelblue", alpha=0.7
        )

    # Add reference lines
    ax.axvline(0.1, color="darkred", linestyle="--", alpha=0.7, label="0.1 threshold")
    ax.axvline(0.2, color="darkred", linestyle=":", alpha=0.7, label="0.2 threshold")

    # Set labels and title
    ax.set_xlabel("Standardized Mean Difference")
    ax.set_title("Covariate Balance")

    # Add legend if we have both before and after
    if len(smd_cols) > 1:
        ax.legend(loc="best")

    # Adjust layout to make room for variable names
    plt.tight_layout()
    logger.debug("Balance plot created successfully")

    return fig


def plot_love_plot(
    results: "MatchResults", threshold: float = 0.1, figsize: tuple[int, int] = (10, 12)
) -> plt.Figure:
    """Create a Love plot showing standardized mean differences.

    Displays a dot plot where each row represents a covariate, with points showing the
    standardized mean difference before and after matching. Lines connect the before/after
    points for each variable. A vertical reference line indicates the balance threshold.
    Variables are alphabetically sorted.

    Args:
        results: MatchResults object containing matching results
        threshold: Threshold line for acceptable balance
        figsize: Figure size

    Returns:
        Matplotlib figure

    """
    # Extract balance statistics from results
    balance_statistics = results.balance_statistics
    if balance_statistics is None:
        raise ValueError("No balance statistics available to plot")
    balance_statistics = balance_statistics.assign(
        smd_before=balance_statistics["smd_before"].abs(),
        smd_after=balance_statistics["smd_after"].abs(),
    )
    if balance_statistics is None:
        raise ValueError("Balance statistics are not available in the results.")

    # Sort by variable name
    sorted_df = balance_statistics.sort_values("variable")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot data
    y = np.arange(len(sorted_df))

    ax.scatter(
        sorted_df["smd_before"], y, label="Before Matching", marker="o", s=50, alpha=0.7
    )
    ax.scatter(
        sorted_df["smd_after"], y, label="After Matching", marker="x", s=50, alpha=0.7
    )

    # Add connecting lines
    for i, (before, after) in enumerate(
        zip(sorted_df["smd_before"], sorted_df["smd_after"], strict=False)
    ):
        ax.plot([before, after], [i, i], "k-", alpha=0.3)

    # Add reference line
    ax.axvline(
        x=threshold,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"{threshold} Threshold",
    )

    # Customize plot
    ax.set_xlabel("Standardized Mean Difference")
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_df["variable"])
    ax.set_title("Love Plot: Standardized Mean Differences")
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_propensity_comparison(
    results: "MatchResults", bins: int = 30, figsize: tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Compare propensity score distributions before and after matching.

    Creates a side-by-side comparison of propensity score distributions before matching (left)
    and after matching (right). Each subplot shows overlapping histograms with density curves
    for treatment and control groups. This visualization helps assess how matching improves
    the overlap between propensity distributions.

    Args:
        results: MatchResults object containing matching results
        bins: Number of histogram bins
        figsize: Figure size

    Returns:
        Matplotlib figure

    """
    # Extract propensity scores and treatment mask from results
    propensity_scores = results.propensity_scores
    if propensity_scores is None:
        raise ValueError(
            "No propensity scores in this result: the design used a covariate "
            "distance, not a propensity score. Use plot_love_plot() or "
            "plot_balance() for covariate balance."
        )

    # Get treatment column and create masks
    treatment_col = results.config.treatment_col
    original_data = results.original_data
    matched_data = results.matched_data

    # Original data masks
    original_treatment_mask = original_data[treatment_col] == 1
    original_treatment_ps = propensity_scores[original_treatment_mask]
    original_control_ps = propensity_scores[~original_treatment_mask]

    # Create a mapping from original index to propensity score
    ps_map = dict(zip(original_data.index, propensity_scores, strict=False))

    # Extract propensity scores for matched data using the mapping
    matched_treatment_ps = [
        ps_map[idx] for idx in matched_data[matched_data[treatment_col] == 1].index
    ]
    matched_control_ps = [
        ps_map[idx] for idx in matched_data[matched_data[treatment_col] == 0].index
    ]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot before matching (left subplot)
    ax1.hist(
        original_treatment_ps,
        bins=bins,
        alpha=0.5,
        label="Treatment",
        density=True,
        color="blue",
    )
    ax1.hist(
        original_control_ps,
        bins=bins,
        alpha=0.5,
        label="Control",
        density=True,
        color="orange",
    )

    # Plot after matching (right subplot)
    ax2.hist(
        matched_treatment_ps,
        bins=bins,
        alpha=0.5,
        label="Treatment",
        density=True,
        color="blue",
    )
    ax2.hist(
        matched_control_ps,
        bins=bins,
        alpha=0.5,
        label="Control",
        density=True,
        color="orange",
    )

    # Customize subplots
    ax1.set_xlabel("Propensity Score")
    ax1.set_ylabel("Density")
    ax1.set_title("Before Matching")
    ax1.legend()

    ax2.set_xlabel("Propensity Score")
    ax2.set_title("After Matching")
    ax2.legend()

    fig.suptitle(
        "Propensity Score Distributions Before and After Matching", fontsize=14
    )
    fig.tight_layout()
    return fig


def plot_matched_pairs_distance(
    results: "MatchResults", bins: int = 30, figsize: tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot histogram of distances between matched pairs.

    Creates a histogram showing the distribution of distances between matched treatment-control
    pairs. A vertical dashed red line indicates the median distance. This visualization helps
    assess the quality of matches and identify potential outliers with large distances.

    Args:
        results: MatchResults object containing matching results
        bins: Number of histogram bins
        figsize: Figure size

    Returns:
        Matplotlib figure

    """
    # Extract match distances directly
    if not hasattr(results, "match_distances") or not results.match_distances:
        # If distances are not available or empty, raise an error or return an empty plot
        logger.warning(
            "Match distances are not available or empty in the results. Cannot plot distance distribution."
        )
        # Create empty figure with message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "Match distances not available",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.set_axis_off()
        return fig

    match_distances = results.match_distances

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram
    ax.hist(match_distances, bins=bins, alpha=0.7)

    # Add median line
    median_distance = np.median(match_distances)
    ax.axvline(
        x=median_distance,
        color="r",
        linestyle="--",
        label=f"Median: {median_distance:.4f}",
    )

    # Customize plot
    ax.set_xlabel("Distance")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Distances Between Matched Pairs")
    ax.legend()

    fig.tight_layout()
    return fig
