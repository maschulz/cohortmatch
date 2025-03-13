"""Data validation utilities for CohortMatch.

This module provides centralized data validation functions to ensure
input data meets the requirements for matching algorithms.
"""

import numpy as np
import pandas as pd

from cohortmatch.utils.logging import get_logger

# Create a logger for this module
logger = get_logger(__name__)


def validate_dataframe_index(
    data: pd.DataFrame, allow_duplicates: bool = False
) -> None:
    """Validate that the dataframe index has an acceptable type and is unique.

    Args:
        data: DataFrame to validate
        allow_duplicates: Whether to allow duplicate indices (for matching with replacement)

    Raises:
        TypeError: If index has mixed types or unsupported types
        ValueError: If index is not unique and duplicates are not allowed

    """
    # Handle empty dataframes - nothing to validate
    if data.empty:
        return

    # Get index values and their types
    index_values = data.index.tolist()
    index_types = {type(idx) for idx in index_values}

    # Check if there are mixed types
    if len(index_types) > 1:
        raise TypeError(
            f"DataFrame index has mixed types: {index_types}. All indices must be of the same type."
        )

    # Check if the type is supported (str or int)
    if index_types:  # Only check if we have values
        index_type = list(index_types)[0]
        if not issubclass(index_type, (str, int, np.integer)):
            raise TypeError(
                f"DataFrame index has unsupported type: {index_type}. Supported types are str and int."
            )

    # Check if index is unique, unless duplicates are explicitly allowed
    if not allow_duplicates and not data.index.is_unique:
        raise ValueError("DataFrame index must be unique")


def validate_data(
    data: pd.DataFrame,
    treatment_col: str,
    covariates: list[str] = None,
    outcomes: list[str] | None = None,
    propensity_col: str | None = None,
    exact_match_cols: list[str] | None = None,
    require_both_groups: bool = True,
) -> None:
    """Validate input data for matching.

    Args:
        data: DataFrame containing the data
        treatment_col: Name of the treatment column
        covariates: List of covariate column names
        outcomes: List of outcome column names
        propensity_col: Name of propensity score column
        exact_match_cols: Columns to use for exact matching
        require_both_groups: Whether to require both treatment and control groups

    Raises:
        ValueError: If there's a validation error

    """
    # Validate dataframe index (ensures unique and supported types)
    validate_dataframe_index(data)

    logger.debug(f"Validating data with {len(data)} observations")

    # Validate treatment column
    validate_treatment_column(data, treatment_col, require_both_groups)

    # Validate covariates if provided
    if covariates is not None and len(covariates) > 0:
        logger.debug(f"Validating {len(covariates)} covariate columns")
        # Check that all covariates exist in the data
        missing_covariates = [col for col in covariates if col not in data.columns]
        if missing_covariates:
            raise ValueError(f"Covariates not found in data: {missing_covariates}")

        # Validate that covariates are numeric and have no missing values
        validate_numeric_columns(data, covariates)
        validate_no_missing_values(data, covariates)

    # Validate outcomes if provided
    if outcomes is not None and len(outcomes) > 0:
        logger.debug(f"Validating {len(outcomes)} outcome columns")
        # Check that all outcomes exist in the data
        missing_outcomes = [col for col in outcomes if col not in data.columns]
        if missing_outcomes:
            raise ValueError(f"Outcomes not found in data: {missing_outcomes}")

        # Validate that outcomes are numeric and have no missing values
        validate_numeric_columns(data, outcomes)
        validate_no_missing_values(data, outcomes)

    # Validate propensity column if provided
    if propensity_col is not None:
        logger.debug(f"Validating propensity score column: {propensity_col}")
        if propensity_col not in data.columns:
            raise ValueError(f"Propensity column '{propensity_col}' not found in data")
        validate_propensity_scores(data, propensity_col)

    # Validate exact match columns if provided
    if exact_match_cols is not None and len(exact_match_cols) > 0:
        logger.debug(f"Validating {len(exact_match_cols)} exact match columns")
        # Check that all exact match columns exist in the data
        missing_exact_cols = [
            col for col in exact_match_cols if col not in data.columns
        ]
        if missing_exact_cols:
            raise ValueError(
                f"Exact match columns not found in data: {missing_exact_cols}"
            )

        # Check for missing values in exact match columns
        validate_no_missing_values(data, exact_match_cols)

    logger.info("Data validation successful")


def validate_treatment_column(
    data: pd.DataFrame, treatment_col: str, require_both_groups: bool = True
) -> None:
    """Validate that treatment column contains only binary values (0/1).

    Args:
        data: DataFrame containing the data
        treatment_col: Name of the treatment indicator column
        require_both_groups: Whether to require both treatment and control units

    Raises:
        ValueError: If treatment column validation fails

    """
    # Check that treatment column contains only binary values
    treatment_values = data[treatment_col].unique()
    if not set(treatment_values).issubset({0, 1}):
        raise ValueError(
            f"Treatment column '{treatment_col}' must contain only binary values (0/1), "
            f"found: {sorted(treatment_values)}"
        )

    # Count treatment and control units
    n_treatment = (data[treatment_col] == 1).sum()
    n_control = (data[treatment_col] == 0).sum()

    # Always check for at least one treated unit
    if n_treatment == 0:
        raise ValueError(f"No treatment units found in '{treatment_col}' (no 1s)")

    # Check for control units only if required
    if require_both_groups and n_control == 0:
        raise ValueError(f"No control units found in '{treatment_col}' (no 0s)")


def validate_numeric_columns(data: pd.DataFrame, columns: list[str]) -> None:
    """Validate that columns contain only numeric data.

    Args:
        data: DataFrame containing the data
        columns: List of column names to check

    Raises:
        ValueError: If any column contains non-numeric data

    """
    for col in columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(
                f"Column '{col}' must contain only numeric values, "
                f"but has dtype {data[col].dtype}"
            )


def validate_no_missing_values(data: pd.DataFrame, columns: list[str]) -> None:
    """Validate that columns have no missing values.

    Args:
        data: DataFrame containing the data
        columns: List of column names to check

    Raises:
        ValueError: If any column contains missing values

    """
    for col in columns:
        if data[col].isna().any():
            n_missing = data[col].isna().sum()
            raise ValueError(
                f"Column '{col}' contains {n_missing} missing values. "
                f"Please handle missing values before matching."
            )


def validate_propensity_scores(data: pd.DataFrame, propensity_col: str) -> None:
    """Validate that propensity scores are between 0 and 1.

    Args:
        data: DataFrame containing the data
        propensity_col: Name of propensity score column

    Raises:
        ValueError: If propensity scores are outside [0, 1]

    """
    p_scores = data[propensity_col]
    if (p_scores < 0).any() or (p_scores > 1).any():
        raise ValueError(
            f"Propensity scores in '{propensity_col}' must be between 0 and 1. "
            f"Found min={p_scores.min()}, max={p_scores.max()}"
        )


def validate_matcher_config(config) -> None:
    """Validate MatcherConfig for required fields and proper values.

    Args:
        config: MatcherConfig object to validate

    Raises:
        ValueError: If configuration fails validation checks

    """
    logger.debug("Validating matcher configuration")

    # Validate required fields
    if not hasattr(config, "treatment_col") or not config.treatment_col:
        raise ValueError("treatment_col is required in MatcherConfig")

    if not hasattr(config, "covariates") or not config.covariates:
        raise ValueError("covariates list is required in MatcherConfig")

    # Validate match method
    valid_match_methods = ["greedy", "optimal", "fast_greedy", "subclass", "cem"]
    if config.match_method not in valid_match_methods:
        raise ValueError(
            f"match_method must be one of {valid_match_methods}, got {config.match_method}"
        )

    # Validate distance method
    valid_distance_methods = ["euclidean", "mahalanobis", "propensity", "logit"]
    if config.distance_method not in valid_distance_methods:
        raise ValueError(
            f"distance_method must be one of {valid_distance_methods}, got {config.distance_method}"
        )

    # Validate ratio
    if config.ratio <= 0:
        raise ValueError(f"ratio must be positive, got {config.ratio}")

    # Validate caliper method
    if config.caliper_method is not None:
        valid_caliper_methods = ["propensity", "logit", "mahalanobis", "euclidean"]
        if config.caliper_method not in valid_caliper_methods:
            raise ValueError(
                f"caliper_method must be one of {valid_caliper_methods}, got "
                f"{config.caliper_method!r}. Per-covariate calipers are not supported yet."
            )
        # We don't explicitly check against data columns here, that happens at runtime
        # This is to allow flexibility. If the method is not a standard one, it's assumed to be a column.
        pass

    # Validate propensity model if estimation is enabled
    if config.estimate_propensity:
        valid_propensity_models = ["logistic", "custom"]
        if config.propensity_model not in valid_propensity_models:
            raise ValueError(
                f"propensity_model must be one of {valid_propensity_models}, got {config.propensity_model}"
            )

        # Validate CV folds
        if config.cv_folds <= 1:
            raise ValueError(f"cv_folds must be greater than 1, got {config.cv_folds}")

    logger.info("Matcher configuration validation successful")
