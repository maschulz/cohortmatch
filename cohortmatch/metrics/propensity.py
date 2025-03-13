"""Propensity score estimation for CohortMatch.

This module provides functions for estimating propensity scores using various models,
as well as utilities for assessing propensity score quality and overlap.
"""

import functools
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Import logger
from cohortmatch.utils.logging import get_logger

# Import validation functions
from cohortmatch.validation import validate_data

# Create a logger for this module
logger = get_logger(__name__)

# Try to import scikit-learn
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning(
        "Scikit-learn is not installed. Some propensity score estimation methods will not be available."
    )


def suppress_warnings(func):
    """Decorator to suppress warnings in a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return wrapper


def estimate_propensity_scores(
    data: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    model_type: str = "logistic",
    model_params: dict[str, Any] | None = None,
    cv: int = 5,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Estimate propensity scores using a classification model.

    Args:
        data: DataFrame containing the data
        treatment_col: Name of the column containing treatment indicators
        covariates: List of column names to use for propensity model
        model_type: "logistic" or "custom" (pass the estimator in model_params["model"])
        model_params: Parameters to pass to the model constructor
        cv: Number of cross-validation folds
        random_state: Random state for reproducibility

    Returns:
        Dictionary with propensity scores, model, and metrics

    """
    # Validate input data
    validate_data(data=data, treatment_col=treatment_col, covariates=covariates)

    # Validate model type
    valid_model_types = {"logistic", "custom"}
    if model_type not in valid_model_types:
        raise ValueError(
            f"Unknown model type: {model_type}. Must be one of: {', '.join(valid_model_types)}"
        )

    # Validate CV parameter
    if cv < 2:
        raise ValueError(
            f"Number of cross-validation folds must be at least 2, got {cv}"
        )

    if not HAS_SKLEARN:
        raise ImportError(
            "Scikit-learn is required for propensity score estimation. "
            "Install it with 'pip install scikit-learn'"
        )

    logger.info(
        f"Estimating propensity scores using {model_type} model with {cv}-fold cross-validation"
    )
    # Create a copy of model_params to avoid modifying the original
    model_params = model_params.copy() if model_params else {}

    # Extract features and treatment indicators
    X = data[covariates].values
    y = data[treatment_col].values

    logger.debug(
        f"Treatment prevalence: {np.mean(y):.3f} ({np.sum(y)} out of {len(y)} units)"
    )

    # Standardize features for logistic regression
    scaler = None
    if model_type == "logistic":
        logger.debug("Standardizing features for logistic regression")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Get the propensity model
    model = get_propensity_model(model_type, model_params, random_state)

    # Use cross-validation to estimate propensity scores without data leakage
    cv_results = estimate_propensity_scores_with_cv(
        X=X,
        y=y,
        model=model,
        cv=cv,
        random_state=random_state,
    )

    final_model = cv_results["final_model"]
    if scaler is not None:
        # ship the scaler with the model so it is usable on raw data
        from sklearn.pipeline import Pipeline

        final_model = Pipeline([("scaler", scaler), ("model", final_model)])

    # Add propensity scores and model to the result
    result = {
        "propensity_scores": cv_results["propensity_scores"],
        "model": final_model,
        "cv_results": cv_results["cv_results"],
        "model_type": model_type,
        "auc": cv_results["auc"],
    }

    logger.info(f"Propensity score estimation complete. AUC: {cv_results['auc']:.3f}")
    return result


def get_propensity_model(
    model_type: str = "logistic",
    model_params: dict[str, Any] | None = None,
    random_state: int | None = None,
) -> Any:
    """Create a propensity score model based on the specified type.

    Args:
        model_type: "logistic" or "custom" (estimator in model_params["model"])
        model_params: Parameters to pass to the model constructor
        random_state: Random state for reproducibility

    Returns:
        A scikit-learn compatible model instance

    """
    if not HAS_SKLEARN:
        raise ImportError(
            "Scikit-learn is required for propensity score estimation. "
            "Install it with 'pip install scikit-learn'"
        )

    model_params = model_params or {}

    if model_type == "logistic":
        if random_state is not None:
            model_params.setdefault("random_state", random_state)
        return LogisticRegression(max_iter=1000, solver="lbfgs", **model_params)
    if model_type == "custom":
        if "model" not in model_params:
            raise ValueError(
                "For custom model type, you must provide a 'model' in model_params"
            )
        return model_params["model"]
    raise ValueError(f"Unknown model type: {model_type}")


def estimate_propensity_scores_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    cv: int = 5,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Estimate propensity scores using cross-validation to prevent overfitting.

    This function uses K-fold cross-validation to estimate propensity scores
    without data leakage, which can occur when the same data is used for
    both fitting the propensity model and subsequent matching.

    Args:
        X: Feature matrix
        y: Treatment indicator
        model: Classification model
        cv: Number of cross-validation folds
        random_state: Random state for reproducibility

    Returns:
        Dictionary with propensity scores, model, and metrics

    """
    # Determine a valid number of folds for the given class counts
    unique, counts = np.unique(y, return_counts=True)
    min_class_count = int(counts.min()) if counts.size > 0 else cv
    if min_class_count < 2:
        raise ValueError(
            "Too few units in the smaller treatment group to cross-fit "
            f"propensity scores (need >= 2, have {min_class_count}). Provide "
            "precomputed propensity_scores or use a covariate distance."
        )
    n_splits = max(2, min(cv, min_class_count))

    # Create cross-validation splitter
    cv_splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    # Initialize array for propensity scores
    propensity_scores = np.zeros_like(y, dtype=float)

    # Storage for per-fold metrics
    aucs = []
    fold_models = []

    logger.debug(f"Starting {cv}-fold cross-validation for propensity score estimation")

    # Perform cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        logger.debug(
            f"Fold {fold_idx + 1}/{cv}: Training on {len(X_train)} samples, testing on {len(X_test)} samples"
        )

        # Fit the model
        model_clone = clone_model(model)

        try:
            model_clone.fit(X_train, y_train)

            # Predict probabilities for test set
            preds = model_clone.predict_proba(X_test)[:, 1]

            # Store propensity scores for this fold
            propensity_scores[test_idx] = preds

            # Calculate AUC for this fold
            fold_auc = roc_auc_score(y_test, preds)
            aucs.append(fold_auc)
            fold_models.append(model_clone)

            logger.debug(f"Fold {fold_idx + 1}/{cv}: AUC = {fold_auc:.3f}")

        except Exception as e:
            logger.error(f"Error in fold {fold_idx + 1}/{cv}: {e!s}")
            raise

    # Train a final model on all data
    logger.debug("Training final model on all data")
    final_model = clone_model(model)
    final_model.fit(X, y)

    # Calculate overall AUC
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    logger.info(f"Cross-validation AUC: {mean_auc:.3f} ± {std_auc:.3f}")

    return {
        "propensity_scores": propensity_scores,
        "final_model": final_model,
        "cv_results": {"fold_aucs": aucs, "mean_auc": mean_auc, "std_auc": std_auc},
        "fold_models": fold_models,
        "auc": mean_auc,
    }


def clone_model(model: Any) -> Any:
    """Clone a scikit-learn model.

    This function attempts to clone a model using scikit-learn's clone function.
    If that fails, it attempts to create a new instance with the same parameters.

    Args:
        model: Model to clone

    Returns:
        Cloned model

    """
    logger.debug(f"Cloning model of type {type(model).__name__}")

    try:
        from sklearn.base import clone

        cloned_model = clone(model)
        logger.debug("Model cloned successfully using sklearn.base.clone")
        return cloned_model
    except (ImportError, TypeError) as e:
        logger.warning(f"Could not clone model using sklearn.base.clone: {e!s}")

        # Fallback option: try to create a new instance with the same parameters
        try:
            logger.debug(
                "Attempting to clone by creating new instance with same parameters"
            )
            cloned_model = model.__class__(**model.get_params())
            logger.debug("Model cloned successfully by creating new instance")
            return cloned_model
        except Exception as e2:
            # Last resort: just return the model itself (not ideal)
            logger.warning(f"Could not clone model by creating new instance: {e2!s}")
            logger.warning("Using original model instance (not recommended)")
            return model


def assess_common_support(
    propensity_scores: np.ndarray, treatment: np.ndarray, bins: int = 20
) -> dict[str, Any]:
    """Assess common support between treatment and control propensity distributions.

    Args:
        propensity_scores: Array of propensity scores
        treatment: Binary treatment indicator array
        bins: Number of bins for histogram

    Returns:
        Dictionary with common support metrics

    """
    # Input validation
    if len(propensity_scores) != len(treatment):
        raise ValueError("Propensity scores and treatment must have the same length")

    if len(propensity_scores) == 0:
        raise ValueError("Propensity scores array is empty")

    # Check that treatment is binary
    unique_treatment = np.unique(treatment)
    if not np.all(np.isin(unique_treatment, [0, 1])):
        raise ValueError(
            f"Treatment must contain only binary values (0/1), found: {unique_treatment}"
        )

    # Check that propensity scores are between 0 and 1
    if np.any(propensity_scores < 0) or np.any(propensity_scores > 1):
        raise ValueError(
            f"Propensity scores must be between 0 and 1, found min={np.min(propensity_scores)}, max={np.max(propensity_scores)}"
        )

    # Check bins parameter
    if bins < 2:
        raise ValueError(f"Number of bins must be at least 2, got {bins}")

    # Split by treatment group
    treated_ps = propensity_scores[treatment == 1]
    control_ps = propensity_scores[treatment == 0]

    # Check that we have both treatment and control units
    if len(treated_ps) == 0:
        raise ValueError("No treatment units found (no 1s in treatment array)")

    if len(control_ps) == 0:
        raise ValueError("No control units found (no 0s in treatment array)")

    # Calculate common support range
    min_treated, max_treated = np.min(treated_ps), np.max(treated_ps)
    min_control, max_control = np.min(control_ps), np.max(control_ps)

    cs_min = max(min_treated, min_control)
    cs_max = min(max_treated, max_control)

    # Create histograms for visualization
    all_range = (min(min_treated, min_control), max(max_treated, max_control))
    hist_treated, bin_edges = np.histogram(
        treated_ps, bins=bins, range=all_range, density=True
    )
    hist_control, _ = np.histogram(control_ps, bins=bins, range=all_range, density=True)

    # Calculate overlap coefficient
    bin_width = (all_range[1] - all_range[0]) / bins
    overlap = np.sum(np.minimum(hist_treated, hist_control) * bin_width)

    return {
        "common_support_min": cs_min,
        "common_support_max": cs_max,
        "overlap_coefficient": overlap,
        "hist_treated": hist_treated,
        "hist_control": hist_control,
        "bin_edges": bin_edges,
    }


@suppress_warnings
def assess_propensity_overlap(
    data: pd.DataFrame,
    propensity_col: str,
    treatment_col: str,
    matched_indices: pd.Index | None = None,
) -> dict[str, float]:
    """Assess the overlap of propensity scores between treatment and control groups.

    This function computes various metrics to evaluate the quality of propensity score
    overlap, which is crucial for valid causal inference. It can evaluate both the
    original data and the matched data if matched indices are provided.

    Args:
        data: DataFrame containing the data
        propensity_col: Name of the propensity score column
        treatment_col: Name of the treatment indicator column (must be binary)
        matched_indices: Optional indices of matched units to assess post-matching overlap

    Returns:
        Dictionary with overlap metrics:
        - ks_statistic: Kolmogorov-Smirnov statistic (smaller is better)
        - ks_pvalue: p-value for the KS test
        - overlap_coefficient: Overlap coefficient between distributions (higher is better)
        - common_support_range: Range of common support as a tuple (min, max)
        - treated_range: Range of treated group propensity scores
        - control_range: Range of control group propensity scores

    """
    # Validate input data
    validate_data(data=data, treatment_col=treatment_col, propensity_col=propensity_col)

    # Use matched data if indices are provided
    if matched_indices is not None:
        data = data.loc[matched_indices]

    # Extract propensity scores and treatment indicators
    ps = data[propensity_col].values
    treatment = data[treatment_col].values

    # Split by treatment group
    treated_ps = ps[treatment == 1]
    control_ps = ps[treatment == 0]

    # Compute KS test
    ks_statistic, ks_pvalue = stats.ks_2samp(treated_ps, control_ps)

    # Get common support using the existing function
    common_support = assess_common_support(ps, treatment)

    # Calculate ranges
    treated_range = (np.min(treated_ps), np.max(treated_ps))
    control_range = (np.min(control_ps), np.max(control_ps))

    # Calculate common support range
    common_support_min = max(treated_range[0], control_range[0])
    common_support_max = min(treated_range[1], control_range[1])
    common_support_range = (common_support_min, common_support_max)

    # Calculate proportion of units in common support
    in_common_support = ((ps >= common_support_min) & (ps <= common_support_max)).mean()

    # Calculate proportion of treated and control units in common support
    treated_in_cs = (
        (treated_ps >= common_support_min) & (treated_ps <= common_support_max)
    ).mean()
    control_in_cs = (
        (control_ps >= common_support_min) & (control_ps <= common_support_max)
    ).mean()

    return {
        "ks_statistic": ks_statistic,
        "ks_pvalue": ks_pvalue,
        "overlap_coefficient": common_support["overlap_coefficient"],
        "common_support_range": common_support_range,
        "treated_range": treated_range,
        "control_range": control_range,
        "prop_in_common_support": in_common_support,
        "prop_treated_in_cs": treated_in_cs,
        "prop_control_in_cs": control_in_cs,
    }
