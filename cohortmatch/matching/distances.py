"""Distance calculation functions for matching algorithms.

This module provides functions for calculating distances between treatment and control units,
which are used by the matching algorithms.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logit
from sklearn.preprocessing import StandardScaler

# Import logger
from cohortmatch.utils.logging import get_logger

# Create a logger for this module
logger = get_logger(__name__)


def calculate_distance_matrix(
    X_treat: np.ndarray,
    X_control: np.ndarray,
    method: str = "euclidean",
    standardize: bool = True,
    weights: np.ndarray | None = None,
    cov_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate distance matrix between treatment and control groups.

    Args:
        X_treat: Array of treatment group features, shape (n_treatment, n_features)
        X_control: Array of control group features, shape (n_control, n_features)
        method: Distance calculation method ('euclidean', 'mahalanobis', 'propensity', 'logit')
        standardize: Whether to standardize features before calculating distances
        weights: Feature weights for euclidean distance, shape (n_features,)
        cov_matrix: Covariance matrix for Mahalanobis distance, shape (n_features, n_features)

    Returns:
        Distance matrix, shape (n_treatment, n_control)

    """
    logger.debug(f"Calculating distance matrix using method: {method}")
    logger.debug(
        f"Input dimensions: X_treat {X_treat.shape}, X_control {X_control.shape}"
    )

    # Validate inputs
    if X_treat.ndim != 2 or X_control.ndim != 2:
        raise ValueError("X_treat and X_control must be 2D arrays")

    if X_treat.shape[1] != X_control.shape[1]:
        raise ValueError("X_treat and X_control must have same number of features")

    if method not in {"euclidean", "mahalanobis", "propensity", "logit"}:
        raise ValueError(f"Unknown distance method: {method}")

    # Ensure numeric types to prevent errors with object arrays from pandas
    if X_treat.dtype == "object":
        X_treat = X_treat.astype(np.float64)
    if X_control.dtype == "object":
        X_control = X_control.astype(np.float64)

    if weights is not None:
        logger.debug(f"Using feature weights with shape: {weights.shape}")
        if len(weights) != X_treat.shape[1]:
            raise ValueError("Weights length must match number of features")

    # Handle propensity-based methods
    if method in ["propensity", "logit"]:
        logger.debug(f"Using propensity-based distance method: {method}")
        return _calculate_propensity_distances(X_treat, X_control, method)

    # Standardize if requested
    if standardize:
        logger.debug("Standardizing data before distance calculation")
        X_treat, X_control = _standardize_data(X_treat, X_control)

    # Apply weights for Euclidean distance
    if method == "euclidean" and weights is not None:
        logger.debug("Applying feature weights to Euclidean distance calculation")
        weights_sqrt = np.sqrt(weights.ravel())
        X_treat = X_treat * weights_sqrt
        X_control = X_control * weights_sqrt

    # Calculate distances
    if method == "euclidean":
        logger.debug("Calculating Euclidean distances")
        distance_matrix = cdist(X_treat, X_control, metric="euclidean")
    else:  # mahalanobis
        logger.debug("Calculating Mahalanobis distances")
        if cov_matrix is None:
            logger.debug("No covariance matrix provided, estimating from data")
            X_combined = np.vstack((X_treat, X_control))
            cov_matrix = np.cov(X_combined, rowvar=False)
        else:
            logger.debug(
                f"Using provided covariance matrix with shape: {cov_matrix.shape}"
            )

        try:
            # Handle scalar covariance for 1D data
            if hasattr(cov_matrix, "ndim") and cov_matrix.ndim == 0:
                cov_matrix = cov_matrix.item()

            if np.isscalar(cov_matrix):
                if cov_matrix <= 0:
                    logger.warning(
                        "Covariance matrix is zero or negative. Using pseudoinverse."
                    )
                    cov_inv = np.linalg.pinv(np.array([[cov_matrix]]))
                else:
                    cov_inv = 1.0 / cov_matrix
            else:
                # Add small regularization term for numerical stability
                cov_inv = np.linalg.inv(cov_matrix + 1e-6 * np.eye(cov_matrix.shape[0]))
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix inversion failed, using pseudoinverse")
            cov_inv = np.linalg.pinv(cov_matrix)

        distance_matrix = cdist(X_treat, X_control, metric="mahalanobis", VI=cov_inv)

    logger.debug(f"Distance matrix calculated with shape: {distance_matrix.shape}")
    logger.debug(
        f"Distance matrix stats: min={distance_matrix.min():.4f}, "
        f"mean={distance_matrix.mean():.4f}, max={distance_matrix.max():.4f}"
    )

    return distance_matrix


def _standardize_data(
    X_treat: np.ndarray, X_control: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize treatment and control data using sklearn's StandardScaler."""
    logger.debug(
        f"Standardizing data: X_treat {X_treat.shape}, X_control {X_control.shape}"
    )

    X_combined = np.vstack((X_treat, X_control))
    scaler = StandardScaler()
    scaler.fit(X_combined)

    # Check for zero variance features
    zero_std_features = np.where(scaler.scale_ < 1e-10)[0]
    if len(zero_std_features) > 0:
        logger.warning(
            f"Found {len(zero_std_features)} feature(s) with near-zero standard deviation. "
            f"These will be set to zero in the standardized data."
        )

    X_treat_std = scaler.transform(X_treat)
    X_control_std = scaler.transform(X_control)

    logger.debug("Standardization complete")

    return X_treat_std, X_control_std


def _calculate_propensity_distances(
    X_treat: np.ndarray, X_control: np.ndarray, method: str
) -> np.ndarray:
    """Calculate distances for propensity-based methods."""
    # Ensure 1D arrays
    X_treat_1d = X_treat.ravel()
    X_control_1d = X_control.ravel()

    logger.debug(
        f"Propensity score ranges - Treatment: [{X_treat_1d.min():.4f}, {X_treat_1d.max():.4f}], "
        f"Control: [{X_control_1d.min():.4f}, {X_control_1d.max():.4f}]"
    )

    # Apply logit transform if needed
    if method == "logit":
        logger.debug("Applying logit transformation with clipping to [1e-6, 1 - 1e-6]")
        # Clip to avoid numerical issues
        X_treat_1d = np.clip(X_treat_1d, 1e-6, 1 - 1e-6)
        X_control_1d = np.clip(X_control_1d, 1e-6, 1 - 1e-6)
        X_treat_1d = logit(X_treat_1d)
        X_control_1d = logit(X_control_1d)

    distance_matrix = cdist(
        X_treat_1d.reshape(-1, 1), X_control_1d.reshape(-1, 1), metric="euclidean"
    )

    logger.debug(
        f"Propensity distance matrix calculated with shape: {distance_matrix.shape}"
    )
    logger.debug(
        f"Distance matrix stats: min={distance_matrix.min():.4f}, "
        f"mean={distance_matrix.mean():.4f}, max={distance_matrix.max():.4f}"
    )

    return distance_matrix
