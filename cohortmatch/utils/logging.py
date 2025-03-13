"""Logging utilities for CohortMatch.

This module provides functions for configuring and using logging
in a consistent way throughout the package.
"""

import logging
import sys
from typing import TextIO


def get_logger(name: str = "cohortmatch") -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Name for the logger, defaults to 'cohortmatch'

    Returns:
        A named logger instance

    """
    return logging.getLogger(name)


def configure_logging(
    level: int | str = logging.INFO,
    format_string: str | None = None,
    stream: TextIO | None = sys.stdout,
    log_file: str | None = None,
    name: str = "cohortmatch",
) -> logging.Logger:
    """Configure logging for CohortMatch.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Format string for log messages
        stream: Stream to output logs to (default: sys.stdout)
        log_file: Optional file path to write logs to
        name: Logger name, defaults to 'cohortmatch'

    Returns:
        Configured logger instance

    """
    logger = logging.getLogger(name)

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set the logging level
    logger.setLevel(level)

    # Default format string if not provided
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Add console handler
    if stream:
        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Don't propagate to root logger
    logger.propagate = False

    return logger


# Create a default logger instance
logger = get_logger()
