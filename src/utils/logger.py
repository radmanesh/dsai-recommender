"""Centralized logging utility with configurable debug levels."""

import os
import logging
import sys
from enum import IntEnum
from typing import Optional


class DebugLevel(IntEnum):
    """Debug levels from least to most verbose."""
    ERROR = 0      # Only errors
    WARNING = 1    # Warnings and errors
    INFO = 2       # Info, warnings, and errors (default)
    DEBUG = 3      # Debug, info, warnings, and errors
    VERBOSE = 4    # Very detailed debug output


# Global logger instance
_logger: Optional[logging.Logger] = None


def get_logger(name: str = "dsai_recommender") -> logging.Logger:
    """
    Get or create the global logger instance.

    Args:
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    global _logger

    if _logger is None:
        _logger = _setup_logger(name)

    return _logger


def _setup_logger(name: str) -> logging.Logger:
    """
    Setup logger with configuration from Config.

    Args:
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to lowest level, handler will filter

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(_get_log_level())

    # Create formatter
    debug_level = _get_debug_level()
    if debug_level >= DebugLevel.VERBOSE:
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)-8s] [%(name)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    elif debug_level >= DebugLevel.DEBUG:
        formatter = logging.Formatter(
            '[%(levelname)-8s] %(name)s: %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '[%(levelname)-8s] %(message)s'
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def _get_debug_level() -> DebugLevel:
    """Get debug level from Config."""
    debug_level_str = os.getenv("DEBUG_LEVEL", "INFO").upper()

    # Try to parse as enum name
    try:
        return DebugLevel[debug_level_str]
    except KeyError:
        pass

    # Try to parse as integer
    try:
        level_int = int(debug_level_str)
        if 0 <= level_int <= 4:
            return DebugLevel(level_int)
    except ValueError:
        pass

    # Default to INFO
    return DebugLevel.INFO


def _get_log_level() -> int:
    """Convert debug level to Python logging level."""
    debug_level = _get_debug_level()

    level_map = {
        DebugLevel.ERROR: logging.ERROR,
        DebugLevel.WARNING: logging.WARNING,
        DebugLevel.INFO: logging.INFO,
        DebugLevel.DEBUG: logging.DEBUG,
        DebugLevel.VERBOSE: logging.DEBUG,
    }

    return level_map[debug_level]


def debug(msg: str, *args, **kwargs):
    """Log debug message."""
    get_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    """Log info message."""
    get_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    """Log warning message."""
    get_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    """Log error message."""
    get_logger().error(msg, *args, **kwargs)


def verbose(msg: str, *args, **kwargs):
    """Log verbose debug message (only if VERBOSE level)."""
    if _get_debug_level() >= DebugLevel.VERBOSE:
        get_logger().debug(f"[VERBOSE] {msg}", *args, **kwargs)

