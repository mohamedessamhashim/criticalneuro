"""Centralized logging for CriticalNeuroMap pipeline."""

import logging
import sys
from pathlib import Path


def setup_logging(log_file: str = "results/pipeline.log", level: str = "INFO") -> logging.Logger:
    """Configure logging to both stdout and file.

    Parameters
    ----------
    log_file : str
        Path to log file.
    level : str
        Logging level (DEBUG, INFO, WARNING).

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(numeric_level)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # File handler
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return logging.getLogger("pipeline")
