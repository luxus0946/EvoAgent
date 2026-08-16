"""Logging configuration: unified format with both console and file output."""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "evoagent",
    level: str = "INFO",
    log_file: str | None = None,
) -> logging.Logger:
    """Initialize the global logger.

    Args:
        name: Logger name
        level: Log level
        log_file: Log file path; console-only output when None

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def new_log_file_path(output_dir: str | Path) -> str:
    """Generate a log filename with a timestamp."""
    return str(
        Path(output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
