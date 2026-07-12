"""Structured logging configuration."""

import logging
import sys

from app.core.config import settings


def setup_logging() -> None:
    """Configure structured logging for the application."""
    logging.basicConfig(
        level=settings.LOG_LEVEL.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


logger = logging.getLogger("jobpatra")
