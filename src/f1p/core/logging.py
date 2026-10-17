"""Logging with Rich formatting; one call configures the whole process."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

LOGGER_NAME = "f1p"


def configure(level: int | str = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        # stderr, so progress never mixes into a command's own (stdout) output
        handler = RichHandler(
            console=Console(stderr=True), rich_tracebacks=True, show_path=False, markup=False
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)
