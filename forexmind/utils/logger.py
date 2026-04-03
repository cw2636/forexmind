"""
ForexMind — Structured Logger
================================
Uses Python's built-in logging wired to Rich for beautiful console output.

Advanced Python concepts:
  - Module-level logger factory
  - Logging handlers and formatters
  - Context-manager based temporary log level override
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)
_initialized = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger with Rich formatting.
    Call once at the top of each module:
        log = get_logger(__name__)
    """
    global _initialized
    if not _initialized:
        _setup_root_logger()
        _initialized = True
    return logging.getLogger(name)


def _setup_root_logger(level: str = "INFO") -> None:
    """Configure the root logger with Rich handler."""
    # Try to read level from settings (graceful fallback if settings aren't ready)
    try:
        from forexmind.config.settings import get_settings
        level = get_settings().app.log_level
    except Exception:
        pass

    handler = RichHandler(
        console=_console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%H:%M:%S]"))

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Avoid duplicate handlers if called multiple times
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "asyncio", "httpx", "httpcore", "oandapyV20", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@contextmanager
def log_level(level: str) -> Generator[None, None, None]:
    """
    Temporarily change the root log level.
    Useful in tests or for verbose debug sessions.

    Usage:
        with log_level("DEBUG"):
            run_indicator_engine()
    """
    root = logging.getLogger()
    original = root.level
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    try:
        yield
    finally:
        root.setLevel(original)
