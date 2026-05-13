"""Rich-aware logging setup, single source of truth for the project."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

try:
    from rich.logging import RichHandler

    _HAVE_RICH = True
except ImportError:  # pragma: no cover - rich is in core deps but be defensive
    _HAVE_RICH = False


_DEF_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> None:
    """Configure root logger. Idempotent: safe to call multiple times."""
    root = logging.getLogger()
    # Drop existing handlers so re-runs in notebooks don't duplicate lines.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    if _HAVE_RICH:
        console_handler: logging.Handler = RichHandler(
            rich_tracebacks=True, markup=False, show_path=False, show_time=True
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(logging.Formatter(_DEF_FMT))
    root.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_DEF_FMT))
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
