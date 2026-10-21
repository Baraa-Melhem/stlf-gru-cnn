"""Structured logging set-up shared by the library and the CLIs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> None:
    """Configure the ``stlf`` logger hierarchy once: console plus an optional log file.

    Library modules only ever call :func:`get_logger`; handlers are attached here, by the
    application entry point, so importing ``stlf`` never changes a host program's logging.
    """
    root = logging.getLogger("stlf")
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False
    formatter = logging.Formatter(_FORMAT, _DATE_FORMAT)

    # Legacy Windows code pages cannot encode every character (e.g. "≤", "—"); never let a log
    # line raise because of the console's encoding.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``stlf`` logger (``name`` is usually ``__name__``)."""
    return logging.getLogger(name if name.startswith("stlf") else f"stlf.{name}")
