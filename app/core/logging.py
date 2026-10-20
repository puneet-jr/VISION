"""Structured logging setup."""
from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once; safe to call repeatedly."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric)
    if not any(getattr(h, "_visionshield", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler._visionshield = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
