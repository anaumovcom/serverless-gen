"""Logging helpers."""
from __future__ import annotations

import logging
import sys


_CONFIGURED = False


def get_logger(name: str = "generation-worker") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        if not root.handlers:
            root.addHandler(handler)
        _CONFIGURED = True
    return logging.getLogger(name)
