"""Uniform logging for every stage.

Each stage call gets a console handler plus a timestamped file under
``<data_root>/logs/<stage>-<timestamp>.log`` so a long SLURM run leaves an
auditable trail even when stdout is lost.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_FMT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    stage: str,
    log_dir: Path | None = None,
    level: int = logging.INFO,
    quiet: bool = False,
) -> Path | None:
    """Configure the root logger. Returns the log file path, if one was opened."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    if not quiet:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(formatter)
        console.setLevel(level)
        root.addHandler(console)

    log_path: Path | None = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = log_dir / f"{stage}-{stamp}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        root.addHandler(file_handler)

    # Third-party loggers are noisy at INFO; they rarely say anything useful.
    for noisy in ("urllib3", "selenium", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return log_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"mecs.{name}")
