from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_dirs_exist(paths: Iterable[Path]) -> None:
    """
    Create directories if they do not exist.
    Safe to run multiple times.
    """
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
