"""
Filesystem path helpers for PileFlow generator outputs.
"""

import os
from datetime import datetime


def ensure_dir(path: str) -> str:
    """
    Create a directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)
    return path


def timestamp_string() -> str:
    """
    Return a timestamp string for run folders.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")