"""
Output writers for simple serialized files.
"""

import json
from typing import Any


def save_json(path: str, payload: dict[str, Any]) -> None:
    """
    Save a dictionary as pretty-printed JSON.
    """
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_text(path: str, text: str) -> None:
    """
    Save plain text to a file.
    """
    with open(path, "w") as f:
        f.write(text)