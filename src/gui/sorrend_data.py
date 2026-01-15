"""
Sorrend data - Settings lookup tables for H-K and SZ-P modes.
Auto-generated from sorrend.xlsx.
Refactored to load data from JSON to avoid large file issues.
"""

import json
import os
from typing import Dict

_DATA_CACHE: Dict[str, Dict[str, str]] = {}
_JSON_PATH = os.path.join(os.path.dirname(__file__), "sorrend_data.json")


def _load_data():
    """Load data from JSON file if not already loaded."""
    if not _DATA_CACHE:
        try:
            with open(_JSON_PATH, "r", encoding="utf-8") as f:
                _DATA_CACHE.update(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # Fallback or error logging could go here
            print(f"Error loading sorrend_data.json: {e}")
            _DATA_CACHE.update({"HK_DATA": {}, "SZP_DATA": {}})


def get_settings(no: int, mode: str = "HK") -> str:
    """
    Get settings string for a given number and mode.

    Args:
        no: Strategy number (0-6143)
        mode: "HK" for H-K mode or "SZP" for SZ-P mode

    Returns:
        Settings string or empty string if not found
    """
    _load_data()

    # JSON keys are strings, so convert integer 'no' to string
    str_no = str(no)

    if mode == "HK":
        return _DATA_CACHE.get("HK_DATA", {}).get(str_no, "")
    if mode == "SZP":
        return _DATA_CACHE.get("SZP_DATA", {}).get(str_no, "")
    return ""


# For backward compatibility if module variables were used directly
# (though we found they weren't widely used) definition properties or just rely on get_settings.
# Implementing __getattr__ for module level lazy access to HK_DATA/SZP_DATA if needed.


def __getattr__(name):
    if name in ("HK_DATA", "SZP_DATA"):
        _load_data()
        return _DATA_CACHE.get(name, {})
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
