"""
GUI Settings Manager - saves and loads window state and user preferences.
Settings are stored in a JSON file in the gui folder (same location as this file).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class SettingsManager:
    """Manages GUI settings persistence."""

    # Default settings
    DEFAULTS = {
        "window": {
            "width": 1600,
            "height": 1080,
            "x": None,  # None = center
            "y": None,
        },
        "paths": {
            "last_folder": "",
            "last_parquet_folder": "",
            "last_convert_folder": "",
            "last_results_output_folder": "",
            "last_analysis_state_folder": "",
            "last_csv_export_folder": "",
            "last_comparison_folder": "",
        },
        "preferences": {
            "language": "EN",
            "dark_mode": True,
            "muted": True,
            "feature_mode": "Original",
        },
    }

    def __init__(self):
        """Initialize settings manager."""
        # Store settings in the gui folder (same directory as this file)
        self.settings_dir = Path(os.path.dirname(__file__))
        self.settings_file = self.settings_dir / "window_config.json"
        self.settings: Dict[str, Any] = {}
        self._load()

    def _load(self):
        """Load settings from JSON file."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                # Merge with defaults (in case new settings were added)
                self.settings = self._merge_defaults(loaded)
            else:
                self.settings = self.DEFAULTS.copy()
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load settings: {e}")
            self.settings = self.DEFAULTS.copy()

    def _merge_defaults(self, loaded: Dict) -> Dict:
        """Merge loaded settings with defaults for missing keys."""
        result = {}
        for key, default_value in self.DEFAULTS.items():
            if key in loaded:
                if isinstance(default_value, dict):
                    # Merge nested dict
                    result[key] = {**default_value, **loaded[key]}
                else:
                    result[key] = loaded[key]
            else:
                result[key] = default_value
        return result

    def save(self):
        """Save settings to JSON file."""
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Warning: Could not save settings: {e}")

    # === Window Settings ===

    def get_window_geometry(self) -> Dict[str, Optional[int]]:
        """Get window geometry (width, height, x, y)."""
        return self.settings.get("window", self.DEFAULTS["window"])

    def set_window_geometry(self, width: int, height: int, x: int, y: int):
        """Save window geometry."""
        self.settings["window"] = {
            "width": width,
            "height": height,
            "x": x,
            "y": y,
        }

    # === Path Settings ===

    def get_last_folder(self) -> str:
        """Get last opened folder path."""
        return self.settings.get("paths", {}).get("last_folder", "")

    def set_last_folder(self, path: str):
        """Save last opened folder path."""
        if "paths" not in self.settings:
            self.settings["paths"] = {}
        self.settings["paths"]["last_folder"] = path

    def get_last_parquet_folder(self) -> str:
        """Get last parquet folder path."""
        return self.settings.get("paths", {}).get("last_parquet_folder", "")

    def set_last_parquet_folder(self, path: str):
        """Save last parquet folder path."""
        if "paths" not in self.settings:
            self.settings["paths"] = {}
        self.settings["paths"]["last_parquet_folder"] = path

    def get_last_convert_folder(self) -> str:
        """Get last convert folder path."""
        return self.settings.get("paths", {}).get("last_convert_folder", "")

    def set_last_convert_folder(self, path: str):
        """Save last convert folder path."""
        if "paths" not in self.settings:
            self.settings["paths"] = {}
        self.settings["paths"]["last_convert_folder"] = path

    def get_last_results_output_folder(self) -> str:
        """Get last results output folder path."""
        return self.settings.get("paths", {}).get("last_results_output_folder", "")

    def set_last_results_output_folder(self, path: str):
        """Save last results output folder path."""
        if "paths" not in self.settings:
            self.settings["paths"] = {}
        self.settings["paths"]["last_results_output_folder"] = path

    def get_last_analysis_state_folder(self) -> str:
        """Get last analysis state folder path."""
        return self.settings.get("paths", {}).get("last_analysis_state_folder", "")

    def set_last_analysis_state_folder(self, path: str):
        """Save last analysis state folder path."""
        if "paths" not in self.settings:
            self.settings["paths"] = {}
        self.settings["paths"]["last_analysis_state_folder"] = path

    def get_last_csv_export_folder(self) -> str:
        """Get last CSV export folder path."""
        return self.settings.get("paths", {}).get("last_csv_export_folder", "")

    def set_last_csv_export_folder(self, path: str):
        """Save last CSV export folder path."""
        if "paths" not in self.settings:
            self.settings["paths"] = {}
        self.settings["paths"]["last_csv_export_folder"] = path

    def get_last_comparison_folder(self) -> str:
        """Get last comparison folder path."""
        return self.settings.get("paths", {}).get("last_comparison_folder", "")

    def set_last_comparison_folder(self, path: str):
        """Save last comparison folder path."""
        if "paths" not in self.settings:
            self.settings["paths"] = {}
        self.settings["paths"]["last_comparison_folder"] = path

    # === Preferences ===

    def get_language(self) -> str:
        """Get language preference."""
        return self.settings.get("preferences", {}).get("language", "EN")

    def set_language(self, lang: str):
        """Save language preference."""
        if "preferences" not in self.settings:
            self.settings["preferences"] = {}
        self.settings["preferences"]["language"] = lang

    def get_dark_mode(self) -> bool:
        """Get dark mode preference."""
        return self.settings.get("preferences", {}).get("dark_mode", True)

    def set_dark_mode(self, enabled: bool):
        """Save dark mode preference."""
        if "preferences" not in self.settings:
            self.settings["preferences"] = {}
        self.settings["preferences"]["dark_mode"] = enabled

    def get_muted(self) -> bool:
        """Get muted preference."""
        return self.settings.get("preferences", {}).get("muted", True)

    def set_muted(self, muted: bool):
        """Save muted preference."""
        if "preferences" not in self.settings:
            self.settings["preferences"] = {}
        self.settings["preferences"]["muted"] = muted

    def get_feature_mode(self) -> str:
        """Get feature mode preference."""
        return self.settings.get("preferences", {}).get("feature_mode", "Original")

    def set_feature_mode(self, mode: str):
        """Save feature mode preference."""
        if "preferences" not in self.settings:
            self.settings["preferences"] = {}
        self.settings["preferences"]["feature_mode"] = mode
