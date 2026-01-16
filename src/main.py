"""
MBO Trading Strategy Analyzer v5 - Desktop Entry Point

Main entry point for the application.
For debug mode with full logging, use main_debug.py instead.
"""
# pylint: disable=wrong-import-position
# Note: Import order is intentional - environment variables and warnings must be set
# before importing customtkinter and application modules.

import multiprocessing
import os
import sys
import warnings
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress PyTorch deprecation warnings (before any imports)
os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")

# Configure Julia/PySR environment (prevents segfaults)
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

# Suppress known PyTorch warnings
warnings.filterwarnings(
    "ignore",
    message="Deallocating Tensor that still has live PyObject references",
)
warnings.filterwarnings(
    "ignore",
    message=r".*argument 'device' of Tensor\.pin_memory\(\) is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*argument 'device' of Tensor\.is_pinned\(\) is deprecated.*",
    category=DeprecationWarning,
)

import customtkinter as ctk

from gui.app import MBOApp
from utils.logging_utils import configure_debug_logging


def main():
    """Fő belépési pont a desktop alkalmazáshoz."""
    # Debug logging (only if MBO_DEBUG_MODE=1)
    log_file = configure_debug_logging()

    # CustomTkinter alapbeállítások
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    # Disable automatic DPI scaling (prevents crashes on multi-monitor setups)
    ctk.deactivate_automatic_dpi_awareness()

    # Alkalmazás indítása
    app = MBOApp()

    # Ha debug mode, jelezzük az app-nak
    if log_file:
        app._log(f"Debug mode enabled. Log: {log_file}")  # pylint: disable=protected-access

    app.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
