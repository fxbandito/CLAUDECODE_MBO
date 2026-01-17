"""
MBO Trading Strategy Analyzer v5 - Desktop Entry Point

Main entry point for the application.
For debug mode with full logging, use main_debug.py instead.

Environment variables:
    MBO_DEBUG_MODE: Set to "1" to enable debug logging
    PYTHON_JULIACALL_HANDLE_SIGNALS: Set to "yes" for Julia/PySR stability
"""
# pylint: disable=wrong-import-position
# Note: Import order is intentional - environment variables and warnings must be set
# before importing customtkinter and application modules.

import multiprocessing
import os
import sys
import warnings
from pathlib import Path

# =============================================================================
# PATH SETUP
# =============================================================================
sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# ENVIRONMENT CONFIGURATION (before any library imports)
# =============================================================================

# Suppress PyTorch deprecation warnings
os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")

# Configure Julia/PySR environment (prevents segfaults on Windows)
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

# =============================================================================
# WARNING FILTERS (suppress known harmless warnings)
# =============================================================================

# PyTorch tensor deallocation warning (harmless, occurs during cleanup)
warnings.filterwarnings(
    "ignore",
    message="Deallocating Tensor that still has live PyObject references",
)

# PyTorch 2.x deprecation warnings (will be fixed in future versions)
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

# =============================================================================
# APPLICATION IMPORTS (after environment setup)
# =============================================================================

import logging
import customtkinter as ctk

from gui.app import MBOApp
from utils.logging_utils import configure_debug_logging
from utils.system_check import SystemChecker


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def is_debug_mode() -> bool:
    """Check if debug mode is enabled via environment variable."""
    return os.environ.get("MBO_DEBUG_MODE", "0") == "1"


def setup_customtkinter() -> None:
    """Configure CustomTkinter appearance and settings."""
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    # Disable automatic DPI scaling (prevents crashes on multi-monitor setups)
    ctk.deactivate_automatic_dpi_awareness()


def run_system_diagnostics(debug_mode: bool) -> SystemChecker:
    """
    Rendszer diagnosztika futtatása.

    Args:
        debug_mode: Ha True, részletes jelentés a log fájlba.

    Returns:
        SystemChecker: A checker objektum a jelentésekhez.
    """
    checker = SystemChecker()
    checker.run_full_check()

    # Debug módban részletes jelentés a log fájlba
    if debug_mode:
        logger = logging.getLogger()
        logger.info("\n%s", checker.get_detailed_report())

        # Figyelmeztetések külön is
        if checker.status.warnings:
            for warning in checker.status.warnings:
                logger.warning("[SYSTEM] %s", warning)

        if checker.status.errors:
            for error in checker.status.errors:
                logger.error("[SYSTEM] %s", error)

    return checker


def main() -> None:
    """
    Fő belépési pont a desktop alkalmazáshoz.

    Inicializálja a debug loggingot (ha engedélyezve van),
    futtatja a rendszer diagnosztikát,
    beállítja a CustomTkinter-t, majd elindítja az alkalmazást.
    """
    # Debug logging csak ha MBO_DEBUG_MODE=1
    debug_mode = is_debug_mode()
    log_file = configure_debug_logging() if debug_mode else None

    # Rendszer diagnosztika futtatása
    checker = run_system_diagnostics(debug_mode)

    # CustomTkinter beállítások
    setup_customtkinter()

    # Alkalmazás létrehozása és indítása
    app = MBOApp()

    # Debug módban jelezzük a log fájl helyét
    if debug_mode and log_file:
        app.log_message(f"Debug mode enabled. Log: {log_file}", level="info")

    # Rendszer információk megjelenítése a GUI-ban
    app.log_message(checker.get_short_report(), level="info")

    # Figyelmeztetések megjelenítése a GUI-ban
    if checker.status.warnings:
        for warning in checker.status.warnings:
            app.log_message(f"System: {warning}", level="warning")

    if checker.status.errors:
        for error in checker.status.errors:
            app.log_message(f"System: {error}", level="error")

    app.mainloop()


if __name__ == "__main__":
    # Windows multiprocessing támogatás (exe-hez szükséges)
    multiprocessing.freeze_support()
    main()
