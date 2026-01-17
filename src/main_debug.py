"""
MBO Trading Strategy Analyzer v5 - Debug Entry Point

This script enables full diagnostic logging:
- All log messages saved to Log/ directory
- stdout/stderr captured to log file
- Unhandled exceptions logged with full traceback
- Third-party library warnings captured

Usage:
    python main_debug.py
    python -m main_debug

Log files are saved to:
    ./Log/mbo_debug_YYYY-MM-DD_HH-MM-SS.log

Environment:
    Sets MBO_DEBUG_MODE=1 before importing main module.
"""

import multiprocessing
import os
import sys


def run_debug() -> None:
    """
    Run the application in debug mode.

    Sets the MBO_DEBUG_MODE environment variable and launches the main app.
    The import is done inside this function to ensure env var is set first.
    """
    # Enable debug mode BEFORE importing main
    os.environ["MBO_DEBUG_MODE"] = "1"

    # Import main module (after env var is set)
    # pylint: disable=import-outside-toplevel
    from main import main as run_app
    run_app()


if __name__ == "__main__":
    # Windows multiprocessing támogatás
    multiprocessing.freeze_support()

    # Ensure we're running from the correct directory
    if not os.path.exists("main.py"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        sys.path.insert(0, script_dir)

    run_debug()
