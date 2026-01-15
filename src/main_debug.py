"""
MBO Trading Strategy Analyzer v5 - Debug Entry Point

This script enables full diagnostic logging:
- All log messages saved to Log/ directory
- stdout/stderr captured to log file
- Unhandled exceptions logged with full traceback
- Third-party library warnings captured

Usage:
    python main_debug.py

Log files are saved to:
    ./Log/mbo_debug_YYYY-MM-DD_HH-MM-SS.log
"""

import multiprocessing
import os

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Enable debug mode BEFORE importing main
    os.environ["MBO_DEBUG_MODE"] = "1"

    # Import and run main
    import main

    main.main()
