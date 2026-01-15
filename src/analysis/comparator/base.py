"""
Base utilities for report comparison functionality.
Contains shared helper functions used across comparison modules.
"""

import os
from datetime import datetime
from typing import List


def is_aggregate_report(file_path: str) -> bool:
    """
    Checks if a markdown file is an aggregate/comparison report by inspecting its content.
    Reads the first few lines to look for specific headers.

    Args:
        file_path: Path to the .md file

    Returns:
        True if it's an aggregate report (Main Data or Main Data AR), False otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            # Read first few lines (header should be at the top)
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                # Check for Main Data Comparison Report header
                if "# Main Data Comparison Report" in line:
                    return True
                # Check for Main Data AR Comparison Report header
                if "# Main Data AR Comparison Report" in line:
                    return True
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return False


def scan_reports(root_folder: str, exclude_ar: bool = True, exclude_mr: bool = True) -> List[str]:
    """
    Recursively finds all .md files in the given folder.

    Args:
        root_folder: Root directory to scan
        exclude_ar: If True, excludes AR_ prefixed files (All Results reports)
        exclude_mr: If True, excludes MR_ prefixed files (Monthly Results reports)

    Returns:
        List of file paths to .md reports
    """
    report_files = []
    # explicitly excluded filenames (legacy check)
    excluded_files = {"README.md", "main_data_comparison.md", "main_data_ar_comparison.md"}

    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".md") and file not in excluded_files:
                # Exclude files with "comparison" in the name (case-insensitive)
                if "comparison" in file.lower():
                    continue

                # Optionally exclude AR_ prefixed files
                if exclude_ar and file.startswith("AR_"):
                    continue

                # Optionally exclude MR_ prefixed files
                if exclude_mr and file.startswith("MR_"):
                    continue

                full_path = os.path.join(root, file)

                # Content-based check: Exclude aggregate reports even if renamed
                if is_aggregate_report(full_path):
                    continue

                report_files.append(full_path)
    return report_files


def scan_ar_reports(root_folder: str) -> List[str]:
    """
    Recursively finds all AR_ prefixed .md files in the given folder.
    These are the All Results reports containing weekly forecast breakdowns.

    Args:
        root_folder: Root directory to scan

    Returns:
        List of file paths to AR_ prefixed .md reports
    """
    ar_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".md") and file.startswith("AR_"):
                # Exclude files with "comparison" in the name (case-insensitive)
                if "comparison" in file.lower():
                    continue

                full_path = os.path.join(root, file)

                # Content-based check: Ensure we don't accidentally pick up an aggregate report
                # (though unlikely to start with AR_, better safe)
                if is_aggregate_report(full_path):
                    continue

                ar_files.append(full_path)
    return ar_files


def scan_mr_reports(root_folder: str) -> List[str]:
    """
    Recursively finds all MR_ prefixed .md files in the given folder.
    These are the Monthly Results reports containing 4-4-5 calendar breakdowns.

    Args:
        root_folder: Root directory to scan

    Returns:
        List of file paths to MR_ prefixed .md reports
    """
    mr_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".md") and file.startswith("MR_"):
                # Exclude files with "comparison" in the name (case-insensitive)
                if "comparison" in file.lower():
                    continue

                full_path = os.path.join(root, file)

                # Content-based check: Ensure we don't accidentally pick up an aggregate report
                if is_aggregate_report(full_path):
                    continue

                mr_files.append(full_path)
    return mr_files


def parse_time_to_seconds(time_str: str) -> float:
    """
    Convert time string (HH:MM:SS or MM:SS) to seconds.

    Args:
        time_str: Time string in format "HH:MM:SS" or "MM:SS"

    Returns:
        Time in seconds as float
    """
    try:
        parts = time_str.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return 0
    except (ValueError, IndexError):
        return 0


def get_timestamp() -> str:
    """Get current timestamp formatted for reports."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if division by zero

    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator
