"""
Comparator module for strategy report comparison functionality.

This module provides functions for comparing strategy analysis reports
across different dimensions:

- Horizon comparison: Compare reports by forecast horizon (1W, 1M, 3M, 6M, 1Y)
- Main Data comparison: Overview of all analyzed reports
- Main Data AR comparison: Weekly forecast breakdown from All Results reports
- Main Data MR comparison: Monthly forecast breakdown from Monthly Results reports (4-4-5 calendar)

Usage:
    from analysis.comparator import scan_reports, parse_report, generate_html_report
    from analysis.comparator import generate_main_data_report
    from analysis.comparator import generate_main_data_ar_report
    from analysis.comparator import generate_main_data_mr_report
"""

# Base utilities
from .base import (
    get_timestamp,
    parse_time_to_seconds,
    safe_divide,
    scan_ar_reports,
    scan_mr_reports,
    scan_reports,
)

# Horizon comparison functions
from .horizon import (
    generate_html_report,
    parse_report,
)

# Main Data comparison functions
from .main_data import (
    generate_main_data_report,
    parse_main_data_report,
)

# Main Data AR comparison functions
from .main_data_ar import (
    generate_main_data_ar_report,
    parse_ar_report,
)

# Main Data MR comparison functions
from .main_data_mr import (
    generate_main_data_mr_report,
    parse_mr_report,
)

__all__ = [
    # Base utilities
    "scan_reports",
    "scan_ar_reports",
    "scan_mr_reports",
    "parse_time_to_seconds",
    "get_timestamp",
    "safe_divide",
    # Horizon comparison
    "parse_report",
    "generate_html_report",
    # Main Data comparison
    "parse_main_data_report",
    "generate_main_data_report",
    # Main Data AR comparison
    "parse_ar_report",
    "generate_main_data_ar_report",
    # Main Data MR comparison
    "parse_mr_report",
    "generate_main_data_mr_report",
]
