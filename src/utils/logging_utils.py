"""
Logging utilities for MBO Trading Strategy Analyzer v5.
Professional logging system with structured output and categorization.

Improvements over v4:
- Cleaner category detection
- Better multiprocess support
- Simplified configuration
"""
# pylint: disable=broad-exception-caught,too-many-locals

import logging
import os
import re
import sys
import traceback
import warnings as py_warnings
from datetime import datetime
from enum import Enum
from typing import Optional


class LogCategory(Enum):
    """Log categories for structured logging."""

    APP = "APP"  # Application lifecycle (start, stop, init)
    UI = "UI"  # User interface events
    DATA = "DATA"  # Data loading, processing, conversion
    ANALYSIS = "ANALYSIS"  # Analysis engine operations
    MODEL = "MODEL"  # Model training, prediction
    REPORT = "REPORT"  # Report generation
    SYSTEM = "SYSTEM"  # System-level events (stdout/stderr capture)
    CONFIG = "CONFIG"  # Configuration changes
    ERROR = "ERROR"  # Errors and exceptions


class LogLevel:
    """
    Log level constants with colors for GUI.

    Levels for GUI display:
    - INFO: Important messages shown in GUI (green) - data loaded, analysis complete, etc.
    - DEBUG: Detailed messages for file log only (not shown in GUI)
    - WARNING: Warnings shown in GUI (orange)
    - ERROR/CRITICAL: Errors shown in GUI (red)
    """

    # GUI display levels
    INFO = "info"          # Important - shown in GUI (green)
    DEBUG = "debug"        # Detailed - file only (not shown in GUI)
    WARNING = "warning"    # Warnings - shown in GUI (orange)
    ERROR = "error"        # Errors - shown in GUI (red)
    CRITICAL = "critical"  # Critical - shown in GUI (red)

    # Legacy aliases for backward compatibility
    NORMAL = "info"

    # Colors for GUI display
    COLORS = {
        "info": "#2ecc71",     # Green - important info
        "debug": None,         # Not shown in GUI
        "warning": "#f39c12",  # Orange
        "error": "#e74c3c",    # Red
        "critical": "#e74c3c", # Red
        "normal": "#2ecc71",   # Legacy alias
    }

    # Which levels should appear in GUI
    GUI_VISIBLE = {"info", "warning", "error", "critical", "normal"}


def create_log_message(event: str, details: Optional[str] = None, **kwargs) -> str:
    """
    Create a standardized log message.

    Args:
        event: Event description
        details: Optional additional details
        **kwargs: Additional key-value pairs to include

    Returns:
        Formatted log message
    """
    msg_parts = [event]

    if details:
        msg_parts.append(f"- {details}")

    if kwargs:
        extras = ", ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
        if extras:
            msg_parts.append(f"[{extras}]")

    return " ".join(msg_parts)


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    Used for capturing stdout/stderr in debug mode.
    """

    def __init__(self, logger: logging.Logger, log_level: int = logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf: str):
        """Write buffer content to logger."""
        for line in buf.rstrip().splitlines():
            stripped = line.rstrip()
            if stripped:  # Skip empty lines
                self.logger.log(self.log_level, stripped)

    def flush(self):
        """Flush the stream (no-op for logger)."""


class ProfessionalFormatter(logging.Formatter):
    """
    Professional log formatter with structured output.

    Output format:
    HH:MM:SS | CATEGORY | LEVEL | message

    Example:
    14:32:15 | ANALYSIS | INFO  | Analysis started - model=LightGBM, strategies=4500
    """

    # Level name padding for alignment
    LEVEL_PADDING = {
        "DEBUG": "DEBUG",
        "INFO": "INFO ",
        "WARNING": "WARN ",
        "ERROR": "ERROR",
        "CRITICAL": "CRIT ",
    }

    # Logger name to category mapping
    CATEGORY_MAP = {
        "root": LogCategory.APP,
        "GUI": LogCategory.UI,
        "__main__": LogCategory.APP,
        "analysis": LogCategory.ANALYSIS,
        "data": LogCategory.DATA,
        "models": LogCategory.MODEL,
        "core": LogCategory.ANALYSIS,
        "STDOUT": LogCategory.SYSTEM,
        "STDERR": LogCategory.ERROR,
    }

    # Patterns to detect category from message content
    CATEGORY_PATTERNS = [
        (re.compile(r"Language changed|Theme changed|clicked|toggle|pressed|tab", re.I), LogCategory.UI),
        (re.compile(r"loaded|converted|file|folder|parquet|excel", re.I), LogCategory.DATA),
        (re.compile(r"analysis|strategies|workers|pool|CPU|GPU", re.I), LogCategory.ANALYSIS),
        (re.compile(r"model|training|prediction|forecast|epoch", re.I), LogCategory.MODEL),
        (re.compile(r"report|export|generated|HTML|markdown", re.I), LogCategory.REPORT),
        (re.compile(r"config|setting|window|version", re.I), LogCategory.CONFIG),
        (re.compile(r"error|failed|exception|crash", re.I), LogCategory.ERROR),
    ]

    def format(self, record: logging.LogRecord) -> str:
        # Get timestamp
        timestamp = self.formatTime(record, "%H:%M:%S")

        # Get padded level name
        level = self.LEVEL_PADDING.get(record.levelname, record.levelname[:5])

        # Determine category
        category = self._get_category(record)

        # Clean up the message
        message = self._clean_message(record.getMessage())

        # Format: timestamp | category | level | message
        return f"{timestamp} | {category.value:8} | {level} | {message}"

    def _get_category(self, record: logging.LogRecord) -> LogCategory:
        """Determine the log category based on logger name and message content."""
        # Check explicit category in record (passed via extra)
        if hasattr(record, "category") and isinstance(record.category, LogCategory):
            return record.category

        # Check logger name - direct mapping
        logger_name = record.name
        if logger_name in self.CATEGORY_MAP:
            return self.CATEGORY_MAP[logger_name]

        # Check for prefix matches (e.g., 'models.statistical.arima')
        for prefix, category in self.CATEGORY_MAP.items():
            if logger_name.startswith(prefix + "."):
                return category

        # Check message patterns
        message = record.getMessage()
        for pattern, category in self.CATEGORY_PATTERNS:
            if pattern.search(message):
                return category

        # Default
        return LogCategory.APP

    def _clean_message(self, message: str) -> str:
        """Clean up the log message for consistency."""
        # Remove "USER ACTION: " prefix
        message = re.sub(r"^USER ACTION:\s*", "", message)
        # Remove redundant "Pressed " prefix
        message = re.sub(r"^Pressed\s+", "Clicked: ", message)
        return message


class SimpleGuiFormatter(logging.Formatter):
    """
    Simplified formatter for GUI log box display.

    Output format:
    [HH:MM:SS] message
    """

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%H:%M:%S")
        message = record.getMessage()
        message = re.sub(r"^USER ACTION:\s*", "", message)
        return f"[{timestamp}] {message}"


class GuiLogHandler(logging.Handler):
    """Custom logging handler that writes to the GUI log box."""

    def __init__(self, log_method):
        super().__init__()
        self.log_method = log_method

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            # Determine level for coloring
            if record.levelno >= logging.ERROR:
                level = LogLevel.CRITICAL
            elif record.levelno >= logging.WARNING:
                level = LogLevel.WARNING
            else:
                level = LogLevel.NORMAL
            self.log_method(msg, level)
        except (AttributeError, RuntimeError, OSError):
            self.handleError(record)


class SafeQueueHandler(logging.Handler):
    """
    A QueueHandler that gracefully handles connection errors.
    Used for multiprocess logging where workers send logs to main process.
    """

    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self._queue_broken = False

    def emit(self, record: logging.LogRecord):
        if self._queue_broken:
            return

        try:
            self.format(record)
            record.exc_info = None
            record.exc_text = None
            self.queue.put_nowait(record)
        except (BrokenPipeError, EOFError, OSError, AttributeError, FileNotFoundError):
            self._queue_broken = True
        except Exception:
            self._queue_broken = True


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with the professional formatter."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def configure_debug_logging(log_dir_base: str = "Log", filename_prefix: str = "mbo_debug") -> Optional[str]:
    """
    Configure enhanced debug logging used in Debug Mode.
    Sets up file logging, captures warnings, and system exceptions.

    Args:
        log_dir_base: Directory to save log files
        filename_prefix: Prefix for log filename

    Returns:
        Path to the log file created, or None if debug mode is off or failed.
    """
    if os.environ.get("MBO_DEBUG_MODE") != "1":
        return None

    try:
        # Create log directory
        log_dir = os.path.join(os.getcwd(), log_dir_base)
        os.makedirs(log_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        current_pid = os.getpid()

        if "worker" in filename_prefix:
            log_file = os.path.join(log_dir, f"{filename_prefix}_{current_pid}_{timestamp}.log")
        else:
            log_file = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.log")

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # File handler with professional formatter
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(ProfessionalFormatter())
        root_logger.addHandler(file_handler)

        # Enable DEBUG level for key application modules
        debug_modules = ["core", "models", "gui", "GUI", "analysis", "data"]
        for module_name in debug_modules:
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(logging.DEBUG)

        # Capture all Python warnings
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.setLevel(logging.DEBUG)
        py_warnings.simplefilter("always")

        # Suppress known PyTorch deprecation warnings
        py_warnings.filterwarnings(
            "ignore",
            message=r".*argument 'device' of Tensor\.pin_memory\(\) is deprecated.*",
            category=DeprecationWarning,
        )
        py_warnings.filterwarnings(
            "ignore",
            message=r".*argument 'device' of Tensor\.is_pinned\(\) is deprecated.*",
            category=DeprecationWarning,
        )

        # Third-party library logging (warnings only)
        third_party_libs = [
            "sklearn", "lightgbm", "xgboost", "catboost",
            "tensorflow", "torch", "keras",
            "pandas", "numpy", "numba",
            "statsmodels", "prophet", "pmdarima",
            "matplotlib", "plotly",
            "multiprocessing", "concurrent",
        ]
        for lib_name in third_party_libs:
            lib_logger = logging.getLogger(lib_name)
            lib_logger.setLevel(logging.WARNING)
            lib_logger.addHandler(file_handler)

        # Exception hook for unhandled exceptions
        original_excepthook = sys.excepthook

        def debug_excepthook(exc_type, exc_value, exc_tb):
            logging.error("=" * 60)
            logging.error("UNHANDLED EXCEPTION")
            logging.error("=" * 60)
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
            for line in tb_lines:
                for subline in line.rstrip().split("\n"):
                    logging.error(subline)
            original_excepthook(exc_type, exc_value, exc_tb)

        sys.excepthook = debug_excepthook

        # Redirect stdout/stderr to logging
        sys.stdout = StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
        sys.stderr = StreamToLogger(logging.getLogger("STDERR"), logging.ERROR)

        # Log startup info
        logging.info("=" * 60)
        logging.info("DEBUG MODE ENABLED - Full diagnostic logging active")
        logging.info("Log file: %s", log_file)
        logging.info("Python version: %s", sys.version)
        logging.info("Working directory: %s", os.getcwd())
        logging.info("=" * 60)

        return log_file

    except (IOError, OSError, AttributeError, ValueError) as e:
        print(f"Failed to setup debug logging: {e}")
        return None


def setup_worker_logging_queue(log_queue):
    """
    Configure worker process to send logs to the main process via a queue.

    Args:
        log_queue: multiprocessing.Queue to send log records to
    """
    if log_queue is None:
        return

    try:
        root = logging.getLogger()
        handler = SafeQueueHandler(log_queue)
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        logging.captureWarnings(True)
    except Exception as e:
        print(f"Failed to setup worker logging queue: {e}")
