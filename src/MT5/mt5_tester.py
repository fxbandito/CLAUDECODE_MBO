"""
MT5 Backtester module for running MetaTrader 5 Strategy Tester from Python.
"""
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional
import logging


class MT5Backtester:
    """
    Wrapper class to run MetaTrader 5 Strategy Tester from Python.
    """

    # MT5 Tester config keys - NO "Test" prefix needed!
    VALID_TESTER_KEYS = {
        'Expert', 'Symbol', 'Period', 'FromDate', 'ToDate', 'Model',
        'Optimization', 'ExecutionMode', 'Deposit', 'Currency', 'Leverage',
        'Visual', 'Report', 'ReplaceReport', 'ForwardMode', 'ProfitInPips',
        'OptimizationCriterion', 'ShutdownTerminal'
    }

    # Default MT5 data folder location
    DEFAULT_DATA_PATH = (
        r"C:\Users\bandi\AppData\Roaming\MetaQuotes\Terminal"
        r"\0DBD99CB9F797FCDB51D89D621359CB6"
    )

    def __init__(
        self,
        terminal_path: str = r"C:\Program Files\MetaTrader 5.3\terminal64.exe",
        data_path: Optional[str] = None
    ):
        """
        Initialize the MT5Backtester.

        Args:
            terminal_path: Full path to the terminal64.exe executable.
            data_path: Path to MT5 data folder (where MQL5 folder is located).
        """
        self.terminal_path = terminal_path
        self.data_path = data_path or self.DEFAULT_DATA_PATH
        if not os.path.exists(self.terminal_path):
            logging.warning("MT5 terminal not found at: %s", self.terminal_path)

    def create_config(
        self,
        params: Dict[str, Any],
        config_path: str = "backtest_config.ini"
    ) -> str:
        """
        Creates an INI configuration file for the MT5 Strategy Tester.

        Args:
            params: Dictionary of parameters for the tester.
            config_path: Path where the INI file should be saved.

        Returns:
            Absolute path to the created config file.
        """
        abs_config_path = os.path.abspath(config_path)

        # Separation of Tester settings and Expert Inputs
        tester_settings = {k: v for k, v in params.items() if k != 'Inputs'}
        inputs = params.get('Inputs', {})

        # Build INI content manually for proper MT5 format
        lines = ['[Tester]']

        for key, value in tester_settings.items():
            str_value = str(value)

            # Handle Leverage format: "1:500" -> "500"
            if key == 'Leverage' and ':' in str_value:
                str_value = str_value.rsplit(':', maxsplit=1)[-1]

            # Convert forward slashes to backslashes for paths
            if key in ('Expert', 'Report') or '/' in str_value:
                str_value = str_value.replace('/', '\\')

            lines.append(f'{key}={str_value}')

        # Add ShutdownTerminal to close MT5 after backtest
        lines.append('ShutdownTerminal=1')

        # Handle EA Inputs in [TesterInputs] section
        if inputs:
            lines.append('')
            lines.append('[TesterInputs]')
            for k, v in inputs.items():
                lines.append(f'{k}={v}')

        # Write to file - MT5 expects UTF-16 LE encoding (with BOM)
        with open(abs_config_path, 'w', encoding='utf-16') as configfile:
            configfile.write('\n'.join(lines))

        return abs_config_path

    def run_backtest(
        self,
        config_path: str,
        timeout: int = None,
        output_folder: Optional[str] = None,
        report_name: Optional[str] = None
    ) -> bool:
        """
        Runs the backtest by executing terminal64.exe with the config file.

        Args:
            config_path: Path to the configuration INI file.
            timeout: Optional timeout in seconds for the backtest.
            output_folder: Folder to copy the report to after completion.
            report_name: Name of the report file (without extension).

        Returns:
            True if subprocess finished successfully, False otherwise.
        """
        if not os.path.exists(self.terminal_path):
            logging.error(
                "MT5 terminal executable not found at %s",
                self.terminal_path
            )
            return False

        abs_config_path = os.path.abspath(config_path)
        cmd = [self.terminal_path, f"/config:{abs_config_path}"]

        logging.info("Starting MT5 Backtest with command: %s", ' '.join(cmd))

        try:
            # Don't use check=True - MT5 may return non-zero even on success
            result = subprocess.run(cmd, timeout=timeout, check=False)

            logging.info("MT5 process finished with return code: %d", result.returncode)

            # Copy report to output folder if specified
            if output_folder and report_name:
                self._copy_report(report_name, output_folder)

            return True
        except subprocess.TimeoutExpired:
            logging.error("MT5 backtest timed out after %d seconds", timeout)
            return False
        except OSError as e:
            logging.error("Error starting MT5 process: %s", e)
            return False

    def _copy_report(self, report_name: str, output_folder: str):
        """Copy the generated report from MT5 folder to output folder."""
        possible_locations = [
            os.path.join(self.data_path, "MQL5", "Files"),
            os.path.join(self.data_path, "Tester"),
            self.data_path,
        ]

        for location in possible_locations:
            source_htm = os.path.join(location, f"{report_name}.htm")
            source_html = os.path.join(location, f"{report_name}.html")

            for source in [source_htm, source_html]:
                if os.path.exists(source):
                    dest = os.path.join(output_folder, os.path.basename(source))
                    try:
                        shutil.copy2(source, dest)
                        logging.info("Report copied to: %s", dest)
                        return
                    except (OSError, IOError) as e:
                        logging.error("Failed to copy report: %s", e)

        logging.warning("Report file not found in expected locations")

    def parse_report_xml(self, report_path: str) -> Dict[str, str]:
        """
        Parses an XML report generated by MT5 Strategy Tester.

        Args:
            report_path: Path to the report file (must be .xml).

        Returns:
            Dictionary containing key metrics extracted from the report.
        """
        if not os.path.exists(report_path):
            logging.error("Report file not found: %s", report_path)
            return {}

        try:
            tree = ET.parse(report_path)
            root = tree.getroot()

            results = {}
            for child in root.iter():
                if child.tag in ['Profit', 'ExpectedPayoff', 'Drawdown',
                                 'Trades', 'SharpeRatio', 'RecoveryFactor',
                                 'ProfitFactor']:
                    results[child.tag] = child.text

            return results

        except ET.ParseError as e:
            logging.error("Failed to parse XML report: %s", e)
            return {}
