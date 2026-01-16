"""
Inspection Engine - Compare forecast reports with actual benchmark results.
"""
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ForecastRecord:
    """A single forecast record from a markdown report."""
    model: str = ""
    forecast_year: str = ""
    horizon: str = ""  # 1 Week, 1 Month, 3 Months, 6 Months, 1 Year
    predicted_rank: int = 0
    strategy_no: int = 0
    predicted_profit: float = 0.0
    currency_pair: str = ""
    training_years: str = ""
    file_path: str = ""


@dataclass
class BenchmarkRecord:
    """A single benchmark record from Excel."""
    strategy_no: int = 0
    actual_profit: float = 0.0
    actual_rank: int = 0
    horizon: str = ""


@dataclass
class ComparisonRecord:
    """Comparison between forecast and benchmark."""
    model: str = ""
    forecast_year: str = ""
    horizon: str = ""
    predicted_rank: int = 0
    actual_rank: int = 0
    rank_diff: int = 0
    predicted_profit: float = 0.0
    actual_profit: float = 0.0
    actual_winner: str = ""
    winner_profit: float = 0.0
    strategy_no: int = 0
    currency_pair: str = ""
    training_years: str = ""


class InspectionEngine:
    """
    Engine for inspecting forecast accuracy against benchmark data.

    Workflow:
    1. Parse markdown reports from a folder
    2. Parse benchmark Excel file
    3. Compare results and calculate accuracy metrics
    4. Generate HTML/MD reports
    """

    # Horizon definitions - weeks to aggregate
    HORIZON_WEEKS = {
        "1 Week": 1,
        "1 Month": 4,
        "3 Months": 13,
        "6 Months": 26,
        "1 Year": 52
    }

    def __init__(self):
        self.forecasts: List[ForecastRecord] = []
        self.benchmarks: Dict[str, List[BenchmarkRecord]] = {}  # horizon -> records
        self.comparisons: List[ComparisonRecord] = []

    def parse_markdown_reports(self, folder_path: str, mode: str = "standard") -> List[ForecastRecord]:
        """
        Parse markdown report files from a folder.

        Args:
            folder_path: Path to folder containing .md files
            mode: "standard" for normal reports, "ar" for AR_ prefixed reports

        Returns:
            List of ForecastRecord objects
        """
        self.forecasts = []

        if not os.path.exists(folder_path):
            return self.forecasts

        md_files = [f for f in os.listdir(folder_path) if f.endswith('.md')]

        # Filter by mode
        if mode == "ar":
            md_files = [f for f in md_files if f.startswith("AR_")]
        else:
            md_files = [f for f in md_files if not f.startswith("AR_")]

        for filename in md_files:
            file_path = os.path.join(folder_path, filename)

            try:
                records = self._parse_single_md_file(file_path)
                self.forecasts.extend(records)
            except Exception as e:
                print(f"Error parsing {filename}: {e}")

        return self.forecasts

    def _parse_single_md_file(self, file_path: str) -> List[ForecastRecord]:
        """Parse a single markdown report file."""
        records = []

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        filename = os.path.basename(file_path)

        # Extract metadata from filename or content
        # Expected format: Model_PAIR_YEARS_YEAR.md or similar
        currency_pair = self._extract_currency_pair(content, filename)
        training_years = self._extract_training_years(content, filename)
        model_name = self._extract_model_name(content, filename)
        forecast_year = self._extract_forecast_year(content, filename)

        # Parse forecast tables - look for markdown tables with predictions
        # Format: | Rank | Strategy | Profit | ... |
        table_pattern = r'\|[^\n]+\|[\n\r]+\|[-:\s|]+\|[\n\r]+((?:\|[^\n]+\|[\n\r]*)+)'

        for table_match in re.finditer(table_pattern, content):
            table_content = table_match.group(0)

            # Determine horizon from context (look for headers before table)
            horizon = self._detect_horizon(content, table_match.start())

            # Parse rows
            rows = re.findall(r'\|([^|]+)\|([^|]+)\|([^|]+)\|', table_content)

            for row in rows:
                if len(row) >= 3:
                    try:
                        # Try to parse rank and strategy
                        rank_str = row[0].strip()
                        strategy_str = row[1].strip()
                        profit_str = row[2].strip()

                        if rank_str.isdigit() and strategy_str.isdigit():
                            record = ForecastRecord(
                                model=model_name,
                                forecast_year=forecast_year,
                                horizon=horizon,
                                predicted_rank=int(rank_str),
                                strategy_no=int(strategy_str),
                                predicted_profit=self._parse_profit(profit_str),
                                currency_pair=currency_pair,
                                training_years=training_years,
                                file_path=file_path
                            )
                            records.append(record)
                    except (ValueError, IndexError):
                        continue

        return records

    def _extract_currency_pair(self, content: str, filename: str) -> str:
        """Extract currency pair from content or filename."""
        # Try filename first
        pairs = ['EURUSD', 'EURJPY', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD']
        for pair in pairs:
            if pair in filename.upper():
                return pair

        # Try content
        match = re.search(r'(EUR|GBP|USD|AUD|JPY|CHF|CAD)(USD|JPY|EUR|GBP|CHF|CAD|AUD)', content.upper())
        if match:
            return match.group(0)

        return "UNKNOWN"

    def _extract_training_years(self, content: str, filename: str) -> str:
        """Extract training years from content or filename."""
        # Look for patterns like 2020-2023 or (2020-2023)
        match = re.search(r'(\d{4})\s*[-–]\s*(\d{4})', content)
        if match:
            return f"{match.group(1)}-{match.group(2)}"

        match = re.search(r'(\d{4})\s*[-–]\s*(\d{4})', filename)
        if match:
            return f"{match.group(1)}-{match.group(2)}"

        return ""

    def _extract_model_name(self, content: str, filename: str) -> str:
        """Extract model name from content or filename."""
        models = ['ARIMA', 'Prophet', 'LSTM', 'XGBoost', 'LightGBM', 'CatBoost',
                  'RandomForest', 'GradientBoosting', 'Linear', 'Ridge', 'Lasso',
                  'ElasticNet', 'SVR', 'KNN', 'MLP', 'Ensemble']

        for model in models:
            if model.lower() in filename.lower() or model.lower() in content.lower():
                return model

        # Try to extract from first line
        first_line = content.split('\n')[0] if content else ""
        if '#' in first_line:
            return first_line.replace('#', '').strip()[:50]

        return os.path.splitext(os.path.basename(filename))[0]

    def _extract_forecast_year(self, content: str, filename: str) -> str:
        """Extract forecast year from content or filename."""
        # Look for 4-digit year near keywords like "forecast", "prediction"
        match = re.search(r'(?:forecast|prediction|year)[:\s]*(\d{4})', content, re.I)
        if match:
            return match.group(1)

        # Try filename - usually last 4-digit number
        years = re.findall(r'\d{4}', filename)
        if years:
            return years[-1]

        return str(datetime.now().year)

    def _detect_horizon(self, content: str, position: int) -> str:
        """Detect the horizon for a table based on nearby headers."""
        # Look backwards for horizon indicators
        search_area = content[max(0, position-500):position]

        horizon_patterns = [
            (r'1\s*week', "1 Week"),
            (r'1\s*month', "1 Month"),
            (r'3\s*months?', "3 Months"),
            (r'6\s*months?', "6 Months"),
            (r'1\s*year', "1 Year"),
            (r'52\s*weeks?', "1 Year"),
        ]

        for pattern, horizon in reversed(horizon_patterns):
            if re.search(pattern, search_area, re.I):
                return horizon

        return "1 Month"  # Default

    def _parse_profit(self, profit_str: str) -> float:
        """Parse profit string to float."""
        try:
            # Remove currency symbols and spaces
            clean = re.sub(r'[^\d.\-,]', '', profit_str)
            clean = clean.replace(',', '.')
            return float(clean)
        except (ValueError, AttributeError):
            return 0.0

    def parse_benchmark_excel(self, excel_path: str) -> Dict[str, List[BenchmarkRecord]]:
        """
        Parse benchmark Excel file with actual results.

        Expected structure: 10 columns per week (No., Profit, ...)

        Args:
            excel_path: Path to Excel file

        Returns:
            Dictionary mapping horizon to list of BenchmarkRecord
        """
        self.benchmarks = {}

        if not os.path.exists(excel_path):
            return self.benchmarks

        try:
            df = pd.read_excel(excel_path, header=None)
        except Exception as e:
            print(f"Error loading Excel: {e}")
            return self.benchmarks

        # Parse weekly data
        weekly_results = self._parse_weekly_columns(df)

        # Aggregate by horizon
        for horizon, weeks in self.HORIZON_WEEKS.items():
            records = self._aggregate_weeks(weekly_results, weeks)
            self.benchmarks[horizon] = records

        return self.benchmarks

    def _parse_weekly_columns(self, df: pd.DataFrame) -> Dict[int, List[Tuple[int, float]]]:
        """
        Parse weekly columns from DataFrame.

        Returns:
            Dictionary mapping week number to list of (strategy_no, profit) tuples
        """
        weekly_results = {}
        cols_per_week = 10

        num_weeks = min(52, df.shape[1] // cols_per_week)

        for week in range(num_weeks):
            start_col = week * cols_per_week

            # Column 0 = Strategy No, Column 1 = Profit (typically)
            try:
                no_col = start_col
                profit_col = start_col + 1

                week_data = []
                for row_idx in range(df.shape[0]):
                    try:
                        strategy_no = int(df.iloc[row_idx, no_col])
                        profit = float(df.iloc[row_idx, profit_col])
                        week_data.append((strategy_no, profit))
                    except (ValueError, TypeError):
                        continue

                if week_data:
                    weekly_results[week + 1] = week_data

            except Exception:
                continue

        return weekly_results

    def _aggregate_weeks(self, weekly_results: Dict[int, List[Tuple[int, float]]],
                         num_weeks: int) -> List[BenchmarkRecord]:
        """
        Aggregate weekly results for a given horizon.

        Args:
            weekly_results: Weekly data
            num_weeks: Number of weeks to aggregate

        Returns:
            List of BenchmarkRecord with actual rank based on aggregated profit
        """
        # Sum profits for each strategy over the weeks
        strategy_profits = {}

        for week in range(1, min(num_weeks + 1, max(weekly_results.keys()) + 1)):
            if week not in weekly_results:
                continue

            for strategy_no, profit in weekly_results[week]:
                if strategy_no not in strategy_profits:
                    strategy_profits[strategy_no] = 0.0
                strategy_profits[strategy_no] += profit

        # Calculate ranks based on total profit (higher profit = better rank)
        sorted_strategies = sorted(strategy_profits.items(), key=lambda x: x[1], reverse=True)

        records = []
        for rank, (strategy_no, total_profit) in enumerate(sorted_strategies, 1):
            records.append(BenchmarkRecord(
                strategy_no=strategy_no,
                actual_profit=total_profit,
                actual_rank=rank,
                horizon=f"{num_weeks} weeks"
            ))

        return records

    def compare_results(self) -> List[ComparisonRecord]:
        """
        Compare forecasts with benchmarks.

        Returns:
            List of ComparisonRecord objects
        """
        self.comparisons = []

        if not self.forecasts or not self.benchmarks:
            return self.comparisons

        for forecast in self.forecasts:
            horizon = forecast.horizon

            if horizon not in self.benchmarks:
                continue

            # Find matching benchmark by strategy number
            benchmark = None
            for b in self.benchmarks[horizon]:
                if b.strategy_no == forecast.strategy_no:
                    benchmark = b
                    break

            if not benchmark:
                continue

            # Find winner for this horizon
            winner = min(self.benchmarks[horizon], key=lambda x: x.actual_rank)

            comparison = ComparisonRecord(
                model=forecast.model,
                forecast_year=forecast.forecast_year,
                horizon=horizon,
                predicted_rank=forecast.predicted_rank,
                actual_rank=benchmark.actual_rank,
                rank_diff=abs(forecast.predicted_rank - benchmark.actual_rank),
                predicted_profit=forecast.predicted_profit,
                actual_profit=benchmark.actual_profit,
                actual_winner=f"Strategy {winner.strategy_no}",
                winner_profit=winner.actual_profit,
                strategy_no=forecast.strategy_no,
                currency_pair=forecast.currency_pair,
                training_years=forecast.training_years
            )
            self.comparisons.append(comparison)

        return self.comparisons

    def generate_html_report(self, output_path: str) -> str:
        """
        Generate HTML report with comparison results.

        Args:
            output_path: Path to save HTML file

        Returns:
            Path to generated file
        """
        if not self.comparisons:
            return ""

        html = self._build_html_report()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return output_path

    def _build_html_report(self) -> str:
        """Build HTML content for report."""
        # Group by model
        by_model = {}
        for comp in self.comparisons:
            if comp.model not in by_model:
                by_model[comp.model] = []
            by_model[comp.model].append(comp)

        rows_html = ""
        for comp in self.comparisons:
            rank_class = "positive" if comp.rank_diff <= 2 else "negative" if comp.rank_diff > 5 else ""
            rows_html += f"""
            <tr>
                <td>{comp.model}</td>
                <td>{comp.forecast_year}</td>
                <td>{comp.horizon}</td>
                <td>{comp.predicted_rank}</td>
                <td>{comp.actual_rank}</td>
                <td class="{rank_class}">{comp.rank_diff}</td>
                <td>{comp.predicted_profit:,.2f}</td>
                <td>{comp.actual_profit:,.2f}</td>
                <td>{comp.actual_winner}</td>
                <td>{comp.winner_profit:,.2f}</td>
            </tr>
            """

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Inspection Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', sans-serif;
            background: #1a1a2e;
            color: #eaeaea;
            padding: 20px;
        }}
        h1 {{ color: #4dabf7; text-align: center; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 10px;
            border: 1px solid #2a2a4a;
            text-align: center;
        }}
        th {{
            background: #0f3460;
            color: #ffd93d;
        }}
        tr:nth-child(even) {{ background: #16213e; }}
        tr:hover {{ background: #1f4068; }}
        .positive {{ color: #00d26a; }}
        .negative {{ color: #ff6b6b; }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin: 20px 0;
        }}
        .summary-box {{
            background: #0f3460;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}
        .summary-value {{
            font-size: 2em;
            font-weight: bold;
            color: #4dabf7;
        }}
        .summary-label {{ color: #a0a0a0; }}
    </style>
</head>
<body>
    <h1>Forecast Inspection Report</h1>

    <div class="summary">
        <div class="summary-box">
            <div class="summary-value">{len(self.comparisons)}</div>
            <div class="summary-label">Total Comparisons</div>
        </div>
        <div class="summary-box">
            <div class="summary-value">{len(by_model)}</div>
            <div class="summary-label">Models</div>
        </div>
        <div class="summary-box">
            <div class="summary-value">{sum(1 for c in self.comparisons if c.rank_diff <= 2)}</div>
            <div class="summary-label">Accurate (±2 rank)</div>
        </div>
        <div class="summary-box">
            <div class="summary-value">{sum(c.rank_diff for c in self.comparisons) / len(self.comparisons):.1f}</div>
            <div class="summary-label">Avg Rank Diff</div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Year</th>
                <th>Horizon</th>
                <th>Pred. Rank</th>
                <th>Actual Rank</th>
                <th>Rank Diff</th>
                <th>Pred. Profit</th>
                <th>Actual Profit</th>
                <th>Winner</th>
                <th>Winner Profit</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

    <p style="text-align: center; color: #a0a0a0; margin-top: 20px;">
        Generated by MBO Inspection Engine
    </p>
</body>
</html>"""
        return html

    def generate_md_report(self, output_path: str) -> str:
        """Generate Markdown report."""
        if not self.comparisons:
            return ""

        lines = [
            "# Forecast Inspection Report",
            "",
            f"**Total Comparisons:** {len(self.comparisons)}",
            "",
            "| Model | Year | Horizon | Pred. Rank | Actual Rank | Rank Diff | Pred. Profit | Actual Profit | Winner |",
            "|-------|------|---------|------------|-------------|-----------|--------------|---------------|--------|",
        ]

        for comp in self.comparisons:
            lines.append(
                f"| {comp.model} | {comp.forecast_year} | {comp.horizon} | "
                f"{comp.predicted_rank} | {comp.actual_rank} | {comp.rank_diff} | "
                f"{comp.predicted_profit:,.2f} | {comp.actual_profit:,.2f} | {comp.actual_winner} |"
            )

        content = '\n'.join(lines)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return output_path
