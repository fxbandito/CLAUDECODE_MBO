"""
Inspection Engine - Compare forecast reports with actual benchmark results.

Enhanced version with:
- Recursive MD file search
- Complex AR_ mode parsing
- Plotly interactive charts
- Detailed Excel parsing
- Auto filename generation
"""
import glob
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field

import pandas as pd

# Optional Plotly import
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class ForecastRecord:
    """A single forecast record from a markdown report."""
    model: str = ""
    forecast_year: int = 0
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
    forecast_year: int = 0
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


@dataclass
class InspectionMetadata:
    """Metadata extracted from reports."""
    pair: str = "Unknown"
    years: str = "Unknown"
    benchmark_year: str = "Unknown"


class InspectionEngine:
    """
    Engine for inspecting forecast accuracy against benchmark data.

    Workflow:
    1. Parse markdown reports from a folder (recursive search)
    2. Parse benchmark Excel file
    3. Compare results and calculate accuracy metrics
    4. Generate HTML/MD reports with Plotly visualizations
    """  # pylint: disable=too-many-instance-attributes

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
        self.benchmarks: Dict[str, pd.DataFrame] = {}  # horizon -> DataFrame
        self.comparisons: List[ComparisonRecord] = []
        self.metadata: InspectionMetadata = InspectionMetadata()

    def parse_markdown_reports(
        self,
        folder_path: str,
        mode: str = "standard",
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Tuple[List[ForecastRecord], InspectionMetadata]:
        """
        Parse markdown report files from a folder (recursive search).

        Args:
            folder_path: Path to folder containing .md files
            mode: "standard" for normal reports, "ar" for AR_ prefixed reports
            progress_callback: Optional callback for progress updates (0-50%)

        Returns:
            Tuple of (List of ForecastRecord objects, InspectionMetadata)
        """
        self.forecasts = []
        self.metadata = InspectionMetadata()

        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return self.forecasts, self.metadata

        # Recursive glob search for MD files
        print(f"Searching in: {folder_path} [Mode: {mode}]")
        md_files = glob.glob(os.path.join(folder_path, "**/*.md"), recursive=True)

        # Filter by mode
        filtered_files = []
        for f in md_files:
            fname = os.path.basename(f)
            if mode == "standard" and not fname.startswith("AR_"):
                filtered_files.append(f)
            elif mode == "ar" and fname.startswith("AR_"):
                filtered_files.append(f)

        total_files = len(filtered_files)
        print(f"Found {total_files} valid markdown files for mode '{mode}'.")

        if total_files == 0:
            return self.forecasts, self.metadata

        for i, file_path in enumerate(filtered_files):
            if progress_callback:
                progress_callback((i / total_files) * 50)

            try:
                records = self._parse_single_md_file(file_path, mode)
                self.forecasts.extend(records)

                # Extract metadata from first valid file
                if self.metadata.pair == "Unknown" and records:
                    self.metadata.pair = records[0].currency_pair
                    self.metadata.years = records[0].training_years

            except (IOError, ValueError, IndexError, RuntimeError) as e:
                print(f"Error parsing {file_path}: {e}")
                continue

        print(f"Total forecasts parsed: {len(self.forecasts)}")
        return self.forecasts, self.metadata

    def _parse_single_md_file(
        self, file_path: str, mode: str = "standard"
    ) -> List[ForecastRecord]:
        """Parse a single markdown report file with complex extraction logic."""
        records = []

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        filename = os.path.basename(file_path)

        # Extract metadata
        currency_pair = self._extract_currency_pair(content, filename)
        training_years = self._extract_training_years(content, filename)
        model_name = self._extract_model_name(content, filename)
        forecast_year = self._extract_forecast_year(
            content, filename, training_years
        )

        # Try to parse Top 10 Strategies table (Standard format)
        table_section = re.search(
            r"## Top 10 Strategies by Horizon\s+(.*?)\n\n",
            content,
            re.DOTALL,
        )

        if table_section:
            records.extend(self._parse_standard_table(
                table_section.group(1), model_name, forecast_year,
                currency_pair, training_years, file_path
            ))
        else:
            # Fallback: AR Report Parsing
            records.extend(self._parse_ar_report(
                content, model_name, forecast_year,
                currency_pair, training_years, file_path
            ))

        return records

    def _parse_standard_table(
        self, table_content: str, model_name: str, forecast_year: int,
        currency_pair: str, training_years: str, file_path: str
    ) -> List[ForecastRecord]:
        """Parse standard Top 10 Strategies table."""
        records = []
        lines = table_content.strip().split("\n")

        # Find data start (after header separator)
        start_idx = 0
        for j, line in enumerate(lines):
            if "|---" in line or "| ---" in line:
                start_idx = j + 1
                break
        if start_idx == 0 and len(lines) > 2:
            start_idx = 2

        def parse_cell(cell_text: str) -> Tuple[Optional[str], Optional[float]]:
            """Parse cell like '#6127 (185.91)' -> ('6127', 185.91)"""
            m = re.search(r"#(\d+).*?\(.*?([\d,.-]+)\)", cell_text)
            if m:
                return m.group(1), float(m.group(2).replace(",", ""))
            return None, None

        horizons = ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"]

        for line in lines[start_idx:]:
            if not line.strip().startswith("|"):
                continue

            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) < 5:
                continue

            try:
                rank = int(parts[0])
            except ValueError:
                continue

            # Parse each horizon column
            for col_idx, horizon in enumerate(horizons):
                if col_idx + 1 < len(parts):
                    strategy, profit = parse_cell(parts[col_idx + 1])
                    if strategy:
                        records.append(ForecastRecord(
                            model=model_name,
                            forecast_year=forecast_year,
                            horizon=horizon,
                            predicted_rank=rank,
                            strategy_no=int(strategy),
                            predicted_profit=profit or 0.0,
                            currency_pair=currency_pair,
                            training_years=training_years,
                            file_path=file_path
                        ))

        return records

    def _parse_ar_report(
        self, content: str, model_name: str, forecast_year: int,
        currency_pair: str, training_years: str, file_path: str
    ) -> List[ForecastRecord]:
        """Parse AR (Auto-Regressive) report format."""
        records = []
        horizons_found = set()

        # 1. Parse Best Strategy Forecast section
        # Format: "- **1 Week**: (No. 6127) 185.91"
        best_forecast_matches = re.finditer(
            r"\*\*(\d+\s*(?:Week|Month|Months|Year)s?)\*\*:\s*"
            r"\(No\.\s*(\d+)\)\s*([\d,.-]+)",
            content,
        )

        for m in best_forecast_matches:
            horizon_raw = m.group(1).strip()
            strategy = m.group(2).strip()
            pred_profit = float(m.group(3).replace(",", ""))

            # Normalize horizon names
            horizon = self._normalize_horizon(horizon_raw)

            if horizon not in horizons_found:
                records.append(ForecastRecord(
                    model=model_name,
                    forecast_year=forecast_year,
                    horizon=horizon,
                    predicted_rank=1,
                    strategy_no=int(strategy),
                    predicted_profit=pred_profit,
                    currency_pair=currency_pair,
                    training_years=training_years,
                    file_path=file_path
                ))
                horizons_found.add(horizon)

        # 2. Fallback: Executive Summary Sentences
        if not horizons_found:
            summary_matches = re.finditer(
                r"overall best strategy \((.*?) horizon\) is "
                r"\*\*Strategy No\. (\d+)\*\*",
                content,
            )
            for m in summary_matches:
                horizon = m.group(1).strip()
                strategy = m.group(2).strip()
                pred_profit = 0.0

                # Try to find profit
                prof_m = re.search(r"Total Profit.*?\|\s*([\d\.-]+)", content)
                if prof_m:
                    try:
                        pred_profit = float(prof_m.group(1))
                    except ValueError:
                        pass

                records.append(ForecastRecord(
                    model=model_name,
                    forecast_year=forecast_year,
                    horizon=horizon,
                    predicted_rank=1,
                    strategy_no=int(strategy),
                    predicted_profit=pred_profit,
                    currency_pair=currency_pair,
                    training_years=training_years,
                    file_path=file_path
                ))

        # 3. Weekly Forecast Breakdown (Week 1 only)
        week1_match = re.search(r"\| Week 1 \| No\. (\d+) \|\s*([\d\.-]+)", content)
        if week1_match and "1 Week" not in horizons_found:
            strategy = week1_match.group(1)
            profit = float(week1_match.group(2))
            records.append(ForecastRecord(
                model=model_name,
                forecast_year=forecast_year,
                horizon="1 Week",
                predicted_rank=1,
                strategy_no=int(strategy),
                predicted_profit=profit,
                currency_pair=currency_pair,
                training_years=training_years,
                file_path=file_path
            ))

        return records

    def _normalize_horizon(self, horizon_raw: str) -> str:
        """Normalize horizon name to standard format."""
        if "Week" in horizon_raw:
            return "1 Week"
        elif "Month" in horizon_raw:
            if "3" in horizon_raw:
                return "3 Months"
            elif "6" in horizon_raw:
                return "6 Months"
            else:
                return "1 Month"
        elif "Year" in horizon_raw:
            return "1 Year"
        return horizon_raw

    def _extract_currency_pair(self, content: str, filename: str) -> str:
        """Extract currency pair from content or filename."""
        # Try filename first - look for pattern like AUDJPY, EURUSD etc.
        pairs = ['EURUSD', 'EURJPY', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF',
                 'USDCAD', 'AUDJPY', 'GBPJPY', 'EURGBP', 'NZDUSD', 'EURAUD']
        for pair in pairs:
            if pair in filename.upper():
                return pair

        # Try content
        match = re.search(
            r'(EUR|GBP|USD|AUD|JPY|CHF|CAD|NZD)(USD|JPY|EUR|GBP|CHF|CAD|AUD|NZD)',
            content.upper()
        )
        if match:
            return match.group(0)

        # Try extracting from filename pattern like Report_PAIR_(years)
        years_match = re.search(r"\(([\d\-_]+)\)", filename)
        if years_match:
            pre_years = filename.split("(")[0]
            clean_name = pre_years.replace("Report_", "").replace("AR_", "").strip("_ ")
            if clean_name:
                return clean_name

        return "Unknown"

    def _extract_training_years(self, content: str, filename: str) -> str:
        """Extract training years from content or filename."""
        # Look for patterns like (2020-2021-2022-2023) in filename
        years_in_filename = re.search(r"\((\d{4}(?:[-_]\d{4})*)\)", filename)
        if years_in_filename:
            return years_in_filename.group(1)

        # Try content
        match = re.search(r'(\d{4})\s*[-–]\s*(\d{4})', content)
        if match:
            return f"{match.group(1)}-{match.group(2)}"

        return "Unknown"

    def _extract_model_name(self, content: str, filename: str) -> str:
        """Extract model name from content or filename."""
        # Try **Method** field in content
        method_match = re.search(r"\*\*Method\*\*: (.*)", content)
        if method_match:
            return method_match.group(1).strip()

        # Try known model names
        models = ['ARIMA', 'Prophet', 'LSTM', 'XGBoost', 'LightGBM', 'CatBoost',
                  'RandomForest', 'GradientBoosting', 'Linear', 'Ridge', 'Lasso',
                  'ElasticNet', 'SVR', 'KNN', 'MLP', 'Ensemble']

        for model in models:
            if model.lower() in filename.lower() or model.lower() in content.lower():
                return model

        # Extract from filename parts
        parts = filename.split("_")
        if len(parts) > 1:
            return parts[0]

        return "Unknown"

    def _extract_forecast_year(self, content: str, filename: str, training_years: str) -> int:
        """Extract forecast year from content, filename, or training years."""
        # Calculate from training years (forecast = last training year + 1)
        if training_years and training_years != "Unknown":
            all_years = re.findall(r"\d{4}", training_years)
            if all_years:
                last_training_year = max(int(y) for y in all_years)
                return last_training_year + 1

        # Try **Date** field
        date_match = re.search(r"\*\*Date\*\*: (.*)", content)
        if date_match:
            try:
                report_date = datetime.strptime(
                    date_match.group(1).strip(), "%Y-%m-%d %H:%M"
                )
                return report_date.year
            except ValueError:
                pass

        # Try filename - last 4-digit number
        years = re.findall(r'\d{4}', filename)
        if years:
            return int(years[-1])

        return datetime.now().year

    def parse_benchmark_excel(
        self,
        file_path: str,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Tuple[Dict[str, pd.DataFrame], str]:
        """
        Parse benchmark Excel file with actual results.

        Args:
            file_path: Path to Excel file
            progress_callback: Optional callback for progress updates (50-80%)

        Returns:
            Tuple of (Dictionary mapping horizon to DataFrame, excel_year)
        """
        self.benchmarks = {}
        excel_year = "Unknown"

        if not os.path.exists(file_path):
            print(f"Excel file not found: {file_path}")
            return self.benchmarks, excel_year

        try:
            print(f"Parsing Benchmark: {file_path}")
            if progress_callback:
                progress_callback(60)

            df = pd.read_excel(file_path)

            n_cols = df.shape[1]
            n_weeks = n_cols // 10
            print(f"Detected {n_cols} columns. Implies {n_weeks} weeks (stride=10).")

            # Parse weekly profits
            weekly_profits = []

            for w in range(n_weeks):
                start_col = w * 10

                if start_col >= n_cols or start_col + 1 >= n_cols:
                    continue

                week_df = df.iloc[1:, [start_col, start_col + 1]].copy()
                week_df.columns = ["Strategy", "Profit"]

                week_df["Strategy"] = (
                    week_df["Strategy"].astype(str).str.replace("#", "").str.strip()
                )
                week_df["Profit"] = pd.to_numeric(week_df["Profit"], errors="coerce").fillna(0)
                week_df = week_df[week_df["Strategy"] != "nan"]

                weekly_profits.append(week_df)

            # Aggregate by horizon
            def aggregate_weeks(start_week: int, end_week: int) -> pd.DataFrame:
                end_week = min(end_week, len(weekly_profits))
                relevant_dfs = weekly_profits[start_week - 1:end_week]

                if not relevant_dfs:
                    return pd.DataFrame(columns=["Strategy", "Actual_Profit"])

                try:
                    big_df = pd.concat(relevant_dfs, ignore_index=True)
                    big_df = big_df[
                        big_df["Strategy"].notna()
                        & (big_df["Strategy"] != "nan")
                        & (big_df["Strategy"] != "")
                    ]

                    merged = big_df.groupby("Strategy", as_index=False)["Profit"].sum()
                    merged.rename(columns={"Profit": "Actual_Profit"}, inplace=True)
                    return merged
                except (ValueError, KeyError, IndexError) as e:
                    print(f"Error in aggregation: {e}")
                    return pd.DataFrame(columns=["Strategy", "Actual_Profit"])

            # Build benchmarks for each horizon
            for horizon, weeks in self.HORIZON_WEEKS.items():
                if n_weeks >= weeks or (horizon == "1 Week" and n_weeks >= 1):
                    bench_df = aggregate_weeks(1, weeks)
                    if not bench_df.empty:
                        bench_df["Actual_Rank"] = bench_df["Actual_Profit"].rank(
                            ascending=False, method="min"
                        )
                        self.benchmarks[horizon] = bench_df

            if progress_callback:
                progress_callback(80)

            # Extract year from filename
            y_match = re.search(r"(\d{4})", os.path.basename(file_path))
            if y_match:
                excel_year = y_match.group(1)
                self.metadata.benchmark_year = excel_year

            return self.benchmarks, excel_year

        except (IOError, ValueError, IndexError, RuntimeError) as e:
            print(f"Error parsing benchmark: {e}")
            return {}, "Error"

    def compare_results(
        self,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[ComparisonRecord]:
        """
        Compare forecasts with benchmarks.

        Args:
            progress_callback: Optional callback for progress updates (80-100%)

        Returns:
            List of ComparisonRecord objects
        """
        self.comparisons = []

        if not self.forecasts or not self.benchmarks:
            print("No forecasts or benchmarks to compare.")
            return self.comparisons

        print("Comparing results...")

        for forecast in self.forecasts:
            horizon = forecast.horizon

            if horizon not in self.benchmarks:
                continue

            bench_df = self.benchmarks[horizon]
            strategy_str = str(forecast.strategy_no)

            # Find matching benchmark
            match = bench_df[bench_df["Strategy"] == strategy_str]

            if match.empty:
                continue

            actual_profit = match.iloc[0]["Actual_Profit"]
            actual_rank = int(match.iloc[0]["Actual_Rank"])

            # Find winner(s) for this horizon
            winners = bench_df[bench_df["Actual_Rank"] == 1]
            if not winners.empty:
                if len(winners) == 1:
                    winner_strategy = winners.iloc[0]["Strategy"]
                else:
                    winner_list = winners["Strategy"].tolist()
                    winner_strategy = ", ".join(str(s) for s in winner_list[:5])
                    if len(winner_list) > 5:
                        winner_strategy += f" (+{len(winner_list) - 5} more)"
                winner_profit = winners.iloc[0]["Actual_Profit"]
            else:
                winner_strategy = "N/A"
                winner_profit = 0.0

            comparison = ComparisonRecord(
                model=forecast.model,
                forecast_year=forecast.forecast_year,
                horizon=horizon,
                predicted_rank=forecast.predicted_rank,
                actual_rank=actual_rank,
                rank_diff=abs(forecast.predicted_rank - actual_rank),
                predicted_profit=forecast.predicted_profit,
                actual_profit=actual_profit,
                actual_winner=f"Strategy {winner_strategy}",
                winner_profit=winner_profit,
                strategy_no=forecast.strategy_no,
                currency_pair=forecast.currency_pair,
                training_years=forecast.training_years
            )
            self.comparisons.append(comparison)

        if progress_callback:
            progress_callback(100)

        print(f"Generated {len(self.comparisons)} comparison records.")
        return self.comparisons

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Convert comparisons to DataFrame for reporting."""
        if not self.comparisons:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "Model": c.model,
                "Forecast_Year": c.forecast_year,
                "Horizon": c.horizon,
                "Predicted_Rank": c.predicted_rank,
                "Strategy": c.strategy_no,
                "Predicted_Profit": c.predicted_profit,
                "Actual_Rank": c.actual_rank,
                "Actual_Profit": c.actual_profit,
                "Rank_Deviation": c.rank_diff,
                "Winner_Strategy": c.actual_winner,
                "Winner_Profit": c.winner_profit
            }
            for c in self.comparisons
        ])

    def generate_html_report(
        self,
        output_path: str,
        metadata: Optional[InspectionMetadata] = None,
        excel_year: Optional[str] = None
    ) -> str:
        """
        Generate interactive HTML report with Plotly visualizations.

        Args:
            output_path: Path to save HTML file
            metadata: Optional metadata override
            excel_year: Optional benchmark year override

        Returns:
            Path to generated file
        """
        if not self.comparisons:
            print("No comparisons to report.")
            return ""

        comparison_df = self.get_comparison_dataframe()
        if comparison_df.empty:
            return ""

        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Use provided or stored metadata
        meta = metadata or self.metadata
        pair = meta.pair
        years = meta.years
        benchmark_year = excel_year or meta.benchmark_year

        # Build HTML content
        if PLOTLY_AVAILABLE:
            html_content = self._build_plotly_html(comparison_df, pair, years, benchmark_year)
        else:
            html_content = self._build_simple_html(comparison_df, pair, years, benchmark_year)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML report saved to: {output_path}")
        return output_path

    def _build_plotly_html(
        self,
        comparison_df: pd.DataFrame,
        pair: str,
        years: str,
        benchmark_year: str
    ) -> str:
        """Build HTML content with Plotly charts."""
        # Dark theme template
        dark_template = dict(
            layout=dict(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e4e4e4"),
                xaxis=dict(
                    gridcolor="rgba(255,255,255,0.1)",
                    zerolinecolor="rgba(255,255,255,0.2)"
                ),
                yaxis=dict(
                    gridcolor="rgba(255,255,255,0.1)",
                    zerolinecolor="rgba(255,255,255,0.2)"
                ),
            )
        )

        # 1. Rank Comparison Scatter Plot
        fig_rank = px.scatter(
            comparison_df,
            x="Predicted_Rank",
            y="Actual_Rank",
            color="Model",
            facet_col="Horizon",
            hover_data=["Strategy", "Predicted_Profit", "Actual_Profit", "Winner_Strategy"],
            title="Predicted vs Actual Rank (Lower Actual Rank = Better Prediction)",
        )
        fig_rank.update_yaxes(autorange="reversed")
        fig_rank.update_layout(**dark_template["layout"])

        # 2. Profit Accuracy Bar Chart (Top 1 only)
        top1_df = comparison_df[comparison_df["Predicted_Rank"] == 1]
        fig_profit = px.bar(
            top1_df,
            x="Model",
            y="Actual_Profit",
            color="Horizon",
            barmode="group",
            title="Actual Profit of Predicted Top 1 Strategy by Horizon",
        )
        fig_profit.update_layout(**dark_template["layout"])

        # 3. Rank Deviation Box Plot
        fig_deviation = px.box(
            comparison_df,
            x="Horizon",
            y="Rank_Deviation",
            color="Model",
            title="Rank Deviation Distribution by Horizon (Lower = Better)",
        )
        fig_deviation.update_layout(**dark_template["layout"])

        # Calculate summary statistics
        summary_html = self._build_summary_html(comparison_df)
        tables_html = self._build_tables_html(comparison_df)

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Inspection Report - {pair}</title>
            <style>
                {self._get_css_styles()}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Model Inspection Report</h1>
                <p class="subtitle">
                    <strong>Pair:</strong> {pair} |
                    <strong>Training Data:</strong> {years} |
                    <strong>Benchmark Year:</strong> {benchmark_year} |
                    <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}
                </p>

                <div class="card">
                    <h2>Summary Statistics</h2>
                    {summary_html}
                </div>

                <div class="card">
                    <h2>Predicted vs Actual Rank Analysis</h2>
                    <div class="chart-container">
                        {fig_rank.to_html(full_html=False, include_plotlyjs="cdn")}
                    </div>
                </div>

                <div class="card">
                    <h2>Actual Profit of Top 1 Predictions</h2>
                    <div class="chart-container">
                        {fig_profit.to_html(full_html=False, include_plotlyjs=False)}
                    </div>
                </div>

                <div class="card">
                    <h2>Rank Deviation Distribution</h2>
                    <div class="chart-container">
                        {fig_deviation.to_html(full_html=False, include_plotlyjs=False)}
                    </div>
                </div>

                <h2 style="color: #fff; margin-top: 30px;">Detailed Results by Model</h2>
                {tables_html}
            </div>
        </body>
        </html>
        """
        return html_content

    def _build_simple_html(
        self,
        comparison_df: pd.DataFrame,
        pair: str,
        years: str,
        benchmark_year: str
    ) -> str:
        """Build simple HTML without Plotly (fallback)."""
        summary_html = self._build_summary_html(comparison_df)
        tables_html = self._build_tables_html(comparison_df)

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Inspection Report - {pair}</title>
            <style>
                {self._get_css_styles()}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Model Inspection Report</h1>
                <p class="subtitle">
                    <strong>Pair:</strong> {pair} |
                    <strong>Training Data:</strong> {years} |
                    <strong>Benchmark Year:</strong> {benchmark_year} |
                    <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}
                </p>

                <div class="card">
                    <h2>Summary Statistics</h2>
                    {summary_html}
                </div>

                <div class="card">
                    <p><em>Note: Install plotly for interactive charts: pip install plotly</em></p>
                </div>

                <h2 style="color: #fff; margin-top: 30px;">Detailed Results by Model</h2>
                {tables_html}
            </div>
        </body>
        </html>
        """

    def _build_summary_html(self, comparison_df: pd.DataFrame) -> str:
        """Build summary statistics HTML."""
        total_predictions = len(comparison_df)
        avg_rank_deviation = comparison_df["Rank_Deviation"].mean()

        predicted_top1 = comparison_df[comparison_df["Predicted_Rank"] == 1]
        top1_hits = predicted_top1[predicted_top1["Actual_Rank"] == 1]
        top1_accuracy = (
            (len(top1_hits) / len(predicted_top1) * 100) if len(predicted_top1) > 0 else 0
        )

        top10_hits = len(
            comparison_df[
                (comparison_df["Predicted_Rank"] <= 10) & (comparison_df["Actual_Rank"] <= 10)
            ]
        )

        return f"""
        <div class="summary-stats">
            <div class="stat-box">
                <h3>{total_predictions}</h3>
                <p>Total Predictions</p>
            </div>
            <div class="stat-box highlight">
                <h3>{avg_rank_deviation:.1f}</h3>
                <p>Avg Rank Deviation</p>
            </div>
            <div class="stat-box {"success" if top1_accuracy > 10 else ""}">
                <h3>{top1_accuracy:.1f}%</h3>
                <p>Top 1 Accuracy</p>
            </div>
            <div class="stat-box success">
                <h3>{top10_hits}</h3>
                <p>Top 10 Hits</p>
            </div>
        </div>
        """

    def _build_tables_html(self, comparison_df: pd.DataFrame) -> str:
        """Build detailed tables HTML by model."""
        tables_html = ""
        models = comparison_df["Model"].unique()

        display_cols = [
            "Forecast_Year", "Horizon", "Predicted_Rank", "Strategy",
            "Predicted_Profit", "Actual_Rank", "Actual_Profit",
            "Rank_Deviation", "Winner_Strategy", "Winner_Profit"
        ]
        display_cols = [c for c in display_cols if c in comparison_df.columns]

        for model in models:
            model_df = comparison_df[comparison_df["Model"] == model].copy()

            # Format values
            if "Actual_Rank" in model_df.columns:
                model_df["Actual_Rank"] = model_df["Actual_Rank"].apply(
                    lambda x: f"{int(x)}." if pd.notna(x) else "N/A"
                )
            if "Rank_Deviation" in model_df.columns:
                model_df["Rank_Deviation"] = model_df["Rank_Deviation"].apply(
                    lambda x: f"{int(x)}" if pd.notna(x) else "N/A"
                )
            if "Predicted_Profit" in model_df.columns:
                model_df["Predicted_Profit"] = model_df["Predicted_Profit"].apply(
                    lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
                )
            if "Actual_Profit" in model_df.columns:
                model_df["Actual_Profit"] = model_df["Actual_Profit"].apply(
                    lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
                )
            if "Winner_Profit" in model_df.columns:
                model_df["Winner_Profit"] = model_df["Winner_Profit"].apply(
                    lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
                )

            rows_html = ""
            for _, row in model_df[display_cols].iterrows():
                cells = "".join([f"<td>{row[col]}</td>" for col in display_cols])
                rows_html += f"<tr>{cells}</tr>"

            headers = "".join(
                [f"<th>{col.replace('_', ' ')}</th>" for col in display_cols]
            )

            tables_html += f"""
            <div class="card">
                <h2>Model: {model}</h2>
                <div class="table-wrapper">
                    <table>
                        <thead><tr>{headers}</tr></thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>
            </div>
            """

        return tables_html

    def _get_css_styles(self) -> str:
        """Return CSS styles for HTML report."""
        return """
            * { box-sizing: border-box; }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                min-height: 100vh;
                color: #e4e4e4;
            }

            h1 {
                color: #fff;
                font-size: 2.2em;
                margin-bottom: 5px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }

            h2 {
                color: #3498db;
                font-size: 1.4em;
                margin: 0 0 15px 0;
            }

            .container { max-width: 1600px; margin: 0 auto; }

            .subtitle {
                color: #aaa;
                margin-bottom: 25px;
                font-size: 0.95em;
            }

            .summary-stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 15px;
                margin-bottom: 25px;
            }

            .stat-box {
                background: linear-gradient(135deg, #3498db, #2980b9);
                color: white;
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
                transition: transform 0.3s ease;
            }

            .stat-box:hover { transform: translateY(-5px); }

            .stat-box.highlight {
                background: linear-gradient(135deg, #e67e22, #d35400);
                box-shadow: 0 4px 15px rgba(230, 126, 34, 0.3);
            }

            .stat-box.success {
                background: linear-gradient(135deg, #27ae60, #1e8449);
                box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
            }

            .stat-box h3 { margin: 0; font-size: 1.8em; font-weight: 700; }
            .stat-box p { margin: 8px 0 0 0; opacity: 0.9; font-size: 0.85em; }

            .card {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                padding: 25px;
                margin-bottom: 20px;
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
                overflow: hidden;
            }

            .chart-container {
                background: rgba(255, 255, 255, 0.03);
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            }

            .table-wrapper {
                overflow-x: auto;
                margin-top: 15px;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9em;
            }

            th, td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }

            th {
                background: rgba(52, 152, 219, 0.3);
                color: #fff;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 0.85em;
                letter-spacing: 0.5px;
            }

            tr:hover { background: rgba(255, 255, 255, 0.05); }
        """

    def generate_md_report(
        self,
        output_path: str,
        metadata: Optional[InspectionMetadata] = None,
        excel_year: Optional[str] = None
    ) -> str:
        """
        Generate Markdown report.

        Args:
            output_path: Path to save MD file
            metadata: Optional metadata override
            excel_year: Optional benchmark year override

        Returns:
            Path to generated file
        """
        if not self.comparisons:
            return ""

        comparison_df = self.get_comparison_dataframe()
        if comparison_df.empty:
            return ""

        meta = metadata or self.metadata
        pair = meta.pair
        years = meta.years
        benchmark_year = excel_year or meta.benchmark_year

        # Calculate statistics
        total_predictions = len(comparison_df)
        avg_rank_deviation = comparison_df["Rank_Deviation"].mean()

        top1_count = len(comparison_df[comparison_df["Predicted_Rank"] == 1])
        top1_hits = comparison_df[
            (comparison_df["Predicted_Rank"] == 1) & (comparison_df["Actual_Rank"] == 1)
        ]
        top1_accuracy = (len(top1_hits) / top1_count * 100) if top1_count > 0 else 0

        top10_hits_count = len(
            comparison_df[
                (comparison_df["Predicted_Rank"] <= 10) & (comparison_df["Actual_Rank"] <= 10)
            ]
        )

        lines = [
            "# Model Inspection Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"**Pair:** {pair}",
            "",
            f"**Training Data:** {years}",
            "",
            f"**Benchmark Year:** {benchmark_year}",
            "",
            "## Summary Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Predictions | {total_predictions} |",
            f"| Avg Rank Deviation | {avg_rank_deviation:.1f} |",
            f"| Top 1 Accuracy | {top1_accuracy:.1f}% |",
            f"| Top 10 Hits | {top10_hits_count} |",
            "",
            "## Detailed Results by Model",
            "",
        ]

        # Add tables per model
        models = comparison_df["Model"].unique()
        for model in models:
            lines.append(f"### Model: {model}")
            lines.append("")
            model_df = comparison_df[comparison_df["Model"] == model].copy()

            display_cols = [
                "Forecast_Year", "Horizon", "Predicted_Rank", "Strategy",
                "Predicted_Profit", "Actual_Rank", "Actual_Profit",
                "Rank_Deviation", "Winner_Strategy", "Winner_Profit"
            ]
            display_cols = [c for c in display_cols if c in model_df.columns]

            # Format values
            if "Actual_Rank" in model_df.columns:
                model_df["Actual_Rank"] = model_df["Actual_Rank"].apply(
                    lambda x: f"{int(x)}." if pd.notna(x) else "N/A"
                )
            if "Rank_Deviation" in model_df.columns:
                model_df["Rank_Deviation"] = model_df["Rank_Deviation"].apply(
                    lambda x: f"{int(x)}" if pd.notna(x) else "N/A"
                )

            lines.append(model_df[display_cols].to_markdown(index=False))
            lines.append("")

        content = '\n'.join(lines)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Markdown report saved to: {output_path}")
        return output_path

    def generate_auto_filename(
        self,
        metadata: Optional[InspectionMetadata] = None,
        excel_year: Optional[str] = None,
        mode: str = "standard"
    ) -> str:
        """
        Generate automatic filename for report.

        Format: Inspect_Pair_(Years)_ExcelYear.html
        AR mode: AR_Inspect_Pair_(Years)_ExcelYear.html

        Args:
            metadata: Optional metadata override
            excel_year: Optional benchmark year override
            mode: "standard" or "ar"

        Returns:
            Generated filename
        """
        meta = metadata or self.metadata
        pair = meta.pair if meta.pair != "Unknown" else "UnknownPair"
        years = meta.years if meta.years != "Unknown" else "UnknownYears"
        b_year = excel_year or meta.benchmark_year or "UnknownYear"

        # Ensure format consistency
        if not years.startswith("("):
            years = f"({years})"

        # Add AR_ prefix for AR mode
        prefix = "AR_Inspect" if mode == "ar" else "Inspect"

        filename = f"{prefix}_{pair}_{years}_{b_year}.html"

        # Sanitize filename
        filename = re.sub(r'[<>:"/\\|?*]', "", filename)

        return filename
