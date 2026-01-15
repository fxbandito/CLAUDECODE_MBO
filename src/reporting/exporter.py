"""Report Exporter for generating analysis reports in various formats."""

# pylint: disable=line-too-long, too-many-lines, import-outside-toplevel
# Note: line-too-long disabled because this file contains embedded HTML/CSS templates
# where line breaks would significantly reduce readability.
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportExporter:
    """Generates and exports analysis reports in Markdown and HTML formats."""

    def _calculate_composite_best(self, results_df):
        """
        Calculates the best performing strategy for each monthly period (4-4-5 calendar).
        Instead of picking one global best, it finds which strategy performed best in that specific month.

        IMPORTANT: The 'Forecasts' column contains CUMULATIVE values, not individual weekly values.
        - forecasts[0] = Week 1 cumulative (just week 1)
        - forecasts[3] = Week 4 cumulative (sum of weeks 1-4)
        - forecasts[7] = Week 8 cumulative (sum of weeks 1-8)
        - etc.

        To get the profit for a specific period (e.g., Month 2 = Weeks 5-8):
        Month 2 profit = forecasts[7] - forecasts[3]  (cumulative at end - cumulative at start)

        Args:
            results_df: DataFrame containing all strategy results, including 'Forecasts' column.

        Returns:
            List of dictionaries with keys: 'Month', 'Weeks', 'Weeks_Count', 'Best_Strategy_ID', 'Forecast'
        """
        import ast
        import numpy as np

        # 4-4-5 pattern for one quarter
        quarter_pattern = [4, 4, 5]
        # Repeat for 4 quarters to get full year pattern
        full_year_pattern = quarter_pattern * 4  # [4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5]

        monthly_data = []
        current_week_idx = 0

        month_names = [
            "Month 1", "Month 2", "Month 3 (Q1 End)",
            "Month 4", "Month 5", "Month 6 (Q2 End)",
            "Month 7", "Month 8", "Month 9 (Q3 End)",
            "Month 10", "Month 11", "Month 12 (Year End)"
        ]

        # Pre-parse all forecasts once into a list of (id, forecast_list) tuples
        all_strategies = []
        for _, row in results_df.iterrows():
            strat_id = row['No.']
            forecasts = row.get('Forecasts', [])
            if isinstance(forecasts, str):
                try:
                    forecasts = ast.literal_eval(forecasts)
                except (ValueError, SyntaxError):
                    forecasts = []

            # Handle numpy arrays
            if isinstance(forecasts, np.ndarray):
                forecasts = forecasts.tolist()

            # Ensure valid list of floats
            if isinstance(forecasts, list) and len(forecasts) >= 52:
                all_strategies.append((strat_id, forecasts))

        # FALLBACK: If no valid Forecasts arrays, generate from cumulative columns
        if not all_strategies:
            forecast_cols = ["Forecast_1W", "Forecast_1M", "Forecast_3M", "Forecast_6M", "Forecast_12M"]
            week_limits = [1, 4, 13, 26, 52]

            for _, row in results_df.iterrows():
                strat_id = row['No.']
                # Generate 52-week cumulative forecast from available columns
                forecasts = []
                for week_idx in range(52):
                    cumulative = 0.0
                    for col, max_week in zip(forecast_cols, week_limits):
                        if col in row and week_idx < max_week:
                            col_value = float(row.get(col, 0) or 0)
                            # Linear interpolation within each period
                            cumulative = col_value * (week_idx + 1) / max_week
                            break
                    forecasts.append(cumulative)
                if forecasts:
                    all_strategies.append((strat_id, forecasts))

        if not all_strategies:
            return []

        for i, weeks_in_month in enumerate(full_year_pattern):
            start_week = current_week_idx
            end_week = current_week_idx + weeks_in_month

            best_month_profit = -float('inf')
            best_month_strat_id = None

            # Find the best strategy for THIS specific month
            # Since Forecasts are CUMULATIVE, we need to calculate the difference
            for strat_id, forecasts in all_strategies:
                # end_week - 1 because forecasts is 0-indexed and end_week is exclusive
                cumulative_at_end = forecasts[end_week - 1]

                # For the first month (start_week = 0), there's no previous cumulative
                if start_week == 0:
                    cumulative_at_start = 0
                else:
                    cumulative_at_start = forecasts[start_week - 1]

                # Month profit = cumulative at end of month - cumulative at start of month
                month_profit = cumulative_at_end - cumulative_at_start

                if month_profit > best_month_profit:
                    best_month_profit = month_profit
                    best_month_strat_id = strat_id

            week_range = f"Weeks {start_week + 1}-{end_week}"

            monthly_data.append({
                "Month": month_names[i],
                "Weeks": week_range,
                "Weeks_Count": weeks_in_month,
                "Best_Strategy_ID": best_month_strat_id,
                "Forecast": best_month_profit
            })

            current_week_idx += weeks_in_month

        return monthly_data

    def create_monthly_html_report(
        self,
        results_df,
        best_strategy_id,
        method_name,
        best_strat_data=None,
        filename_base=None,
        params=None,
        execution_time=None,
        ranking_mode=None,
    ):
        """
        Generates a monthly forecast report (MR_) in HTML format using 4-4-5 aggregation.
        Styled to match the AR_ (All Results) report format with interactive charts.
        """
        import plotly.graph_objects as go
        import plotly.io as pio

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Determine filename with MR_ prefix
        if filename_base:
            if filename_base.startswith("AR_"):
                mr_filename_base = "MR_" + filename_base[3:]
            elif filename_base.startswith("MR_"):
                mr_filename_base = filename_base
            else:
                mr_filename_base = f"MR_{filename_base}"
            filename = os.path.join(self.output_dir, f"{mr_filename_base}.html")
        else:
            filename = os.path.join(self.output_dir, f"MR_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.html")

        if results_df is None or results_df.empty:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("<html><head></head><body><div class='container'><div class='card'><h1>Error</h1><p>No results data found.</p></div></div></body></html>")
            return filename

        # Calculate Composite 4-4-5 Monthly Best
        monthly_data = self._calculate_composite_best(results_df)

        if not monthly_data:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("<html><head></head><body><div class='container'><div class='card'><h1>Error</h1><p>Insufficient forecast data length to generate 4-4-5 report.</p></div></div></body></html>")
            return filename

        # Aggregation Totals
        total_yearly_forecast = sum(row['Forecast'] for row in monthly_data)

        # Prepare Data for Charts
        months = [row['Month'] for row in monthly_data]
        profits = [row['Forecast'] for row in monthly_data]
        strategy_ids = [row['Best_Strategy_ID'] for row in monthly_data]

        charts_html = ""
        plotly_config = {"responsive": True, "displayModeBar": True}

        # ============ Chart 1: Monthly Profit Forecast Bar Chart ============
        colors = ["#2ecc71" if p >= 0 else "#e74c3c" for p in profits]

        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                x=months,
                y=profits,
                # Show both profit and strategy No. on bars
                text=[f"#{s}<br>{p:,.0f}" for p, s in zip(profits, strategy_ids)],
                textposition='auto',
                marker=dict(color=colors, line=dict(width=1, color="white")),
                hovertemplate="<b>%{x}</b><br>Profit: %{y:,.2f}<br>Strategy: #%{customdata}<extra></extra>",
                customdata=strategy_ids,
            )
        )

        fig_bar.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

        fig_bar.update_layout(
            title=dict(text="Monthly Profit Forecast (4-4-5 Calendar)", font=dict(color="white", size=16)),
            xaxis=dict(title="Month", color="white", gridcolor="rgba(255,255,255,0.1)", tickangle=-45),
            yaxis=dict(title="Predicted Profit", color="white", gridcolor="rgba(255,255,255,0.1)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=60, r=30, t=50, b=80),
        )
        charts_html += f'<div class="chart-container">{pio.to_html(fig_bar, full_html=False, include_plotlyjs="cdn", config=plotly_config)}</div>'

        # ============ Chart 2: Strategy Distribution Pie Chart ============
        strategy_counts = {}
        for strat_id in strategy_ids:
            strategy_counts[strat_id] = strategy_counts.get(strat_id, 0) + 1

        sorted_counts = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)
        top_strategies = sorted_counts[:5]
        other_count = sum(c for _, c in sorted_counts[5:])

        labels = [f"Strategy {s}" for s, _ in top_strategies]
        values = [c for _, c in top_strategies]
        if other_count > 0:
            labels.append("Other")
            values.append(other_count)

        colors_pie = ["#f39c12", "#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#95a5a6"]

        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    marker=dict(colors=colors_pie[: len(labels)], line=dict(color="white", width=2)),
                    textinfo="label+percent",
                    textfont=dict(color="white", size=12),
                    hovertemplate="%{label}<br>Months: %{value}<br>%{percent}<extra></extra>",
                )
            ]
        )

        fig_pie.update_layout(
            title=dict(text="Strategy Dominance (Best Strategy per Month)", font=dict(color="white", size=16)),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=True,
            legend=dict(font=dict(color="white")),
            margin=dict(l=30, r=30, t=50, b=30),
        )
        charts_html += f'<div class="chart-container">{pio.to_html(fig_pie, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'

        # ============ Chart 3: Cumulative Monthly Profit ============
        cumulative = []
        cumsum = 0
        for p in profits:
            cumsum += p
            cumulative.append(cumsum)

        fig_cumulative = go.Figure()
        fig_cumulative.add_trace(
            go.Scatter(
                x=months,
                y=cumulative,
                mode="lines+markers+text",
                name="Cumulative Profit",
                line=dict(color="#f39c12", width=3),
                marker=dict(size=12, color="#f39c12", line=dict(width=2, color="white")),
                fill="tozeroy",
                fillcolor="rgba(243, 156, 18, 0.1)",
                # Show strategy No. as text labels on markers
                text=[f"#{s}" for s in strategy_ids],
                textposition="top center",
                textfont=dict(size=10, color="white"),
                hovertemplate="<b>%{x}</b><br>Cumulative: %{y:,.2f}<br>Strategy: #%{customdata}<br>Month Profit: %{meta:,.2f}<extra></extra>",
                customdata=strategy_ids,
                meta=profits,
            )
        )

        fig_cumulative.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

        fig_cumulative.update_layout(
            title=dict(text="Cumulative Yearly Profit Progression", font=dict(color="white", size=16)),
            xaxis=dict(title="Month", color="white", gridcolor="rgba(255,255,255,0.1)", tickangle=-45),
            yaxis=dict(title="Cumulative Profit", color="white", gridcolor="rgba(255,255,255,0.1)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=60, r=30, t=70, b=80),
        )
        charts_html += f'<div class="chart-container">{pio.to_html(fig_cumulative, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'

        # ============ Chart 4: Quarterly Summary ============
        quarterly_profits = []
        quarterly_strategies = []  # Dominant strategy per quarter
        quarterly_labels = ["Q1 (M1-M3)", "Q2 (M4-M6)", "Q3 (M7-M9)", "Q4 (M10-M12)"]
        for q in range(4):
            q_profit = sum(profits[q*3:(q+1)*3])
            quarterly_profits.append(q_profit)
            # Find dominant strategy in quarter (most frequent or best performer)
            q_strategies = strategy_ids[q*3:(q+1)*3]
            # Count occurrences
            from collections import Counter
            counter = Counter(q_strategies)
            dominant_strat = counter.most_common(1)[0][0] if counter else "N/A"
            quarterly_strategies.append(dominant_strat)

        q_colors = ["#2ecc71" if p >= 0 else "#e74c3c" for p in quarterly_profits]

        # Create quarterly strategy info for hover
        quarterly_strat_info = []
        for q in range(4):
            q_strats = strategy_ids[q*3:(q+1)*3]
            strat_str = ", ".join([f"#{s}" for s in q_strats])
            quarterly_strat_info.append(strat_str)

        fig_quarterly = go.Figure()
        fig_quarterly.add_trace(
            go.Bar(
                x=quarterly_labels,
                y=quarterly_profits,
                # Show profit and dominant strategy on bars
                text=[f"#{s}<br>{p:,.0f}" for p, s in zip(quarterly_profits, quarterly_strategies)],
                textposition='auto',
                marker=dict(color=q_colors, line=dict(width=2, color="white")),
                hovertemplate="<b>%{x}</b><br>Profit: %{y:,.2f}<br>Dominant Strategy: #%{customdata}<br>Monthly Strategies: %{meta}<extra></extra>",
                customdata=quarterly_strategies,
                meta=quarterly_strat_info,
            )
        )

        fig_quarterly.update_layout(
            title=dict(text="Quarterly Profit Summary", font=dict(color="white", size=16)),
            xaxis=dict(title="Quarter", color="white", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Predicted Profit", color="white", gridcolor="rgba(255,255,255,0.1)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=60, r=30, t=50, b=50),
        )
        charts_html += f'<div class="chart-container">{pio.to_html(fig_quarterly, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'

        # Generate Table Rows (with color coding)
        table_rows = ""
        for row in monthly_data:
            profit = row['Forecast']
            best_id = row.get('Best_Strategy_ID', 'N/A')
            profit_class = "positive" if profit >= 0 else "negative"
            table_rows += f'<tr><td>{row["Month"]}</td><td>{row["Weeks"]}</td><td>{row["Weeks_Count"]}</td><td>#{best_id}</td><td class="{profit_class}">{profit:,.2f}</td></tr>'

        # Totals Row
        total_class = "positive" if total_yearly_forecast >= 0 else "negative"
        table_rows += f'<tr style="font-weight:bold; background-color: rgba(243, 156, 18, 0.2);"><td>TOTAL YEAR</td><td>Weeks 1-52</td><td>52</td><td>Composite</td><td class="{total_class}">{total_yearly_forecast:,.2f}</td></tr>'

        # Model Settings HTML
        settings_html = ""
        if params and len(params) > 0:
            settings_rows = "".join([f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in params.items()])
            settings_html = f"""
            <div class="card">
                <h2>Model Settings</h2>
                <p>Configuration parameters used for the analysis.</p>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr><th>Parameter</th><th>Value</th></tr>
                        </thead>
                        <tbody>
                            {settings_rows}
                        </tbody>
                    </table>
                </div>
            </div>
            """

        # Financial Metrics HTML
        metrics_html = ""
        if best_strat_data is not None:
            from analysis.metrics import FinancialMetrics
            metrics = FinancialMetrics.calculate_all(best_strat_data)

            win_rate = metrics.get("Win Rate", 0) * 100
            win_rate_class = "success" if win_rate >= 50 else "highlight"
            sharpe = metrics.get("Sharpe Ratio", 0)
            sharpe_class = "success" if sharpe >= 1 else ("" if sharpe >= 0 else "highlight")

            metrics_html = f"""
            <div class="card">
                <h2>Financial Metrics (Global Best Strategy {best_strategy_id})</h2>
                <p>Key performance indicators for the best performing strategy.</p>
                <div class="summary-stats">
                    <div class="stat-box {win_rate_class}">
                        <h3>{metrics.get("Total Profit", 0):,.2f}</h3>
                        <p>Total Profit</p>
                    </div>
                    <div class="stat-box {win_rate_class}">
                        <h3>{win_rate:.1f}%</h3>
                        <p>Win Rate</p>
                    </div>
                    <div class="stat-box">
                        <h3>{metrics.get("Profit Factor", 0):.2f}</h3>
                        <p>Profit Factor</p>
                    </div>
                    <div class="stat-box {sharpe_class}">
                        <h3>{sharpe:.2f}</h3>
                        <p>Sharpe Ratio</p>
                    </div>
                </div>
            </div>
            """

        exec_time_html = f" | <strong>Generating time:</strong> {execution_time}" if execution_time else ""
        ranking_html = f" | <strong>Ranking:</strong> {ranking_mode}" if ranking_mode else ""

        # Prepare logo HTML (same as AR_ report)
        logo_html = ""
        try:
            import base64

            assets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gui", "assets", "dark_logo.png")
            if os.path.exists(assets_path):
                with open(assets_path, "rb") as img_f:
                    b64_data = base64.b64encode(img_f.read()).decode("utf-8")
                    logo_html = f'<img src="data:image/png;base64,{b64_data}" class="header-logo" alt="MBO Logo">'
        except OSError as e:
            logger.debug("Logo embedding failed: %s", e)

        # Best and worst months
        best_month_idx = profits.index(max(profits))
        worst_month_idx = profits.index(min(profits))
        best_month = months[best_month_idx]
        worst_month = months[worst_month_idx]

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Monthly Forecast Report (MR) - {method_name}</title>
            <style>
                * {{
                    box-sizing: border-box;
                }}

                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                    min-height: 100vh;
                    color: #e4e4e4;
                }}

                h1 {{
                    color: #fff;
                    font-size: 2.2em;
                    margin: 0 0 10px 0;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }}

                .header-section {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    background: rgba(255, 255, 255, 0.05);
                    backdrop-filter: blur(10px);
                    padding: 25px;
                    margin-bottom: 25px;
                    border-radius: 16px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }}

                .header-content {{
                    flex: 1;
                }}

                .header-logo-wrapper {{
                    background: rgba(255, 255, 255, 0.1);
                    padding: 10px;
                    border-radius: 12px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    margin-left: 20px;
                }}

                .header-logo {{
                    height: 50px;
                    width: auto;
                    display: block;
                    filter: drop-shadow(0px 0px 5px rgba(0,0,0,0.5));
                }}

                h2 {{
                    color: #f39c12;
                    font-size: 1.4em;
                    margin: 0 0 15px 0;
                }}

                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}

                .subtitle {{
                    color: #aaa;
                    margin: 0;
                    font-size: 0.95em;
                }}

                /* Summary Stats */
                .summary-stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                    gap: 15px;
                    margin-bottom: 25px;
                }}

                .stat-box {{
                    background: linear-gradient(135deg, #3498db, #2980b9);
                    color: white;
                    padding: 20px;
                    border-radius: 12px;
                    text-align: center;
                    box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }}

                .stat-box:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 8px 25px rgba(52, 152, 219, 0.4);
                }}

                .stat-box.highlight {{
                    background: linear-gradient(135deg, #e67e22, #d35400);
                    box-shadow: 0 4px 15px rgba(230, 126, 34, 0.3);
                }}

                .stat-box.success {{
                    background: linear-gradient(135deg, #27ae60, #1e8449);
                    box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
                }}

                .stat-box h3 {{
                    margin: 0;
                    font-size: 1.8em;
                    font-weight: 700;
                }}

                .stat-box p {{
                    margin: 8px 0 0 0;
                    opacity: 0.9;
                    font-size: 0.85em;
                }}

                /* Cards */
                .card {{
                    background: rgba(255, 255, 255, 0.05);
                    backdrop-filter: blur(10px);
                    padding: 25px;
                    margin-bottom: 20px;
                    border-radius: 16px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    overflow: hidden;
                }}

                .card p {{
                    color: #bbb;
                    margin: 0 0 15px 0;
                    font-size: 0.95em;
                }}

                /* Modern Table */
                .table-wrapper {{
                    overflow-x: auto;
                    border-radius: 12px;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
                    max-height: 600px;
                    overflow-y: auto;
                }}

                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background: rgba(30, 30, 50, 0.8);
                    font-size: 0.9em;
                }}

                thead {{
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }}

                th {{
                    background: linear-gradient(135deg, #f39c12, #d68910);
                    color: white;
                    padding: 16px 12px;
                    text-align: left;
                    font-weight: 600;
                    white-space: nowrap;
                    border: none;
                }}

                th:first-child {{
                    border-radius: 12px 0 0 0;
                }}

                th:last-child {{
                    border-radius: 0 12px 0 0;
                }}

                td {{
                    padding: 14px 12px;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                    color: #e4e4e4;
                    vertical-align: middle;
                }}

                tbody tr:nth-child(even) {{
                    background: rgba(255, 255, 255, 0.02);
                }}

                tbody tr {{
                    transition: background 0.2s ease;
                }}

                tbody tr:hover {{
                    background: rgba(243, 156, 18, 0.15);
                }}

                /* Column colors */
                td:first-child {{
                    color: #f39c12;
                    font-weight: 600;
                }}

                td:nth-child(4) {{
                    color: #3498db;
                    font-weight: 600;
                }}

                td.positive {{
                    color: #2ecc71;
                    font-weight: 600;
                }}

                td.negative {{
                    color: #e74c3c;
                    font-weight: 600;
                }}

                /* Badge */
                .badge {{
                    background: linear-gradient(135deg, #f39c12, #d68910);
                    color: white;
                    padding: 6px 14px;
                    border-radius: 20px;
                    font-size: 0.9em;
                    font-weight: 600;
                    display: inline-block;
                }}

                /* Scrollbar styling */
                ::-webkit-scrollbar {{
                    width: 8px;
                    height: 8px;
                }}

                ::-webkit-scrollbar-track {{
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 4px;
                }}

                ::-webkit-scrollbar-thumb {{
                    background: rgba(243, 156, 18, 0.5);
                    border-radius: 4px;
                }}

                ::-webkit-scrollbar-thumb:hover {{
                    background: rgba(243, 156, 18, 0.7);
                }}

                /* Info box */
                .info-box {{
                    background: rgba(243, 156, 18, 0.1);
                    border-left: 4px solid #f39c12;
                    padding: 15px 20px;
                    margin-bottom: 20px;
                    border-radius: 0 8px 8px 0;
                }}

                .info-box p {{
                    margin: 0;
                    color: #e4e4e4;
                }}

                /* Chart containers */
                .chart-container {{
                    background: rgba(255, 255, 255, 0.03);
                    padding: 20px;
                    border-radius: 12px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    max-width: 100%;
                    overflow-x: auto;
                    overflow-y: hidden;
                }}

                .chart-container .plotly-graph-div {{
                    width: 100% !important;
                    max-width: 100% !important;
                }}

                .charts-grid {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                    margin-bottom: 20px;
                }}

                @media (max-width: 1200px) {{
                    .charts-grid {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header-section">
                    <div class="header-content">
                        <h1>Monthly Forecast Report (MR) - 4-4-5 Calendar</h1>
                        <p class="subtitle"><strong>Date:</strong> {timestamp}{exec_time_html}{ranking_html} | <strong>Method:</strong> {method_name}</p>
                    </div>
                    <div class="header-logo-wrapper">
                        {logo_html}
                    </div>
                </div>

                <div class="card">
                    <h2>Executive Summary</h2>
                    <p>This report shows the 12-month forecast using the <span class="badge">4-4-5 Financial Calendar</span> aggregation method.</p>
                    <p>Each month uses the best performing strategy for that specific period (composite approach).</p>
                    <div class="summary-stats">
                        <div class="stat-box highlight">
                            <h3>{total_yearly_forecast:,.0f}</h3>
                            <p>Total Yearly Profit</p>
                        </div>
                        <div class="stat-box success">
                            <h3>{best_month}</h3>
                            <p>Best Month</p>
                        </div>
                        <div class="stat-box">
                            <h3>{worst_month}</h3>
                            <p>Worst Month</p>
                        </div>
                        <div class="stat-box">
                            <h3>4-4-5</h3>
                            <p>Calendar Logic</p>
                        </div>
                    </div>
                </div>

                {metrics_html}

                {settings_html}

                {charts_html}

                <div class="card">
                    <h2>12-Month Forecast Data Table</h2>
                    <div class="info-box">
                        <p><strong>4-4-5 Calendar:</strong> Each quarter has 4 weeks, 4 weeks, and 5 weeks. This ensures 52 weeks map to exactly 12 months.</p>
                    </div>
                    <div class="table-wrapper">
                        <table>
                            <thead>
                                <tr>
                                    <th>Month</th>
                                    <th>Week Range</th>
                                    <th>Weeks Count</th>
                                    <th>Best Strategy</th>
                                    <th>Predicted Profit</th>
                                </tr>
                            </thead>
                            <tbody>
                                {table_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)

        return filename

    def create_monthly_markdown_report(
        self,
        results_df,
        best_strategy_id,
        method_name,
        best_strat_data=None,
        filename_base=None,
        params=None,
        execution_time=None,
        ranking_mode=None,
    ):
        """
        Generates a monthly forecast report (MR_) using 4-4-5 aggregation.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Determine filename with MR_ prefix
        if filename_base:
            # If filename_base starts with AR_, replace it with MR_
            if filename_base.startswith("AR_"):
                mr_filename_base = "MR_" + filename_base[3:]
            elif filename_base.startswith("MR_"):
                mr_filename_base = filename_base
            else:
                mr_filename_base = f"MR_{filename_base}"
            filename = os.path.join(self.output_dir, f"{mr_filename_base}.md")
        else:
            filename = os.path.join(self.output_dir, f"MR_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.md")

        if results_df is None or results_df.empty:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("# Monthly Forecast Report (MR)\n\nError: No results data found.")
            return filename

        # Calculate Composite 4-4-5 Monthly Best
        monthly_data = self._calculate_composite_best(results_df)

        if not monthly_data:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("# Monthly Forecast Report (MR)\n\nError: Insufficient forecast data length to generate 4-4-5 report.")
            return filename

        # Format Aggregation Table
        monthly_table_rows = ""
        total_yearly_forecast = 0

        for row in monthly_data:
            profit = row['Forecast']
            total_yearly_forecast += profit
            # Use 'Best_Strategy_ID' from the composite calculation
            best_id = row.get('Best_Strategy_ID', 'N/A')
            monthly_table_rows += f"| {row['Month']} | {row['Weeks']} | {row['Weeks_Count']} | #{best_id} | {profit:,.2f} |\n"

        # Add Totals Row
        monthly_table_rows += f"| **TOTAL YEAR** | **Weeks 1-52** | **52** | **Composite** | **{total_yearly_forecast:,.2f}** |\n"

        # Monthly Forecast Table Section
        forecast_table_section = f"""
## 12-Month Forecast (4-4-5 Calendar)
*Note: The 'Best Strategy' column indicates which strategy performed best for that specific month.*

| Month | Week Range | Weeks | Best Strategy | Predicted Profit |
|-------|------------|-------|---------------|------------------|
{monthly_table_rows}
"""

        # Model Settings Section (Reused)
        settings_section = ""
        if params and len(params) > 0:
            settings_lines = [f"| **{key}** | {value} |" for key, value in params.items()]
            settings_section = (
                """
## Model Settings
| Parameter | Value |
|-----------|-------|
"""
                + "\n".join(settings_lines)
                + "\n"
            )

        # 2. Financial Metrics
        metrics_section = ""
        if best_strat_data is not None:
            from analysis.metrics import FinancialMetrics
            metrics = FinancialMetrics.calculate_all(best_strat_data)
            metrics_section = f"""
## Financial Metrics (Strategy {best_strategy_id})
| Metric | Value |
|--------|-------|
| **Total Profit** | {metrics.get("Total Profit", 0):.2f} |
| **Win Rate** | {metrics.get("Win Rate", 0) * 100:.1f}% |
| **Profit Factor** | {metrics.get("Profit Factor", 0):.2f} |
| **Sharpe Ratio** | {metrics.get("Sharpe Ratio", 0):.2f} |
| **Sortino Ratio** | {metrics.get("Sortino Ratio", 0):.2f} |
| **Max Drawdown** | {metrics.get("Max Drawdown ($)", 0):.2f} |
| **Avg Trade** | {metrics.get("Avg Trade", 0):.2f} |
"""

        # 3. Top 10 Ranked (Same logic as AR)
        top_ranked_section = ""
        if ranking_mode and ranking_mode != "forecast" and ranking_mode != "Standard" and "Ranking_Score" in results_df.columns:
            top_ranked = results_df.head(10)
            rows = ""
            for i, (_, row) in enumerate(top_ranked.iterrows()):
                s_score = f"{row.get('Stability_Score', 0):.2f}" if "Stability_Score" in row else "N/A"
                r_score = f"{row.get('Ranking_Score', 0):.2f}"
                f_1m = f"{row.get('Forecast_1M', 0):.2f}"
                rows += f"| {i+1} | #{int(row['No.'])} | {f_1m} | {s_score} | {r_score} |\n"

            top_ranked_section = f"""
## Top 10 Strategies (Ranked by {ranking_mode})
| Rank | Strategy | Forecast 1M | Stability | Score |
|------|----------|-------------|-----------|-------|
{rows}
"""

        header_info = []
        if execution_time:
            header_info.append(f"**Generation Time**: {execution_time}")
        if ranking_mode:
            header_info.append(f"**Ranking Mode**: {ranking_mode}")
        header_str = "\n".join(header_info)

        md_content = f"""# Monthly Forecast Report (MR)
**Date**: {timestamp}
{header_str}
**Method**: {method_name}

## Executive Summary
This report provides a 12-month forecast breakdown based on the **4-4-5 Financial Calendar** standard.
The 52-week forecast has been aggregated into monthly buckets.
The top performing strategy for the next 12 months (aggregated) is **Strategy No. {best_strategy_id}**.

### Annual Outlook
- **Total Predicted Yearly Profit**: {total_yearly_forecast:,.2f}
- **Aggregation Logic**: 4 weeks - 4 weeks - 5 weeks (Quarterly Pattern)

{settings_section}

{metrics_section}

{top_ranked_section}

{forecast_table_section}

> **Note**: This aggregation assumes the forecast horizon starts at the beginning of a 4-week fiscal period.
> The monthly values are derived purely from summing the corresponding weekly predictions.
"""

        with open(filename, "w", encoding="utf-8") as f:
            f.write(md_content)

        return filename

    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        # Use exist_ok=True to avoid TOCTOU race condition
        os.makedirs(output_dir, exist_ok=True)

    def _get_top_strategies_by_horizon(self, df, horizon_col, n=10):
        if horizon_col not in df.columns:
            return ["N/A"] * n
        # Filter out NaNs to strictly sort valid values
        valid_df = df.dropna(subset=[horizon_col])
        if valid_df.empty:
            sorted_df = df.head(n)  # Fallback to unsorted first n
        else:
            sorted_df = valid_df.sort_values(horizon_col, ascending=False).head(n)

        return [f"#{int(row.get('No.', 0))} ({row.get(horizon_col, 0):.2f})" for _, row in sorted_df.iterrows()]

    def _get_best_row_safe(self, df, col):
        """Safely get the row with the max value in col, handling NaNs/empty."""
        if df.empty:
            return None  # Should be handled by caller
        if col not in df.columns:
            return df.iloc[0]

        valid_s = df[col].dropna()
        if valid_s.empty:
            return df.iloc[0]

        try:
            return df.loc[valid_s.idxmax()]
        except (KeyError, ValueError):
            return df.iloc[0]

    def create_markdown_report(
        self,
        results_df,
        best_strategy_id,
        method_name,
        _image_paths,
        best_strat_data=None,
        filename_base=None,
        params=None,
        execution_time=None,
        ranking_mode=None,
    ):
        """
        Generates a markdown report.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        if filename_base:
            filename = os.path.join(self.output_dir, f"{filename_base}.md")
        else:
            filename = os.path.join(self.output_dir, f"Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.md")

        # Calculate Best per Horizon
        best_1w = self._get_best_row_safe(results_df, "Forecast_1W")
        best_1m = self._get_best_row_safe(results_df, "Forecast_1M")
        best_3m = self._get_best_row_safe(results_df, "Forecast_3M")
        best_6m = self._get_best_row_safe(results_df, "Forecast_6M")
        best_1y = self._get_best_row_safe(results_df, "Forecast_12M")

        # Model Settings Section
        settings_section = ""
        if params and len(params) > 0:
            settings_lines = [f"| **{key}** | {value} |" for key, value in params.items()]
            settings_section = (
                """
## Model Settings
| Parameter | Value |
|-----------|-------|
"""
                + "\n".join(settings_lines)
                + "\n"
            )

        # Calculate Metrics if data is provided
        metrics_section = ""
        if best_strat_data is not None:
            from analysis.metrics import FinancialMetrics

            metrics = FinancialMetrics.calculate_all(best_strat_data)

            metrics_section = f"""
## Financial Metrics (Strategy {best_strategy_id})
| Metric | Value |
|--------|-------|
| **Total Profit** | {metrics.get("Total Profit", 0):.2f} |
| **Win Rate** | {metrics.get("Win Rate", 0) * 100:.1f}% |
| **Profit Factor** | {metrics.get("Profit Factor", 0):.2f} |
| **Sharpe Ratio** | {metrics.get("Sharpe Ratio", 0):.2f} |
| **Sortino Ratio** | {metrics.get("Sortino Ratio", 0):.2f} |
| **Max Drawdown** | {metrics.get("Max Drawdown ($)", 0):.2f} |
| **Avg Trade** | {metrics.get("Avg Trade", 0):.2f} |
"""


        # Top 10 Ranked Section
        top_ranked_section = ""
        if ranking_mode != "forecast" and "Ranking_Score" in results_df.columns:
            # Results are already sorted by ranking logic in apply_ranking
            top_ranked = results_df.head(10)
            rows = ""
            for i, (_, row) in enumerate(top_ranked.iterrows()):
                s_score = f"{row.get('Stability_Score', 0):.2f}" if "Stability_Score" in row else "N/A"
                r_score = f"{row.get('Ranking_Score', 0):.2f}"
                f_1m = f"{row.get('Forecast_1M', 0):.2f}"
                rows += f"| {i+1} | #{int(row['No.'])} | {f_1m} | {s_score} | {r_score} |\n"

            top_ranked_section = f"""
## Top 10 Strategies (Ranked by {ranking_mode})
| Rank | Strategy | Forecast 1M | Stability | Score |
|------|----------|-------------|-----------|-------|
{rows}
"""

        # Prepare Top 10 Data
        top_1w = self._get_top_strategies_by_horizon(results_df, "Forecast_1W")
        top_1m = self._get_top_strategies_by_horizon(results_df, "Forecast_1M")
        top_3m = self._get_top_strategies_by_horizon(results_df, "Forecast_3M")
        top_6m = self._get_top_strategies_by_horizon(results_df, "Forecast_6M")
        top_1y = self._get_top_strategies_by_horizon(results_df, "Forecast_12M")

        top_table_rows = ""
        for i in range(min(10, len(results_df))):
            # Handle cases where lists might be shorter than 10 (unlikely but safe)
            v_1w = top_1w[i] if i < len(top_1w) else ""
            v_1m = top_1m[i] if i < len(top_1m) else ""
            v_3m = top_3m[i] if i < len(top_3m) else ""
            v_6m = top_6m[i] if i < len(top_6m) else ""
            v_1y = top_1y[i] if i < len(top_1y) else ""
            top_table_rows += f"| {i + 1} | {v_1w} | {v_1m} | {v_3m} | {v_6m} | {v_1y} |\n"

        exec_time_line = f"**Generating time**: {execution_time}" if execution_time else ""
        ranking_line = f"**Ranking Mode**: {ranking_mode}" if ranking_mode else ""

        header_meta = [line for line in [exec_time_line, ranking_line] if line]
        header_meta_str = "\n".join(header_meta)

        md_content = f"""# Trading Strategy Analysis Report
**Date**: {timestamp}
{header_meta_str}
**Method**: {method_name}

## Executive Summary
The analysis evaluated {len(results_df)} strategies.
The top performing strategy for the next month is **Strategy No. {best_strategy_id}**.

### Best Strategy Forecast
- **1 Week**: (No. {int(best_1w["No."])}) {best_1w["Forecast_1W"]:.2f}
- **1 Month**: (No. {int(best_1m["No."])}) {best_1m["Forecast_1M"]:.2f}
- **3 Months**: (No. {int(best_3m["No."])}) {best_3m["Forecast_3M"]:.2f}
- **6 Months**: (No. {int(best_6m["No."])}) {best_6m["Forecast_6M"]:.2f}
- **1 Year**: (No. {int(best_1y["No."])}) {best_1y["Forecast_12M"]:.2f}

{settings_section}

{metrics_section}

{top_ranked_section}

## Top 10 Strategies by Horizon
| Rank | 1 Week | 1 Month | 3 Months | 6 Months | 1 Year |
|------|--------|---------|----------|----------|--------|
{top_table_rows}
"""

        with open(filename, "w", encoding="utf-8") as f:
            f.write(md_content)

        return filename

    def create_html_report(
        self,
        results_df,
        best_strategy_id,
        method_name,
        best_strat_data=None,
        forecast_values=None,
        filename_base=None,
        params=None,
        execution_time=None,
        ranking_mode=None,
    ):
        """Generates an interactive HTML report."""
        import plotly.io as pio

        from analysis.metrics import FinancialMetrics
        from reporting.visualizer import Visualizer

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        if filename_base:
            filename = os.path.join(self.output_dir, f"{filename_base}.html")
        else:
            filename = os.path.join(self.output_dir, f"Interactive_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.html")

        # Model Settings
        settings_html = ""
        if params and len(params) > 0:
            settings_rows = "".join([f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in params.items()])
            settings_html = f"""
            <div class="card">
                <h2>Model Settings</h2>
                <p>Configuration parameters used for the analysis.</p>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr><th>Parameter</th><th>Value</th></tr>
                        </thead>
                        <tbody>
                            {settings_rows}
                        </tbody>
                    </table>
                </div>
            </div>
            """

        # Calculate Best per Horizon for HTML
        best_1w = self._get_best_row_safe(results_df, "Forecast_1W")
        best_1m = self._get_best_row_safe(results_df, "Forecast_1M")
        best_3m = self._get_best_row_safe(results_df, "Forecast_3M")
        best_6m = self._get_best_row_safe(results_df, "Forecast_6M")
        best_1y = self._get_best_row_safe(results_df, "Forecast_12M")

        forecast_html = f"""
        <div class="card">
            <h2>Best Strategy Forecast by Horizon</h2>
            <p>Top performing strategies for each forecast horizon.</p>
            <div class="summary-stats">
                <div class="stat-box">
                    <h3>{best_1w["Forecast_1W"]:.2f}</h3>
                    <p>1 Week (No. {int(best_1w["No."])})</p>
                </div>
                <div class="stat-box highlight">
                    <h3>{best_1m["Forecast_1M"]:.2f}</h3>
                    <p>1 Month (No. {int(best_1m["No."])})</p>
                </div>
                <div class="stat-box">
                    <h3>{best_3m["Forecast_3M"]:.2f}</h3>
                    <p>3 Months (No. {int(best_3m["No."])})</p>
                </div>
                <div class="stat-box success">
                    <h3>{best_6m["Forecast_6M"]:.2f}</h3>
                    <p>6 Months (No. {int(best_6m["No."])})</p>
                </div>
                <div class="stat-box success">
                    <h3>{best_1y["Forecast_12M"]:.2f}</h3>
                    <p>1 Year (No. {int(best_1y["No."])})</p>
                </div>
            </div>
        </div>
        """

        # Metrics
        metrics_html = ""
        charts_html = ""

        if best_strat_data is not None:
            metrics = FinancialMetrics.calculate_all(best_strat_data)

            # Determine win rate color class
            win_rate = metrics.get("Win Rate", 0) * 100
            win_rate_class = "success" if win_rate >= 50 else "highlight"

            # Determine sharpe ratio color class
            sharpe = metrics.get("Sharpe Ratio", 0)
            sharpe_class = "success" if sharpe >= 1 else ("" if sharpe >= 0 else "highlight")

            # Metrics Table with modern styling
            metrics_html = f"""
            <div class="card">
                <h2>Financial Metrics (Strategy {best_strategy_id})</h2>
                <p>Key performance indicators for the selected strategy.</p>
                <div class="summary-stats">
                    <div class="stat-box {win_rate_class}">
                        <h3>{metrics.get("Total Profit", 0):,.2f}</h3>
                        <p>Total Profit</p>
                    </div>
                    <div class="stat-box {win_rate_class}">
                        <h3>{win_rate:.1f}%</h3>
                        <p>Win Rate</p>
                    </div>
                    <div class="stat-box">
                        <h3>{metrics.get("Profit Factor", 0):.2f}</h3>
                        <p>Profit Factor</p>
                    </div>
                    <div class="stat-box {sharpe_class}">
                        <h3>{sharpe:.2f}</h3>
                        <p>Sharpe Ratio</p>
                    </div>
                </div>
                <div class="table-wrapper" style="margin-top: 20px;">
                    <table>
                        <thead>
                            <tr><th>Metric</th><th>Value</th></tr>
                        </thead>
                        <tbody>
                            <tr><td>Total Profit</td><td>{metrics.get("Total Profit", 0):,.2f}</td></tr>
                            <tr><td>Win Rate</td><td>{win_rate:.1f}%</td></tr>
                            <tr><td>Profit Factor</td><td>{metrics.get("Profit Factor", 0):.2f}</td></tr>
                            <tr><td>Sharpe Ratio</td><td>{sharpe:.2f}</td></tr>
                            <tr><td>Sortino Ratio</td><td>{metrics.get("Sortino Ratio", 0):.2f}</td></tr>
                            <tr><td>Max Drawdown</td><td>{metrics.get("Max Drawdown ($)", 0):,.2f}</td></tr>
                            <tr><td>Avg Trade</td><td>{metrics.get("Avg Trade", 0):.2f}</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
            """

            # Interactive Charts
            viz = Visualizer(self.output_dir)

            # Common plotly config for responsive charts
            plotly_config = {"responsive": True, "displayModeBar": True}
            plotlyjs_included = False

            # Forecast Chart
            if forecast_values:
                fig_forecast = viz.create_interactive_forecast(best_strat_data, forecast_values, best_strategy_id, method_name)
                fig_forecast.update_layout(autosize=True, width=None)
                charts_html += f'<div class="chart-container">{pio.to_html(fig_forecast, full_html=False, include_plotlyjs="cdn", config=plotly_config)}</div>'
                plotlyjs_included = True

            # Drawdown Chart
            fig_dd = viz.create_interactive_drawdown(best_strat_data, best_strategy_id)
            fig_dd.update_layout(autosize=True, width=None)
            include_js = "cdn" if not plotlyjs_included else False
            charts_html += f'<div class="chart-container">{pio.to_html(fig_dd, full_html=False, include_plotlyjs=include_js, config=plotly_config)}</div>'
            plotlyjs_included = True

            # Returns Distribution
            fig_dist = viz.create_interactive_returns_distribution(best_strat_data, best_strategy_id)
            fig_dist.update_layout(autosize=True, width=None)
            charts_html += (
                f'<div class="chart-container">{pio.to_html(fig_dist, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'
            )

            # Rolling Metrics
            fig_rolling = viz.create_interactive_rolling_metrics(best_strat_data, best_strategy_id)
            fig_rolling.update_layout(autosize=True, width=None)
            charts_html += f'<div class="chart-container">{pio.to_html(fig_rolling, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'

            # Lag Plot
            fig_lag = viz.create_interactive_lag_plot(best_strat_data, best_strategy_id)
            fig_lag.update_layout(autosize=True, width=None)
            charts_html += (
                f'<div class="chart-container">{pio.to_html(fig_lag, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'
            )

            # Decomposition
            fig_decomp = viz.create_interactive_decomposition(best_strat_data, best_strategy_id)
            if fig_decomp:
                fig_decomp.update_layout(autosize=True, width=None)
                charts_html += f'<div class="chart-container">{pio.to_html(fig_decomp, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'

            # Heatmap
            monthly_returns = FinancialMetrics.get_monthly_returns(best_strat_data)
            if not monthly_returns.empty:
                fig_heat = viz.create_interactive_heatmap(monthly_returns, best_strategy_id)
                fig_heat.update_layout(autosize=True, width=None)
                charts_html += f'<div class="chart-container">{pio.to_html(fig_heat, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'

        # Top 10 Ranked HTML
        top_ranked_html = ""
        if ranking_mode != "forecast" and "Ranking_Score" in results_df.columns:
            top_ranked = results_df.head(10)
            rows = ""
            for i, (_, row) in enumerate(top_ranked.iterrows()):
                s_score = f"{row.get('Stability_Score', 0):.2f}" if "Stability_Score" in row else "N/A"
                r_score = f"{row.get('Ranking_Score', 0):.2f}"
                f_1m = f"{row.get('Forecast_1M', 0):.2f}"
                rows += f"<tr><td>{i+1}</td><td>#{int(row['No.'])}</td><td>{f_1m}</td><td>{s_score}</td><td>{r_score}</td></tr>"

            top_ranked_html = f"""
            <div class="card">
                <h2>Top 10 Strategies (Ranked by {ranking_mode})</h2>
                <p>Strategies ranked according to the selected mode ({ranking_mode}).</p>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr><th>Rank</th><th>Strategy</th><th>Forecast 1M</th><th>Stability</th><th>Score</th></tr>
                        </thead>
                        <tbody>
                            {rows}
                        </tbody>
                    </table>
                </div>
            </div>
            """

        # Prepare Top 10 Data for HTML
        top_1w = self._get_top_strategies_by_horizon(results_df, "Forecast_1W")
        top_1m = self._get_top_strategies_by_horizon(results_df, "Forecast_1M")
        top_3m = self._get_top_strategies_by_horizon(results_df, "Forecast_3M")
        top_6m = self._get_top_strategies_by_horizon(results_df, "Forecast_6M")
        top_1y = self._get_top_strategies_by_horizon(results_df, "Forecast_12M")

        # Top Strategies Table with modern styling
        top_table_html = """<table>
            <thead>
                <tr><th>Rank</th><th>1 Week</th><th>1 Month</th><th>3 Months</th><th>6 Months</th><th>1 Year</th></tr>
            </thead>
            <tbody>"""
        for i in range(min(10, len(results_df))):
            v_1w = top_1w[i] if i < len(top_1w) else ""
            v_1m = top_1m[i] if i < len(top_1m) else ""
            v_3m = top_3m[i] if i < len(top_3m) else ""
            v_6m = top_6m[i] if i < len(top_6m) else ""
            v_1y = top_1y[i] if i < len(top_1y) else ""
            top_table_html += f"<tr><td>{i + 1}</td><td>{v_1w}</td><td>{v_1m}</td><td>{v_3m}</td><td>{v_6m}</td><td>{v_1y}</td></tr>"
        top_table_html += "</tbody></table>"

        exec_time_html = f" | <strong>Generating time:</strong> {execution_time}" if execution_time else ""
        ranking_html = f" | <strong>Ranking:</strong> {ranking_mode}" if ranking_mode else ""

        # Prepare logo HTML
        logo_html = ""
        try:
            import base64

            # Path to src/gui/assets/dark_logo.png (relative to src/reporting/exporter.py)
            assets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gui", "assets", "dark_logo.png")
            if os.path.exists(assets_path):
                with open(assets_path, "rb") as img_f:
                    b64_data = base64.b64encode(img_f.read()).decode("utf-8")
                    logo_html = f'<img src="data:image/png;base64,{b64_data}" class="header-logo" alt="MBO Logo">'
        except OSError as e:
            logger.debug("Logo embedding failed: %s", e)

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Trading Strategy Report - Strategy {best_strategy_id}</title>
            <style>
                * {{
                    box-sizing: border-box;
                }}

                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                    min-height: 100vh;
                    color: #e4e4e4;
                }}

                h1 {{
                    color: #fff;
                    font-size: 2.2em;
                    margin: 0 0 10px 0;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }}

                .header-section {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    background: rgba(255, 255, 255, 0.05);
                    backdrop-filter: blur(10px);
                    padding: 25px;
                    margin-bottom: 25px;
                    border-radius: 16px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }}

                .header-content {{
                    flex: 1;
                }}

                .header-logo-wrapper {{
                    background: rgba(255, 255, 255, 0.1);
                    padding: 10px;
                    border-radius: 12px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    margin-left: 20px;
                }}

                .header-logo {{
                    height: 50px;
                    width: auto;
                    display: block;
                    filter: drop-shadow(0px 0px 5px rgba(0,0,0,0.5));
                }}

                h2 {{
                    color: #3498db;
                    font-size: 1.4em;
                    margin: 0 0 15px 0;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}

                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                }}

                .subtitle {{
                    color: #aaa;
                    margin: 0;
                    font-size: 0.95em;
                }}

                /* Summary Stats */
                .summary-stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                    gap: 15px;
                    margin-bottom: 25px;
                }}

                .stat-box {{
                    background: linear-gradient(135deg, #3498db, #2980b9);
                    color: white;
                    padding: 20px;
                    border-radius: 12px;
                    text-align: center;
                    box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }}

                .stat-box:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 8px 25px rgba(52, 152, 219, 0.4);
                }}

                .stat-box.highlight {{
                    background: linear-gradient(135deg, #e67e22, #d35400);
                    box-shadow: 0 4px 15px rgba(230, 126, 34, 0.3);
                }}

                .stat-box.highlight:hover {{
                    box-shadow: 0 8px 25px rgba(230, 126, 34, 0.4);
                }}

                .stat-box.success {{
                    background: linear-gradient(135deg, #27ae60, #1e8449);
                    box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
                }}

                .stat-box.success:hover {{
                    box-shadow: 0 8px 25px rgba(39, 174, 96, 0.4);
                }}

                .stat-box h3 {{
                    margin: 0;
                    font-size: 1.8em;
                    font-weight: 700;
                }}

                .stat-box p {{
                    margin: 8px 0 0 0;
                    opacity: 0.9;
                    font-size: 0.85em;
                }}

                /* Cards */
                .card {{
                    background: rgba(255, 255, 255, 0.05);
                    backdrop-filter: blur(10px);
                    padding: 25px;
                    margin-bottom: 20px;
                    border-radius: 16px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    overflow: hidden;
                }}

                .card p {{
                    color: #bbb;
                    margin: 0 0 15px 0;
                    font-size: 0.95em;
                }}

                .card ul {{
                    list-style: none;
                    padding: 0;
                    margin: 0;
                }}

                .card ul li {{
                    padding: 10px 0;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                    color: #e4e4e4;
                }}

                .card ul li:last-child {{
                    border-bottom: none;
                }}

                .card ul li strong {{
                    color: #3498db;
                    min-width: 120px;
                    display: inline-block;
                }}

                /* Chart containers */
                .chart-container {{
                    background: rgba(255, 255, 255, 0.03);
                    padding: 20px;
                    border-radius: 12px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    max-width: 100%;
                    overflow-x: auto;
                    overflow-y: hidden;
                }}

                .chart-container .plotly-graph-div {{
                    width: 100% !important;
                    max-width: 100% !important;
                }}

                .chart-container .js-plotly-plot {{
                    width: 100% !important;
                }}

                .chart-container .svg-container {{
                    width: 100% !important;
                    max-width: 100% !important;
                }}

                /* Charts grid */
                .charts-grid {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                    margin-bottom: 20px;
                }}

                @media (max-width: 1200px) {{
                    .charts-grid {{
                        grid-template-columns: 1fr;
                    }}
                }}

                /* Modern Table */
                .table-wrapper {{
                    overflow-x: auto;
                    border-radius: 12px;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
                }}

                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background: rgba(30, 30, 50, 0.8);
                    font-size: 0.9em;
                }}

                th {{
                    background: linear-gradient(135deg, #3498db, #2980b9);
                    color: white;
                    padding: 16px 12px;
                    text-align: left;
                    font-weight: 600;
                    white-space: nowrap;
                    border: none;
                }}

                th:first-child {{
                    border-radius: 12px 0 0 0;
                }}

                th:last-child {{
                    border-radius: 0 12px 0 0;
                }}

                td {{
                    padding: 14px 12px;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                    color: #e4e4e4;
                    vertical-align: middle;
                }}

                tbody tr:nth-child(even) {{
                    background: rgba(255, 255, 255, 0.02);
                }}

                tbody tr {{
                    transition: background 0.2s ease;
                }}

                tbody tr:hover {{
                    background: rgba(52, 152, 219, 0.15);
                }}

                /* Specific column colors */
                td:first-child {{
                    color: #e67e22;
                    font-weight: 600;
                }}

                /* Scrollbar styling */
                ::-webkit-scrollbar {{
                    width: 8px;
                    height: 8px;
                }}

                ::-webkit-scrollbar-track {{
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 4px;
                }}

                ::-webkit-scrollbar-thumb {{
                    background: rgba(52, 152, 219, 0.5);
                    border-radius: 4px;
                }}

                ::-webkit-scrollbar-thumb:hover {{
                    background: rgba(52, 152, 219, 0.7);
                }}

                /* Badge */
                .badge {{
                    background: linear-gradient(135deg, #e74c3c, #c0392b);
                    color: white;
                    padding: 6px 14px;
                    border-radius: 20px;
                    font-size: 0.9em;
                    font-weight: 600;
                    display: inline-block;
                }}

                .badge.success {{
                    background: linear-gradient(135deg, #27ae60, #1e8449);
                }}

                .badge.info {{
                    background: linear-gradient(135deg, #3498db, #2980b9);
                }}

                /* Section title with icon */
                .section-title {{
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin-bottom: 20px;
                }}

                .section-title h2 {{
                    margin: 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header-section">
                    <div class="header-content">
                        <h1>Trading Strategy Analysis Report</h1>
                        <p class="subtitle"><strong>Date:</strong> {timestamp}{exec_time_html}{ranking_html} | <strong>Method:</strong> {method_name}</p>
                    </div>
                    <div class="header-logo-wrapper">
                        {logo_html}
                    </div>
                </div>

                <div class="card">
                    <h2>Executive Summary</h2>
                    <p>Analysis completed successfully. The top performing strategy is <span class="badge highlight">Strategy No. {best_strategy_id}</span></p>
                </div>

                {metrics_html}

                {forecast_html}

                {settings_html}

                <div class="section-title">
                    <h2>Interactive Visualizations</h2>
                </div>
                {charts_html}

                {top_ranked_html}

                <div class="card">
                    <h2>Top 10 Strategies by Horizon</h2>
                    <p>Performance ranking across different forecast horizons.</p>
                    <div class="table-wrapper">
                        {top_table_html}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)

        return filename

    def create_all_results_report(
        self, results_df, method_name, best_strat_data=None, filename_base=None, params=None,
        execution_time=None, forecast_horizon=52, ranking_mode=None
    ):
        """
        Generates a detailed report with weekly breakdown of forecasts.
        Shows individual week forecasts (Week 1, Week 2, etc.) not cumulative.

        Args:
            results_df: DataFrame with strategy results including 'Forecasts' column
            method_name: Name of the analysis method used
            best_strat_data: Historical data for the best strategy
            filename_base: Base name for output files (should end with _all)
            params: Model parameters dictionary
            execution_time: Time taken for analysis
            forecast_horizon: Number of weeks to forecast (default 52)
            ranking_mode: Ranking mode used (Standard, StabilityWeighted, RiskAdjusted)

        Returns:
            Tuple of (md_path, html_path)
        """
        from analysis.metrics import FinancialMetrics

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # File naming follows Generate Reports pattern: filename_base.md / filename_base.html
        if filename_base:
            md_filename = os.path.join(self.output_dir, f"{filename_base}.md")
            html_filename = os.path.join(self.output_dir, f"{filename_base}.html")
        else:
            md_filename = os.path.join(self.output_dir, f"AllResults_{datetime.now().strftime('%Y%m%d_%H%M')}.md")
            html_filename = os.path.join(self.output_dir, f"AllResults_{datetime.now().strftime('%Y%m%d_%H%M')}.html")

        # Get best strategy info
        best_strategy_id = int(results_df.iloc[0]["No."]) if not results_df.empty else 0

        import numpy as np

        # Check if Forecasts column exists with valid data
        has_forecasts = "Forecasts" in results_df.columns
        if has_forecasts:
            # Verify at least one row has valid forecast array
            sample_forecasts = results_df.iloc[0].get("Forecasts", None) if not results_df.empty else None
            has_forecasts = isinstance(sample_forecasts, (list, tuple, np.ndarray)) and len(sample_forecasts) > 0

        # Extract weekly forecasts from all strategies
        weekly_best_strategies = []

        if has_forecasts:
            # Use detailed weekly forecasts from Forecasts column
            for week_idx in range(forecast_horizon):
                best_for_week = None
                best_value = float("-inf")

                for _, row in results_df.iterrows():
                    forecasts = row.get("Forecasts", [])
                    if isinstance(forecasts, (list, tuple, np.ndarray)) and len(forecasts) > week_idx:
                        week_value = float(forecasts[week_idx])
                        if week_value > best_value:
                            best_value = week_value
                            best_for_week = {"week": week_idx + 1, "strategy_no": int(row["No."]), "profit": week_value}

                if best_for_week:
                    weekly_best_strategies.append(best_for_week)
                else:
                    weekly_best_strategies.append({"week": week_idx + 1, "strategy_no": 0, "profit": 0.0})
        else:
            # FALLBACK: Use cumulative forecast columns (Forecast_1W, Forecast_1M, etc.)
            # Generate approximated weekly data from available columns
            forecast_cols = ["Forecast_1W", "Forecast_1M", "Forecast_3M", "Forecast_6M", "Forecast_12M"]
            week_ranges = [1, 4, 13, 26, 52]  # Approximate weeks for each forecast

            for week_idx in range(forecast_horizon):
                best_for_week = None
                best_value = float("-inf")

                for _, row in results_df.iterrows():
                    # Find appropriate forecast column for this week
                    week_value = 0.0
                    for col, max_week in zip(forecast_cols, week_ranges):
                        if col in row and week_idx < max_week:
                            # Approximate weekly value from cumulative
                            cumulative = float(row.get(col, 0) or 0)
                            week_value = cumulative / max_week  # Average per week
                            break

                    if week_value > best_value:
                        best_value = week_value
                        best_for_week = {"week": week_idx + 1, "strategy_no": int(row["No."]), "profit": week_value}

                if best_for_week:
                    weekly_best_strategies.append(best_for_week)
                else:
                    weekly_best_strategies.append({"week": week_idx + 1, "strategy_no": 0, "profit": 0.0})

        # Calculate metrics if data available
        metrics = {}
        if best_strat_data is not None:
            metrics = FinancialMetrics.calculate_all(best_strat_data)

        # Calculate strategy frequency for pie chart
        strategy_counts = {}
        for week_data in weekly_best_strategies:
            strat_no = week_data["strategy_no"]
            strategy_counts[strat_no] = strategy_counts.get(strat_no, 0) + 1

        # Get best strategy's forecasts for cumulative comparison
        best_strategy_forecasts = []
        if not results_df.empty:
            best_row = results_df[results_df["No."] == best_strategy_id]
            if not best_row.empty:
                if has_forecasts and "Forecasts" in best_row.columns:
                    forecasts = best_row.iloc[0]["Forecasts"]
                    if isinstance(forecasts, (list, tuple, np.ndarray)):
                        best_strategy_forecasts = list(forecasts)
                else:
                    # Fallback: generate approximated forecasts from cumulative columns
                    row = best_row.iloc[0]
                    forecast_cols = ["Forecast_1W", "Forecast_1M", "Forecast_3M", "Forecast_6M", "Forecast_12M"]
                    week_ranges = [1, 4, 13, 26, 52]
                    for week_idx in range(forecast_horizon):
                        for col, max_week in zip(forecast_cols, week_ranges):
                            if col in row and week_idx < max_week:
                                cumulative = float(row.get(col, 0) or 0)
                                best_strategy_forecasts.append(cumulative / max_week)
                                break

        # --- Generate Markdown Report ---
        md_content = self._generate_all_results_markdown(
            timestamp, method_name, best_strategy_id, params, metrics, weekly_best_strategies,
            execution_time, forecast_horizon, ranking_mode
        )

        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(md_content)

        # --- Generate HTML Report ---
        html_content = self._generate_all_results_html(
            timestamp,
            method_name,
            best_strategy_id,
            params,
            metrics,
            weekly_best_strategies,
            execution_time,
            forecast_horizon,
            strategy_counts,
            best_strategy_forecasts,
            ranking_mode,
        )

        with open(html_filename, "w", encoding="utf-8") as f:
            f.write(html_content)

        return md_filename, html_filename

    def _generate_all_results_markdown(
        self, timestamp, method_name, best_strategy_id, params, metrics, weekly_best,
        execution_time, forecast_horizon, ranking_mode=None
    ):
        """Generate markdown content for All Results report."""

        # Model Settings Section
        settings_section = ""
        if params and len(params) > 0:
            settings_lines = [f"| **{key}** | {value} |" for key, value in params.items()]
            settings_section = (
                """
## Model Settings
| Parameter | Value |
|-----------|-------|
"""
                + "\n".join(settings_lines)
                + "\n"
            )

        # Metrics Section
        metrics_section = ""
        if metrics:
            metrics_section = f"""
## Financial Metrics (Best Strategy: {best_strategy_id})
| Metric | Value |
|--------|-------|
| **Total Profit** | {metrics.get("Total Profit", 0):.2f} |
| **Win Rate** | {metrics.get("Win Rate", 0) * 100:.1f}% |
| **Profit Factor** | {metrics.get("Profit Factor", 0):.2f} |
| **Sharpe Ratio** | {metrics.get("Sharpe Ratio", 0):.2f} |
| **Sortino Ratio** | {metrics.get("Sortino Ratio", 0):.2f} |
| **Max Drawdown** | {metrics.get("Max Drawdown ($)", 0):.2f} |
| **Avg Trade** | {metrics.get("Avg Trade", 0):.2f} |
"""

        # Weekly Breakdown Section
        weekly_rows = ""
        for week_data in weekly_best:
            weekly_rows += f"| Week {week_data['week']} | No. {week_data['strategy_no']} | {week_data['profit']:.2f} |\n"

        exec_time_line = f"**Generating time**: {execution_time}\n" if execution_time else ""
        ranking_line = f"**Ranking Mode**: {ranking_mode}\n" if ranking_mode else ""

        md_content = f"""# Trading Strategy Analysis Report - All Results
**Date**: {timestamp}
{exec_time_line}{ranking_line}**Method**: {method_name}
**Forecast Horizon**: {forecast_horizon} weeks

## Executive Summary
This report shows the best performing strategy for each individual week forecast.
The overall best strategy (1 Month horizon) is **Strategy No. {best_strategy_id}**.

{settings_section}

{metrics_section}

## Weekly Forecast Breakdown
Best strategy for each individual week (not cumulative):

| Week | Best Strategy | Predicted Profit |
|------|---------------|------------------|
{weekly_rows}
"""
        return md_content

    def _generate_all_results_html(
        self,
        timestamp,
        method_name,
        best_strategy_id,
        params,
        metrics,
        weekly_best,
        execution_time,
        forecast_horizon,
        strategy_counts=None,
        best_strategy_forecasts=None,
        ranking_mode=None,
    ):
        """Generate HTML content for All Results report with modern dark theme and interactive charts."""
        import plotly.graph_objects as go
        import plotly.io as pio

        # Initialize defaults
        if strategy_counts is None:
            strategy_counts = {}
        if best_strategy_forecasts is None:
            best_strategy_forecasts = []

        charts_html = ""
        plotly_config = {"responsive": True, "displayModeBar": True}

        # ============ Chart 1: Weekly Profit Forecast Line Chart ============
        if weekly_best:
            weeks = [w["week"] for w in weekly_best]
            profits = [w["profit"] for w in weekly_best]
            colors = ["#2ecc71" if p >= 0 else "#e74c3c" for p in profits]

            fig_line = go.Figure()
            fig_line.add_trace(
                go.Scatter(
                    x=weeks,
                    y=profits,
                    mode="lines+markers",
                    name="Weekly Profit",
                    line=dict(color="#f39c12", width=2),
                    marker=dict(size=8, color=colors, line=dict(width=1, color="white")),
                    fill="tozeroy",
                    fillcolor="rgba(243, 156, 18, 0.1)",
                    hovertemplate="Week %{x}<br>Profit: %{y:.2f}<extra></extra>",
                )
            )

            # Add zero line
            fig_line.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

            fig_line.update_layout(
                title=dict(text="Weekly Profit Forecast", font=dict(color="white", size=16)),
                xaxis=dict(title="Week", color="white", gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(title="Predicted Profit", color="white", gridcolor="rgba(255,255,255,0.1)"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                hovermode="x unified",
                showlegend=False,
                margin=dict(l=60, r=30, t=50, b=50),
            )
            charts_html += (
                f'<div class="chart-container">{pio.to_html(fig_line, full_html=False, include_plotlyjs="cdn", config=plotly_config)}</div>'
            )

        # ============ Chart 2: Strategy Frequency Pie Chart ============
        if strategy_counts:
            # Sort by count, group small ones as "Other"
            sorted_counts = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)
            top_strategies = sorted_counts[:5]
            other_count = sum(c for _, c in sorted_counts[5:])

            labels = [f"Strategy {s}" for s, _ in top_strategies]
            values = [c for _, c in top_strategies]
            if other_count > 0:
                labels.append("Other")
                values.append(other_count)

            colors_pie = ["#f39c12", "#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#95a5a6"]

            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.4,
                        marker=dict(colors=colors_pie[: len(labels)], line=dict(color="white", width=2)),
                        textinfo="label+percent",
                        textfont=dict(color="white", size=12),
                        hovertemplate="%{label}<br>Weeks: %{value}<br>%{percent}<extra></extra>",
                    )
                ]
            )

            fig_pie.update_layout(
                title=dict(text="Strategy Dominance (Best Strategy per Week)", font=dict(color="white", size=16)),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                showlegend=True,
                legend=dict(font=dict(color="white")),
                margin=dict(l=30, r=30, t=50, b=30),
            )
            charts_html += (
                f'<div class="chart-container">{pio.to_html(fig_pie, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'
            )

        # ============ Chart 3: Cumulative Profit Comparison ============
        if weekly_best and best_strategy_forecasts:
            # "Perfect switching" cumulative - following best strategy each week
            perfect_cumulative = []
            cumsum = 0
            for w in weekly_best:
                cumsum += w["profit"]
                perfect_cumulative.append(cumsum)

            # Single best strategy cumulative
            single_cumulative = []
            cumsum = 0
            for val in best_strategy_forecasts[: len(weekly_best)]:
                cumsum += val
                single_cumulative.append(cumsum)

            weeks = list(range(1, len(weekly_best) + 1))

            fig_cumulative = go.Figure()
            fig_cumulative.add_trace(
                go.Scatter(
                    x=weeks,
                    y=perfect_cumulative,
                    mode="lines",
                    name="Perfect Switching (Best each week)",
                    line=dict(color="#2ecc71", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(46, 204, 113, 0.1)",
                    hovertemplate="Week %{x}<br>Cumulative: %{y:.2f}<extra></extra>",
                )
            )
            fig_cumulative.add_trace(
                go.Scatter(
                    x=weeks,
                    y=single_cumulative,
                    mode="lines",
                    name=f"Strategy {best_strategy_id} Only",
                    line=dict(color="#f39c12", width=3, dash="dash"),
                    hovertemplate="Week %{x}<br>Cumulative: %{y:.2f}<extra></extra>",
                )
            )

            # Calculate and display advantage annotation
            advantage = 0
            advantage_pct = 0
            # Check all lists are non-empty to avoid IndexError
            if perfect_cumulative and single_cumulative and weeks:
                advantage = perfect_cumulative[-1] - single_cumulative[-1]
                # Safe division - avoid division by zero
                if single_cumulative[-1] != 0:
                    advantage_pct = (advantage / abs(single_cumulative[-1])) * 100

                # Add annotation showing the advantage
                advantage_color = "#2ecc71" if advantage >= 0 else "#e74c3c"
                advantage_sign = "+" if advantage >= 0 else ""
                fig_cumulative.add_annotation(
                    x=weeks[-1],
                    y=max(perfect_cumulative[-1], single_cumulative[-1]) * 1.05,
                    text=f"Advantage: {advantage_sign}{advantage:.2f} ({advantage_sign}{advantage_pct:.1f}%)",
                    showarrow=False,
                    font=dict(size=14, color=advantage_color, family="Arial Black"),
                    bgcolor="rgba(0,0,0,0.5)",
                    borderpad=4,
                )

            fig_cumulative.update_layout(
                title=dict(text="Cumulative Profit: Perfect Switching vs Single Strategy", font=dict(color="white", size=16)),
                xaxis=dict(title="Week", color="white", gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(title="Cumulative Profit", color="white", gridcolor="rgba(255,255,255,0.1)"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                hovermode="x unified",
                legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.3)", x=0.02, y=0.98),
                margin=dict(l=60, r=30, t=50, b=50),
            )
            charts_html += f'<div class="chart-container">{pio.to_html(fig_cumulative, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'

        # ============ Chart 4: Strategy Switches Timeline ============
        if weekly_best:
            # Assign colors to strategies
            unique_strategies = list({w["strategy_no"] for w in weekly_best})
            color_palette = ["#f39c12", "#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#1abc9c", "#e91e63", "#00bcd4"]
            strategy_colors = {s: color_palette[i % len(color_palette)] for i, s in enumerate(unique_strategies)}

            # Create timeline data
            weeks = [w["week"] for w in weekly_best]
            strategies = [w["strategy_no"] for w in weekly_best]
            colors_timeline = [strategy_colors[s] for s in strategies]

            fig_timeline = go.Figure()

            # Add bars for each week showing which strategy
            fig_timeline.add_trace(
                go.Bar(
                    x=weeks,
                    y=[1] * len(weeks),
                    marker=dict(color=colors_timeline, line=dict(width=0)),
                    hovertemplate="Week %{x}<br>Strategy: %{customdata}<extra></extra>",
                    customdata=strategies,
                    showlegend=False,
                )
            )

            # Add strategy switch markers
            switches = []
            for i in range(1, len(strategies)):
                if strategies[i] != strategies[i - 1]:
                    switches.append(weeks[i])

            if switches:
                fig_timeline.add_trace(
                    go.Scatter(
                        x=switches,
                        y=[1.1] * len(switches),
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=12, color="white"),
                        name=f"{len(switches)} Strategy Switches",
                        hovertemplate="Switch at Week %{x}<extra></extra>",
                    )
                )

            fig_timeline.update_layout(
                title=dict(
                    text=f"Strategy Timeline ({len(switches)} switches across {forecast_horizon} weeks)", font=dict(color="white", size=16)
                ),
                xaxis=dict(title="Week", color="white", gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(visible=False, range=[0, 1.3]),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                showlegend=True,
                legend=dict(font=dict(color="white")),
                margin=dict(l=30, r=30, t=50, b=50),
                bargap=0,
            )
            charts_html += f'<div class="chart-container">{pio.to_html(fig_timeline, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'

        # Model Settings HTML
        settings_html = ""
        if params and len(params) > 0:
            settings_rows = "".join([f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in params.items()])
            settings_html = f"""
            <div class="card">
                <h2>Model Settings</h2>
                <p>Configuration parameters used for the analysis.</p>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr><th>Parameter</th><th>Value</th></tr>
                        </thead>
                        <tbody>
                            {settings_rows}
                        </tbody>
                    </table>
                </div>
            </div>
            """

        # Metrics HTML
        metrics_html = ""
        if metrics:
            win_rate = metrics.get("Win Rate", 0) * 100
            win_rate_class = "success" if win_rate >= 50 else "highlight"
            sharpe = metrics.get("Sharpe Ratio", 0)
            sharpe_class = "success" if sharpe >= 1 else ("" if sharpe >= 0 else "highlight")

            metrics_html = f"""
            <div class="card">
                <h2>Financial Metrics (Best Strategy: {best_strategy_id})</h2>
                <p>Key performance indicators for the best performing strategy.</p>
                <div class="summary-stats">
                    <div class="stat-box {win_rate_class}">
                        <h3>{metrics.get("Total Profit", 0):,.2f}</h3>
                        <p>Total Profit</p>
                    </div>
                    <div class="stat-box {win_rate_class}">
                        <h3>{win_rate:.1f}%</h3>
                        <p>Win Rate</p>
                    </div>
                    <div class="stat-box">
                        <h3>{metrics.get("Profit Factor", 0):.2f}</h3>
                        <p>Profit Factor</p>
                    </div>
                    <div class="stat-box {sharpe_class}">
                        <h3>{sharpe:.2f}</h3>
                        <p>Sharpe Ratio</p>
                    </div>
                </div>
            </div>
            """

        # Weekly Breakdown Table
        weekly_rows = ""
        for week_data in weekly_best:
            profit_class = "positive" if week_data["profit"] >= 0 else "negative"
            weekly_rows += f"""
                <tr>
                    <td>Week {week_data["week"]}</td>
                    <td>No. {week_data["strategy_no"]}</td>
                    <td class="{profit_class}">{week_data["profit"]:.2f}</td>
                </tr>"""

        exec_time_html = f" | <strong>Generating time:</strong> {execution_time}" if execution_time else ""
        ranking_html = f" | <strong>Ranking:</strong> {ranking_mode}" if ranking_mode else ""

        # Prepare logo HTML
        logo_html = ""
        try:
            import base64

            # Path to src/gui/assets/dark_logo.png
            assets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gui", "assets", "dark_logo.png")
            if os.path.exists(assets_path):
                with open(assets_path, "rb") as img_f:
                    b64_data = base64.b64encode(img_f.read()).decode("utf-8")
                    logo_html = f'<img src="data:image/png;base64,{b64_data}" class="header-logo" alt="MBO Logo">'
        except OSError as e:
            logger.debug("Logo embedding failed: %s", e)

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>All Results Report - {method_name}</title>
            <style>
                * {{
                    box-sizing: border-box;
                }}

                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                    min-height: 100vh;
                    color: #e4e4e4;
                }}

                h1 {{
                    color: #fff;
                    font-size: 2.2em;
                    margin: 0 0 10px 0;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }}

                .header-section {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    background: rgba(255, 255, 255, 0.05);
                    backdrop-filter: blur(10px);
                    padding: 25px;
                    margin-bottom: 25px;
                    border-radius: 16px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }}

                .header-content {{
                    flex: 1;
                }}

                .header-logo-wrapper {{
                    background: rgba(255, 255, 255, 0.1);
                    padding: 10px;
                    border-radius: 12px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    margin-left: 20px;
                }}

                .header-logo {{
                    height: 50px;
                    width: auto;
                    display: block;
                    filter: drop-shadow(0px 0px 5px rgba(0,0,0,0.5));
                }}

                h2 {{
                    color: #f39c12;
                    font-size: 1.4em;
                    margin: 0 0 15px 0;
                }}

                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}

                .subtitle {{
                    color: #aaa;
                    margin: 0;
                    font-size: 0.95em;
                }}

                /* Summary Stats */
                .summary-stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                    gap: 15px;
                    margin-bottom: 25px;
                }}

                .stat-box {{
                    background: linear-gradient(135deg, #3498db, #2980b9);
                    color: white;
                    padding: 20px;
                    border-radius: 12px;
                    text-align: center;
                    box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }}

                .stat-box:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 8px 25px rgba(52, 152, 219, 0.4);
                }}

                .stat-box.highlight {{
                    background: linear-gradient(135deg, #e67e22, #d35400);
                    box-shadow: 0 4px 15px rgba(230, 126, 34, 0.3);
                }}

                .stat-box.success {{
                    background: linear-gradient(135deg, #27ae60, #1e8449);
                    box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
                }}

                .stat-box h3 {{
                    margin: 0;
                    font-size: 1.8em;
                    font-weight: 700;
                }}

                .stat-box p {{
                    margin: 8px 0 0 0;
                    opacity: 0.9;
                    font-size: 0.85em;
                }}

                /* Cards */
                .card {{
                    background: rgba(255, 255, 255, 0.05);
                    backdrop-filter: blur(10px);
                    padding: 25px;
                    margin-bottom: 20px;
                    border-radius: 16px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    overflow: hidden;
                }}

                .card p {{
                    color: #bbb;
                    margin: 0 0 15px 0;
                    font-size: 0.95em;
                }}

                /* Modern Table */
                .table-wrapper {{
                    overflow-x: auto;
                    border-radius: 12px;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
                    max-height: 600px;
                    overflow-y: auto;
                }}

                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background: rgba(30, 30, 50, 0.8);
                    font-size: 0.9em;
                }}

                thead {{
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }}

                th {{
                    background: linear-gradient(135deg, #f39c12, #d68910);
                    color: white;
                    padding: 16px 12px;
                    text-align: left;
                    font-weight: 600;
                    white-space: nowrap;
                    border: none;
                }}

                th:first-child {{
                    border-radius: 12px 0 0 0;
                }}

                th:last-child {{
                    border-radius: 0 12px 0 0;
                }}

                td {{
                    padding: 14px 12px;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                    color: #e4e4e4;
                    vertical-align: middle;
                }}

                tbody tr:nth-child(even) {{
                    background: rgba(255, 255, 255, 0.02);
                }}

                tbody tr {{
                    transition: background 0.2s ease;
                }}

                tbody tr:hover {{
                    background: rgba(243, 156, 18, 0.15);
                }}

                /* Column colors */
                td:first-child {{
                    color: #f39c12;
                    font-weight: 600;
                }}

                td:nth-child(2) {{
                    color: #3498db;
                    font-weight: 600;
                }}

                td.positive {{
                    color: #2ecc71;
                    font-weight: 600;
                }}

                td.negative {{
                    color: #e74c3c;
                    font-weight: 600;
                }}

                /* Badge */
                .badge {{
                    background: linear-gradient(135deg, #f39c12, #d68910);
                    color: white;
                    padding: 6px 14px;
                    border-radius: 20px;
                    font-size: 0.9em;
                    font-weight: 600;
                    display: inline-block;
                }}

                /* Scrollbar styling */
                ::-webkit-scrollbar {{
                    width: 8px;
                    height: 8px;
                }}

                ::-webkit-scrollbar-track {{
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 4px;
                }}

                ::-webkit-scrollbar-thumb {{
                    background: rgba(243, 156, 18, 0.5);
                    border-radius: 4px;
                }}

                ::-webkit-scrollbar-thumb:hover {{
                    background: rgba(243, 156, 18, 0.7);
                }}

                /* Info box */
                .info-box {{
                    background: rgba(243, 156, 18, 0.1);
                    border-left: 4px solid #f39c12;
                    padding: 15px 20px;
                    margin-bottom: 20px;
                    border-radius: 0 8px 8px 0;
                }}

                .info-box p {{
                    margin: 0;
                    color: #e4e4e4;
                }}

                /* Chart containers */
                .chart-container {{
                    background: rgba(255, 255, 255, 0.03);
                    padding: 20px;
                    border-radius: 12px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    max-width: 100%;
                    overflow-x: auto;
                    overflow-y: hidden;
                }}

                .chart-container .plotly-graph-div {{
                    width: 100% !important;
                    max-width: 100% !important;
                }}

                .charts-grid {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                    margin-bottom: 20px;
                }}

                @media (max-width: 1200px) {{
                    .charts-grid {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header-section">
                    <div class="header-content">
                        <h1>Trading Strategy Analysis - All Results</h1>
                        <p class="subtitle"><strong>Date:</strong> {timestamp}{exec_time_html}{ranking_html} | <strong>Method:</strong> {method_name} | <strong>Horizon:</strong> {forecast_horizon} weeks</p>
                    </div>
                    <div class="header-logo-wrapper">
                        {logo_html}
                    </div>
                </div>

                <div class="card">
                    <h2>Executive Summary</h2>
                    <p>This report shows the best performing strategy for each individual week forecast (not cumulative).</p>
                    <p>The overall best strategy (1 Month horizon) is <span class="badge">Strategy No. {best_strategy_id}</span></p>
                </div>

                {metrics_html}

                {settings_html}

                {charts_html}

                <div class="card">
                    <h2>Weekly Forecast Breakdown</h2>
                    <div class="info-box">
                        <p>Each row shows the best strategy for that specific week's forecast only (individual week profit, not cumulative).</p>
                    </div>
                    <div class="table-wrapper">
                        <table>
                            <thead>
                                <tr>
                                    <th>Week</th>
                                    <th>Best Strategy</th>
                                    <th>Predicted Profit</th>
                                </tr>
                            </thead>
                            <tbody>
                                {weekly_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        return html_content
