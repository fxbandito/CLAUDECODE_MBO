"""
Main Data MR comparison functionality.
Compares Monthly Results (MR_) reports that contain 4-4-5 calendar breakdowns.
"""

# pylint: disable=line-too-long
# ruff: noqa: E501
# Note: This file contains embedded HTML/CSS templates where line breaks
# would significantly reduce readability. The long lines are intentional.
import html
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Tuple, cast

import pandas as pd

from .base import scan_mr_reports


def parse_mr_report(file_path: str) -> Dict[str, Any]:
    """
    Parses an MR_ markdown report to extract method info and monthly forecast data.

    Args:
        file_path: Path to the MR_ prefixed markdown report

    Returns:
        Dictionary with file info, method, generating_time, monthly_data
    """
    data = {
        "file": os.path.basename(file_path),
        "path": file_path,
        "method": "Unknown",
        "generating_time": "N/A",
        "monthly_data": [],
        "total_yearly_profit": 0.0,
    }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")

        # Extract Method - multiple patterns supported
        method_match = re.search(r"\*\*Method\*\*:\s*(.+)", content)
        if method_match:
            data["method"] = method_match.group(1).strip()

        # Extract Generation Time - multiple patterns
        time_match = re.search(r"\*\*Generation Time\*\*:\s*(.+)", content)
        if not time_match:
            time_match = re.search(r"\*\*Generating time\*\*:\s*(.+)", content)
        if time_match:
            data["generating_time"] = time_match.group(1).strip()

        # Extract Total Yearly Profit - multiple patterns
        yearly_match = re.search(r"\*\*Total Predicted Yearly Profit\*\*:\s*([\d,.-]+)", content)
        if not yearly_match:
            yearly_match = re.search(r"\*\*Total Yearly Profit\*\*:\s*([\d,.-]+)", content)
        if yearly_match:
            try:
                data["total_yearly_profit"] = float(yearly_match.group(1).replace(",", ""))
            except ValueError:
                pass

        # Parse Monthly Forecast table
        # Looking for table with Month | Week Range | Weeks | Best Strategy | Predicted Profit
        in_table = False
        for line in lines:
            # Detect table header - multiple formats
            if "| Month |" in line and ("Best Strategy" in line or "Strategy" in line):
                in_table = True
                continue
            # Skip separator line
            if in_table and "|---" in line:
                continue

            if in_table and line.strip().startswith("|"):
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 5:
                    month = parts[0].replace("**", "")  # Remove bold markers
                    week_range = parts[1].replace("**", "")
                    weeks_count = parts[2].replace("**", "")

                    # Extract strategy number (handle #1234, No. 1234, or just numbers)
                    strat_text = parts[3].replace("**", "")
                    strat_match = re.search(r"#?(\d+)", strat_text)
                    strategy_no = int(strat_match.group(1)) if strat_match else 0

                    # Extract profit (handle bold markers and commas)
                    profit_text = parts[4].replace("**", "").replace(",", "")
                    try:
                        profit = float(profit_text)
                    except ValueError:
                        profit = 0.0

                    # Skip TOTAL row
                    if "TOTAL" in month.upper():
                        continue

                    # Skip Composite entries
                    if "Composite" in strat_text:
                        continue

                    cast(List[Dict[str, Any]], data["monthly_data"]).append({
                        "month": month,
                        "week_range": week_range,
                        "weeks_count": weeks_count,
                        "strategy_no": strategy_no,
                        "profit": profit
                    })
            elif in_table and not line.strip().startswith("|") and line.strip():
                # Check if this is just a note/comment line starting with >
                if line.strip().startswith(">"):
                    continue
                break

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error parsing MR report {file_path}: {e}")

    return data


def generate_main_data_mr_report(root_folder: str) -> Tuple[bool, str, str]:
    """
    Generates the Main Data MR Comparison report from all MR_ prefixed files.
    Analyzes monthly forecast data across all models using 4-4-5 calendar.

    Args:
        root_folder: Root folder containing the MR_ report files

    Returns:
        Tuple of (Success, Message, HTML Path)
    """
    # pylint: disable=import-outside-toplevel
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio

    # Scan for MR_ reports
    mr_files = scan_mr_reports(root_folder)
    if not mr_files:
        return False, "No MR_ prefixed .md files found.", ""

    # Parse all MR reports
    parsed_reports = []
    for report_file in mr_files:
        parsed = parse_mr_report(report_file)
        if parsed["monthly_data"]:
            parsed_reports.append(parsed)

    if not parsed_reports:
        return False, "Failed to parse any MR reports.", ""

    # Sort by method name
    parsed_reports.sort(key=lambda x: x["method"])

    # =========================================================================
    # DATA AGGREGATION
    # =========================================================================

    total_reports = len(parsed_reports)
    unique_methods = len({r["method"] for r in parsed_reports})

    # Monthly consensus: which strategy is best for each month
    monthly_consensus: Dict[int, Counter] = {m: Counter() for m in range(1, 13)}
    method_monthly_data: Dict[str, Dict[int, Dict[str, Any]]] = {}
    strategy_total_appearances: Counter[int] = Counter()
    method_yearly_profits: Dict[str, float] = {}
    method_stability: Dict[str, int] = {}

    for report in parsed_reports:
        method = report["method"]
        method_monthly_data[method] = {}
        method_yearly_profits[method] = report["total_yearly_profit"]
        prev_strategy = None
        switch_count = 0

        for idx, month_data in enumerate(report["monthly_data"]):
            month_num = idx + 1  # 1-12
            strat = month_data["strategy_no"]
            profit = month_data["profit"]

            monthly_consensus[month_num][strat] += 1
            method_monthly_data[method][month_num] = {
                "strategy_no": strat,
                "profit": profit,
                "month_name": month_data["month"]
            }
            strategy_total_appearances[strat] += 1

            if prev_strategy is not None and strat != prev_strategy:
                switch_count += 1
            prev_strategy = strat

        method_stability[method] = switch_count

    # =========================================================================
    # CALCULATE METRICS
    # =========================================================================

    unique_strategies = len(strategy_total_appearances)
    most_dominant_strategy = strategy_total_appearances.most_common(1)[0] if strategy_total_appearances else (0, 0)

    monthly_consensus_strength = []
    monthly_best_strategies = []
    month_names = [
        "Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6",
        "Month 7", "Month 8", "Month 9", "Month 10", "Month 11", "Month 12"
    ]

    for month in range(1, 13):
        if monthly_consensus[month]:
            best_strat, best_count = monthly_consensus[month].most_common(1)[0]
            consensus_pct = (best_count / total_reports) * 100
            monthly_consensus_strength.append({
                "month": month,
                "month_name": month_names[month - 1],
                "best_strategy": best_strat,
                "votes": best_count,
                "consensus_pct": consensus_pct
            })
            monthly_best_strategies.append(best_strat)

    avg_consensus = (
        sum(m["consensus_pct"] for m in monthly_consensus_strength) / len(monthly_consensus_strength)
        if monthly_consensus_strength else 0
    )

    most_stable_method = min(method_stability.items(), key=lambda x: x[1]) if method_stability else ("N/A", 0)
    best_yearly_method = max(method_yearly_profits.items(), key=lambda x: x[1]) if method_yearly_profits else ("N/A", 0)
    avg_yearly_profit = sum(method_yearly_profits.values()) / len(method_yearly_profits) if method_yearly_profits else 0

    # =========================================================================
    # GENERATE VISUALIZATIONS
    # =========================================================================

    charts_html = ""
    plotly_config = {"responsive": True, "displayModeBar": True}

    # Chart 1: Monthly Consensus Strength
    if monthly_consensus_strength:
        months = [m["month_name"] for m in monthly_consensus_strength]
        consensus_pcts = [m["consensus_pct"] for m in monthly_consensus_strength]
        best_strats = [f"#{m['best_strategy']}" for m in monthly_consensus_strength]
        colors = ["#2ecc71" if p >= 50 else "#f39c12" if p >= 25 else "#e74c3c" for p in consensus_pcts]

        fig_consensus = go.Figure()
        fig_consensus.add_trace(
            go.Bar(
                x=months,
                y=consensus_pcts,
                marker_color=colors,
                text=best_strats,
                textposition="outside",
                hovertemplate="%{x}<br>Consensus: %{y:.1f}%<br>Best: %{text}<extra></extra>",
            )
        )
        fig_consensus.add_hline(y=avg_consensus, line_dash="dash", line_color="white", annotation_text=f"Avg: {avg_consensus:.1f}%")
        fig_consensus.update_layout(
            title=dict(text="Monthly Consensus Strength (% of Models Agreeing)", font=dict(color="white", size=16)),
            xaxis=dict(title="Month", color="white", gridcolor="rgba(255,255,255,0.1)", tickangle=-45),
            yaxis=dict(title="Consensus %", color="white", gridcolor="rgba(255,255,255,0.1)", range=[0, 105]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
        )
        charts_html += f'<div class="chart-container">{pio.to_html(fig_consensus, full_html=False, include_plotlyjs="cdn", config=plotly_config)}</div>'

    # Chart 2: Yearly Profit Comparison by Method
    if method_yearly_profits:
        profit_df = pd.DataFrame([
            {"Method": m, "Yearly Profit": p}
            for m, p in sorted(method_yearly_profits.items(), key=lambda x: x[1], reverse=True)
        ])
        colors = ["#2ecc71" if p >= 0 else "#e74c3c" for p in profit_df["Yearly Profit"]]

        fig_yearly = go.Figure()
        fig_yearly.add_trace(
            go.Bar(
                x=profit_df["Method"],
                y=profit_df["Yearly Profit"],
                marker_color=colors,
                text=[f"{p:,.0f}" for p in profit_df["Yearly Profit"]],
                textposition="auto",
                hovertemplate="%{x}<br>Yearly Profit: %{y:,.2f}<extra></extra>",
            )
        )
        fig_yearly.add_hline(y=avg_yearly_profit, line_dash="dash", line_color="#f39c12",
                            annotation_text=f"Avg: {avg_yearly_profit:,.0f}")
        fig_yearly.update_layout(
            title=dict(text="Total Yearly Profit by Method (4-4-5 Calendar)", font=dict(color="white", size=16)),
            xaxis=dict(title="Method", color="white", gridcolor="rgba(255,255,255,0.1)", tickangle=-45),
            yaxis=dict(title="Yearly Profit", color="white", gridcolor="rgba(255,255,255,0.1)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
        )
        charts_html += f'<div class="chart-container">{pio.to_html(fig_yearly, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'

    # Chart 3: Monthly Heatmap (Method x Month)
    if method_monthly_data:
        methods = sorted(method_monthly_data.keys())
        all_strategies_in_data = set()
        for method_data in method_monthly_data.values():
            for month_info in method_data.values():
                all_strategies_in_data.add(month_info["strategy_no"])

        strategy_list = sorted(all_strategies_in_data)
        strategy_to_idx = {s: i for i, s in enumerate(strategy_list)}

        heatmap_data = []
        hover_texts = []
        for method in methods:
            row = []
            hover_row = []
            for month in range(1, 13):
                if month in method_monthly_data[method]:
                    strat = method_monthly_data[method][month]["strategy_no"]
                    profit = method_monthly_data[method][month]["profit"]
                    row.append(strategy_to_idx.get(strat, 0))
                    hover_row.append(f"Month {month}<br>Strategy: #{strat}<br>Profit: {profit:,.2f}")
                else:
                    row.append(-1)
                    hover_row.append(f"Month {month}<br>No data")
            heatmap_data.append(row)
            hover_texts.append(hover_row)

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=heatmap_data,
                x=month_names,
                y=methods,
                colorscale="Viridis",
                hovertext=hover_texts,
                hoverinfo="text",
                showscale=True,
                colorbar=dict(
                    title="Strategy Index",
                    tickvals=list(range(min(len(strategy_list), 20))),
                    ticktext=[f"#{s}" for s in strategy_list[:20]]
                ),
            )
        )
        fig_heatmap.update_layout(
            title=dict(text="Strategy Selection Heatmap (Method x Month)", font=dict(color="white", size=16)),
            xaxis=dict(title="Month", color="white", gridcolor="rgba(255,255,255,0.1)", tickangle=-45),
            yaxis=dict(title="Method", color="white", gridcolor="rgba(255,255,255,0.1)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=max(400, len(methods) * 25),
        )
        charts_html += f'<div class="chart-container">{pio.to_html(fig_heatmap, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'

    # Chart 4: Model Stability Ranking
    if method_stability:
        stability_df = pd.DataFrame([
            {"Method": m, "Switches": s, "Stability": 11 - s}  # 11 = max switches for 12 months
            for m, s in sorted(method_stability.items(), key=lambda x: x[1])
        ])
        fig_stability = px.bar(
            stability_df,
            x="Switches",
            y="Method",
            orientation="h",
            color="Switches",
            color_continuous_scale="RdYlGn_r",
            title="Model Stability Ranking (Fewer Switches = More Stable)",
        )
        fig_stability.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            yaxis={"categoryorder": "total descending"},
            title=dict(font=dict(color="white", size=16)),
            xaxis=dict(title="Number of Strategy Switches", color="white"),
            yaxis_title=dict(text="Method", font=dict(color="white")),
        )
        charts_html += f'<div class="chart-container">{pio.to_html(fig_stability, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'

    # Chart 5: Strategy Dominance Over Months
    if monthly_best_strategies:
        strategy_month_counts = {}
        for month in range(1, 13):
            for strat, count in monthly_consensus[month].items():
                if strat not in strategy_month_counts:
                    strategy_month_counts[strat] = [0] * 12
                strategy_month_counts[strat][month - 1] = count

        top_strategies = [s for s, _ in strategy_total_appearances.most_common(10)]
        fig_dominance = go.Figure()
        for strat in top_strategies:
            if strat in strategy_month_counts:
                fig_dominance.add_trace(
                    go.Scatter(
                        x=month_names,
                        y=strategy_month_counts[strat],
                        mode="lines",
                        name=f"Strategy {strat}",
                        stackgroup="one",
                        hovertemplate=f"Strategy {strat}<br>%{{x}}<br>Votes: %{{y}}<extra></extra>",
                    )
                )
        fig_dominance.update_layout(
            title=dict(text="Strategy Dominance Over Months (Top 10)", font=dict(color="white", size=16)),
            xaxis=dict(title="Month", color="white", gridcolor="rgba(255,255,255,0.1)", tickangle=-45),
            yaxis=dict(title="Number of Models", color="white", gridcolor="rgba(255,255,255,0.1)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            legend=dict(font=dict(color="white")),
        )
        charts_html += f'<div class="chart-container">{pio.to_html(fig_dominance, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'

    # Chart 6: Strategy Frequency (Top 20)
    top_20_strategies = strategy_total_appearances.most_common(20)
    if top_20_strategies:
        strat_df = pd.DataFrame(top_20_strategies, columns=["Strategy", "Appearances"])
        strat_df["Strategy"] = strat_df["Strategy"].apply(lambda x: f"#{x}")
        fig_freq = px.bar(
            strat_df,
            x="Strategy",
            y="Appearances",
            color="Appearances",
            color_continuous_scale="Viridis",
            title="Top 20 Most Selected Strategies (All Months, All Models)",
        )
        fig_freq.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title=dict(font=dict(color="white", size=16)),
            xaxis=dict(color="white"),
            yaxis=dict(color="white"),
        )
        charts_html += f'<div class="chart-container">{pio.to_html(fig_freq, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'

    # Chart 7: Method Agreement Matrix
    if len(method_monthly_data) > 1:
        methods = sorted(method_monthly_data.keys())
        agreement_matrix = []
        for m1 in methods:
            agree_row: List[Any] = []
            for m2 in methods:
                if m1 == m2:
                    agree_row.append(100)
                else:
                    agree_count = 0
                    total_count = 0
                    for month in range(1, 13):
                        if month in method_monthly_data[m1] and month in method_monthly_data[m2]:
                            total_count += 1
                            if method_monthly_data[m1][month]["strategy_no"] == method_monthly_data[m2][month]["strategy_no"]:
                                agree_count += 1
                    agreement = (agree_count / total_count * 100) if total_count > 0 else 0
                    agree_row.append(round(agreement, 1))
            agreement_matrix.append(agree_row)

        fig_agreement = px.imshow(
            agreement_matrix,
            x=methods,
            y=methods,
            color_continuous_scale="Greens",
            text_auto=True,
            title="Method Agreement Matrix (% Same Strategy Chosen)",
        )
        fig_agreement.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title=dict(font=dict(color="white", size=16)),
            height=max(400, len(methods) * 30),
        )
        charts_html += f'<div class="chart-container">{pio.to_html(fig_agreement, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'

    # =========================================================================
    # GENERATE MARKDOWN REPORT
    # =========================================================================

    md_output = [
        "# Main Data MR Comparison Report",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total MR Reports Analyzed**: {total_reports}",
        f"**Unique Methods**: {unique_methods}",
        f"**Unique Strategies Across All Data**: {unique_strategies}",
        "",
        "## Summary Statistics",
        f"- **Most Dominant Strategy**: #{most_dominant_strategy[0]} ({most_dominant_strategy[1]} appearances)",
        f"- **Average Monthly Consensus**: {avg_consensus:.1f}%",
        f"- **Most Stable Model**: {most_stable_method[0]} ({most_stable_method[1]} switches)",
        f"- **Best Yearly Profit**: {best_yearly_method[0]} ({best_yearly_method[1]:,.2f})",
        f"- **Average Yearly Profit**: {avg_yearly_profit:,.2f}",
        "",
        "## Monthly Consensus Data (4-4-5 Calendar)",
        "| Month | Best Strategy | Votes | Consensus % |",
        "|-------|---------------|-------|-------------|",
    ]

    for m in monthly_consensus_strength:
        md_output.append(f"| {m['month_name']} | #{m['best_strategy']} | {m['votes']} | {m['consensus_pct']:.1f}% |")

    md_output.extend(["", "## Model Stability Ranking", "| Method | Strategy Switches |", "|--------|-------------------|"])
    for method, switches in sorted(method_stability.items(), key=lambda x: x[1]):
        md_output.append(f"| {method} | {switches} |")

    md_output.extend(["", "## Yearly Profit by Method", "| Method | Yearly Profit |", "|--------|---------------|"])
    for method, profit in sorted(method_yearly_profits.items(), key=lambda x: x[1], reverse=True):
        md_output.append(f"| {method} | {profit:,.2f} |")

    md_output.extend(["", "## Processed MR Reports", "| File | Method | Months | Gen. Time |", "|------|--------|--------|-----------|"])
    for report in parsed_reports:
        md_output.append(f"| {report['file']} | {report['method']} | {len(report['monthly_data'])} | {report['generating_time']} |")

    md_path = os.path.join(root_folder, "main_data_mr_comparison.md")
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_output))
    except Exception as e:  # pylint: disable=broad-exception-caught
        return False, f"Failed to write Markdown: {e}", ""

    # =========================================================================
    # GENERATE HTML REPORT
    # =========================================================================

    html_content = _generate_main_data_mr_html(
        total_reports=total_reports,
        unique_methods=unique_methods,
        most_dominant_strategy=most_dominant_strategy,
        avg_consensus=avg_consensus,
        most_stable_method=most_stable_method,
        best_yearly_method=best_yearly_method,
        avg_yearly_profit=avg_yearly_profit,
        charts_html=charts_html,
        monthly_consensus_strength=monthly_consensus_strength,
        method_stability=method_stability,
        method_yearly_profits=method_yearly_profits,
        parsed_reports=parsed_reports,
    )

    html_path = os.path.join(root_folder, "main_data_mr_comparison.html")
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return False, f"Failed to write HTML: {e}", ""

    return True, f"Successfully generated:\n{md_path}\n{html_path}", html_path


def _generate_main_data_mr_html(
    total_reports,
    unique_methods,
    most_dominant_strategy,
    avg_consensus,
    most_stable_method,
    best_yearly_method,
    avg_yearly_profit,
    charts_html,
    monthly_consensus_strength,
    method_stability,
    method_yearly_profits,
    parsed_reports,
) -> str:
    """Generate the full HTML content for the Main Data MR comparison report."""

    # Build table rows
    monthly_table_rows = ""
    for m in monthly_consensus_strength:
        consensus_class = "high" if m["consensus_pct"] >= 50 else "medium" if m["consensus_pct"] >= 25 else "low"
        monthly_table_rows += f"""
            <tr>
                <td>{m["month_name"]}</td>
                <td class="strategy">#{m["best_strategy"]}</td>
                <td>{m["votes"]}</td>
                <td class="{consensus_class}">{m["consensus_pct"]:.1f}%</td>
            </tr>"""

    stability_table_rows = ""
    for method, switches in sorted(method_stability.items(), key=lambda x: x[1]):
        stability_class = "stable" if switches < 3 else "moderate" if switches < 6 else "unstable"
        stability_table_rows += f"""
            <tr>
                <td class="method">{html.escape(method)}</td>
                <td class="{stability_class}">{switches}</td>
            </tr>"""

    profit_table_rows = ""
    for method, profit in sorted(method_yearly_profits.items(), key=lambda x: x[1], reverse=True):
        profit_class = "positive" if profit >= 0 else "negative"
        profit_table_rows += f"""
            <tr>
                <td class="method">{html.escape(method)}</td>
                <td class="{profit_class}">{profit:,.2f}</td>
            </tr>"""

    reports_table_rows = ""
    for report in parsed_reports:
        reports_table_rows += f"""
            <tr>
                <td>{html.escape(report["file"])}</td>
                <td class="method">{html.escape(report["method"])}</td>
                <td>{len(report["monthly_data"])}</td>
                <td>{html.escape(report["generating_time"])}</td>
            </tr>"""

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Main Data MR Comparison</title>
        <style>
            * {{ box-sizing: border-box; }}
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); min-height: 100vh; color: #e4e4e4; }}
            h1 {{ color: #fff; font-size: 2.2em; margin-bottom: 5px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
            h2 {{ color: #9b59b6; font-size: 1.4em; margin: 0 0 15px 0; }}
            .container {{ max-width: 1600px; margin: 0 auto; }}
            .subtitle {{ color: #aaa; margin-bottom: 25px; }}
            .summary-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 25px; }}
            .stat-box {{ background: linear-gradient(135deg, #9b59b6, #8e44ad); color: white; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(155, 89, 182, 0.3); transition: transform 0.3s ease; }}
            .stat-box:hover {{ transform: translateY(-5px); }}
            .stat-box.highlight {{ background: linear-gradient(135deg, #e67e22, #d35400); }}
            .stat-box.success {{ background: linear-gradient(135deg, #27ae60, #1e8449); }}
            .stat-box h3 {{ margin: 0; font-size: 1.8em; font-weight: 700; }}
            .stat-box p {{ margin: 8px 0 0 0; opacity: 0.9; font-size: 0.85em; }}
            .card {{ background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); padding: 25px; margin-bottom: 20px; border-radius: 16px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.1); }}
            .card p {{ color: #aaa; margin: 0 0 15px 0; }}
            .chart-container {{ background: rgba(255, 255, 255, 0.03); padding: 20px; border-radius: 12px; margin-bottom: 20px; overflow-x: auto; }}
            .charts-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 20px; }}
            @media (max-width: 1200px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
            .table-wrapper {{ overflow-x: auto; border-radius: 12px; max-height: 500px; overflow-y: auto; }}
            table {{ width: 100%; border-collapse: collapse; background: rgba(30, 30, 50, 0.8); font-size: 0.9em; }}
            thead {{ position: sticky; top: 0; z-index: 10; }}
            th {{ background: linear-gradient(135deg, #9b59b6, #8e44ad); color: white; padding: 14px 12px; text-align: left; font-weight: 600; }}
            td {{ padding: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); color: #e4e4e4; }}
            tbody tr:hover {{ background: rgba(155, 89, 182, 0.15); }}
            td.strategy {{ color: #e67e22; font-weight: 600; }}
            td.method {{ color: #3498db; font-weight: 600; }}
            td.high {{ color: #2ecc71; font-weight: 600; }}
            td.medium {{ color: #f39c12; font-weight: 600; }}
            td.low {{ color: #e74c3c; font-weight: 600; }}
            td.stable {{ color: #2ecc71; }}
            td.moderate {{ color: #f39c12; }}
            td.unstable {{ color: #e74c3c; }}
            td.positive {{ color: #2ecc71; font-weight: 600; }}
            td.negative {{ color: #e74c3c; font-weight: 600; }}
            .info-box {{ background: rgba(155, 89, 182, 0.1); border-left: 4px solid #9b59b6; padding: 15px 20px; margin-bottom: 20px; border-radius: 0 8px 8px 0; }}
            .badge {{ background: linear-gradient(135deg, #9b59b6, #8e44ad); color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; }}
            ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
            ::-webkit-scrollbar-track {{ background: rgba(255,255,255,0.05); border-radius: 4px; }}
            ::-webkit-scrollbar-thumb {{ background: rgba(155,89,182,0.5); border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Main Data MR Comparison Report</h1>
            <p class="subtitle">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Monthly Forecast Analysis (4-4-5 Calendar) Across All Models</p>

            <div class="summary-stats">
                <div class="stat-box"><h3>{total_reports}</h3><p>MR Reports Analyzed</p></div>
                <div class="stat-box"><h3>{unique_methods}</h3><p>Unique Methods</p></div>
                <div class="stat-box highlight"><h3>#{most_dominant_strategy[0]}</h3><p>Most Dominant Strategy</p></div>
                <div class="stat-box success"><h3>{avg_consensus:.1f}%</h3><p>Avg Monthly Consensus</p></div>
                <div class="stat-box"><h3>{avg_yearly_profit:,.0f}</h3><p>Avg Yearly Profit</p></div>
            </div>

            <div class="card">
                <h2>Executive Summary</h2>
                <div class="info-box">
                    <p>This report aggregates monthly forecast data from <strong>{total_reports}</strong> MR (Monthly Results) reports using the <span class="badge">4-4-5 Calendar</span> method.</p>
                    <p>The most frequently selected strategy across all months and models is <strong>#{most_dominant_strategy[0]}</strong> with <strong>{most_dominant_strategy[1]}</strong> total appearances.</p>
                    <p>The most stable model (fewest strategy switches) is <strong>{most_stable_method[0]}</strong> with only <strong>{most_stable_method[1]}</strong> changes over 12 months.</p>
                    <p>The best performing model by yearly profit is <strong>{best_yearly_method[0]}</strong> with <strong>{best_yearly_method[1]:,.2f}</strong> total profit.</p>
                </div>
            </div>

            {charts_html}

            <div class="charts-grid">
                <div class="card">
                    <h2>Monthly Consensus Details</h2>
                    <p>Best strategy per month based on model agreement (4-4-5 calendar).</p>
                    <div class="table-wrapper">
                        <table><thead><tr><th>Month</th><th>Best Strategy</th><th>Votes</th><th>Consensus</th></tr></thead>
                        <tbody>{monthly_table_rows}</tbody></table>
                    </div>
                </div>
                <div class="card">
                    <h2>Model Stability Ranking</h2>
                    <p>Models sorted by number of strategy switches (fewer = more stable).</p>
                    <div class="table-wrapper">
                        <table><thead><tr><th>Method</th><th>Switches</th></tr></thead>
                        <tbody>{stability_table_rows}</tbody></table>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>Yearly Profit by Method</h2>
                <p>Total predicted yearly profit for each model.</p>
                <div class="table-wrapper">
                    <table><thead><tr><th>Method</th><th>Yearly Profit</th></tr></thead>
                    <tbody>{profit_table_rows}</tbody></table>
                </div>
            </div>

            <div class="card">
                <h2>Processed MR Reports</h2>
                <p>List of all analyzed Monthly Results reports.</p>
                <div class="table-wrapper">
                    <table><thead><tr><th>File</th><th>Method</th><th>Months</th><th>Gen. Time</th></tr></thead>
                    <tbody>{reports_table_rows}</tbody></table>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content
