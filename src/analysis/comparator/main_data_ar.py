"""
Main Data AR comparison functionality.
Compares All Results (AR_) reports that contain weekly forecast breakdowns.
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

from .base import scan_ar_reports


def parse_ar_report(file_path: str) -> Dict[str, Any]:
    """
    Parses an AR_ markdown report to extract method info and weekly forecast data.

    Args:
        file_path: Path to the AR_ prefixed markdown report

    Returns:
        Dictionary with file info, method, forecast_horizon, generating_time, weekly_data
    """
    data = {
        "file": os.path.basename(file_path),
        "path": file_path,
        "method": "Unknown",
        "forecast_horizon": 52,
        "generating_time": "N/A",
        "weekly_data": [],
    }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")

        # Extract Method
        method_match = re.search(r"\*\*Method\*\*:\s*(.+)", content)
        if method_match:
            data["method"] = method_match.group(1).strip()

        # Extract Forecast Horizon
        horizon_match = re.search(r"\*\*Forecast Horizon\*\*:\s*(\d+)", content)
        if horizon_match:
            data["forecast_horizon"] = int(horizon_match.group(1))

        # Extract Generating Time
        time_match = re.search(r"\*\*Generating time\*\*:\s*(.+)", content)
        if time_match:
            data["generating_time"] = time_match.group(1).strip()

        # Parse Weekly Forecast Breakdown table
        in_table = False
        for line in lines:
            # Detect table header - look for "Week" column header
            if "| Week |" in line and ("Best Strategy" in line or "Strategy" in line):
                in_table = True
                continue
            # Skip separator line
            if in_table and "|---" in line:
                continue

            if in_table and line.strip().startswith("|"):
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 3:
                    week_match = re.search(r"Week\s*(\d+)", parts[0])
                    # Handle both "No. XXXX" and just numbers in strategy column
                    strat_match = re.search(r"No\.\s*(\d+)", parts[1])
                    if not strat_match:
                        # Try plain number
                        strat_match = re.search(r"(\d+)", parts[1])
                    try:
                        profit = float(parts[2].replace(",", ""))
                    except ValueError:
                        profit = 0.0

                    if week_match and strat_match:
                        cast(List[Dict[str, Any]], data["weekly_data"]).append(
                            {"week": int(week_match.group(1)), "strategy_no": int(strat_match.group(1)), "profit": profit}
                        )
            elif in_table and not line.strip().startswith("|") and line.strip():
                break

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error parsing AR report {file_path}: {e}")

    return data


def generate_main_data_ar_report(root_folder: str) -> Tuple[bool, str, str]:
    """
    Generates the Main Data AR Comparison report from all AR_ prefixed files.
    Analyzes weekly forecast data across all models.

    Args:
        root_folder: Root folder containing the AR_ report files

    Returns:
        Tuple of (Success, Message, HTML Path)
    """
    # pylint: disable=import-outside-toplevel
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio

    # Scan for AR_ reports
    ar_files = scan_ar_reports(root_folder)
    if not ar_files:
        return False, "No AR_ prefixed .md files found.", ""

    # Parse all AR reports
    parsed_reports = []
    for report_file in ar_files:
        parsed = parse_ar_report(report_file)
        if parsed["weekly_data"]:
            parsed_reports.append(parsed)

    if not parsed_reports:
        return False, "Failed to parse any AR reports.", ""

    # Sort by method name
    parsed_reports.sort(key=lambda x: x["method"])

    # =========================================================================
    # DATA AGGREGATION
    # =========================================================================

    max_weeks = max(len(r["weekly_data"]) for r in parsed_reports)
    weekly_consensus: Dict[int, Counter] = {w: Counter() for w in range(1, max_weeks + 1)}
    method_weekly_data: Dict[str, Dict[int, Dict[str, Any]]] = {}
    strategy_total_appearances: Counter[int] = Counter()
    method_stability: Dict[str, int] = {}

    for report in parsed_reports:
        method = report["method"]
        method_weekly_data[method] = {}
        prev_strategy = None
        switch_count = 0

        for week_data in report["weekly_data"]:
            week = week_data["week"]
            strat = week_data["strategy_no"]
            profit = week_data["profit"]

            weekly_consensus[week][strat] += 1
            method_weekly_data[method][week] = {"strategy_no": strat, "profit": profit}
            strategy_total_appearances[strat] += 1

            if prev_strategy is not None and strat != prev_strategy:
                switch_count += 1
            prev_strategy = strat

        method_stability[method] = switch_count

    # =========================================================================
    # CALCULATE METRICS
    # =========================================================================

    total_reports = len(parsed_reports)
    total_weeks = max_weeks
    unique_methods = len({r["method"] for r in parsed_reports})
    unique_strategies = len(strategy_total_appearances)

    most_dominant_strategy = strategy_total_appearances.most_common(1)[0] if strategy_total_appearances else (0, 0)

    weekly_consensus_strength = []
    weekly_best_strategies = []
    for week in range(1, max_weeks + 1):
        if weekly_consensus[week]:
            best_strat, best_count = weekly_consensus[week].most_common(1)[0]
            consensus_pct = (best_count / total_reports) * 100
            weekly_consensus_strength.append(
                {"week": week, "best_strategy": best_strat, "votes": best_count, "consensus_pct": consensus_pct}
            )
            weekly_best_strategies.append(best_strat)

    avg_consensus = (
        sum(w["consensus_pct"] for w in weekly_consensus_strength) / len(weekly_consensus_strength) if weekly_consensus_strength else 0
    )
    most_stable_method = min(method_stability.items(), key=lambda x: x[1]) if method_stability else ("N/A", 0)

    # =========================================================================
    # GENERATE VISUALIZATIONS
    # =========================================================================

    charts_html = ""
    plotly_config = {"responsive": True, "displayModeBar": True}

    # Chart 1: Weekly Consensus Strength
    if weekly_consensus_strength:
        weeks = [w["week"] for w in weekly_consensus_strength]
        consensus_pcts = [w["consensus_pct"] for w in weekly_consensus_strength]
        best_strats = [f"#{w['best_strategy']}" for w in weekly_consensus_strength]
        colors = ["#2ecc71" if p >= 50 else "#f39c12" if p >= 25 else "#e74c3c" for p in consensus_pcts]

        fig_consensus = go.Figure()
        fig_consensus.add_trace(
            go.Bar(
                x=weeks,
                y=consensus_pcts,
                marker_color=colors,
                text=best_strats,
                textposition="outside",
                hovertemplate="Week %{x}<br>Consensus: %{y:.1f}%<br>Best: %{text}<extra></extra>",
            )
        )
        fig_consensus.add_hline(y=avg_consensus, line_dash="dash", line_color="white", annotation_text=f"Avg: {avg_consensus:.1f}%")
        fig_consensus.update_layout(
            title=dict(text="Weekly Consensus Strength (% of Models Agreeing)", font=dict(color="white", size=16)),
            xaxis=dict(title="Week", color="white", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Consensus %", color="white", gridcolor="rgba(255,255,255,0.1)", range=[0, 105]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
        )
        charts_html += f'<div class="chart-container">{pio.to_html(fig_consensus, full_html=False, include_plotlyjs="cdn", config=plotly_config)}</div>'

    # Chart 2: Master Heatmap
    if method_weekly_data:
        methods = sorted(method_weekly_data.keys())
        all_strategies_in_data = set()
        for method_data in method_weekly_data.values():
            for week_info in method_data.values():
                all_strategies_in_data.add(week_info["strategy_no"])

        strategy_list = sorted(all_strategies_in_data)
        strategy_to_idx = {s: i for i, s in enumerate(strategy_list)}

        heatmap_data = []
        hover_texts = []
        for method in methods:
            row = []
            hover_row = []
            for week in range(1, max_weeks + 1):
                if week in method_weekly_data[method]:
                    strat = method_weekly_data[method][week]["strategy_no"]
                    profit = method_weekly_data[method][week]["profit"]
                    row.append(strategy_to_idx.get(strat, 0))
                    hover_row.append(f"Week {week}<br>Strategy: {strat}<br>Profit: {profit:.2f}")
                else:
                    row.append(-1)
                    hover_row.append(f"Week {week}<br>No data")
            heatmap_data.append(row)
            hover_texts.append(hover_row)

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=heatmap_data,
                x=list(range(1, max_weeks + 1)),
                y=methods,
                colorscale="Viridis",
                hovertext=hover_texts,
                hoverinfo="text",
                showscale=True,
                colorbar=dict(
                    title="Strategy Index", tickvals=list(range(len(strategy_list))), ticktext=[f"#{s}" for s in strategy_list[:20]]
                ),
            )
        )
        fig_heatmap.update_layout(
            title=dict(text="Strategy Selection Heatmap (Method × Week)", font=dict(color="white", size=16)),
            xaxis=dict(title="Week", color="white", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Method", color="white", gridcolor="rgba(255,255,255,0.1)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=max(400, len(methods) * 25),
        )
        charts_html += (
            f'<div class="chart-container">{pio.to_html(fig_heatmap, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'
        )

    # Chart 3: Model Stability Ranking
    if method_stability:
        stability_df = pd.DataFrame(
            [{"Method": m, "Switches": s, "Stability": max_weeks - 1 - s} for m, s in sorted(method_stability.items(), key=lambda x: x[1])]
        )
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

    # Chart 4: Strategy Dominance Over Time
    if weekly_best_strategies:
        weeks = list(range(1, len(weekly_best_strategies) + 1))
        strategy_week_counts = {}
        for week in range(1, max_weeks + 1):
            for strat, count in weekly_consensus[week].items():
                if strat not in strategy_week_counts:
                    strategy_week_counts[strat] = [0] * max_weeks
                strategy_week_counts[strat][week - 1] = count

        top_strategies = [s for s, _ in strategy_total_appearances.most_common(10)]
        fig_dominance = go.Figure()
        for strat in top_strategies:
            if strat in strategy_week_counts:
                fig_dominance.add_trace(
                    go.Scatter(
                        x=weeks,
                        y=strategy_week_counts[strat],
                        mode="lines",
                        name=f"Strategy {strat}",
                        stackgroup="one",
                        hovertemplate=f"Strategy {strat}<br>Week %{{x}}<br>Votes: %{{y}}<extra></extra>",
                    )
                )
        fig_dominance.update_layout(
            title=dict(text="Strategy Dominance Over Time (Top 10)", font=dict(color="white", size=16)),
            xaxis=dict(title="Week", color="white", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Number of Models", color="white", gridcolor="rgba(255,255,255,0.1)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            legend=dict(font=dict(color="white")),
        )
        charts_html += f'<div class="chart-container">{pio.to_html(fig_dominance, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'

    # Chart 5: Strategy Frequency Sunburst
    if strategy_total_appearances:
        sunburst_data = []
        for report in parsed_reports:
            method = report["method"]
            strat_counts = Counter(w["strategy_no"] for w in report["weekly_data"])
            for strat, count in strat_counts.most_common(5):
                sunburst_data.append({"Method": method, "Strategy": f"#{strat}", "Count": count})

        if sunburst_data:
            sun_df = pd.DataFrame(sunburst_data)
            fig_sunburst = px.sunburst(
                sun_df, path=["Method", "Strategy"], values="Count", title="Strategy Distribution by Method (Sunburst)"
            )
            fig_sunburst.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                title=dict(font=dict(color="white", size=16)),
            )
            charts_html += f'<div class="chart-container">{pio.to_html(fig_sunburst, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'

    # Chart 6: Perfect Switching Comparison
    if method_weekly_data:
        perfect_switching_profit = []
        for week in range(1, max_weeks + 1):
            best_profit_this_week = float("-inf")
            for method_data in method_weekly_data.values():
                if week in method_data:
                    profit = method_data[week]["profit"]
                    if profit > best_profit_this_week:
                        best_profit_this_week = profit
            perfect_switching_profit.append(best_profit_this_week if best_profit_this_week > float("-inf") else 0)

        perfect_cumulative = []
        cumsum: float = 0.0
        for p in perfect_switching_profit:
            cumsum += p
            perfect_cumulative.append(cumsum)

        method_cumulatives: Dict[str, List[float]] = {}
        for method, week_data in method_weekly_data.items():
            run_cumsum: float = 0.0
            method_cumulatives[method] = []
            for week in range(1, max_weeks + 1):
                if week in week_data:
                    run_cumsum += week_data[week]["profit"]
                method_cumulatives[method].append(run_cumsum)

        avg_cumulative = []
        for week_idx in range(max_weeks):
            week_vals = [mc[week_idx] for mc in method_cumulatives.values() if len(mc) > week_idx]
            avg_cumulative.append(sum(week_vals) / len(week_vals) if week_vals else 0)

        weeks = list(range(1, max_weeks + 1))
        fig_perfect = go.Figure()
        fig_perfect.add_trace(
            go.Scatter(
                x=weeks,
                y=perfect_cumulative,
                mode="lines",
                name="Perfect Switching (Best each week)",
                line=dict(color="#2ecc71", width=3),
                fill="tozeroy",
                fillcolor="rgba(46, 204, 113, 0.1)",
            )
        )
        fig_perfect.add_trace(
            go.Scatter(
                x=weeks, y=avg_cumulative, mode="lines", name="Average Model Performance", line=dict(color="#f39c12", width=3, dash="dash")
            )
        )

        # Check all lists are non-empty to avoid IndexError
        if perfect_cumulative and avg_cumulative and weeks:
            advantage = perfect_cumulative[-1] - avg_cumulative[-1]
            adv_pct = (advantage / abs(avg_cumulative[-1]) * 100) if avg_cumulative[-1] != 0 else 0
            adv_color = "#2ecc71" if advantage >= 0 else "#e74c3c"
            adv_sign = "+" if advantage >= 0 else ""
            fig_perfect.add_annotation(
                x=weeks[-1],
                y=max(perfect_cumulative[-1], avg_cumulative[-1]) * 1.05,
                text=f"Advantage: {adv_sign}{advantage:.2f} ({adv_sign}{adv_pct:.1f}%)",
                showarrow=False,
                font=dict(size=14, color=adv_color, family="Arial Black"),
                bgcolor="rgba(0,0,0,0.5)",
                borderpad=4,
            )

        fig_perfect.update_layout(
            title=dict(text="Perfect Switching vs Average Model Performance", font=dict(color="white", size=16)),
            xaxis=dict(title="Week", color="white", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Cumulative Profit", color="white", gridcolor="rgba(255,255,255,0.1)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            legend=dict(font=dict(color="white"), x=0.02, y=0.98),
        )
        charts_html += (
            f'<div class="chart-container">{pio.to_html(fig_perfect, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'
        )

    # Chart 7: Strategy Appearance Frequency
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
            title="Top 20 Most Selected Strategies (All Weeks, All Models)",
        )
        fig_freq.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title=dict(font=dict(color="white", size=16)),
            xaxis=dict(color="white"),
            yaxis=dict(color="white"),
        )
        charts_html += (
            f'<div class="chart-container">{pio.to_html(fig_freq, full_html=False, include_plotlyjs=False, config=plotly_config)}</div>'
        )

    # Chart 8: Method Agreement Matrix
    if len(method_weekly_data) > 1:
        methods = sorted(method_weekly_data.keys())
        agreement_matrix = []
        for m1 in methods:
            agree_row: List[Any] = []
            for m2 in methods:
                if m1 == m2:
                    agree_row.append(100)
                else:
                    agree_count = 0
                    total_count = 0
                    for week in range(1, max_weeks + 1):
                        if week in method_weekly_data[m1] and week in method_weekly_data[m2]:
                            total_count += 1
                            if method_weekly_data[m1][week]["strategy_no"] == method_weekly_data[m2][week]["strategy_no"]:
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
        "# Main Data AR Comparison Report",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total AR Reports Analyzed**: {total_reports}",
        f"**Unique Methods**: {unique_methods}",
        f"**Forecast Weeks**: {total_weeks}",
        f"**Unique Strategies Across All Data**: {unique_strategies}",
        "",
        "## Summary Statistics",
        f"- **Most Dominant Strategy**: #{most_dominant_strategy[0]} ({most_dominant_strategy[1]} appearances)",
        f"- **Average Weekly Consensus**: {avg_consensus:.1f}%",
        f"- **Most Stable Model**: {most_stable_method[0]} ({most_stable_method[1]} switches)",
        "",
        "## Weekly Consensus Data",
        "| Week | Best Strategy | Votes | Consensus % |",
        "|------|---------------|-------|-------------|",
    ]

    for w in weekly_consensus_strength:
        md_output.append(f"| {w['week']} | #{w['best_strategy']} | {w['votes']} | {w['consensus_pct']:.1f}% |")

    md_output.extend(["", "## Model Stability Ranking", "| Method | Strategy Switches |", "|--------|-------------------|"])
    for method, switches in sorted(method_stability.items(), key=lambda x: x[1]):
        md_output.append(f"| {method} | {switches} |")

    md_output.extend(["", "## Processed AR Reports", "| File | Method | Weeks | Gen. Time |", "|------|--------|-------|-----------|"])
    for report in parsed_reports:
        md_output.append(f"| {report['file']} | {report['method']} | {len(report['weekly_data'])} | {report['generating_time']} |")

    md_path = os.path.join(root_folder, "main_data_ar_comparison.md")
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_output))
    except Exception as e:  # pylint: disable=broad-exception-caught
        return False, f"Failed to write Markdown: {e}", ""

    # =========================================================================
    # GENERATE HTML REPORT
    # =========================================================================

    html_content = _generate_main_data_ar_html(
        total_reports=total_reports,
        unique_methods=unique_methods,
        total_weeks=total_weeks,
        most_dominant_strategy=most_dominant_strategy,
        avg_consensus=avg_consensus,
        most_stable_method=most_stable_method,
        charts_html=charts_html,
        weekly_consensus_strength=weekly_consensus_strength,
        method_stability=method_stability,
        parsed_reports=parsed_reports,
    )

    html_path = os.path.join(root_folder, "main_data_ar_comparison.html")
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return False, f"Failed to write HTML: {e}", ""

    return True, f"Successfully generated:\n{md_path}\n{html_path}", html_path


def _generate_main_data_ar_html(
    total_reports,
    unique_methods,
    total_weeks,
    most_dominant_strategy,
    avg_consensus,
    most_stable_method,
    charts_html,
    weekly_consensus_strength,
    method_stability,
    parsed_reports,
) -> str:
    """Generate the full HTML content for the Main Data AR comparison report."""

    # Build table rows
    weekly_table_rows = ""
    for w in weekly_consensus_strength:
        consensus_class = "high" if w["consensus_pct"] >= 50 else "medium" if w["consensus_pct"] >= 25 else "low"
        weekly_table_rows += f"""
            <tr>
                <td>Week {w["week"]}</td>
                <td class="strategy">#{w["best_strategy"]}</td>
                <td>{w["votes"]}</td>
                <td class="{consensus_class}">{w["consensus_pct"]:.1f}%</td>
            </tr>"""

    stability_table_rows = ""
    for method, switches in sorted(method_stability.items(), key=lambda x: x[1]):
        stability_class = "stable" if switches < 10 else "moderate" if switches < 25 else "unstable"
        stability_table_rows += f"""
            <tr>
                <td class="method">{html.escape(method)}</td>
                <td class="{stability_class}">{switches}</td>
            </tr>"""

    reports_table_rows = ""
    for report in parsed_reports:
        reports_table_rows += f"""
            <tr>
                <td>{html.escape(report["file"])}</td>
                <td class="method">{html.escape(report["method"])}</td>
                <td>{len(report["weekly_data"])}</td>
                <td>{html.escape(report["generating_time"])}</td>
            </tr>"""

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Main Data AR Comparison</title>
        <style>
            * {{ box-sizing: border-box; }}
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 50%, #415a77 100%); min-height: 100vh; color: #e4e4e4; }}
            h1 {{ color: #fff; font-size: 2.2em; margin-bottom: 5px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
            h2 {{ color: #778da9; font-size: 1.4em; margin: 0 0 15px 0; }}
            .container {{ max-width: 1600px; margin: 0 auto; }}
            .subtitle {{ color: #aaa; margin-bottom: 25px; }}
            .summary-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 25px; }}
            .stat-box {{ background: linear-gradient(135deg, #778da9, #415a77); color: white; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(119, 141, 169, 0.3); transition: transform 0.3s ease; }}
            .stat-box:hover {{ transform: translateY(-5px); }}
            .stat-box.highlight {{ background: linear-gradient(135deg, #e67e22, #d35400); }}
            .stat-box.success {{ background: linear-gradient(135deg, #27ae60, #1e8449); }}
            .stat-box h3 {{ margin: 0; font-size: 2em; font-weight: 700; }}
            .stat-box p {{ margin: 8px 0 0 0; opacity: 0.9; font-size: 0.85em; }}
            .card {{ background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); padding: 25px; margin-bottom: 20px; border-radius: 16px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.1); }}
            .card p {{ color: #aaa; margin: 0 0 15px 0; }}
            .chart-container {{ background: rgba(255, 255, 255, 0.03); padding: 20px; border-radius: 12px; margin-bottom: 20px; overflow-x: auto; }}
            .charts-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 20px; }}
            @media (max-width: 1200px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
            .table-wrapper {{ overflow-x: auto; border-radius: 12px; max-height: 500px; overflow-y: auto; }}
            table {{ width: 100%; border-collapse: collapse; background: rgba(30, 30, 50, 0.8); font-size: 0.9em; }}
            thead {{ position: sticky; top: 0; z-index: 10; }}
            th {{ background: linear-gradient(135deg, #778da9, #415a77); color: white; padding: 14px 12px; text-align: left; font-weight: 600; }}
            td {{ padding: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); color: #e4e4e4; }}
            tbody tr:hover {{ background: rgba(119, 141, 169, 0.15); }}
            td.strategy {{ color: #e67e22; font-weight: 600; }}
            td.method {{ color: #3498db; font-weight: 600; }}
            td.high {{ color: #2ecc71; font-weight: 600; }}
            td.medium {{ color: #f39c12; font-weight: 600; }}
            td.low {{ color: #e74c3c; font-weight: 600; }}
            td.stable {{ color: #2ecc71; }}
            td.moderate {{ color: #f39c12; }}
            td.unstable {{ color: #e74c3c; }}
            .info-box {{ background: rgba(119, 141, 169, 0.1); border-left: 4px solid #778da9; padding: 15px 20px; margin-bottom: 20px; border-radius: 0 8px 8px 0; }}
            ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
            ::-webkit-scrollbar-track {{ background: rgba(255,255,255,0.05); border-radius: 4px; }}
            ::-webkit-scrollbar-thumb {{ background: rgba(119,141,169,0.5); border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Main Data AR Comparison Report</h1>
            <p class="subtitle">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Weekly Forecast Analysis Across All Models</p>

            <div class="summary-stats">
                <div class="stat-box"><h3>{total_reports}</h3><p>AR Reports Analyzed</p></div>
                <div class="stat-box"><h3>{unique_methods}</h3><p>Unique Methods</p></div>
                <div class="stat-box"><h3>{total_weeks}</h3><p>Forecast Weeks</p></div>
                <div class="stat-box highlight"><h3>#{most_dominant_strategy[0]}</h3><p>Most Dominant Strategy</p></div>
                <div class="stat-box success"><h3>{avg_consensus:.1f}%</h3><p>Avg Weekly Consensus</p></div>
            </div>

            <div class="card">
                <h2>Executive Summary</h2>
                <div class="info-box">
                    <p>This report aggregates weekly forecast data from <strong>{total_reports}</strong> AR (All Results) reports.</p>
                    <p>The most frequently selected strategy across all weeks and models is <strong>#{most_dominant_strategy[0]}</strong> with <strong>{most_dominant_strategy[1]}</strong> total appearances.</p>
                    <p>The most stable model (fewest strategy switches) is <strong>{most_stable_method[0]}</strong> with only <strong>{most_stable_method[1]}</strong> changes over {total_weeks} weeks.</p>
                </div>
            </div>

            {charts_html}

            <div class="charts-grid">
                <div class="card">
                    <h2>Weekly Consensus Details</h2>
                    <p>Best strategy per week based on model agreement.</p>
                    <div class="table-wrapper">
                        <table><thead><tr><th>Week</th><th>Best Strategy</th><th>Votes</th><th>Consensus</th></tr></thead>
                        <tbody>{weekly_table_rows}</tbody></table>
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
                <h2>Processed AR Reports</h2>
                <p>List of all analyzed All Results reports.</p>
                <div class="table-wrapper">
                    <table><thead><tr><th>File</th><th>Method</th><th>Weeks</th><th>Gen. Time</th></tr></thead>
                    <tbody>{reports_table_rows}</tbody></table>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content
