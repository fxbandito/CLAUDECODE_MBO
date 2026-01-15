"""
Main Data comparison functionality.
Compares strategy reports to generate an overview of all analyzed reports.
"""

# pylint: disable=line-too-long
# Note: This file contains embedded HTML/CSS templates where line breaks
# would significantly reduce readability. The long lines are intentional.
import html
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Tuple

import pandas as pd

from .base import parse_time_to_seconds, scan_reports


def parse_main_data_report(file_path: str) -> Dict[str, Any]:
    """
    Parses a markdown report to extract specific fields for the Main Data Comparison.
    Fields: File Name, Method, Ranking Mode, Generating time, Model Settings, Best Strategy No.

    Args:
        file_path: Path to the markdown report

    Returns:
        Dictionary with parsed report data
    """
    data = {
        "Filename": os.path.splitext(os.path.basename(file_path))[0],
        "Method": "Unknown",
        "Ranking Mode": "Unknown",
        "Generating Time": "Unknown",
        "Model Settings": "",
        "Best Strategy": "Unknown",
    }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 1. Regex Extraction for simple fields
        content = "".join(lines)

        method_match = re.search(r"\*\*Method\*\*: (.+)", content)
        if method_match:
            data["Method"] = method_match.group(1).strip()

        ranking_match = re.search(r"\*\*Ranking Mode\*\*: (.+)", content)
        if ranking_match:
            data["Ranking Mode"] = ranking_match.group(1).strip()

        time_match = re.search(r"\*\*Generating time\*\*: (.+)", content)
        if time_match:
            data["Generating Time"] = time_match.group(1).strip()

        # 2. Extract Best Strategy
        strat_match = re.search(r"The top performing strategy for the next month is \*\*(Strategy No\. \d+)\*\*", content)
        if strat_match:
            data["Best Strategy"] = strat_match.group(1).strip()

        # 3. Extract Model Settings
        settings_parts = []
        in_settings = False
        for line in lines:
            if "## Model Settings" in line:
                in_settings = True
                continue

            if in_settings:
                if line.strip().startswith("#"):
                    break

                if line.strip().startswith("|") and "Parameter" not in line and "---" not in line:
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if len(parts) >= 2:
                        key = parts[0].replace("**", "").strip()
                        val = parts[1].strip()
                        settings_parts.append(f"{key}: {val}")

        if settings_parts:
            data["Model Settings"] = ", ".join(settings_parts)

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error parsing {file_path} for Main Data: {e}")

    return data


def generate_main_data_report(root_folder: str) -> Tuple[bool, str, str]:
    """
    Generates the Main Data Comparison report (HTML and Markdown).

    Args:
        root_folder: Root folder containing the report files

    Returns:
        Tuple of (Success, Message, HTML Path)
    """
    # pylint: disable=import-outside-toplevel
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio

    report_files = scan_reports(root_folder)
    if not report_files:
        return False, "No .md files found.", ""

    parsed_data = []
    for report_file in report_files:
        parsed_data.append(parse_main_data_report(report_file))

    if not parsed_data:
        return False, "Failed to parse any reports.", ""

    # Sort data by Filename for consistency
    parsed_data.sort(key=lambda x: x["Filename"])

    # --- Generate Visualizations ---

    # Chart 1: Best Strategy Frequency (Bar Chart)
    strategy_counts = Counter([d["Best Strategy"] for d in parsed_data if d["Best Strategy"] != "Unknown"])
    strat_df = pd.DataFrame(strategy_counts.most_common(20), columns=["Strategy", "Count"])

    fig_strategy = px.bar(
        strat_df,
        x="Strategy",
        y="Count",
        title="Top 20 Best Strategy Frequency (Most Selected Strategies)",
        color="Count",
        color_continuous_scale="Viridis",
    )
    fig_strategy.update_layout(template="plotly_white", xaxis_tickangle=-45)
    chart_strategy_html = pio.to_html(fig_strategy, full_html=False)

    # Chart 2: Generating Time Distribution (Box Plot by Method)
    time_data = []
    for d in parsed_data:
        seconds = parse_time_to_seconds(d["Generating Time"])
        if seconds > 0:
            time_data.append({"Method": d["Method"], "Time (seconds)": seconds, "Time String": d["Generating Time"]})

    if time_data:
        time_df = pd.DataFrame(time_data)
        fig_time = px.box(
            time_df, x="Method", y="Time (seconds)", title="Generating Time Distribution by Method", color="Method", points="all"
        )
        fig_time.update_layout(template="plotly_white", xaxis_tickangle=-45, showlegend=False)
        chart_time_html = pio.to_html(fig_time, full_html=False)
    else:
        chart_time_html = "<p>No time data available.</p>"

    # Chart 3: Strategy Heatmap (Method vs Strategy)
    all_methods = sorted({d["Method"] for d in parsed_data})
    top_strategies = [s for s, _ in strategy_counts.most_common(15)]

    method_strategy_matrix: Dict[str, Dict[str, int]] = {}
    for method in all_methods:
        method_strategy_matrix[method] = {}
        for strat in top_strategies:
            method_strategy_matrix[method][strat] = 0

    for d in parsed_data:
        if d["Method"] in method_strategy_matrix and d["Best Strategy"] in top_strategies:
            method_strategy_matrix[d["Method"]][d["Best Strategy"]] += 1

    heatmap_data = []
    for method in all_methods:
        row = [method_strategy_matrix[method].get(s, 0) for s in top_strategies]
        heatmap_data.append(row)

    if heatmap_data and top_strategies:
        fig_heatmap = px.imshow(
            heatmap_data,
            x=top_strategies,
            y=all_methods,
            title="Strategy Selection Heatmap (Method vs Top 15 Strategies)",
            color_continuous_scale="YlOrRd",
            text_auto=True,
            aspect="auto",
        )
        fig_heatmap.update_layout(template="plotly_white")
        chart_heatmap_html = pio.to_html(fig_heatmap, full_html=False)
    else:
        chart_heatmap_html = "<p>No heatmap data available.</p>"

    # Chart 4: Model Settings Impact
    settings_impact_data = []
    for d in parsed_data:
        if d["Model Settings"] and d["Best Strategy"] != "Unknown":
            strat_num = d["Best Strategy"].replace("Strategy No. ", "")
            settings_impact_data.append(
                {
                    "Method": d["Method"],
                    "Settings": d["Model Settings"][:50] + "..." if len(d["Model Settings"]) > 50 else d["Model Settings"],
                    "Strategy": strat_num,
                    "Full Settings": d["Model Settings"],
                }
            )

    if settings_impact_data:
        settings_df = pd.DataFrame(settings_impact_data)
        method_counts = settings_df["Method"].value_counts()
        methods_with_variations = method_counts[method_counts > 1].head(8).index.tolist()

        filtered_df = settings_df[settings_df["Method"].isin(methods_with_variations)]

        if not filtered_df.empty:
            fig_settings = px.histogram(
                filtered_df,
                x="Method",
                color="Strategy",
                title="Strategy Selection by Method (Color = Different Strategies)",
                barmode="stack",
            )
            fig_settings.update_layout(template="plotly_white", xaxis_tickangle=-45)
            chart_settings_html = pio.to_html(fig_settings, full_html=False)
        else:
            chart_settings_html = "<p>Not enough variation in settings.</p>"
    else:
        chart_settings_html = "<p>No settings impact data available.</p>"

    # Chart 5: Execution Time vs Model Complexity
    complexity_data = []
    for d in parsed_data:
        seconds = parse_time_to_seconds(d["Generating Time"])
        if seconds > 0:
            settings = d["Model Settings"]
            param_count = settings.count(":") if settings else 0
            complexity_data.append(
                {
                    "Method": d["Method"],
                    "Time (seconds)": seconds,
                    "Parameter Count": param_count,
                    "Settings": settings[:80] + "..." if len(settings) > 80 else settings,
                }
            )

    if complexity_data:
        complexity_df = pd.DataFrame(complexity_data)
        fig_complexity = px.scatter(
            complexity_df,
            x="Parameter Count",
            y="Time (seconds)",
            color="Method",
            title="Execution Time vs Model Complexity (Parameter Count)",
            hover_data=["Settings"],
            size="Time (seconds)",
            size_max=20,
        )
        fig_complexity.update_layout(template="plotly_white")
        chart_complexity_html = pio.to_html(fig_complexity, full_html=False)
    else:
        chart_complexity_html = "<p>No complexity data available.</p>"

    # Chart 6: Strategy Consensus Score (Gauge)
    total_reports = len([d for d in parsed_data if d["Best Strategy"] != "Unknown"])
    most_common_strategy = "N/A"
    if total_reports > 0 and strategy_counts:
        most_common_strategy, most_common_count = strategy_counts.most_common(1)[0]
        consensus_percentage = (most_common_count / total_reports) * 100

        top3_count = sum([c for _, c in strategy_counts.most_common(3)])
        top3_percentage = (top3_count / total_reports) * 100

        fig_gauge = go.Figure()

        fig_gauge.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=consensus_percentage,
                title={"text": f"Top Strategy Consensus<br><span style='font-size:0.7em;color:gray'>{most_common_strategy}</span>"},
                delta={"reference": 50, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar": {"color": "#e67e22"},
                    "steps": [
                        {"range": [0, 25], "color": "#ffcccc"},
                        {"range": [25, 50], "color": "#ffffcc"},
                        {"range": [50, 75], "color": "#ccffcc"},
                        {"range": [75, 100], "color": "#99ff99"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50},
                },
                domain={"x": [0, 0.45], "y": [0, 1]},
            )
        )

        fig_gauge.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=top3_percentage,
                title={"text": "Top 3 Strategies Combined"},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar": {"color": "#3498db"},
                    "steps": [
                        {"range": [0, 33], "color": "#e8f4f8"},
                        {"range": [33, 66], "color": "#b8dce8"},
                        {"range": [66, 100], "color": "#88c4d8"},
                    ],
                },
                domain={"x": [0.55, 1], "y": [0, 1]},
            )
        )

        fig_gauge.update_layout(title="Strategy Consensus Score", template="plotly_white", height=350)
        chart_gauge_html = pio.to_html(fig_gauge, full_html=False)
    else:
        chart_gauge_html = "<p>No consensus data available.</p>"

    # --- Generate Markdown ---
    md_output = [
        "# Main Data Comparison Report",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Reports**: {len(parsed_data)}",
        "",
        "| Filename | Method | Ranking Mode | Generating Time | Model Settings | Best Strategy No |",
        "|---|---|---|---|---|---|",
    ]

    for d in parsed_data:
        settings = d["Model Settings"].replace("|", "/")
        md_output.append(
            f"| {d['Filename']} | {d['Method']} | {d['Ranking Mode']} | {d['Generating Time']} | {settings} | {d['Best Strategy']} |"
        )

    md_path = os.path.join(root_folder, "main_data_comparison.md")
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_output))
    except Exception as e:  # pylint: disable=broad-exception-caught
        return False, f"Failed to write Markdown: {e}", ""

    # --- Generate HTML ---
    html_content = _generate_main_data_html(
        parsed_data=parsed_data,
        strategy_counts=strategy_counts,
        most_common_strategy=most_common_strategy,
        chart_gauge_html=chart_gauge_html,
        chart_strategy_html=chart_strategy_html,
        chart_time_html=chart_time_html,
        chart_heatmap_html=chart_heatmap_html,
        chart_settings_html=chart_settings_html,
        chart_complexity_html=chart_complexity_html,
    )

    html_path = os.path.join(root_folder, "main_data_comparison.html")
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return False, f"Failed to write HTML: {e}", ""

    return True, f"{md_path}\n{html_path}", html_path


def _generate_main_data_html(
    parsed_data,
    strategy_counts,
    most_common_strategy,
    chart_gauge_html,
    chart_strategy_html,
    chart_time_html,
    chart_heatmap_html,
    chart_settings_html,
    chart_complexity_html,
) -> str:
    """Generate the full HTML content for the Main Data comparison report."""

    # Build table rows
    table_rows = ""
    for d in parsed_data:
        table_rows += f"""
            <tr>
                <td title="{html.escape(d["Filename"])}">{html.escape(d["Filename"])}</td>
                <td>{html.escape(d["Method"])}</td>
                <td>{html.escape(d["Ranking Mode"])}</td>
                <td>{html.escape(d["Generating Time"])}</td>
                <td title="{html.escape(d["Model Settings"])}">{html.escape(d["Model Settings"])}</td>
                <td>{html.escape(d["Best Strategy"])}</td>
            </tr>"""

    file_count_text = f"({len(parsed_data)} analyzed files)"
    top_strategy_display = most_common_strategy.replace("Strategy No. ", "#") if strategy_counts else "N/A"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Main Data Comparison</title>
        <style>
            * {{ box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0; padding: 20px;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                min-height: 100vh; color: #e4e4e4;
            }}
            h1 {{ color: #fff; font-size: 2.2em; margin-bottom: 5px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
            h2 {{ color: #e67e22; font-size: 1.4em; margin: 0 0 10px 0; }}
            .container {{ max-width: 1600px; margin: 0 auto; }}
            .subtitle {{ color: #aaa; margin-bottom: 25px; }}
            .summary-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 25px; }}
            .stat-box {{ background: linear-gradient(135deg, #e67e22, #d35400); color: white; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(230, 126, 34, 0.3); transition: transform 0.3s ease; }}
            .stat-box:hover {{ transform: translateY(-5px); }}
            .stat-box h3 {{ margin: 0; font-size: 2.2em; font-weight: 700; }}
            .stat-box p {{ margin: 8px 0 0 0; opacity: 0.9; font-size: 0.9em; color: white; }}
            .card {{ background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); padding: 25px; margin-bottom: 20px; border-radius: 16px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.1); overflow: hidden; }}
            .card p {{ color: #aaa; margin: 0 0 15px 0; font-size: 0.95em; }}
            .chart-container {{ width: 100%; overflow-x: auto; overflow-y: hidden; }}
            .chart-container > div {{ min-width: 100%; }}
            .charts-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 20px; }}
            @media (max-width: 1200px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
            .table-wrapper {{ overflow-x: auto; border-radius: 12px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2); }}
            table {{ width: 100%; border-collapse: collapse; background: rgba(30, 30, 50, 0.8); font-size: 0.9em; }}
            thead {{ position: sticky; top: 0; z-index: 10; }}
            th {{ background: linear-gradient(135deg, #e67e22, #d35400); color: white; padding: 16px 12px; text-align: left; font-weight: 600; cursor: pointer; user-select: none; white-space: nowrap; transition: background 0.3s ease; border: none; }}
            th:hover {{ background: linear-gradient(135deg, #f39c12, #e67e22); }}
            th:first-child {{ border-radius: 12px 0 0 0; }}
            th:last-child {{ border-radius: 0 12px 0 0; }}
            td {{ padding: 14px 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); color: #e4e4e4; vertical-align: middle; }}
            tbody tr:nth-child(even) {{ background: rgba(255, 255, 255, 0.02); }}
            tbody tr {{ transition: background 0.2s ease; }}
            tbody tr:hover {{ background: rgba(230, 126, 34, 0.15); }}
            td:nth-child(1) {{ font-weight: 500; color: #fff; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
            td:nth-child(2) {{ color: #3498db; font-weight: 600; }}
            td:nth-child(3) {{ color: #9b59b6; }}
            td:nth-child(4) {{ font-family: 'Consolas', 'Monaco', monospace; color: #2ecc71; }}
            td:nth-child(5) {{ font-size: 0.85em; color: #bbb; max-width: 350px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
            td:nth-child(6) {{ color: #e67e22; font-weight: 600; }}
            .search-container {{ margin-bottom: 20px; }}
            .search-input {{ width: 100%; max-width: 400px; padding: 14px 20px; font-size: 1em; border: 2px solid rgba(255, 255, 255, 0.1); border-radius: 10px; background: rgba(255, 255, 255, 0.05); color: #fff; transition: all 0.3s ease; }}
            .search-input:focus {{ outline: none; border-color: #e67e22; background: rgba(255, 255, 255, 0.08); box-shadow: 0 0 20px rgba(230, 126, 34, 0.2); }}
            .search-input::placeholder {{ color: #777; }}
            .row-count {{ display: inline-block; background: rgba(230, 126, 34, 0.2); color: #e67e22; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; margin-left: 10px; }}
            ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
            ::-webkit-scrollbar-track {{ background: rgba(255, 255, 255, 0.05); border-radius: 4px; }}
            ::-webkit-scrollbar-thumb {{ background: rgba(230, 126, 34, 0.5); border-radius: 4px; }}
            ::-webkit-scrollbar-thumb:hover {{ background: rgba(230, 126, 34, 0.7); }}
        </style>
        <script>
            function sortTable(n) {{
                var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
                table = document.getElementById("dataTable");
                switching = true;
                dir = "asc";
                var headers = table.getElementsByTagName("th");
                for (var h = 0; h < headers.length; h++) {{
                    headers[h].classList.remove('sort-asc', 'sort-desc');
                }}
                while (switching) {{
                    switching = false;
                    rows = table.rows;
                    for (i = 1; i < (rows.length - 1); i++) {{
                        shouldSwitch = false;
                        x = rows[i].getElementsByTagName("TD")[n];
                        y = rows[i + 1].getElementsByTagName("TD")[n];
                        var xVal = x.innerHTML.toLowerCase();
                        var yVal = y.innerHTML.toLowerCase();
                        if (n === 3 || n === 5) {{
                            var xNum = parseFloat(xVal.replace(/[^0-9.]/g, '')) || 0;
                            var yNum = parseFloat(yVal.replace(/[^0-9.]/g, '')) || 0;
                            if (!isNaN(xNum) && !isNaN(yNum)) {{
                                xVal = xNum;
                                yVal = yNum;
                            }}
                        }}
                        if (dir == "asc") {{
                            if (xVal > yVal) {{ shouldSwitch = true; break; }}
                        }} else if (dir == "desc") {{
                            if (xVal < yVal) {{ shouldSwitch = true; break; }}
                        }}
                    }}
                    if (shouldSwitch) {{
                        rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                        switching = true;
                        switchcount++;
                    }} else {{
                        if (switchcount == 0 && dir == "asc") {{
                            dir = "desc";
                            switching = true;
                        }}
                    }}
                }}
                headers[n].classList.add(dir === 'asc' ? 'sort-asc' : 'sort-desc');
            }}
            function filterTable() {{
                var input, filter, table, tr, td, i, j, txtValue, visibleCount = 0;
                input = document.getElementById("searchInput");
                filter = input.value.toUpperCase();
                table = document.getElementById("dataTable");
                tr = table.getElementsByTagName("tr");
                for (i = 1; i < tr.length; i++) {{
                    tr[i].style.display = "none";
                    td = tr[i].getElementsByTagName("td");
                    for (j = 0; j < td.length; j++) {{
                        if (td[j]) {{
                            txtValue = td[j].textContent || td[j].innerText;
                            if (txtValue.toUpperCase().indexOf(filter) > -1) {{
                                tr[i].style.display = "";
                                visibleCount++;
                                break;
                            }}
                        }}
                    }}
                }}
                var countEl = document.getElementById("visibleCount");
                if (countEl) {{ countEl.textContent = visibleCount + " rows"; }}
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Main Data Comparison</h1>
            <p class="subtitle">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {file_count_text}</p>

            <div class="summary-stats">
                <div class="stat-box"><h3>{len(parsed_data)}</h3><p>Total Reports</p></div>
                <div class="stat-box"><h3>{len({d["Method"] for d in parsed_data})}</h3><p>Unique Methods</p></div>
                <div class="stat-box"><h3>{len(strategy_counts)}</h3><p>Unique Strategies</p></div>
                <div class="stat-box"><h3>{top_strategy_display}</h3><p>Top Strategy</p></div>
            </div>

            <div class="card">
                <h2>Strategy Consensus Score</h2>
                <p>How strongly do the models agree on the best strategy?</p>
                <div class="chart-container">{chart_gauge_html}</div>
            </div>

            <div class="charts-grid">
                <div class="card"><h2>Best Strategy Frequency</h2><p>Which strategies are most frequently selected as the best?</p><div class="chart-container">{chart_strategy_html}</div></div>
                <div class="card"><h2>Generating Time Distribution</h2><p>How long does each method take to generate results?</p><div class="chart-container">{chart_time_html}</div></div>
            </div>

            <div class="card"><h2>Strategy Selection Heatmap</h2><p>Which methods select which strategies? Darker = More selections.</p><div class="chart-container">{chart_heatmap_html}</div></div>

            <div class="charts-grid">
                <div class="card"><h2>Model Settings Impact</h2><p>How do different settings affect strategy selection within each method?</p><div class="chart-container">{chart_settings_html}</div></div>
                <div class="card"><h2>Execution Time vs Complexity</h2><p>Does model complexity (parameter count) correlate with execution time?</p><div class="chart-container">{chart_complexity_html}</div></div>
            </div>

            <div class="card">
                <h2>Detailed Data Table <span class="row-count" id="visibleCount">{len(parsed_data)} rows</span></h2>
                <div class="search-container">
                    <input type="text" id="searchInput" class="search-input" onkeyup="filterTable()" placeholder="Search for filenames, methods, strategies...">
                </div>
                <div class="table-wrapper">
                    <table id="dataTable">
                        <thead>
                            <tr>
                                <th onclick="sortTable(0)">Filename</th>
                                <th onclick="sortTable(1)">Method</th>
                                <th onclick="sortTable(2)">Ranking Mode</th>
                                <th onclick="sortTable(3)">Gen. Time</th>
                                <th onclick="sortTable(4)">Model Settings</th>
                                <th onclick="sortTable(5)">Best Strategy</th>
                            </tr>
                        </thead>
                        <tbody>{table_rows}</tbody>
                    </table>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content
