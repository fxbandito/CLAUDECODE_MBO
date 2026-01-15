"""
Horizon-based comparison functionality.
Compares strategy reports across different forecast horizons (1W, 1M, 3M, 6M, 1Y).
"""

# pylint: disable=line-too-long
# Note: This file contains embedded HTML/CSS templates where line breaks
# would significantly reduce readability. The long lines are intentional.
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd


def parse_report(file_path: str, horizon: str = "1 Month") -> Dict[str, Any]:
    """
    Parses a single markdown report to extract Method, Top Strategy, and Top 10.

    Args:
        file_path: Path to the markdown report
        horizon: Forecast horizon (e.g., "1 Week", "1 Month", "3 Months")

    Returns:
        Dictionary with file info, method, top_strategy, and top_10 list
    """
    parent_dir = os.path.basename(os.path.dirname(file_path))

    data: Dict[str, Any] = {"file": os.path.basename(file_path), "path": file_path, "method": f"Unknown ({parent_dir})", "top_strategy": None, "top_10": []}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Extract Method (Line 3 usually, but let's search first few lines)
        for i in range(min(10, len(lines))):
            if "**Method**:" in lines[i]:
                raw_method = lines[i].split("**Method**:")[1].strip()
                data["method"] = f"{raw_method} ({parent_dir})"
                break

        # Extract Best Strategy Forecast for the specific horizon
        # Looking for: "### Best Strategy Forecast" then "- **{horizon}**: Strategy No. X"
        in_forecast_section = False
        for line in lines:
            if "### Best Strategy Forecast" in line:
                in_forecast_section = True
                continue

            if in_forecast_section:
                if line.strip().startswith("#"):  # Next section
                    in_forecast_section = False
                    break

                if f"- **{horizon}**:" in line:
                    # Match "Strategy No. X" or "(No. X)"
                    match = re.search(r"(?:Strategy No\.|No\.) (\d+)", line)
                    if match:
                        data["top_strategy"] = int(match.group(1))
                    break

        # Extract Top 10 Strategies by Horizon
        # Find "## Top 10 Strategies by Horizon" line
        top_10_start = -1
        for i, line in enumerate(lines):
            if "## Top 10 Strategies by Horizon" in line:
                top_10_start = i
                break

        if top_10_start != -1:
            # Parse the table header to find the column index for the horizon
            header_index = -1
            for j in range(top_10_start, len(lines)):
                if "| Rank |" in lines[j]:
                    header_index = j
                    header_line = lines[j]
                    break

            if header_index != -1:
                headers = [h.strip() for h in header_line.split("|")]

                try:
                    col_index = -1
                    for idx, h in enumerate(headers):
                        if h == horizon:
                            col_index = idx
                            break

                    if col_index != -1:
                        # Parse rows
                        for k in range(header_index + 2, len(lines)):  # Skip separator line
                            line = lines[k].strip()
                            if not line or not line.startswith("|"):
                                break

                            parts = [p.strip() for p in line.split("|")]
                            if len(parts) > col_index:
                                val = parts[col_index]
                                # Extract number from val which might be "#5" or "#5 (123.4)"
                                match = re.search(r"#?(\d+)", val)
                                if match:
                                    data["top_10"].append(int(match.group(1)))
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"Error parsing table: {e}")

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error parsing {file_path}: {e}")

    return data


def generate_html_report(parsed_data: List[Dict[str, Any]], output_path: str, horizon: str = "1 Month") -> Tuple[bool, str]:
    """
    Generates an HTML report from the parsed data.

    Args:
        parsed_data: List of parsed report dictionaries
        output_path: Path to write the HTML report
        horizon: Forecast horizon for the report title

    Returns:
        Tuple of (success: bool, message: str)
    """
    if not parsed_data:
        return False, "No data to generate report."

    # pylint: disable=import-outside-toplevel
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio

    # 1. Statistics for Top Strategy (Winner)
    winners = [d["top_strategy"] for d in parsed_data if d["top_strategy"] is not None]
    winner_counts = Counter(winners)

    # Map strategies to models that chose them
    strategy_to_models: Dict[int, List[str]] = {}
    for d in parsed_data:
        strat = d["top_strategy"]
        if strat is not None:
            if strat not in strategy_to_models:
                strategy_to_models[strat] = []
            strategy_to_models[strat].append(d["method"])

    # 2. Statistics for Top 10 Appearances
    all_top_10 = []
    for d in parsed_data:
        all_top_10.extend(d["top_10"])
    top_10_counts = Counter(all_top_10)

    # --- Generate Charts ---

    # Chart 1: Consensus (Winners)
    winner_df = pd.DataFrame(winner_counts.most_common(), columns=["Strategy", "Count"])
    winner_df["Strategy"] = winner_df["Strategy"].astype(str)

    fig_winner = px.bar(
        winner_df,
        x="Strategy",
        y="Count",
        title=f"Top Strategy Consensus (Wins) - {horizon}",
        color="Count",
        color_continuous_scale="Viridis",
    )
    fig_winner.update_layout(template="plotly_white")
    chart_winner_html = pio.to_html(fig_winner, full_html=False)

    # Chart 2: Consistency (Top 10)
    top10_df = pd.DataFrame(top_10_counts.most_common(20), columns=["Strategy", "Count"])
    top10_df["Strategy"] = top10_df["Strategy"].astype(str)

    fig_consistency = px.bar(
        top10_df, x="Strategy", y="Count", title=f"Top 10 Consistency ({horizon})", color="Count", color_continuous_scale="Blues"
    )
    fig_consistency.update_layout(template="plotly_white")
    chart_consistency_html = pio.to_html(fig_consistency, full_html=False)

    # Chart 3: Method Performance (Pie Chart)
    method_wins = []
    for d in parsed_data:
        if d["top_strategy"] is not None:
            method_wins.append(d["method"].split("(")[0].strip())

    if method_wins:
        method_counts = Counter(method_wins)
        method_df = pd.DataFrame(method_counts.most_common(), columns=["Method", "Count"])
        fig_method = px.pie(
            method_df,
            values="Count",
            names="Method",
            title=f"Winning Methods Distribution - {horizon}",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        chart_method_html = pio.to_html(fig_method, full_html=False)
    else:
        chart_method_html = "<p>No winning methods data available.</p>"

    # Chart 4: Weighted Score Leaderboard
    strategy_scores: Dict[int, int] = {}
    for d in parsed_data:
        for rank, strat in enumerate(d["top_10"]):
            if rank >= 10:
                break
            points = 10 - rank
            strategy_scores[strat] = strategy_scores.get(strat, 0) + points

    score_df = pd.DataFrame(list(strategy_scores.items()), columns=["Strategy", "Score"])
    score_df = score_df.sort_values(by="Score", ascending=False).head(20)
    score_df["Strategy"] = score_df["Strategy"].astype(str)

    fig_score = px.bar(
        score_df,
        x="Score",
        y="Strategy",
        orientation="h",
        title=f"Weighted Score Leaderboard - {horizon}",
        color="Score",
        color_continuous_scale="Magma",
    )
    fig_score.update_layout(template="plotly_white", yaxis={"categoryorder": "total ascending"})
    chart_score_html = pio.to_html(fig_score, full_html=False)

    # Chart 5: Rank Distribution (Box Plot)
    rank_data = []
    top_strategies_set = set(score_df["Strategy"].astype(int).tolist()) if not score_df.empty else set()

    for d in parsed_data:
        for rank, strat in enumerate(d["top_10"]):
            if strat in top_strategies_set:
                rank_data.append({"Strategy": str(strat), "Rank": rank + 1})

    if rank_data:
        rank_df = pd.DataFrame(rank_data)
        fig_rank = px.box(
            rank_df, x="Strategy", y="Rank", title=f"Rank Volatility (Top Strategies) - {horizon}", points="all", color="Strategy"
        )
        fig_rank.update_yaxes(autorange="reversed")
        fig_rank.update_layout(template="plotly_white")
        chart_rank_html = pio.to_html(fig_rank, full_html=False)
    else:
        chart_rank_html = "<p>No rank data available.</p>"

    # Chart 6: Method Similarity Heatmap
    method_sets: Dict[str, set] = {}
    for d in parsed_data:
        m_name = d["method"].split("(")[0].strip()
        if m_name not in method_sets:
            method_sets[m_name] = set()
        method_sets[m_name].update(d["top_10"])

    unique_methods = list(method_sets.keys())
    if len(unique_methods) > 1:
        similarity_matrix = []
        for m1 in unique_methods:
            row: List[float | int] = []
            for m2 in unique_methods:
                set1 = method_sets[m1]
                set2 = method_sets[m2]
                if not set1 or not set2:
                    row.append(0)
                else:
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    jaccard = intersection / union if union > 0 else 0
                    row.append(round(float(jaccard), 2))
            similarity_matrix.append(row)

        fig_heatmap = px.imshow(
            similarity_matrix,
            x=unique_methods,
            y=unique_methods,
            title=f"Method Similarity (Jaccard Index) - {horizon}",
            color_continuous_scale="RdBu_r",
            text_auto=True,
        )
        chart_heatmap_html = pio.to_html(fig_heatmap, full_html=False)
    else:
        chart_heatmap_html = "<p>Not enough unique methods for comparison.</p>"

    # Chart 7: Sankey Diagram - Method → Strategy Flow
    method_names = list({d["method"].split("(")[0].strip() for d in parsed_data})
    strategy_names = list({str(d["top_strategy"]) for d in parsed_data if d["top_strategy"] is not None})

    all_nodes = method_names + [f"Strategy {s}" for s in strategy_names]
    node_colors = ["#3498db"] * len(method_names) + ["#e74c3c"] * len(strategy_names)

    method_to_strategy_count: Dict[Tuple[str, str], int] = {}
    for d in parsed_data:
        if d["top_strategy"] is not None:
            m_name = d["method"].split("(")[0].strip()
            s_name = str(d["top_strategy"])
            key = (m_name, s_name)
            method_to_strategy_count[key] = method_to_strategy_count.get(key, 0) + 1

    sankey_source = []
    sankey_target = []
    sankey_value = []
    for (m_name, s_name), count in method_to_strategy_count.items():
        if m_name in method_names and s_name in strategy_names:
            sankey_source.append(method_names.index(m_name))
            sankey_target.append(len(method_names) + strategy_names.index(s_name))
            sankey_value.append(count)

    if sankey_source:
        fig_sankey = go.Figure(
            data=[
                go.Sankey(
                    node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_nodes, color=node_colors),
                    link=dict(source=sankey_source, target=sankey_target, value=sankey_value, color="rgba(52, 152, 219, 0.4)"),
                )
            ]
        )
        fig_sankey.update_layout(title_text=f"Method → Strategy Flow (Sankey) - {horizon}", template="plotly_white")
        chart_sankey_html = pio.to_html(fig_sankey, full_html=False)
    else:
        chart_sankey_html = "<p>No data available for Sankey diagram.</p>"

    # Chart 8: Rank Stability Line Chart
    rank_stability_data = []
    for d in parsed_data:
        m_name = d["method"].split("(")[0].strip()
        for rank, strat in enumerate(d["top_10"][:10]):
            rank_stability_data.append({"Method": m_name, "Strategy": str(strat), "Rank": rank + 1})

    if rank_stability_data:
        rank_stab_df = pd.DataFrame(rank_stability_data)
        top_strats = rank_stab_df["Strategy"].value_counts().head(10).index.tolist()
        rank_stab_df = rank_stab_df[rank_stab_df["Strategy"].isin(top_strats)]

        if not rank_stab_df.empty:
            fig_rank_stability = px.line(
                rank_stab_df, x="Method", y="Rank", color="Strategy", markers=True, title=f"Rank Stability Across Methods - {horizon}"
            )
            fig_rank_stability.update_yaxes(autorange="reversed")
            fig_rank_stability.update_layout(template="plotly_white", xaxis_tickangle=-45)
            chart_rank_stability_html = pio.to_html(fig_rank_stability, full_html=False)
        else:
            chart_rank_stability_html = "<p>No rank stability data available.</p>"
    else:
        chart_rank_stability_html = "<p>No rank stability data available.</p>"

    # Chart 9: Confidence Score
    confidence_data = []
    total_methods = len({d["method"].split("(")[0].strip() for d in parsed_data})

    for strat, count in winner_counts.most_common(15):
        confidence = (count / total_methods) * 100 if total_methods > 0 else 0
        confidence_data.append({"Strategy": f"Strategy {strat}", "Confidence": round(confidence, 1), "Votes": count})

    if confidence_data:
        conf_df = pd.DataFrame(confidence_data)
        fig_confidence = px.bar(
            conf_df,
            x="Strategy",
            y="Confidence",
            text="Confidence",
            title=f"Consensus Confidence Score (%) - {horizon}",
            color="Confidence",
            color_continuous_scale="RdYlGn",
        )
        fig_confidence.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_confidence.update_layout(template="plotly_white", yaxis_title="Confidence (%)")
        chart_confidence_html = pio.to_html(fig_confidence, full_html=False)
    else:
        chart_confidence_html = "<p>No confidence data available.</p>"

    # Chart 10: Method Agreement Matrix
    method_top1_map: Dict[str, List[int]] = {}
    for d in parsed_data:
        m_name = d["method"].split("(")[0].strip()
        if m_name not in method_top1_map:
            method_top1_map[m_name] = []
        if d["top_strategy"] is not None:
            method_top1_map[m_name].append(d["top_strategy"])

    unique_methods_agreement = list(method_top1_map.keys())
    if len(unique_methods_agreement) > 1:
        agreement_matrix = []
        for m1 in unique_methods_agreement:
            agreement_row: List[float | int] = []
            for m2 in unique_methods_agreement:
                if m1 == m2:
                    agreement_row.append(100)
                else:
                    set1 = set(method_top1_map[m1])
                    set2 = set(method_top1_map[m2])
                    if set1 and set2:
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        agreement = (intersection / union) * 100 if union > 0 else 0
                        agreement_row.append(round(float(agreement), 1))
                    else:
                        agreement_row.append(0)
            agreement_matrix.append(agreement_row)

        fig_agreement = px.imshow(
            agreement_matrix,
            x=unique_methods_agreement,
            y=unique_methods_agreement,
            title=f"Method Agreement Matrix (%) - {horizon}",
            color_continuous_scale="Greens",
            text_auto=True,
        )
        fig_agreement.update_layout(template="plotly_white")
        chart_agreement_html = pio.to_html(fig_agreement, full_html=False)
    else:
        chart_agreement_html = "<p>Not enough methods for agreement matrix.</p>"

    # Chart 11: Strategy Coverage (Upset-style)
    method_strategy_sets: Dict[str, set] = {}
    for d in parsed_data:
        m_name = d["method"].split("(")[0].strip()
        if m_name not in method_strategy_sets:
            method_strategy_sets[m_name] = set()
        if d["top_strategy"] is not None:
            method_strategy_sets[m_name].add(d["top_strategy"])

    all_strategies_upset = set()
    for strategies in method_strategy_sets.values():
        all_strategies_upset.update(strategies)

    upset_data = []
    for strat in all_strategies_upset:
        methods_with_strat = [m for m, s in method_strategy_sets.items() if strat in s]
        upset_data.append(
            {"Strategy": f"Strategy {strat}", "Method Count": len(methods_with_strat), "Methods": ", ".join(methods_with_strat)}
        )

    if upset_data:
        upset_df = pd.DataFrame(upset_data)
        upset_df = upset_df.sort_values("Method Count", ascending=True).tail(20)

        fig_upset = px.bar(
            upset_df,
            x="Method Count",
            y="Strategy",
            orientation="h",
            title=f"Strategy Coverage (Upset-style) - {horizon}",
            color="Method Count",
            color_continuous_scale="Purples",
            hover_data=["Methods"],
        )
        fig_upset.update_layout(template="plotly_white", yaxis={"categoryorder": "total ascending"})
        chart_upset_html = pio.to_html(fig_upset, full_html=False)
    else:
        chart_upset_html = "<p>No upset data available.</p>"

    # Chart 12: Top 3 Overlap Analysis
    top3_overlap: Dict[int, Dict[str, Any]] = {}
    for d in parsed_data:
        m_name = d["method"].split("(")[0].strip()
        top3 = d["top_10"][:3] if len(d["top_10"]) >= 3 else d["top_10"]
        for strat in top3:
            if strat not in top3_overlap:
                top3_overlap[strat] = {"methods": [], "count": 0}
            top3_overlap[strat]["methods"].append(m_name)
            top3_overlap[strat]["count"] += 1

    venn_data = []
    # Explicitly hint the lambda or use a typed helper if needed, but usually Dict hint is enough
    # casting x[1]["count"] to int helps mypy know it's comparable
    sorted_items = sorted(top3_overlap.items(), key=lambda x: int(x[1]["count"]), reverse=True)
    for strat, info in sorted_items[:15]:
        unique_methods_count = len(set(info["methods"]))
        venn_data.append({"Strategy": f"Strategy {strat}", "Appearances in Top 3": info["count"], "Unique Methods": unique_methods_count})

    if venn_data:
        venn_df = pd.DataFrame(venn_data)
        fig_venn = px.bar(
            venn_df,
            x="Strategy",
            y=["Appearances in Top 3", "Unique Methods"],
            title=f"Top 3 Strategy Overlap Analysis - {horizon}",
            barmode="group",
            color_discrete_sequence=["#3498db", "#e74c3c"],
        )
        fig_venn.update_layout(template="plotly_white", xaxis_tickangle=-45, legend_title="Metric")
        chart_venn_html = pio.to_html(fig_venn, full_html=False)
    else:
        chart_venn_html = "<p>No overlap data available.</p>"

    # Sort winners by frequency
    sorted_winners = winner_counts.most_common()

    # Calculate summary statistics
    total_reports = len(parsed_data)
    unique_methods_count = len({d["method"].split("(")[0].strip() for d in parsed_data})
    unique_strategies_count = len(winner_counts)
    top_strategy = sorted_winners[0][0] if sorted_winners else "N/A"
    top_strategy_votes = sorted_winners[0][1] if sorted_winners else 0

    # Generate HTML
    html = _generate_horizon_html(
        horizon=horizon,
        total_reports=total_reports,
        unique_methods=unique_methods_count,
        unique_strategies=unique_strategies_count,
        top_strategy=top_strategy,
        top_strategy_votes=top_strategy_votes,
        sorted_winners=sorted_winners,
        strategy_to_models=strategy_to_models,
        top_10_counts=top_10_counts,
        parsed_data=parsed_data,
        chart_winner_html=chart_winner_html,
        chart_consistency_html=chart_consistency_html,
        chart_method_html=chart_method_html,
        chart_score_html=chart_score_html,
        chart_rank_html=chart_rank_html,
        chart_confidence_html=chart_confidence_html,
        chart_heatmap_html=chart_heatmap_html,
        chart_sankey_html=chart_sankey_html,
        chart_rank_stability_html=chart_rank_stability_html,
        chart_agreement_html=chart_agreement_html,
        chart_upset_html=chart_upset_html,
        chart_venn_html=chart_venn_html,
    )

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return True, output_path
    except Exception as e:  # pylint: disable=broad-exception-caught
        return False, str(e)


def _generate_horizon_html(
    horizon,
    total_reports,
    unique_methods,
    unique_strategies,
    top_strategy,
    top_strategy_votes,
    sorted_winners,
    strategy_to_models,
    top_10_counts,
    parsed_data,
    chart_winner_html,
    chart_consistency_html,
    chart_method_html,
    chart_score_html,
    chart_rank_html,
    chart_confidence_html,
    chart_heatmap_html,
    chart_sankey_html,
    chart_rank_stability_html,
    chart_agreement_html,
    chart_upset_html,
    chart_venn_html,
) -> str:
    """Generate the full HTML content for the horizon comparison report."""

    # Build winners table rows
    winners_table_rows = ""
    for rank, (strat, count) in enumerate(sorted_winners, 1):
        models = ", ".join(strategy_to_models.get(strat, []))
        winners_table_rows += f"""
            <tr>
                <td>{rank}</td>
                <td>Strategy No. {strat}</td>
                <td><span class="badge">{count}</span></td>
                <td class="model-list">{models}</td>
            </tr>"""

    # Build consistency table rows
    sorted_top_10 = top_10_counts.most_common()
    hidden_count_consistency = max(0, len(sorted_top_10) - 30)
    consistency_table_rows = ""
    for rank, (strat, count) in enumerate(sorted_top_10, 1):
        row_class = "expandable-row" if rank > 30 else ""
        consistency_table_rows += f"""
            <tr class="{row_class}">
                <td>{rank}</td>
                <td>Strategy No. {strat}</td>
                <td>{count}</td>
            </tr>"""

    # Build processed reports table rows
    hidden_count_processed = max(0, len(parsed_data) - 30)
    processed_table_rows = ""
    for idx, d in enumerate(parsed_data, 1):
        row_class = "expandable-row" if idx > 30 else ""
        processed_table_rows += f"""
            <tr class="{row_class}">
                <td>{d["file"]}</td>
                <td>{d["method"]}</td>
                <td>{d["top_strategy"] if d["top_strategy"] else "N/A"}</td>
            </tr>"""

    # Consistency show more button
    consistency_button = ""
    if hidden_count_consistency > 0:
        consistency_button = f"""
            <button id="btn-consistency" class="show-more-btn" onclick="toggleRows('consistency-table', 'btn-consistency')">
                Show More ({hidden_count_consistency} more rows)
            </button>"""

    # Processed show more button
    processed_button = ""
    if hidden_count_processed > 0:
        processed_button = f"""
            <button id="btn-processed" class="show-more-btn" onclick="toggleRows('processed-table', 'btn-processed')">
                Show More ({hidden_count_processed} more rows)
            </button>"""

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Strategy Comparison Report - {horizon}</title>
        <style>
            * {{ box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0; padding: 20px;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                min-height: 100vh; color: #e4e4e4;
            }}
            h1 {{ color: #fff; font-size: 2.2em; margin-bottom: 5px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
            h2 {{ color: #3498db; font-size: 1.4em; margin: 0 0 10px 0; display: flex; align-items: center; gap: 10px; }}
            .container {{ max-width: 1600px; margin: 0 auto; }}
            .subtitle {{ color: #aaa; margin-bottom: 25px; }}
            .summary-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 25px; }}
            .stat-box {{ background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3); transition: transform 0.3s ease; }}
            .stat-box:hover {{ transform: translateY(-5px); }}
            .stat-box.highlight {{ background: linear-gradient(135deg, #e67e22, #d35400); }}
            .stat-box h3 {{ margin: 0; font-size: 2.2em; font-weight: 700; }}
            .stat-box p {{ margin: 8px 0 0 0; opacity: 0.9; font-size: 0.9em; }}
            .card {{ background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); padding: 25px; margin-bottom: 20px; border-radius: 16px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.1); overflow: hidden; }}
            .card p {{ color: #aaa; margin: 0 0 15px 0; font-size: 0.95em; }}
            .chart-container {{ width: 100%; overflow-x: auto; overflow-y: hidden; margin-bottom: 20px; }}
            .charts-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 20px; }}
            @media (max-width: 1200px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
            .table-wrapper {{ overflow-x: auto; border-radius: 12px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2); }}
            table {{ width: 100%; border-collapse: collapse; background: rgba(30, 30, 50, 0.8); font-size: 0.9em; }}
            th {{ background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 16px 12px; text-align: left; font-weight: 600; white-space: nowrap; border: none; }}
            th:first-child {{ border-radius: 12px 0 0 0; }}
            th:last-child {{ border-radius: 0 12px 0 0; }}
            td {{ padding: 14px 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); color: #e4e4e4; vertical-align: middle; }}
            tbody tr:nth-child(even) {{ background: rgba(255, 255, 255, 0.02); }}
            tbody tr {{ transition: background 0.2s ease; }}
            tbody tr:hover {{ background: rgba(52, 152, 219, 0.15); }}
            .badge {{ background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; padding: 6px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600; display: inline-block; }}
            .model-list {{ font-size: 0.85em; color: #aaa; max-width: 400px; }}
            td:nth-child(2) {{ color: #e67e22; font-weight: 600; }}
            .expandable-row {{ display: none; }}
            .expandable-row.visible {{ display: table-row; }}
            .show-more-btn {{ display: inline-block; margin: 15px 0; padding: 12px 24px; background: linear-gradient(135deg, #3498db, #2980b9); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 600; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3); }}
            .show-more-btn:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4); }}
            .show-more-btn.expanded {{ background: linear-gradient(135deg, #e74c3c, #c0392b); }}
            .row-count {{ display: inline-block; background: rgba(52, 152, 219, 0.2); color: #3498db; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; margin-left: 10px; }}
            ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
            ::-webkit-scrollbar-track {{ background: rgba(255, 255, 255, 0.05); border-radius: 4px; }}
            ::-webkit-scrollbar-thumb {{ background: rgba(52, 152, 219, 0.5); border-radius: 4px; }}
            ::-webkit-scrollbar-thumb:hover {{ background: rgba(52, 152, 219, 0.7); }}
        </style>
        <script>
            function toggleRows(tableId, btnId) {{
                const table = document.getElementById(tableId);
                const btn = document.getElementById(btnId);
                const hiddenRows = table.querySelectorAll('.expandable-row');
                const isExpanded = btn.classList.contains('expanded');
                hiddenRows.forEach(row => {{
                    if (isExpanded) {{ row.classList.remove('visible'); }} else {{ row.classList.add('visible'); }}
                }});
                if (isExpanded) {{
                    btn.textContent = 'Show More (' + hiddenRows.length + ' more rows)';
                    btn.classList.remove('expanded');
                }} else {{
                    btn.textContent = 'Show Less';
                    btn.classList.add('expanded');
                }}
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Strategy Comparison Report - {horizon}</h1>
            <p class="subtitle">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Horizon Analysis</p>

            <div class="summary-stats">
                <div class="stat-box"><h3>{total_reports}</h3><p>Total Reports</p></div>
                <div class="stat-box"><h3>{unique_methods}</h3><p>Unique Methods</p></div>
                <div class="stat-box"><h3>{unique_strategies}</h3><p>Unique Strategies</p></div>
                <div class="stat-box highlight"><h3>#{top_strategy}</h3><p>Top Strategy ({top_strategy_votes} votes)</p></div>
            </div>

            <div class="card">
                <h2>Top Strategy Consensus</h2>
                <p>Which strategies are most frequently selected as winners across all methods?</p>
                <div class="chart-container">{chart_winner_html}</div>
                <div class="table-wrapper">
                    <table>
                        <thead><tr><th>Rank</th><th>Strategy No.</th><th>Votes</th><th>Models (Voters)</th></tr></thead>
                        <tbody>{winners_table_rows}</tbody>
                    </table>
                </div>
            </div>

            <div class="card">
                <h2>Top 10 Consistency</h2>
                <p>How often does each strategy appear in the Top 10 rankings?</p>
                <div class="chart-container">{chart_consistency_html}</div>
                <div class="table-wrapper">
                    <table id="consistency-table">
                        <thead><tr><th>Rank</th><th>Strategy No.</th><th>Appearances</th></tr></thead>
                        <tbody>{consistency_table_rows}</tbody>
                    </table>
                </div>
                {consistency_button}
            </div>

            <div class="charts-grid">
                <div class="card"><h2>Method Performance</h2><p>Distribution of winning methods.</p><div class="chart-container">{chart_method_html}</div></div>
                <div class="card"><h2>Weighted Score Leaderboard</h2><p>Rank 1 = 10pts, Rank 10 = 1pt.</p><div class="chart-container">{chart_score_html}</div></div>
            </div>

            <div class="charts-grid">
                <div class="card"><h2>Rank Volatility</h2><p>Range of ranks for top strategies.</p><div class="chart-container">{chart_rank_html}</div></div>
                <div class="card"><h2>Consensus Confidence</h2><p>Agreement percentage on winning strategies.</p><div class="chart-container">{chart_confidence_html}</div></div>
            </div>

            <div class="card"><h2>Method Similarity Heatmap</h2><p>Jaccard Similarity Index (0-1) between Top 10 lists of different methods.</p><div class="chart-container">{chart_heatmap_html}</div></div>

            <div class="charts-grid">
                <div class="card"><h2>Method - Strategy Flow</h2><p>Blue = Methods, Red = Strategies.</p><div class="chart-container">{chart_sankey_html}</div></div>
                <div class="card"><h2>Rank Stability</h2><p>How strategies rank across methods.</p><div class="chart-container">{chart_rank_stability_html}</div></div>
            </div>

            <div class="charts-grid">
                <div class="card"><h2>Method Agreement Matrix</h2><p>Pairwise agreement between methods.</p><div class="chart-container">{chart_agreement_html}</div></div>
                <div class="card"><h2>Strategy Coverage</h2><p>Methods selecting each strategy.</p><div class="chart-container">{chart_upset_html}</div></div>
            </div>

            <div class="card"><h2>Top 3 Strategy Overlap Analysis</h2><p>Strategies appearing in Top 3 positions. Blue = Total appearances, Red = Unique methods.</p><div class="chart-container">{chart_venn_html}</div></div>

            <div class="card">
                <h2>Processed Reports <span class="row-count">{len(parsed_data)} files</span></h2>
                <div class="table-wrapper">
                    <table id="processed-table">
                        <thead><tr><th>File</th><th>Method</th><th>Winner Strategy</th></tr></thead>
                        <tbody>{processed_table_rows}</tbody>
                    </table>
                </div>
                {processed_button}
            </div>
        </div>
    </body>
    </html>
    """
    return html
