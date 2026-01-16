"""
Results Tab - MBO Trading Strategy Analyzer
Mixin osztály a Results tab funkcionalitásához.
"""
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=too-many-statements,too-many-locals

import logging
import os
import pickle
import threading
import traceback
from tkinter import filedialog, messagebox

import customtkinter as ctk
import pandas as pd

from data.processor import DataProcessor
from reporting.exporter import ReportExporter  # pylint: disable=wrong-import-order


class ResultsMixin:
    """Mixin a Results tab funkcionalitásához."""

    def _create_results_tab(self) -> ctk.CTkFrame:
        """Results tab - elemzési eredmények megjelenítése."""
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")

        # === ROW 1: Ranking controls ===
        row1 = ctk.CTkFrame(frame, fg_color="transparent", height=40)
        row1.pack(fill="x", pady=(5, 3))

        # Ranking Mode
        ctk.CTkLabel(
            row1,
            text="Ranking Mode:",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold")
        ).pack(side="left", padx=(0, 5))

        self.ranking_mode_var = ctk.StringVar(value="Standard")
        self.ranking_mode_combo = ctk.CTkComboBox(
            row1,
            values=["Standard", "Stability Weighted", "Risk Adjusted"],
            variable=self.ranking_mode_var,
            width=180,
            height=28,
            font=ctk.CTkFont(family="Arial", size=10),
            command=self._on_ranking_mode_change,
            state="readonly"
        )
        self.ranking_mode_combo.pack(side="left", padx=(0, 10))

        # Sort info label
        self.ranking_desc_label = ctk.CTkLabel(
            row1,
            text="Sort by 1M Forecast only",
            font=ctk.CTkFont(family="Arial", size=10),
            text_color="gray"
        )
        self.ranking_desc_label.pack(side="left", padx=(0, 10))

        # Apply Ranking button - zöld mint a screenshoton
        self.btn_apply_ranking = ctk.CTkButton(
            row1,
            text="Apply Ranking",
            width=120,
            height=28,
            fg_color="#2ecc71",
            hover_color="#27ae60",
            text_color="white",
            text_color_disabled="#cccccc",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            state="disabled",
            command=self._on_apply_ranking
        )
        self.btn_apply_ranking.pack(side="left", padx=(0, 15))

        # Current ranking mode display
        self.lbl_current_ranking = ctk.CTkLabel(
            row1,
            text="",
            font=ctk.CTkFont(family="Arial", size=11, weight="bold"),
            text_color="#2ecc71"
        )
        self.lbl_current_ranking.pack(side="right", padx=(0, 20))

        # Help button - szürke
        self.btn_metrics_help = ctk.CTkButton(
            row1,
            text="?",
            width=30,
            height=28,
            fg_color="#555555",
            hover_color="#666666",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            command=self._show_ranking_help
        )
        self.btn_metrics_help.pack(side="right", padx=(0, 5))

        # === ROW 2: Export controls ===
        row2 = ctk.CTkFrame(frame, fg_color="transparent", height=40)
        row2.pack(fill="x", pady=3)

        # Output Folder
        ctk.CTkLabel(
            row2,
            text="Output Folder:",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold")
        ).pack(side="left", padx=(0, 5))

        self.output_folder_var = ctk.StringVar(value="")
        self.output_folder_entry = ctk.CTkEntry(
            row2,
            textvariable=self.output_folder_var,
            width=250,
            height=28,
            font=ctk.CTkFont(family="Arial", size=10)
        )
        self.output_folder_entry.pack(side="left", padx=(0, 8))

        # Browse button - kék
        self.btn_browse_output = ctk.CTkButton(
            row2,
            text="Browse",
            width=70,
            height=28,
            fg_color="#3498db",
            hover_color="#2980b9",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            command=self._on_browse_output
        )
        self.btn_browse_output.pack(side="left", padx=(0, 10))

        # Generate Report button - zöld SOLID (mint az eredetiben)
        self.btn_generate_report = ctk.CTkButton(
            row2,
            text="Generate Report",
            state="disabled",
            fg_color="green",
            hover_color="#006400",
            text_color="white",
            text_color_disabled="#cccccc",
            font=ctk.CTkFont(family="Arial", size=14, weight="bold"),
            command=self._on_generate_report
        )
        self.btn_generate_report.pack(side="left", padx=20)

        # Export CSV button - default stílus (mint az eredetiben)
        self.btn_export_csv = ctk.CTkButton(
            row2,
            text="Export CSV",
            state="disabled",
            text_color_disabled="#cccccc",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            command=self._on_export_csv
        )
        self.btn_export_csv.pack(side="left", padx=5)

        # All Results button - sárga SOLID (mint az eredetiben)
        self.btn_all_results = ctk.CTkButton(
            row2,
            text="All Results",
            width=95,
            fg_color="#f39c12",
            hover_color="#d68910",
            text_color="black",
            text_color_disabled="#666666",
            font=ctk.CTkFont(family="Arial", size=11, weight="bold"),
            state="disabled",
            command=self._on_show_all_results
        )
        self.btn_all_results.pack(side="left", padx=5)

        # Monthly Results button - lila SOLID (mint az eredetiben)
        self.btn_monthly_results = ctk.CTkButton(
            row2,
            text="Monthly Results",
            width=115,
            fg_color="#9b59b6",
            hover_color="#8e44ad",
            text_color="white",
            text_color_disabled="#cccccc",
            font=ctk.CTkFont(family="Arial", size=11, weight="bold"),
            state="disabled",
            command=self._on_show_monthly_results
        )
        self.btn_monthly_results.pack(side="left", padx=5)

        # Load Analysis State button - lila (mint az eredetiben)
        self.btn_load_state = ctk.CTkButton(
            row2,
            text="Load Analysis State",
            width=180,
            fg_color="#8e44ad",
            hover_color="#732d91",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            command=self._on_load_analysis_state
        )
        self.btn_load_state.pack(side="right", padx=10)

        # Show Params button - sötétkék (mint az eredetiben)
        self.btn_show_params = ctk.CTkButton(
            row2,
            text="Show Params",
            width=100,
            fg_color="#34495e",
            hover_color="#2c3e50",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            command=self._on_show_params
        )
        self.btn_show_params.pack(side="right", padx=5)

        # === Results display area ===
        results_frame = ctk.CTkFrame(frame, corner_radius=8)
        results_frame.pack(fill="both", expand=True, pady=(8, 5))

        # Header for results
        header_frame = ctk.CTkFrame(results_frame, fg_color="transparent", height=35)
        header_frame.pack(fill="x", padx=10, pady=(8, 0))

        self.results_info_label = ctk.CTkLabel(
            header_frame,
            text="No results available. Run an analysis first.",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.results_info_label.pack(side="left")

        self.results_count_label = ctk.CTkLabel(
            header_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.results_count_label.pack(side="right")

        # Results table (using CTkTextbox for now, later can be CTkTable)
        self.results_text = ctk.CTkTextbox(
            results_frame,
            font=ctk.CTkFont(family="Consolas", size=12),
            fg_color="#1a1a2e"
        )
        self.results_text.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        self.results_text.insert("1.0", self._get_results_placeholder())
        self.results_text.configure(state="disabled")

        # Initialize ranking state
        self.current_ranking_mode = "forecast"
        self.results_with_stability = None

        # Initialize output folder from saved settings
        saved_output_folder = getattr(self, "last_results_output_folder", "")
        if saved_output_folder:
            self.output_folder_var.set(saved_output_folder)

        return frame

    def _get_results_placeholder(self) -> str:
        """Placeholder szöveg az eredmények területére."""
        return """
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    ANALYSIS RESULTS                                           ║
╠══════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                              ║
║   Run an analysis from the Analysis tab to see results here.                                ║
║                                                                                              ║
║   Results will include:                                                                      ║
║   • Strategy rankings by predicted profit                                                   ║
║   • Forecast accuracy metrics (MAPE, RMSE, MAE)                                             ║
║   • Historical vs predicted performance comparison                                           ║
║   • Confidence intervals for predictions                                                     ║
║                                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
"""

    # === Results tab eseménykezelők ===

    def _on_ranking_mode_change(self, value):
        """Handle ranking mode dropdown change."""
        descriptions = {
            "Standard": "Sort by 1M Forecast only",
            "Stability Weighted": "60% forecast + 40% stability (percentile)",
            "Risk Adjusted": "50% forecast + 30% Sharpe + 20% consistency",
        }
        self.ranking_desc_label.configure(text=descriptions.get(value, ""))

    def _on_apply_ranking(self):
        """Apply the selected ranking mode to results."""
        self.sound.play_button_click()
        logging.info("Ranking mode applied")

        # Use results_df (current/original) if results_with_stability is missing
        has_stability = (
            hasattr(self, "results_with_stability") and self.results_with_stability is not None
        )
        base_df = self.results_with_stability if has_stability else self.results_df

        if base_df is None or base_df.empty:
            messagebox.showwarning("Warning", "No results to rank.")
            return

        # Map combo value to mode key
        mode_map = {
            "Standard": "forecast",
            "Stability Weighted": "stability",
            "Risk Adjusted": "risk_adjusted"
        }
        mode = mode_map.get(self.ranking_mode_combo.get(), "forecast")

        # Check if metrics need calculation
        required_cols = ["Stability_Score"]
        needs_calc = mode in ["stability", "risk_adjusted"] and not all(
            c in base_df.columns for c in required_cols
        )

        if needs_calc:
            # Run calculation in background thread to avoid UI freeze
            self._run_ranking_async(base_df, mode)
        else:
            # No calculation needed, apply ranking directly
            self._log("Metrics already present, skipping calculation.", "debug")
            self._finalize_ranking(base_df, mode)

    def _run_ranking_async(self, base_df, mode):
        """Run stability metrics calculation in background thread."""
        # Disable button during calculation
        self.btn_apply_ranking.configure(state="disabled")
        self._log("Calculating stability metrics... Please wait.")

        def calculate():
            try:
                ts_data = None

                # 1. Try to get data from DataTab (if loaded)
                if hasattr(self, "get_data"):
                    ts_data = self.get_data()
                    if ts_data is not None:
                        self.after(0, lambda: self._log("Using currently loaded data from Data Tab.", "debug"))

                # 2. If not found, try to reconstruct from all_strategies_data
                if ts_data is None:
                    all_strat_data = None
                    if hasattr(self, "all_strategies_data") and self.all_strategies_data:
                        all_strat_data = self.all_strategies_data
                    elif hasattr(self, "last_results") and self.last_results:
                        all_strat_data = self.last_results.get("all_strategies_data", {})

                    if all_strat_data:
                        self.after(0, lambda: self._log(
                            "Data not loaded in Data Tab. "
                            "Attempting to reconstruct from saved strategy data...",
                            "debug"
                        ))
                        ts_data = self._reconstruct_ts_data_from_dict(all_strat_data)
                        if ts_data is not None:
                            cnt = len(ts_data["No."].unique())
                            self.after(0, lambda: self._log(
                                f"Reconstructed data for {cnt} strategies.",
                                "debug"
                            ))

                if ts_data is not None:
                    # Proceed with calculation
                    result_df = DataProcessor.calculate_stability_metrics(ts_data, base_df)
                    self.after(0, lambda: self._on_ranking_complete(result_df, mode))
                else:
                    err_msg = (
                        "No time-series data available. Please load the original data file "
                        "or ensure analysis state contains history."
                    )
                    self.after(0, lambda: self._on_ranking_error(err_msg))

            except (ValueError, KeyError, TypeError, IndexError) as e:
                traceback.print_exc()
                self.after(0, lambda err=e: self._on_ranking_error(str(err)))

        thread = threading.Thread(target=calculate, daemon=True)
        thread.start()

    def _reconstruct_ts_data_from_dict(self, all_strategies_dict):
        """Reconstruct a raw_data DataFrame from a dictionary of strategy data."""
        if not all_strategies_dict:
            return None

        try:
            dfs = []
            for strat_id, strat_data in all_strategies_dict.items():
                if hasattr(strat_data, "copy"):
                    df = strat_data.copy()
                    df["No."] = strat_id
                    dfs.append(df)

            if dfs:
                return pd.concat(dfs, ignore_index=True)
            return None
        except (ValueError, TypeError, IndexError) as e:
            self.after(0, lambda err=e: self._log(f"Error reconstructing data: {err}"))
            return None

    def _on_ranking_complete(self, result_df, mode):
        """Called when background ranking calculation completes."""
        self.results_with_stability = result_df
        self._log("Stability metrics calculated.")
        self._finalize_ranking(result_df, mode)
        self.btn_apply_ranking.configure(state="normal")

    def _on_ranking_error(self, error_msg):
        """Called when background ranking calculation fails."""
        self._log(f"Error calculating metrics: {error_msg}")
        self.btn_apply_ranking.configure(state="normal")

    def _finalize_ranking(self, base_df, mode):
        """Apply ranking and update display."""
        # Apply ranking
        ranked_df, sort_col = DataProcessor.apply_ranking(
            base_df, ranking_mode=mode, sort_column="Forecast_1M"
        )

        # Update stored results
        self.results_df = ranked_df
        self.current_ranking_mode = mode

        # Update last_results for report generation
        if hasattr(self, "last_results") and self.last_results:
            self.last_results["results"] = ranked_df
            self.last_results["ranking_mode"] = mode

        # Update display
        self._update_results_display(ranked_df, sort_col)

        # Update current ranking label
        mode_names = {
            "forecast": "Forecast Only",
            "stability": "Stability Weighted",
            "risk_adjusted": "Risk Adjusted",
        }
        self.lbl_current_ranking.configure(
            text=f"[Current: {mode_names.get(mode, mode)}]"
        )

        self._log(f"Ranking applied: {mode_names.get(mode, mode)}")

    def _update_results_display(self, df, sort_col):
        """Update the results text display with ranked data."""
        self.results_text.configure(state="normal")
        self.results_text.delete("0.0", "end")

        # Display columns
        display_cols = [
            "Rank", "No.", "Forecast_1W", "Forecast_1M", "Forecast_3M",
            "Forecast_6M", "Forecast_12M", "Method",
        ]

        # Hide Rank in Standard mode
        if self.current_ranking_mode == "forecast" and "Rank" in display_cols:
            display_cols.remove("Rank")

        # Add optional columns if present
        for col in ["Stability_Score", "Ranking_Score", "Sharpe_Ratio",
                    "Activity_Consistency", "Profit_Consistency", "Data_Confidence"]:
            if col in df.columns:
                display_cols.append(col)

        # Filter to existing columns
        display_cols = [c for c in display_cols if c in df.columns]

        # Header Aliases
        header_map = {
            "Stability_Score": "SS", "Ranking_Score": "RS", "Sharpe_Ratio": "SR",
            "Activity_Consistency": "AC", "Profit_Consistency": "PC", "Data_Confidence": "DC",
        }

        # Calculate column widths
        col_widths = {}
        for col in display_cols:
            header_text = header_map.get(col, col)
            header_len = len(header_text)
            data_len = 0
            if not df.empty:
                def fmt(x):
                    return f"{x:,.2f}" if isinstance(x, (int, float)) else str(x)
                sample_data = df.head(50)[col].apply(fmt)
                if not sample_data.empty:
                    data_len = sample_data.str.len().max()
            col_widths[col] = max(header_len, data_len, 5) + 2

        # Header
        header_parts = []
        for col in display_cols:
            width = col_widths[col]
            text = header_map.get(col, col)
            header_parts.append(f"{text:^{width}}")

        header = "|".join(header_parts)

        self.results_text.insert("end", f"{'=' * len(header)}\n")
        self.results_text.insert("end", f"{header}\n")
        self.results_text.insert("end", f"{'=' * len(header)}\n")

        # Top 50 rows
        for _, row in df.head(50).iterrows():
            row_parts = []
            for col in display_cols:
                width = col_widths[col]
                val = row.get(col, "")
                if col in ["Rank", "No."]:
                    try:
                        val_str = f"{int(val)}"
                    except (ValueError, TypeError):
                        val_str = str(val)
                elif isinstance(val, (int, float)):
                    val_str = f"{val:,.2f}"
                else:
                    val_str = str(val)
                row_parts.append(f"{val_str:^{width}}")
            row_str = "|".join(row_parts)
            self.results_text.insert("end", f"{row_str}\n")

        self.results_text.insert("end", f"\nTotal strategies: {len(df)}\n")
        self.results_text.insert("end", f"Sorted by: {sort_col}\n")
        self.results_text.configure(state="disabled")

    def _on_browse_output(self):
        """Open folder browser for output directory."""
        self.sound.play_button_click()
        initial_dir = getattr(self, "last_results_output_folder", "") or None
        folder = filedialog.askdirectory(initialdir=initial_dir)
        if folder:
            self.last_results_output_folder = folder
            if hasattr(self, "save_window_state"):
                self.save_window_state()
            self.output_folder_var.set(folder)
            self._log(f"Output folder set to: {folder}")

    def _on_generate_report(self, auto_mode=False, base_dir=None, forced_suffix=None):
        """Generate analysis report."""
        if not auto_mode:
            self.sound.play_button_click()
            logging.info("Generating analysis report")

        if not hasattr(self, "last_results") or not self.last_results:
            if not auto_mode:
                messagebox.showwarning("Warning", "No analysis results to save!")
            else:
                self._log("Auto Report Error: No results to save.")
            return None

        # Get Output Directory
        base_output_dir = base_dir if base_dir else self.output_folder_var.get()
        if not base_output_dir:
            if not auto_mode:
                messagebox.showwarning("Warning", "Please select an output folder first!")
            else:
                self._log("Auto Report Error: No output folder selected.")
            return None

        # Threading Logic
        if not auto_mode:
            threading.Thread(
                target=self._generate_report_bg,
                args=(base_output_dir, auto_mode, forced_suffix),
                daemon=True
            ).start()
            return None
        return self._generate_report_bg(base_output_dir, auto_mode, forced_suffix)

    def _generate_report_bg(self, base_output_dir, auto_mode, forced_suffix=None):
        """Background thread logic for report generation."""
        try:
            self.after(0, lambda: self._log("Generating report..."))

            data = self.last_results
            method = data["method"]
            filename_base = data["filename_base"]
            best_strat_id = data["best_strat_id"]
            best_strat_data = data["best_strat_data"]
            results_df = data["results"]
            forecast_values = data.get("forecast_values")

            # Get ranking mode for folder naming
            ranking_mode = data.get("ranking_mode", "forecast")
            ranking_suffix_map = {
                "forecast": "Standard",
                "stability": "StabilityWeighted",
                "risk_adjusted": "RiskAdjusted"
            }
            ranking_suffix = ranking_suffix_map.get(ranking_mode, "Standard")

            # Determine Best Strategy from CURRENT sorted results
            if not results_df.empty:
                top_row = results_df.iloc[0]
                new_best_id = int(top_row["No."])
                if new_best_id != best_strat_id:
                    self.after(0, lambda: self._log(
                        f"Ranking changed best strategy from {best_strat_id} to {new_best_id}"
                    ))
                    best_strat_id = new_best_id
                    self.after(0, lambda: self._log(
                        f"Re-simulating strategy {best_strat_id} for accurate charts..."
                    ))
                    try:
                        new_best_data, new_forecasts = self._reconstruct_strategy_data(
                            best_strat_id, data
                        )
                        if new_best_data is not None:
                            best_strat_data = new_best_data
                            forecast_values = new_forecasts
                            self.after(0, lambda: self._log("Re-simulation successful."))
                    except (ValueError, KeyError, IndexError, TypeError) as e:
                        self.after(0, lambda err=e: self._log(f"Error re-simulating: {err}"))

            # Create specific subfolder
            currency_pair = filename_base.split("_")[0]
            folder_name = f"{method}_{currency_pair}_{ranking_suffix}"
            original_folder_name = folder_name

            # Auto Mode: Unique naming
            if auto_mode:
                if forced_suffix:
                    folder_name = f"{original_folder_name}{forced_suffix}"
                else:
                    counter = 1
                    while True:
                        folder_name = f"{original_folder_name}_A{counter:02d}"
                        if not os.path.exists(os.path.join(base_output_dir, folder_name)):
                            break
                        counter += 1

            report_dir = os.path.join(base_output_dir, folder_name)
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)

            # Instantiate exporter
            current_exporter = ReportExporter(report_dir)

            # Get forecast values for HTML report
            if forecast_values is None:
                best_row = results_df[results_df["No."] == best_strat_id]
                if not best_row.empty and "Forecasts" in best_row.columns:
                    forecast_values = best_row.iloc[0]["Forecasts"]
                    if not isinstance(forecast_values, list) or len(forecast_values) == 0:
                        forecast_values = None

            # Generate Reports (standard MD and HTML only)
            md_path = current_exporter.create_markdown_report(
                results_df, best_strat_id, method, [],
                best_strat_data=best_strat_data,
                filename_base=f"{method}_{filename_base}",
                params=data["params"],
                execution_time=data.get("execution_time", "N/A"),
                ranking_mode=ranking_suffix,
            )

            html_path = current_exporter.create_html_report(
                results_df, best_strat_id, method,
                best_strat_data=best_strat_data,
                forecast_values=forecast_values,
                filename_base=f"{method}_{filename_base}",
                params=data["params"],
                execution_time=data.get("execution_time", "N/A"),
                ranking_mode=ranking_suffix,
            )

            self.after(0, lambda: self._log(f"Reports saved in: {report_dir}"))
            self.after(0, lambda: self._log(f"MD: {os.path.basename(md_path)}", "debug"))
            if html_path:
                self.after(0, lambda: self._log(f"HTML: {os.path.basename(html_path)}", "debug"))

            # Auto Mode: Generate All Results report
            if auto_mode:
                try:
                    self.after(0, lambda: self._log("Generating All Results report (auto)...", "debug"))
                    forecast_horizon = data.get("params", {}).get("forecast_horizon", 52)
                    ar_filename_base = f"AR_{method}_{filename_base}"
                    ar_md_path, ar_html_path = current_exporter.create_all_results_report(
                        results_df=results_df,
                        method_name=method,
                        best_strat_data=best_strat_data,
                        filename_base=ar_filename_base,
                        params=data.get("params", {}),
                        execution_time=data.get("execution_time", "N/A"),
                        forecast_horizon=forecast_horizon,
                        ranking_mode=ranking_suffix,
                    )
                    self.after(0, lambda: self._log(f"AR MD: {os.path.basename(ar_md_path)}", "debug"))
                    self.after(0, lambda: self._log(f"AR HTML: {os.path.basename(ar_html_path)}", "debug"))
                except (OSError, ValueError, RuntimeError) as ar_e:
                    self.after(0, lambda: self._log(f"Error generating AR report: {ar_e}", "warning"))

            return report_dir

        except (OSError, ValueError, RuntimeError) as e:
            self.after(0, lambda err=e: self._log(f"Error generating report: {err}"))
            traceback.print_exc()
            return None

    def _reconstruct_strategy_data(self, strat_id, context_data):
        """Get strategy data for report generation."""
        try:
            # FAST PATH: Check if we have pre-saved data
            all_strat_data = context_data.get("all_strategies_data", {})
            if all_strat_data and strat_id in all_strat_data:
                self._log(f"Using cached historical data for strategy {strat_id}")
                strat_data = all_strat_data[strat_id]
                results_df = context_data.get("results")
                forecasts = None
                if results_df is not None:
                    strat_row = results_df[results_df["No."] == strat_id]
                    if not strat_row.empty and "Forecasts" in strat_row.columns:
                        forecasts = strat_row.iloc[0]["Forecasts"]
                return strat_data, forecasts

            # SLOW PATH: Re-simulate if no cached data
            self._log(f"No cached data for strategy {strat_id}, attempting re-simulation...")
            if not hasattr(self, "get_data"):
                logging.error("Cannot access data loader to re-simulate.")
                return None, None

            raw_data = self.get_data()
            if raw_data is None:
                return None, None

            strat_data = None
            if isinstance(raw_data, dict):
                strat_data = raw_data.get(strat_id)
            elif isinstance(raw_data, pd.DataFrame):
                if "No." in raw_data.columns:
                    strat_data = raw_data[raw_data["No."] == strat_id].copy()
                else:
                    strat_data = raw_data.copy()

            if strat_data is None or (hasattr(strat_data, "empty") and strat_data.empty):
                logging.error("Could not find raw data for strategy %s", strat_id)
                return None, None

            params = context_data.get("params", {})
            method = context_data.get("method", "ARIMA")
            use_gpu = params.get("use_gpu", False)
            max_horizon = params.get("forecast_horizon", 52)

            # Lazy import to avoid circular dependency
            from analysis.engine import analyze_strategy  # pylint: disable=import-outside-toplevel

            forecasts = analyze_strategy(
                strat_id, strat_data, method, params,
                use_gpu=use_gpu, max_horizon=max_horizon, n_jobs=1
            )

            if forecasts is not None:
                return strat_data, forecasts
            return None, None

        except (ValueError, KeyError, IndexError, RuntimeError, ImportError) as e:
            logging.error("Re-simulation error: %s", e)
            traceback.print_exc()
            return None, None

    def _on_export_csv(self):
        """Export results to CSV file."""
        self.sound.play_button_click()
        logging.info("Exporting results to CSV")
        if not hasattr(self, "results_df") or self.results_df is None or self.results_df.empty:
            messagebox.showwarning("Warning", "No results to export.")
            return

        # Get initial directory from last export or output folder
        initial_dir = getattr(self, "last_csv_export_folder", "") or self.output_folder_var.get()
        if not initial_dir or not os.path.exists(initial_dir):
            initial_dir = os.getcwd()

        try:
            file_path = filedialog.asksaveasfilename(
                initialdir=initial_dir,
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export Results to CSV"
            )
            if file_path:
                # Remember the folder for next time
                self.last_csv_export_folder = os.path.dirname(file_path)
                if hasattr(self, "save_window_state"):
                    self.save_window_state()

                self.results_df.to_csv(file_path, index=False)
                self._log(f"Results exported to: {file_path}")
        except (OSError, ValueError) as e:
            self._log(f"Error exporting results: {e}")

    def _on_show_all_results(self):
        """Generate All Results report with weekly breakdown of forecasts."""
        self.sound.play_button_click()
        logging.info("Generating all results report")

        if not hasattr(self, "last_results") or not self.last_results:
            messagebox.showwarning("Warning", "No analysis results available!")
            return

        base_output_dir = self.output_folder_var.get()
        if not base_output_dir:
            messagebox.showwarning("Warning", "Please select an output folder first!")
            return

        threading.Thread(
            target=self._generate_all_results_bg,
            args=(base_output_dir,),
            daemon=True
        ).start()

    def _generate_all_results_bg(self, base_output_dir):
        """Background thread logic for All Results report."""
        try:
            self.after(0, lambda: self._log("Generating All Results report..."))

            data = self.last_results
            method = data["method"]
            filename_base = data["filename_base"]
            best_strat_data = data["best_strat_data"]
            results_df = data["results"]
            params = data.get("params", {})
            forecast_horizon = params.get("forecast_horizon", 52)

            ranking_mode = data.get("ranking_mode", "forecast")
            ranking_suffix_map = {
                "forecast": "Standard",
                "stability": "StabilityWeighted",
                "risk_adjusted": "RiskAdjusted"
            }
            ranking_suffix = ranking_suffix_map.get(ranking_mode, "Standard")

            currency_pair = filename_base.split("_")[0]
            folder_name = f"{method}_{currency_pair}_{ranking_suffix}"
            report_dir = os.path.join(base_output_dir, folder_name)
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)

            exporter = ReportExporter(report_dir)
            md_path, html_path = exporter.create_all_results_report(
                results_df=results_df,
                method_name=method,
                best_strat_data=best_strat_data,
                filename_base=f"AR_{method}_{filename_base}",
                params=params,
                execution_time=data.get("execution_time", "N/A"),
                forecast_horizon=forecast_horizon,
                ranking_mode=ranking_suffix,
            )

            self.after(0, lambda: self._log(f"All Results reports saved in: {report_dir}"))
            self.after(0, lambda: self._log(f"MD: {os.path.basename(md_path)}", "debug"))
            self.after(0, lambda: self._log(f"HTML: {os.path.basename(html_path)}", "debug"))

        except (OSError, ValueError, RuntimeError) as e:
            self.after(0, lambda: self._log(f"Error generating All Results report: {e}", "error"))
            traceback.print_exc()

    def _on_show_monthly_results(self):
        """Generate Monthly Results report with 4-4-5 calendar breakdown."""
        self.sound.play_button_click()
        logging.info("Generating monthly results report")

        if not hasattr(self, "last_results") or not self.last_results:
            messagebox.showwarning("Warning", "No analysis results available!")
            return

        base_output_dir = self.output_folder_var.get()
        if not base_output_dir:
            messagebox.showwarning("Warning", "Please select an output folder first!")
            return

        threading.Thread(
            target=self._generate_monthly_results_bg,
            args=(base_output_dir,),
            daemon=True
        ).start()

    def _generate_monthly_results_bg(self, base_output_dir):
        """Background thread logic for Monthly Results report."""
        try:
            self.after(0, lambda: self._log("Generating Monthly Results report..."))

            data = self.last_results
            method = data["method"]
            filename_base = data["filename_base"]
            best_strat_id = data["best_strat_id"]
            best_strat_data = data["best_strat_data"]
            results_df = data["results"]
            params = data.get("params", {})

            ranking_mode = data.get("ranking_mode", "forecast")
            ranking_suffix_map = {
                "forecast": "Standard",
                "stability": "StabilityWeighted",
                "risk_adjusted": "RiskAdjusted"
            }
            ranking_suffix = ranking_suffix_map.get(ranking_mode, "Standard")

            currency_pair = filename_base.split("_")[0]
            folder_name = f"{method}_{currency_pair}_{ranking_suffix}"
            report_dir = os.path.join(base_output_dir, folder_name)
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)

            exporter = ReportExporter(report_dir)
            mr_filename_base = f"MR_{method}_{filename_base}"

            md_path = exporter.create_monthly_markdown_report(
                results_df=results_df,
                best_strategy_id=best_strat_id,
                method_name=method,
                best_strat_data=best_strat_data,
                filename_base=mr_filename_base,
                params=params,
                execution_time=data.get("execution_time", "N/A"),
                ranking_mode=ranking_suffix,
            )

            html_path = exporter.create_monthly_html_report(
                results_df=results_df,
                best_strategy_id=best_strat_id,
                method_name=method,
                best_strat_data=best_strat_data,
                filename_base=mr_filename_base,
                params=params,
                execution_time=data.get("execution_time", "N/A"),
                ranking_mode=ranking_suffix,
            )

            self.after(0, lambda: self._log(f"Monthly Results reports saved in: {report_dir}"))
            self.after(0, lambda: self._log(f"MD: {os.path.basename(md_path)}", "debug"))
            self.after(0, lambda: self._log(f"HTML: {os.path.basename(html_path)}", "debug"))

        except (OSError, ValueError, RuntimeError) as e:
            self.after(0, lambda: self._log(f"Error generating Monthly Results report: {e}", "error"))
            traceback.print_exc()

    def _on_load_analysis_state(self):
        """Load a previously saved analysis state from a pickle file."""
        self.sound.play_button_click()
        logging.info("Loading analysis state from file")

        initial_dir = getattr(self, "last_analysis_state_folder", "")
        if not initial_dir or not os.path.exists(initial_dir):
            initial_dir = os.path.join(os.getcwd(), "Data_raw")
            if not os.path.exists(initial_dir):
                initial_dir = os.getcwd()

        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Load Analysis State",
            filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
        )

        if file_path:
            self.last_analysis_state_folder = os.path.dirname(file_path)
            if hasattr(self, "save_window_state"):
                self.save_window_state()
        else:
            return

        try:
            self._log(f"Loading analysis state from: {os.path.basename(file_path)}...")

            with open(file_path, "rb") as f:
                loaded_data = pickle.load(f)

            # Validate loaded data
            if not isinstance(loaded_data, dict) or "results" not in loaded_data:
                raise ValueError("Invalid analysis state file. Missing 'results' data.")

            # Version handling
            data_version = loaded_data.get("version", "0.9")
            self._log(f"Loading data format version: {data_version}", "debug")

            if data_version == "0.9":
                loaded_data.setdefault("ranking_mode", "forecast")
                loaded_data.setdefault("execution_time", "N/A")
                loaded_data.setdefault("all_strategies_data", {})
            elif data_version == "1.0":
                loaded_data.setdefault("all_strategies_data", {})

            # Backfill missing parameters
            method = loaded_data.get("method", "")
            saved_params = loaded_data.get("params", {})
            if method and saved_params is not None and hasattr(self, "get_full_model_params"):
                full_params = self.get_full_model_params(method, saved_params)
                if len(full_params) > len(saved_params):
                    loaded_data["params"] = full_params
                    backfilled = len(full_params) - len(saved_params)
                    self._log(f"Backfilled {backfilled} missing parameters with defaults", "debug")

            # Restore state
            self.last_results = loaded_data
            self.results_df = loaded_data.get("results")
            self.all_strategies_data = loaded_data.get("all_strategies_data", {})

            ranking_mode = loaded_data.get("ranking_mode", "forecast")
            self.current_ranking_mode = ranking_mode

            self.results_with_stability = None
            if "Stability_Score" in self.results_df.columns:
                self.results_with_stability = self.results_df

            # Log statistics
            num_strategies = len(self.results_df) if self.results_df is not None else 0
            num_with_history = len(self.all_strategies_data)
            self._log(
                f"State loaded: {loaded_data.get('method', 'Unknown')} - {num_strategies} strategies"
            )
            self._log(f"Strategies with full history: {num_with_history}", "debug")

            # Update Display
            sort_col = "Forecast_1M"
            if "Ranking_Score" in self.results_df.columns:
                sort_col = "Ranking_Score"

            self._update_results_display(self.results_df, sort_col)

            # Enable Buttons
            self.btn_generate_report.configure(state="normal")
            self.btn_export_csv.configure(state="normal")
            self._enable_ranking_controls()

            # Update ranking label
            mode_names = {
                "forecast": "Forecast Only",
                "stability": "Stability Weighted",
                "risk_adjusted": "Risk Adjusted",
            }
            self.lbl_current_ranking.configure(
                text=f"[Current: {mode_names.get(ranking_mode, ranking_mode)}]"
            )

        except (pickle.UnpicklingError, ValueError, OSError, EOFError) as e:
            self._log(f"Error loading state: {e}", "error")
            logging.error("Error loading state: %s", e)

    def _enable_ranking_controls(self):
        """Enable ranking controls after analysis completes."""
        self.btn_apply_ranking.configure(state="normal")
        self.btn_all_results.configure(state="normal")
        self.btn_monthly_results.configure(state="normal")
        self.lbl_current_ranking.configure(text="[Current: Standard]")

    def _on_show_params(self):
        """Show parameters from the loaded analysis state."""
        self.sound.play_button_click()

        if not hasattr(self, "last_results") or not self.last_results:
            messagebox.showinfo("Info", "No analysis results loaded.")
            return

        params = self.last_results.get("params", {})
        method = self.last_results.get("method", "Unknown Model")

        if not params:
            messagebox.showinfo("Info", "No parameters found in loaded state.")
            return

        # Format parameters for display
        msg = f"Model Parameters: {method}\n\nParameters:\n"
        for k, v in params.items():
            msg += f"- {k}: {v}\n"

        self._show_info_popup("Model Parameters", msg)

    def _show_info_popup(self, title, message):
        """Show custom info popup."""
        popup = ctk.CTkToplevel(self)
        popup.title(title)
        popup.geometry("500x400")
        popup.transient(self)
        popup.grab_set()

        textbox = ctk.CTkTextbox(popup, font=ctk.CTkFont(family="Arial", size=12))
        textbox.pack(fill="both", expand=True, padx=10, pady=10)
        textbox.insert("0.0", message)
        textbox.configure(state="disabled")

        ctk.CTkButton(
            popup, text="Close", command=popup.destroy,
            font=ctk.CTkFont(family="Arial", size=12, weight="bold")
        ).pack(pady=10)

    def _show_ranking_help(self):
        """Show popup explaining stability metrics."""
        self.sound.play_button_click()
        msg = """Stability Metrics Explanation:

SS - Stability Score (0-1):
  Composite score combining consistency, profit, and risk.
  Higher is better. Used for 'Stability Weighted' ranking.

AC - Activity Consistency (0-1):
  Measures how regularly the strategy trades.
  1.0 = Trades every week uniformly.

PC - Profit Consistency (0-1):
  Win rate among active trading weeks.
  > 0.5 means more profitable weeks than losing weeks.

SR - Sharpe Ratio:
  Risk-adjusted return (Mean Profit / Std Dev).

DC - Data Confidence (0-1):
  Confidence based on history length.
  1.0 = 2+ years of data.

RS - Ranking Score:
  Final score used for sorting in 'Weighted' modes.

RANKING MODES:

Standard:
  Sort by 1M Forecast only.

Stability Weighted:
  60% forecast + 40% stability (percentile).

Risk Adjusted:
  50% forecast + 30% Sharpe + 20% consistency.

Click "Apply Ranking" to re-sort the results table."""

        self._show_info_popup("Metrics Help", msg)
