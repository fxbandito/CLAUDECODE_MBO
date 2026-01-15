"""
Results Tab - MBO Trading Strategy Analyzer
Mixin osztály a Results tab funkcionalitásához.
"""
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init

from tkinter import filedialog

import customtkinter as ctk


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
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(0, 5))

        self.ranking_mode_var = ctk.StringVar(value="forecast")
        self.ranking_mode_combo = ctk.CTkComboBox(
            row1,
            values=["forecast", "historical", "combined", "risk-adjusted"],
            variable=self.ranking_mode_var,
            width=160,
            height=28,
            font=ctk.CTkFont(size=10)
        )
        self.ranking_mode_combo.pack(side="left", padx=(0, 10))

        # Sort info label
        ctk.CTkLabel(
            row1,
            text="Sort by predicted profit only",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(side="left", padx=(0, 10))

        # Apply Ranking button - zöld mint a screenshoton
        self.btn_apply_ranking = ctk.CTkButton(
            row1,
            text="Apply Ranking",
            width=100,
            height=28,
            fg_color="#2ecc71",
            hover_color="#27ae60",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_apply_ranking
        )
        self.btn_apply_ranking.pack(side="left", padx=(0, 15))

        # Help button - szürke
        ctk.CTkButton(
            row1,
            text="?",
            width=26,
            height=26,
            corner_radius=13,
            fg_color="#555555",
            hover_color="#666666",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._show_ranking_help
        ).pack(side="right", padx=(0, 5))

        # === ROW 2: Export controls ===
        row2 = ctk.CTkFrame(frame, fg_color="transparent", height=40)
        row2.pack(fill="x", pady=3)

        # Output Folder
        ctk.CTkLabel(
            row2,
            text="Output Folder:",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(0, 5))

        self.output_folder_var = ctk.StringVar(value="")
        self.output_folder_entry = ctk.CTkEntry(
            row2,
            textvariable=self.output_folder_var,
            width=250,
            height=28,
            font=ctk.CTkFont(size=10)
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
            font=ctk.CTkFont(size=11, weight="bold"),
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
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._on_generate_report
        )
        self.btn_generate_report.pack(side="left", padx=20)

        # Export CSV button - default stílus (mint az eredetiben)
        self.btn_export_csv = ctk.CTkButton(
            row2,
            text="Export CSV",
            state="disabled",
            font=ctk.CTkFont(size=12, weight="bold"),
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
            font=ctk.CTkFont(size=11, weight="bold"),
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
            font=ctk.CTkFont(size=11, weight="bold"),
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
            font=ctk.CTkFont(size=12, weight="bold"),
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
            font=ctk.CTkFont(size=12, weight="bold"),
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
            font=ctk.CTkFont(family="Consolas", size=10),
        )
        self.results_text.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        self.results_text.insert("1.0", self._get_results_placeholder())
        self.results_text.configure(state="disabled")

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

    def _on_apply_ranking(self):
        """Ranking alkalmazása."""
        self.sound.play_button_click()
        mode = self.ranking_mode_var.get()
        self._log(f"Applying ranking mode: {mode}")

    def _on_browse_output(self):
        """Output folder kiválasztása."""
        self.sound.play_button_click()
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder_var.set(folder)
            self._log(f"Output folder set to: {folder}")

    def _on_generate_report(self):
        """Report generálása."""
        self.sound.play_button_click()
        if not self.output_folder_var.get():
            self._log("Please select an output folder first!", "warning")
            return
        self._log("Generating report...")

    def _on_export_csv(self):
        """CSV exportálás."""
        self.sound.play_button_click()
        self._log("Exporting to CSV...")

    def _on_show_all_results(self):
        """Összes eredmény megjelenítése."""
        self.sound.play_button_click()
        self._log("Showing all results...")

    def _on_show_monthly_results(self):
        """Havi eredmények megjelenítése."""
        self.sound.play_button_click()
        self._log("Showing monthly results...")

    def _on_load_analysis_state(self):
        """Elemzési állapot betöltése."""
        self.sound.play_button_click()
        file_path = filedialog.askopenfilename(
            title="Load Analysis State",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            self._log(f"Loading analysis state from: {file_path}")

    def _on_show_params(self):
        """Paraméterek megjelenítése popup-ban."""
        self.sound.play_button_click()
        popup = ctk.CTkToplevel(self)
        popup.title("Analysis Parameters")
        popup.geometry("600x400")
        popup.transient(self)
        popup.grab_set()

        text = ctk.CTkTextbox(popup, font=ctk.CTkFont(family="Consolas", size=11))
        text.pack(fill="both", expand=True, padx=15, pady=15)
        text.insert("1.0", """ANALYSIS PARAMETERS

No analysis has been run yet.

When you run an analysis, this will show:
• Selected model and category
• Model parameters used
• Data preprocessing settings
• Forecast horizon
• CPU/GPU settings
• Panel/Dual mode settings
""")
        text.configure(state="disabled")

        ctk.CTkButton(popup, text="Close", command=popup.destroy).pack(pady=10)

    def _show_ranking_help(self):
        """Ranking help popup."""
        self.sound.play_button_click()
        popup = ctk.CTkToplevel(self)
        popup.title("Ranking Mode - Help")
        popup.geometry("550x350")
        popup.transient(self)
        popup.grab_set()

        text = ctk.CTkTextbox(popup, font=ctk.CTkFont(size=12))
        text.pack(fill="both", expand=True, padx=15, pady=15)
        text.insert("1.0", """RANKING MODES

forecast:
  Sort strategies by predicted future profit.
  Best for forward-looking analysis.

historical:
  Sort strategies by historical performance.
  Best for validation against known data.

combined:
  Weighted combination of forecast and historical.
  Balanced approach.

risk-adjusted:
  Sort by risk-adjusted returns (Sharpe-like ratio).
  Penalizes high volatility strategies.

Click "Apply Ranking" to re-sort the results table.""")
        text.configure(state="disabled")

        ctk.CTkButton(popup, text="Close", command=popup.destroy).pack(pady=10)
