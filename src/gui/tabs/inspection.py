"""
Inspection Tab - Compare forecast reports with actual benchmark results.
"""
# pylint: disable=broad-exception-caught

import os
import sys
import subprocess
import threading
from tkinter import filedialog, ttk
import customtkinter as ctk

from gui.translate import tr


class InspectionTabMixin:
    """Mixin class for Inspection tab functionality."""

    def _create_inspection_tab(self) -> ctk.CTkFrame:
        """Create the Inspection tab content."""
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")

        # Main container with padding
        main_container = ctk.CTkFrame(frame, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20, pady=10)

        # === TOP SECTION: Left side (inputs+buttons) + Right side (MT5 Tester) ===
        top_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        top_frame.pack(fill="x", pady=(0, 10))

        # Left side container
        left_side = ctk.CTkFrame(top_frame, fg_color="transparent")
        left_side.pack(side="left", fill="both")

        # --- Row 1: Reports Folder ---
        row1 = ctk.CTkFrame(left_side, fg_color="transparent")
        row1.pack(fill="x", pady=2)

        ctk.CTkLabel(
            row1, text=tr("Reports Folder:"),
            font=ctk.CTkFont(size=11), width=120, anchor="e"
        ).pack(side="left")

        self.inspect_folder_entry = ctk.CTkEntry(row1, height=28, width=400)
        self.inspect_folder_entry.pack(side="left", padx=5)

        ctk.CTkButton(
            row1, text="...", width=30, height=28,
            command=self._browse_inspect_folder
        ).pack(side="left")

        # --- Row 2: Benchmark Excel ---
        row2 = ctk.CTkFrame(left_side, fg_color="transparent")
        row2.pack(fill="x", pady=2)

        ctk.CTkLabel(
            row2, text=tr("Benchmark Excel:"),
            font=ctk.CTkFont(size=11), width=120, anchor="e"
        ).pack(side="left")

        self.inspect_excel_entry = ctk.CTkEntry(row2, height=28, width=400)
        self.inspect_excel_entry.pack(side="left", padx=5)

        ctk.CTkButton(
            row2, text="...", width=30, height=28,
            command=self._browse_inspect_excel
        ).pack(side="left")

        # --- Row 3: Action Buttons ---
        row3 = ctk.CTkFrame(left_side, fg_color="transparent")
        row3.pack(fill="x", pady=(10, 0))

        # Spacer to align with inputs
        ctk.CTkLabel(row3, text="", width=125).pack(side="left")

        self.btn_run_inspection = ctk.CTkButton(
            row3, text=tr("Run Inspection"), width=130, height=32,
            fg_color="#2ecc71", hover_color="#27ae60",
            text_color="white", font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self._run_inspection("standard")
        )
        self.btn_run_inspection.pack(side="left", padx=(0, 10))

        self.btn_run_inspection_ar = ctk.CTkButton(
            row3, text=tr("Run Inspection AR"), width=150, height=32,
            fg_color="#3498db", hover_color="#2980b9",
            text_color="white", font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self._run_inspection("ar")
        )
        self.btn_run_inspection_ar.pack(side="left", padx=(0, 10))

        self.btn_export_inspect = ctk.CTkButton(
            row3, text=tr("Export Report"), width=120, height=32,
            fg_color="#9b59b6", hover_color="#8e44ad",
            text_color="white", text_color_disabled="#cccccc",
            font=ctk.CTkFont(size=11, weight="bold"),
            state="disabled", command=self._export_inspection_report
        )
        self.btn_export_inspect.pack(side="left")

        # Right side - MT5 Tester big button (blue, tall and wide)
        self.btn_mt5_tester = ctk.CTkButton(
            top_frame, text="MT5 Tester", width=240, height=95,
            fg_color="#3498db", hover_color="#2980b9",
            text_color="white", font=ctk.CTkFont(size=16, weight="bold"),
            command=self._launch_mt5_tester
        )
        self.btn_mt5_tester.pack(side="right", padx=(20, 0))

        # === RESULTS TABLE ===
        table_frame = ctk.CTkFrame(main_container, corner_radius=5)
        table_frame.pack(fill="both", expand=True, pady=(10, 0))

        # Title
        ctk.CTkLabel(
            table_frame,
            text=tr("Inspection Results:"),
            font=ctk.CTkFont(size=11, weight="bold"),
            anchor="w"
        ).pack(anchor="w", padx=10, pady=5)

        # Treeview with scrollbar
        tree_container = ctk.CTkFrame(table_frame, fg_color="#1a1a2e")
        tree_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Style for Treeview (dark theme)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Inspect.Treeview",
            background="#1a1a2e",
            foreground="#eaeaea",
            fieldbackground="#1a1a2e",
            rowheight=25,
            font=('Consolas', 10)
        )
        style.configure(
            "Inspect.Treeview.Heading",
            background="#0f3460",
            foreground="#ffd93d",
            font=('Arial', 10, 'bold')
        )
        style.map(
            "Inspect.Treeview",
            background=[('selected', '#3498db')],
            foreground=[('selected', 'white')]
        )

        # Columns
        columns = (
            "model", "year", "horizon", "pred_rank", "actual_rank",
            "rank_diff", "pred_profit", "actual_profit", "winner", "winner_profit"
        )

        self.inspect_tree = ttk.Treeview(
            tree_container,
            columns=columns,
            show="headings",
            style="Inspect.Treeview"
        )

        # Configure columns
        col_configs = [
            ("model", "Model", 100),
            ("year", "Year", 60),
            ("horizon", "Horizon", 80),
            ("pred_rank", "Pred. Rank", 80),
            ("actual_rank", "Actual Rank", 80),
            ("rank_diff", "Rank Diff", 70),
            ("pred_profit", "Pred. Profit", 100),
            ("actual_profit", "Actual Profit", 100),
            ("winner", "Winner", 100),
            ("winner_profit", "Winner Profit", 100)
        ]

        for col_id, heading, width in col_configs:
            self.inspect_tree.heading(col_id, text=heading, command=lambda c=col_id: self._sort_inspect_tree(c))
            self.inspect_tree.column(col_id, width=width, anchor="center")

        # Scrollbars
        v_scroll = ttk.Scrollbar(tree_container, orient="vertical", command=self.inspect_tree.yview)
        h_scroll = ttk.Scrollbar(tree_container, orient="horizontal", command=self.inspect_tree.xview)
        self.inspect_tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        # Grid layout for tree and scrollbars
        self.inspect_tree.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)

        # Store engine reference
        self._inspection_engine = None
        self._inspect_sort_reverse = {}

        return frame

    def _browse_inspect_folder(self):
        """Browse for reports folder."""
        folder = filedialog.askdirectory(
            title=tr("Select Reports Folder"),
            initialdir=self.inspect_folder_entry.get() or os.path.expanduser("~")
        )
        if folder:
            self.inspect_folder_entry.delete(0, "end")
            self.inspect_folder_entry.insert(0, folder)

    def _browse_inspect_excel(self):
        """Browse for benchmark Excel file."""
        filepath = filedialog.askopenfilename(
            title=tr("Select Benchmark Excel"),
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
            initialdir=os.path.dirname(self.inspect_excel_entry.get()) or os.path.expanduser("~")
        )
        if filepath:
            self.inspect_excel_entry.delete(0, "end")
            self.inspect_excel_entry.insert(0, filepath)

    def _run_inspection(self, mode: str):
        """Run the inspection process."""
        folder_path = self.inspect_folder_entry.get().strip()
        excel_path = self.inspect_excel_entry.get().strip()

        if not folder_path:
            self._log(tr("Please select a reports folder."))
            return

        if not excel_path:
            self._log(tr("Please select a benchmark Excel file."))
            return

        if not os.path.exists(folder_path):
            self._log(tr("Reports folder does not exist."))
            return

        if not os.path.exists(excel_path):
            self._log(tr("Benchmark Excel file does not exist."))
            return

        # Disable buttons during processing
        self.btn_run_inspection.configure(state="disabled")
        self.btn_run_inspection_ar.configure(state="disabled")
        self.btn_export_inspect.configure(state="disabled")

        # Clear previous results
        for item in self.inspect_tree.get_children():
            self.inspect_tree.delete(item)

        self._log(f"Starting inspection ({mode} mode)...")

        # Run in thread
        thread = threading.Thread(
            target=self._run_inspection_thread,
            args=(folder_path, excel_path, mode),
            daemon=True
        )
        thread.start()

    def _run_inspection_thread(self, folder_path: str, excel_path: str, mode: str):
        """Run inspection in background thread."""
        try:
            from analysis.inspection import InspectionEngine

            engine = InspectionEngine()

            # Parse reports
            engine.parse_markdown_reports(folder_path, mode)

            # Parse benchmark
            engine.parse_benchmark_excel(excel_path)

            # Compare
            comparisons = engine.compare_results()

            # Update UI in main thread
            self.after(0, lambda: self._display_inspection_results(engine, comparisons))

        except Exception as e:
            self.after(0, lambda: self._on_inspection_error(str(e)))

    def _display_inspection_results(self, engine, comparisons):
        """Display results in the tree view."""
        self._inspection_engine = engine

        # Clear tree
        for item in self.inspect_tree.get_children():
            self.inspect_tree.delete(item)

        # Add results
        for comp in comparisons:
            # Color tag based on rank diff
            if comp.rank_diff <= 2:
                tag = "good"
            elif comp.rank_diff > 5:
                tag = "bad"
            else:
                tag = ""

            self.inspect_tree.insert("", "end", values=(
                comp.model,
                comp.forecast_year,
                comp.horizon,
                comp.predicted_rank,
                comp.actual_rank,
                comp.rank_diff,
                f"{comp.predicted_profit:,.2f}",
                f"{comp.actual_profit:,.2f}",
                comp.actual_winner,
                f"{comp.winner_profit:,.2f}"
            ), tags=(tag,))

        # Configure tags
        self.inspect_tree.tag_configure("good", foreground="#00d26a")
        self.inspect_tree.tag_configure("bad", foreground="#ff6b6b")

        # Re-enable buttons
        self.btn_run_inspection.configure(state="normal")
        self.btn_run_inspection_ar.configure(state="normal")

        if comparisons:
            self.btn_export_inspect.configure(state="normal")
            self._log(f"Inspection complete. {len(comparisons)} comparisons found.")
        else:
            self._log("No matching comparisons found.")

    def _on_inspection_error(self, error_msg: str):
        """Handle inspection error."""
        self._log(f"Inspection error: {error_msg}")
        self.btn_run_inspection.configure(state="normal")
        self.btn_run_inspection_ar.configure(state="normal")

    def _sort_inspect_tree(self, col):
        """Sort treeview by column."""
        items = [(self.inspect_tree.set(item, col), item) for item in self.inspect_tree.get_children('')]

        # Toggle sort direction
        reverse = self._inspect_sort_reverse.get(col, False)
        self._inspect_sort_reverse[col] = not reverse

        # Try numeric sort first
        try:
            items.sort(key=lambda x: float(x[0].replace(',', '')), reverse=reverse)
        except ValueError:
            items.sort(key=lambda x: x[0], reverse=reverse)

        for index, (_, item) in enumerate(items):
            self.inspect_tree.move(item, '', index)

    def _export_inspection_report(self):
        """Export inspection results to HTML/MD."""
        if not self._inspection_engine or not self._inspection_engine.comparisons:
            self._log("No results to export.")
            return

        filepath = filedialog.asksaveasfilename(
            title=tr("Save Inspection Report"),
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("Markdown files", "*.md"), ("All files", "*.*")]
        )

        if not filepath:
            return

        try:
            if filepath.endswith('.md'):
                self._inspection_engine.generate_md_report(filepath)
            else:
                self._inspection_engine.generate_html_report(filepath)

            self._log(f"Report exported to: {filepath}")

            # Open the report
            if os.path.exists(filepath):
                os.startfile(filepath)

        except Exception as e:
            self._log(f"Export error: {e}")

    def _launch_mt5_tester(self):
        """Launch MT5 Tester GUI as independent process."""
        try:
            # src/gui/tabs -> src/gui -> src -> src/MT5
            src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            mt5_gui_path = os.path.join(src_dir, "MT5", "mt5_gui.py")

            if not os.path.exists(mt5_gui_path):
                self._log(f"MT5 GUI not found at: {mt5_gui_path}")
                return

            # Launch as completely independent process (not using 'with' intentionally -
            # we don't want to wait for the subprocess to finish)
            if sys.platform == "win32":
                # CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS = fully independent
                subprocess.Popen(  # pylint: disable=consider-using-with
                    [sys.executable, mt5_gui_path],
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                    close_fds=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                subprocess.Popen(  # pylint: disable=consider-using-with
                    [sys.executable, mt5_gui_path],
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            self._log("MT5 Tester launched.")

        except Exception as e:
            self._log(f"Failed to launch MT5 Tester: {e}")
