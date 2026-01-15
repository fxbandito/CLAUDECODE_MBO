"""
Comparison Tab - MBO Trading Strategy Analyzer
Mixin class for Comparison tab functionality.
"""
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init

import os
from tkinter import TclError, filedialog, messagebox

import customtkinter as ctk

from gui.translate import tr


class ComparisonMixin:
    """Mixin for Comparison tab functionality."""

    def _create_comparison_tab(self) -> ctk.CTkFrame:
        """Create the Comparison tab UI."""
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")

        # === ROW 1: Folder selection and buttons ===
        row1 = ctk.CTkFrame(frame, fg_color="transparent", height=40)
        row1.pack(fill="x", pady=(10, 5), padx=10)

        # Label
        ctk.CTkLabel(
            row1,
            text=tr("Select Reports Folder:"),
            font=ctk.CTkFont(family="Arial", size=12, weight="bold")
        ).pack(side="left", padx=(0, 10))

        # Select Folder button - blue
        self.btn_compare_select_folder = ctk.CTkButton(
            row1,
            text=tr("Select Folder"),
            width=110,
            height=28,
            fg_color="#1f538d",
            hover_color="#163d66",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            command=self._on_compare_select_folder
        )
        self.btn_compare_select_folder.pack(side="left", padx=(0, 20))

        # Horizon buttons - teal/green
        horizons = [
            ("1 Week", "1 Week"),
            ("1 Month", "1 Month"),
            ("3 Months", "3 Months"),
            ("6 Months", "6 Months"),
            ("1 Year", "1 Year"),
        ]

        self.horizon_buttons = {}
        for text, horizon_key in horizons:
            btn = ctk.CTkButton(
                row1,
                text=tr(text),
                width=85,
                height=28,
                fg_color="#27ae60",
                hover_color="#1e8449",
                font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
                command=lambda h=horizon_key: self._on_generate_horizon_report(h)
            )
            btn.pack(side="left", padx=3)
            self.horizon_buttons[text] = btn

        # Main Data button - orange
        self.btn_main_data = ctk.CTkButton(
            row1,
            text=tr("Main Data"),
            width=95,
            height=28,
            fg_color="#e67e22",
            hover_color="#d35400",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            command=self._on_main_data_comparison
        )
        self.btn_main_data.pack(side="left", padx=(15, 2))

        # Help button for Main Data
        self.btn_main_data_help = ctk.CTkButton(
            row1,
            text="?",
            width=25,
            height=25,
            corner_radius=12,
            fg_color="#666666",
            hover_color="#888888",
            font=ctk.CTkFont(family="Arial", size=11, weight="bold"),
            command=self._show_main_data_help
        )
        self.btn_main_data_help.pack(side="left", padx=(0, 8))

        # Main Data AR button - slate blue
        self.btn_main_data_ar = ctk.CTkButton(
            row1,
            text=tr("Main Data AR"),
            width=110,
            height=28,
            fg_color="#778da9",
            hover_color="#415a77",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            command=self._on_main_data_ar_comparison
        )
        self.btn_main_data_ar.pack(side="left", padx=(0, 2))

        # Help button for Main Data AR
        self.btn_main_data_ar_help = ctk.CTkButton(
            row1,
            text="?",
            width=25,
            height=25,
            corner_radius=12,
            fg_color="#666666",
            hover_color="#888888",
            font=ctk.CTkFont(family="Arial", size=11, weight="bold"),
            command=self._show_main_data_ar_help
        )
        self.btn_main_data_ar_help.pack(side="left", padx=(0, 8))

        # Main Data MR button - purple
        self.btn_main_data_mr = ctk.CTkButton(
            row1,
            text=tr("Main Data MR"),
            width=115,
            height=28,
            fg_color="#9b59b6",
            hover_color="#8e44ad",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            command=self._on_main_data_mr_comparison
        )
        self.btn_main_data_mr.pack(side="left", padx=(0, 2))

        # Help button for Main Data MR
        self.btn_main_data_mr_help = ctk.CTkButton(
            row1,
            text="?",
            width=25,
            height=25,
            corner_radius=12,
            fg_color="#666666",
            hover_color="#888888",
            font=ctk.CTkFont(family="Arial", size=11, weight="bold"),
            command=self._show_main_data_mr_help
        )
        self.btn_main_data_mr_help.pack(side="left", padx=(0, 5))

        # === ROW 2: Selection status ===
        row2 = ctk.CTkFrame(frame, fg_color="transparent", height=25)
        row2.pack(fill="x", pady=(0, 5), padx=10)

        self.compare_folder_label = ctk.CTkLabel(
            row2,
            text=tr("No selection"),
            font=ctk.CTkFont(family="Arial", size=11),
            text_color="gray"
        )
        self.compare_folder_label.pack(side="left")

        # === MAIN AREA: Log/Output textbox ===
        log_frame = ctk.CTkFrame(frame, corner_radius=8)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        self.compare_log = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(family="Consolas", size=12),
            fg_color="#1a1a2e"
        )
        self.compare_log.pack(fill="both", expand=True, padx=10, pady=10)
        self.compare_log.insert("1.0", tr("Select a folder and click generate..."))

        # Initialize state
        self.compare_folder = ""

        return frame

    # === EVENT HANDLERS ===

    def _on_compare_select_folder(self):
        """Select folder for comparison reports."""
        self.sound.play_button_click()
        initial_dir = self.settings.get_last_comparison_folder() or None
        folder = filedialog.askdirectory(
            title=tr("Select Reports Folder"),
            initialdir=initial_dir
        )
        if folder:
            self.compare_folder = folder
            self.settings.set_last_comparison_folder(folder)
            self._save_settings_now()
            self.compare_folder_label.configure(text=folder, text_color="white")
            self._compare_log(f"{tr('Selected folder:')} {folder}")

    def _on_generate_horizon_report(self, horizon: str):
        """Generate comparison report for a specific horizon."""
        self.sound.play_button_click()
        if not self.compare_folder:
            messagebox.showwarning(tr("Warning"), tr("Please select a folder first!"))
            return

        try:
            # Import comparator module
            from analysis.comparator import scan_reports, parse_report, generate_html_report

            self._compare_log(f"\n{tr('Scanning reports for')} {horizon}...")
            report_files = scan_reports(self.compare_folder)

            if not report_files:
                self._compare_log(tr("No valid reports found."))
                return

            self._compare_log(f"{len(report_files)} {tr('reports found. Processing...')}")

            parsed_data = []
            for file_path in report_files:
                data = parse_report(file_path, horizon)
                parsed_data.append(data)
                self._compare_log(f"- {tr('Processed:')} {os.path.basename(file_path)}")

            output_path = os.path.join(
                self.compare_folder,
                f"comparison_report_{horizon.replace(' ', '_').lower()}.html"
            )
            success, msg = generate_html_report(parsed_data, output_path, horizon)

            if success:
                self._compare_log(f"\n{tr('Result ready:')}\n{msg}")
                self.sound.play_model_complete()
            else:
                self._compare_log(f"\n{tr('Error generating report:')} {msg}")

        except ImportError:
            self._compare_log(f"\n{tr('Comparator module not available.')}")
        except Exception as e:
            self._compare_log(f"\n{tr('Error:')} {e}")

    def _on_main_data_comparison(self):
        """Run the Main Data Comparison."""
        self.sound.play_button_click()
        if not self.compare_folder:
            messagebox.showwarning(tr("Warning"), tr("Please select a folder first!"))
            return

        try:
            from analysis.comparator import generate_main_data_report

            self._compare_log(f"\n{tr('Starting Main Data Comparison...')}")
            success, msg, _ = generate_main_data_report(self.compare_folder)

            if success:
                self._compare_log(f"{tr('Main Data Comparison Complete!')}")
                self._compare_log(f"{tr('Files created:')}\n{msg}")
                self.sound.play_model_complete()
            else:
                self._compare_log(f"{tr('Error:')} {msg}")

        except ImportError:
            self._compare_log(f"\n{tr('Comparator module not available.')}")
        except Exception as e:
            self._compare_log(f"\n{tr('Error:')} {e}")

    def _on_main_data_ar_comparison(self):
        """Run the Main Data AR Comparison."""
        self.sound.play_button_click()
        if not self.compare_folder:
            messagebox.showwarning(tr("Warning"), tr("Please select a folder first!"))
            return

        try:
            from analysis.comparator import generate_main_data_ar_report

            self._compare_log(f"\n{tr('Starting Main Data AR Comparison...')}")
            self._compare_log(f"{tr('Scanning for AR_ prefixed reports...')}")
            success, msg, _ = generate_main_data_ar_report(self.compare_folder)

            if success:
                self._compare_log(f"\n{tr('Main Data AR Comparison Complete!')}")
                self._compare_log(f"{tr('Files created:')}\n{msg}")
                self.sound.play_model_complete()
            else:
                self._compare_log(f"\n{tr('Error:')} {msg}")

        except ImportError:
            self._compare_log(f"\n{tr('Comparator module not available.')}")
        except Exception as e:
            self._compare_log(f"\n{tr('Error:')} {e}")

    def _on_main_data_mr_comparison(self):
        """Run the Main Data MR Comparison."""
        self.sound.play_button_click()
        if not self.compare_folder:
            messagebox.showwarning(tr("Warning"), tr("Please select a folder first!"))
            return

        try:
            from analysis.comparator import generate_main_data_mr_report

            self._compare_log(f"\n{tr('Starting Main Data MR Comparison...')}")
            self._compare_log(f"{tr('Scanning for MR_ prefixed reports...')}")
            success, msg, _ = generate_main_data_mr_report(self.compare_folder)

            if success:
                self._compare_log(f"\n{tr('Main Data MR Comparison Complete!')}")
                self._compare_log(f"{tr('Files created:')}\n{msg}")
                self.sound.play_model_complete()
            else:
                self._compare_log(f"\n{tr('Error:')} {msg}")

        except ImportError:
            self._compare_log(f"\n{tr('Comparator module not available.')}")
        except Exception as e:
            self._compare_log(f"\n{tr('Error:')} {e}")

    def _compare_log(self, message: str):
        """Add message to comparison log."""
        self.compare_log.insert("end", f"{message}\n")
        self.compare_log.see("end")

    # === HELP POPUPS ===

    def _show_main_data_help(self):
        """Show help popup for Main Data button."""
        self.sound.play_button_click()
        self._show_compare_help_popup(
            tr("Main Data - Help"),
            """Main Data Comparison

Purpose:
Compares standard analysis reports (without AR_ prefix) across multiple
currency pairs/strategies.

How it works:
1. Scans the selected folder for .md report files (excludes AR_ prefixed files)
2. Extracts key metrics from each report:
   - Strategy performance metrics
   - Forecast values for different time horizons
   - Model parameters and settings
3. Generates a consolidated HTML comparison report

Output:
- Creates 'main_data_comparison.html' in the selected folder
- Includes sortable tables and visual comparisons

When to use:
Use this when you want to compare results from multiple single-strategy
analyses to find the best performing strategies across different configurations."""
        )

    def _show_main_data_ar_help(self):
        """Show help popup for Main Data AR button."""
        self.sound.play_button_click()
        self._show_compare_help_popup(
            tr("Main Data AR - Help"),
            """Main Data AR Comparison

Purpose:
Compares All Results (AR_) reports that contain data for ALL strategies
in a single file.

How it works:
1. Scans the selected folder for AR_ prefixed .md report files
2. Parses multi-strategy data from each AR_ report:
   - All strategy IDs and their metrics
   - Combined forecast data for all strategies
   - Performance rankings
3. Generates a comprehensive comparison across all strategies

Output:
- Creates 'main_data_ar_comparison.html' in the selected folder
- Includes all strategies from AR_ reports with full metrics

When to use:
Use this when you have run analyses with 'All Results' mode enabled,
which generates AR_ prefixed reports containing all strategies in a single file.

Difference from Main Data:
- Main Data: One strategy per report file
- Main Data AR: All strategies in one AR_ report file"""
        )

    def _show_main_data_mr_help(self):
        """Show help popup for Main Data MR button."""
        self.sound.play_button_click()
        self._show_compare_help_popup(
            tr("Main Data MR - Help"),
            """Main Data MR Comparison

Purpose:
Compares Monthly Results (MR_) reports that contain monthly forecast
breakdowns using the 4-4-5 financial calendar.

How it works:
1. Scans the selected folder for MR_ prefixed .md report files
2. Parses monthly data from each MR_ report:
   - 12 months of forecast data (4-4-5 calendar: 4+4+5 weeks per quarter)
   - Best strategy for each month
   - Yearly profit totals
3. Generates a comprehensive comparison across all months and models

Output:
- Creates 'main_data_mr_comparison.html' in the selected folder
- Includes monthly consensus, stability ranking, and profit comparisons

When to use:
Use this when you have generated Monthly Results reports using the
'Monthly Results' button on the Reports tab.

4-4-5 Calendar:
- Each quarter has 3 months: 4 weeks, 4 weeks, 5 weeks
- Total: 52 weeks = 12 months
- Used in retail/financial planning for consistent month-over-month comparisons

Difference from Main Data AR:
- Main Data AR: Weekly breakdown (52 weeks)
- Main Data MR: Monthly breakdown (12 months, 4-4-5 calendar)"""
        )

    def _show_compare_help_popup(self, title: str, message: str):
        """Show a help popup window."""
        # Check if popup already exists
        if hasattr(self, "compare_help_window") and self.compare_help_window is not None:
            try:
                if self.compare_help_window.winfo_exists():
                    self.compare_help_window.lift()
                    self.compare_help_window.focus_force()
                    return
            except (TclError, AttributeError):
                pass

        popup = ctk.CTkToplevel(self)
        self.compare_help_window = popup
        popup.title(title)
        popup.geometry("550x450")
        popup.transient(self)
        popup.grab_set()

        textbox = ctk.CTkTextbox(
            popup,
            font=ctk.CTkFont(family="Arial", size=12),
            wrap="word"
        )
        textbox.pack(fill="both", expand=True, padx=15, pady=15)
        textbox.insert("1.0", message)
        textbox.configure(state="disabled")

        ctk.CTkButton(
            popup,
            text=tr("Close"),
            command=popup.destroy,
            font=ctk.CTkFont(family="Arial", size=12, weight="bold")
        ).pack(pady=10)

        popup.focus_force()
