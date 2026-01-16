"""
Data Loading Tab - MBO Trading Strategy Analyzer
Mixin osztaly a Data Loading tab funkcioinalisatasahoz.
"""
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=broad-exception-caught

import gc
import os
import threading
from tkinter import filedialog

import customtkinter as ctk
import pandas as pd

from data.loader import DataLoader
from data.processor import DataProcessor
from gui.translate import tr


class DataLoadingMixin:
    """Mixin a Data Loading tab funkcioinalisatasahoz."""

    def _create_data_tab(self) -> ctk.CTkFrame:
        """Data Loading tab - teljes implementacio."""
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")

        # Felso sor: folder info + gombok
        top_frame = ctk.CTkFrame(frame, fg_color="transparent", height=50)
        top_frame.pack(fill="x", pady=(10, 5))

        self.folder_label = ctk.CTkLabel(
            top_frame,
            text=tr("No folder selected"),
            font=ctk.CTkFont(size=11),
            anchor="w"
        )
        self.folder_label.pack(side="left")

        self.btn_open_folder = ctk.CTkButton(
            top_frame,
            text=tr("Open Folder"),
            width=100,
            height=28,
            fg_color="#3498db",
            hover_color="#2980b9",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_open_folder
        )
        self.btn_open_folder.pack(side="right", padx=5)

        self.btn_open_parquet = ctk.CTkButton(
            top_frame,
            text=tr("Open Parquet"),
            width=100,
            height=28,
            fg_color="#3498db",
            hover_color="#2980b9",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_open_parquet
        )
        self.btn_open_parquet.pack(side="right", padx=5)

        self.btn_convert = ctk.CTkButton(
            top_frame,
            text=tr("Convert Excel to Parquet"),
            width=160,
            height=28,
            fg_color="#9b59b6",
            hover_color="#8e44ad",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_convert_excel
        )
        self.btn_convert.pack(side="right", padx=5)

        # Feature Mode sor
        feature_frame = ctk.CTkFrame(frame, fg_color="transparent", height=40)
        feature_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(
            feature_frame,
            text=tr("Feature Mode:"),
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(0, 10))

        self.feature_var = ctk.StringVar(value="Original")

        self.radio_original = ctk.CTkRadioButton(
            feature_frame,
            text=tr("Original"),
            variable=self.feature_var,
            value="Original",
            command=self._on_feature_mode_change
        )
        self.radio_original.pack(side="left", padx=10)

        self.radio_forward = ctk.CTkRadioButton(
            feature_frame,
            text=tr("Forward Calc"),
            variable=self.feature_var,
            value="Forward Calc",
            command=self._on_feature_mode_change
        )
        self.radio_forward.pack(side="left", padx=10)

        self.radio_rolling = ctk.CTkRadioButton(
            feature_frame,
            text=tr("Rolling Window"),
            variable=self.feature_var,
            value="Rolling Window",
            command=self._on_feature_mode_change
        )
        self.radio_rolling.pack(side="left", padx=10)

        ctk.CTkButton(
            feature_frame,
            text="?",
            width=25,
            height=25,
            corner_radius=12,
            fg_color="#3498db",
            hover_color="#2980b9",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._show_feature_help
        ).pack(side="left", padx=10)

        self.feature_info_label = ctk.CTkLabel(
            feature_frame,
            text=tr("No additional features - raw data only"),
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.feature_info_label.pack(side="left", padx=20)

        # Fo terulet: bal - fajl lista, jobb - adat elonezet
        main_frame = ctk.CTkFrame(frame, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, pady=10)
        main_frame.grid_columnconfigure(0, weight=1, minsize=200)
        main_frame.grid_columnconfigure(1, weight=4)
        main_frame.grid_rowconfigure(0, weight=1)

        # Bal: Files lista
        files_frame = ctk.CTkFrame(main_frame, corner_radius=5)
        files_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        ctk.CTkLabel(
            files_frame,
            text=tr("Files:"),
            font=ctk.CTkFont(size=11),
            anchor="w"
        ).pack(anchor="w", padx=10, pady=5)

        self.files_textbox = ctk.CTkTextbox(
            files_frame,
            font=ctk.CTkFont(family="Consolas", size=10),
            fg_color="#1a1a2e"
        )
        self.files_textbox.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        # Jobb: Adat elonezet
        preview_frame = ctk.CTkFrame(main_frame, corner_radius=5)
        preview_frame.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(
            preview_frame,
            text=tr("Data Preview:"),
            font=ctk.CTkFont(size=11),
            anchor="w"
        ).pack(anchor="w", padx=10, pady=5)

        self.data_preview = ctk.CTkTextbox(
            preview_frame,
            font=ctk.CTkFont(family="Consolas", size=10),
            fg_color="#1a1a2e"
        )
        self.data_preview.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        return frame

    # === DATA TAB ESEMÉNYEK ===

    def _update_feature_info(self):
        """Update feature info label based on current mode."""
        mode = self.feature_var.get()
        info_texts = {
            "Original": tr("No additional features - raw data only"),
            "Forward Calc": tr("Features from entire history (expanding window)"),
            "Rolling Window": tr("Features from rolling 13-week window (dynamic)")
        }
        self.feature_info_label.configure(text=info_texts.get(mode, ""))

    def _on_feature_mode_change(self):
        """Feature mode váltás."""
        self.sound.play_button_click()
        mode = self.feature_var.get()
        self._update_feature_info()
        self._log(f"Feature mode: {mode}", "debug")
        self._save_settings_now()

        if self.raw_data is not None and not self.raw_data.empty:
            threading.Thread(target=self._recalculate_features, daemon=True).start()

    def _recalculate_features(self):
        """Feature-ök újraszámolása háttérszálon."""
        try:
            self.after(0, lambda: self._log(tr("Recalculating features...")))

            with self.data_lock:
                if self.processed_data is not None:
                    del self.processed_data
                    self.processed_data = None
            gc.collect()

            processed = DataProcessor.clean_data(self.raw_data.copy())
            processed = self._apply_feature_mode(processed)

            with self.data_lock:
                self.processed_data = processed

            feat_count = DataProcessor.get_feature_count(processed)
            if feat_count > 0:
                feat_msg = f"{tr('Features added:')} {feat_count} {tr('new columns')}"
                self.after(0, lambda m=feat_msg: self._log(m))

            self.after(0, self._update_data_ui)

        except Exception as e:
            self.after(0, lambda: self._log(f"{tr('Feature calculation error:')} {e}", "warning"))

    def _apply_feature_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature mode alkalmazása."""
        mode = self.feature_var.get()

        if mode == "Forward Calc":
            return DataProcessor.add_features_forward(df)
        if mode == "Rolling Window":
            return DataProcessor.add_features_rolling(df, window=13)
        return df

    def _on_open_folder(self):
        """Mappa megnyitása és fájlok betöltése."""
        self.sound.play_button_click()
        initial_dir = self.settings.get_last_folder() or None
        folder = filedialog.askdirectory(
            title="Select Data Folder",
            initialdir=initial_dir
        )
        if folder:
            self.settings.set_last_folder(folder)
            self._save_settings_now()
            self.folder_label.configure(text=folder)
            self._log(f"{tr('Loading folder:')} {folder}")
            threading.Thread(target=self._load_folder, args=(folder,), daemon=True).start()

    def _load_folder(self, folder: str):
        """Mappa betöltése háttérszálon."""
        try:
            self.after(0, lambda: self.set_progress(0.1))
            self.raw_data = DataLoader.load_folder(folder)

            if self.raw_data is not None and not self.raw_data.empty:
                self.after(0, lambda: self.set_progress(0.5))
                processed = DataProcessor.clean_data(self.raw_data)
                processed = self._apply_feature_mode(processed)

                with self.data_lock:
                    self.processed_data = processed

                self.after(0, lambda: self.set_progress(0.9))
                self.after(0, self._update_data_ui)

                summary = DataProcessor.get_data_summary(processed)
                msg = f"{tr('Loaded:')} {summary['rows']} {tr('rows')}, {summary['strategies']} {tr('strategies')}, {summary['files']} {tr('files')}"
                if summary['features'] > 0:
                    msg += f", {summary['features']} features"
                self.after(0, lambda m=msg: self._log(m))
                self.after(0, self.sound.play_model_complete)
            else:
                self.after(0, lambda: self._log(tr("No data found in folder."), "warning"))

            self.after(0, lambda: self.set_progress(0))

        except Exception as e:
            self.after(0, lambda: self._log(f"{tr('Error loading folder:')} {e}", "critical"))
            self.after(0, lambda: self.set_progress(0))

    def _on_open_parquet(self):
        """Parquet fájl(ok) megnyitása."""
        self.sound.play_button_click()
        initial_dir = self.settings.get_last_parquet_folder() or None
        files = filedialog.askopenfilenames(
            title="Select Parquet Files",
            filetypes=[("Parquet files", "*.parquet"), ("All files", "*.*")],
            initialdir=initial_dir
        )
        if files:
            self.settings.set_last_parquet_folder(os.path.dirname(files[0]))
            self._save_settings_now()
            self.folder_label.configure(text=f"{len(files)} {tr('parquet file(s) selected')}")
            self._log(f"{tr('Loading files...')} ({len(files)} parquet)")
            threading.Thread(target=self._load_parquet_files, args=(files,), daemon=True).start()

    def _load_parquet_files(self, files: tuple):
        """Parquet fájlok betöltése háttérszálon."""
        try:
            self.after(0, lambda: self.set_progress(0.1))
            self.raw_data = DataLoader.load_parquet_files(list(files))

            if self.raw_data is not None and not self.raw_data.empty:
                self.after(0, lambda: self.set_progress(0.5))
                processed = DataProcessor.clean_data(self.raw_data)
                processed = self._apply_feature_mode(processed)

                with self.data_lock:
                    self.processed_data = processed

                self.after(0, lambda: self.set_progress(0.9))
                self.after(0, self._update_data_ui)

                summary = DataProcessor.get_data_summary(processed)
                msg = f"{tr('Loaded:')} {summary['rows']} {tr('rows')}, {summary['strategies']} {tr('strategies')}"
                if summary['features'] > 0:
                    msg += f", {summary['features']} features"
                self.after(0, lambda m=msg: self._log(m))
                self.after(0, self.sound.play_model_complete)
            else:
                self.after(0, lambda: self._log(tr("No data in parquet files."), "warning"))

            self.after(0, lambda: self.set_progress(0))

        except Exception as e:
            self.after(0, lambda: self._log(f"{tr('Error loading parquet:')} {e}", "critical"))
            self.after(0, lambda: self.set_progress(0))

    def _on_convert_excel(self):
        """Excel fájlok Parquet-ba konvertálása."""
        self.sound.play_button_click()
        initial_dir = self.settings.get_last_convert_folder() or None
        folder = filedialog.askdirectory(
            title="Select folder with Excel files to convert",
            initialdir=initial_dir
        )
        if folder:
            self.settings.set_last_convert_folder(folder)
            self._save_settings_now()
            self._log(f"{tr('Converting Excel files in:')} {folder}")
            threading.Thread(target=self._convert_excel, args=(folder,), daemon=True).start()

    def _convert_excel(self, folder: str):
        """Excel konverzió háttérszálon."""
        try:
            self.after(0, lambda: self.set_progress(0.2))
            output_path, rows = DataLoader.convert_excel_to_parquet(folder)
            self.after(0, lambda: self.set_progress(1.0))

            filename = os.path.basename(output_path)
            msg = f"{tr('Conversion complete! Saved:')} {filename} ({rows} {tr('rows')})"
            self.after(0, lambda m=msg: self._log(m))
            self.after(0, self.sound.play_model_complete)
            self.after(500, lambda: self.set_progress(0))

        except Exception as e:
            self.after(0, lambda: self._log(f"{tr('Conversion error:')} {e}", "critical"))
            self.after(0, lambda: self.set_progress(0))

    def _update_data_ui(self):
        """Data tab UI frissítése betöltött adatokkal."""
        if self.processed_data is None or self.processed_data.empty:
            return

        self.files_textbox.delete("1.0", "end")
        if "SourceFile" in self.processed_data.columns:
            files = self.processed_data["SourceFile"].unique()
            for f in sorted(files):
                self.files_textbox.insert("end", f"- {f}\n")

        self.data_preview.delete("1.0", "end")
        cols = list(self.processed_data.columns)
        header = " | ".join(str(c)[:12].ljust(12) for c in cols[:10])
        if len(cols) > 10:
            header += " | ..."
        self.data_preview.insert("end", header + "\n")
        self.data_preview.insert("end", "-" * len(header) + "\n")

        for i in range(min(50, len(self.processed_data))):
            row = self.processed_data.iloc[i]
            values = [str(row[c])[:12].ljust(12) for c in cols[:10]]
            line = " | ".join(values)
            if len(cols) > 10:
                line += " | ..."
            self.data_preview.insert("end", line + "\n")

    def _show_feature_help(self):
        """Feature mode help popup megjelenítése."""
        self.sound.play_button_click()
        popup = ctk.CTkToplevel(self)
        popup.title(tr("Feature Mode - Help"))
        popup.geometry("500x400")
        popup.transient(self)
        popup.grab_set()

        text = ctk.CTkTextbox(popup, font=ctk.CTkFont(size=12))
        text.pack(fill="both", expand=True, padx=15, pady=15)

        help_text = """FEATURE MODES

1. ORIGINAL
   No additional features - uses only raw data columns.
   Best for: Quick analysis, baseline testing.
   Speed: Fastest

2. FORWARD CALC (Expanding Window)
   Features calculated from ENTIRE history up to each point.
   Features: weeks_count, active_ratio, profit_consistency,
   total_profit, cumulative_trades, volatility, sharpe_ratio, max_drawdown
   Best for: Strategy profiling, long-term patterns.
   Speed: Medium

3. ROLLING WINDOW (13 weeks)
   Features calculated from sliding 13-week window.
   Features: rolling_active_ratio, rolling_profit_consistency,
   rolling_sharpe, rolling_volatility, rolling_avg_profit,
   rolling_momentum_4w, rolling_momentum_13w, rolling_max_dd
   Best for: Capturing recent trends, time-series ML.
   Speed: Slowest
"""
        text.insert("1.0", help_text)
        text.configure(state="disabled")

        ctk.CTkButton(popup, text=tr("Close"), command=popup.destroy).pack(pady=10)
