"""
Analysis Tab - MBO Trading Strategy Analyzer
Mixin osztály az Analysis tab funkcionalitásához.
"""
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=too-many-statements,import-outside-toplevel

import customtkinter as ctk

from gui.translate import tr
from analysis.engine import get_resource_manager
from models import (
    get_categories,
    get_models_in_category,
    get_param_defaults,
    get_param_options,
    supports_gpu,
    supports_batch,
    supports_forward_calc,
    supports_rolling_window,
    supports_panel_mode,
    supports_dual_mode,
)


class AnalysisMixin:
    """Mixin az Analysis tab funkcionalitásához."""

    def _create_analysis_tab(self) -> ctk.CTkFrame:
        """Analysis tab - teljes implementáció a screenshot alapján."""
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")

        # === ROW 1: Fő vezérlők ===
        row1 = ctk.CTkFrame(frame, fg_color="transparent", height=40)
        row1.pack(fill="x", pady=(5, 3))

        # Auto gomb - kék
        self.btn_auto = ctk.CTkButton(
            row1,
            text="Auto",
            width=60,
            height=28,
            fg_color="#3498db",
            hover_color="#2980b9",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_auto_click
        )
        self.btn_auto.pack(side="left", padx=(0, 10))

        # Category dropdown
        ctk.CTkLabel(
            row1,
            text="Category:",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(0, 5))

        categories = get_categories()
        self.category_var = ctk.StringVar(value=categories[0] if categories else "")
        self.category_combo = ctk.CTkComboBox(
            row1,
            values=categories,
            variable=self.category_var,
            width=300,
            height=28,
            font=ctk.CTkFont(size=10),
            command=self._on_category_change
        )
        self.category_combo.pack(side="left", padx=(0, 15))

        # Model dropdown
        ctk.CTkLabel(
            row1,
            text="Model:",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(0, 5))

        initial_models = get_models_in_category(self.category_var.get()) if categories else []
        self.model_var = ctk.StringVar(value=initial_models[0] if initial_models else "")
        self.model_combo = ctk.CTkComboBox(
            row1,
            values=initial_models,
            variable=self.model_var,
            width=200,
            height=28,
            font=ctk.CTkFont(size=10),
            command=self._on_model_change
        )
        self.model_combo.pack(side="left", padx=(0, 5))

        # Help gomb (?) - sötét lila SOLID (mint az eredetiben)
        self.btn_model_help = ctk.CTkButton(
            row1,
            text="?",
            width=28,
            height=28,
            corner_radius=14,
            fg_color="#3d3d5c",
            hover_color="#4d4d6c",
            text_color="white",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._show_model_help
        )
        self.btn_model_help.pack(side="left", padx=5)

        # BATCH MODE gomb
        self.btn_batch_mode = ctk.CTkButton(
            row1,
            text="BATCH MODE",
            width=95,
            height=28,
            fg_color="#444444",
            hover_color="#555555",
            font=ctk.CTkFont(size=10, weight="bold"),
            state="disabled"
        )
        self.btn_batch_mode.pack(side="left", padx=(0, 10))

        # Jobb oldali kontrollok frame
        right_controls = ctk.CTkFrame(row1, fg_color="transparent")
        right_controls.pack(side="right")

        # Run Analysis gomb - zöld SOLID (mint az eredetiben)
        self.btn_run = ctk.CTkButton(
            right_controls,
            text="Run Analysis",
            width=180,
            height=28,
            fg_color="green",
            hover_color="#006400",
            text_color="white",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._on_run_analysis
        )
        self.btn_run.pack(side="right", padx=5)

        # Pause gomb - sárga SOLID (mint az eredetiben)
        self.btn_pause = ctk.CTkButton(
            right_controls,
            text="Pause",
            width=120,
            height=28,
            fg_color="#f39c12",
            hover_color="#d35400",
            text_color="white",
            text_color_disabled="#cccccc",
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled",
            command=self._on_pause_analysis
        )
        self.btn_pause.pack(side="right", padx=5)

        # Stop gomb - piros SOLID (mint az eredetiben)
        self.btn_stop = ctk.CTkButton(
            right_controls,
            text="Stop",
            width=140,
            height=28,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            text_color="white",
            text_color_disabled="#cccccc",
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled",
            command=self._on_stop_analysis
        )
        self.btn_stop.pack(side="right", padx=5)

        # Shutdown checkbox
        self.var_shutdown = ctk.BooleanVar(value=False)
        self.chk_shutdown = ctk.CTkCheckBox(
            right_controls,
            text="Shutdown",
            variable=self.var_shutdown,
            font=ctk.CTkFont(size=11),
            text_color="#e74c3c"
        )
        self.chk_shutdown.pack(side="right", padx=(0, 15))

        # === ROW 2: Teljesítmény vezérlők ===
        row2 = ctk.CTkFrame(frame, fg_color="transparent", height=40)
        row2.pack(fill="x", pady=3)

        # CPU Power slider - ResourceManager-ből olvassuk az értéket
        ctk.CTkLabel(
            row2,
            text="CPU Power:",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(0, 5))

        res_mgr = get_resource_manager()
        initial_cpu = res_mgr.cpu_percentage

        self.cpu_slider = ctk.CTkSlider(
            row2,
            from_=10,
            to=100,
            number_of_steps=18,
            width=150,
            command=self._on_cpu_change
        )
        self.cpu_slider.set(initial_cpu)
        self.cpu_slider.pack(side="left", padx=(0, 5))

        cores = res_mgr.physical_cores
        used_cores = res_mgr.get_n_jobs()
        self.cpu_label = ctk.CTkLabel(
            row2,
            text=f"{initial_cpu}% ({used_cores} cores)",
            font=ctk.CTkFont(size=11),
            width=100
        )
        self.cpu_label.pack(side="left", padx=(0, 20))

        # Use GPU switch - ResourceManager-ből olvassuk az értéket
        initial_gpu = res_mgr.gpu_enabled
        self.var_gpu = ctk.BooleanVar(value=initial_gpu)
        self.switch_gpu = ctk.CTkSwitch(
            row2,
            text="Use GPU",
            variable=self.var_gpu,
            font=ctk.CTkFont(size=11),
            command=self._on_gpu_toggle
        )
        if initial_gpu:
            self.switch_gpu.select()
        self.switch_gpu.pack(side="left", padx=(0, 20))

        # Panel Mode checkbox
        self.var_panel_mode = ctk.BooleanVar(value=False)
        self.chk_panel_mode = ctk.CTkCheckBox(
            row2,
            text="Panel Mode",
            variable=self.var_panel_mode,
            font=ctk.CTkFont(size=11),
            command=self._on_panel_mode_change
        )
        self.chk_panel_mode.pack(side="left", padx=(0, 3))

        ctk.CTkButton(
            row2, text="?", width=24, height=24, corner_radius=12,
            fg_color="#3d3d5c", hover_color="#4d4d6c",
            text_color="white",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._show_panel_help
        ).pack(side="left", padx=(0, 15))

        # Dual Model checkbox
        self.var_dual_model = ctk.BooleanVar(value=False)
        self.chk_dual_model = ctk.CTkCheckBox(
            row2,
            text="Dual Model",
            variable=self.var_dual_model,
            font=ctk.CTkFont(size=11),
            command=self._on_dual_model_change
        )
        self.chk_dual_model.pack(side="left", padx=(0, 5))

        ctk.CTkButton(
            row2, text="?", width=24, height=24, corner_radius=12,
            fg_color="#3d3d5c", hover_color="#4d4d6c",
            text_color="white",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._show_dual_help
        ).pack(side="left", padx=(0, 20))

        # Forecast Horizon
        ctk.CTkLabel(
            row2,
            text="Forecast Horizon:",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(0, 5))

        self.horizon_slider = ctk.CTkSlider(
            row2,
            from_=1,
            to=104,
            number_of_steps=103,
            width=150,
            command=self._on_horizon_change
        )
        self.horizon_slider.set(52)
        self.horizon_slider.pack(side="left", padx=(0, 5))

        self.horizon_label = ctk.CTkLabel(
            row2,
            text="52",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=30
        )
        self.horizon_label.pack(side="left", padx=(0, 20))

        # Time display
        self.time_label = ctk.CTkLabel(
            row2,
            text="Time: 00:00:00",
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold")
        )
        self.time_label.pack(side="right", padx=(0, 10))

        # === ROW 3: Paraméterek (dinamikus) ===
        self.params_frame = ctk.CTkFrame(frame, fg_color="transparent", height=50)
        self.params_frame.pack(fill="x", pady=5)

        self.param_widgets = {}  # Widget referenciák tárolása
        self._update_param_ui()  # Első model paramétereinek betöltése

        # === Model Documentation - keretes szekció ===
        doc_frame = ctk.CTkFrame(frame, corner_radius=8)
        doc_frame.pack(fill="both", expand=True, pady=(10, 5))

        doc_label = ctk.CTkLabel(
            doc_frame,
            text="Model Documentation",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        doc_label.pack(fill="x", padx=10, pady=(8, 5))

        self.model_doc_text = ctk.CTkTextbox(
            doc_frame,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#1a1a2e"
        )
        self.model_doc_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.model_doc_text.insert("1.0", tr("Select a model to see its documentation here."))
        self.model_doc_text.configure(state="disabled")

        # Induló állapot beállítása
        self._update_model_ui_state()

        return frame

    def _update_param_ui(self):
        """Paraméter UI frissítése az aktuális modell alapján."""
        # Régi widgetek törlése
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        self.param_widgets.clear()

        model_name = self.model_var.get()
        if not model_name:
            return

        defaults = get_param_defaults(model_name)
        options = get_param_options(model_name)

        if not defaults:
            ctk.CTkLabel(
                self.params_frame,
                text=tr("No configurable parameters for this model."),
                font=ctk.CTkFont(size=11),
                text_color="gray"
            ).pack(side="left")
            return

        # Paraméterek megjelenítése
        for key, default_val in defaults.items():
            # Label
            label_text = key.replace("_", " ").title() + ":"
            ctk.CTkLabel(
                self.params_frame,
                text=label_text,
                font=ctk.CTkFont(size=10)
            ).pack(side="left", padx=(0, 3))

            # Ha van option lista -> ComboBox, különben Entry
            if key in options:
                widget = ctk.CTkComboBox(
                    self.params_frame,
                    values=options[key],
                    width=80,
                    height=28
                )
                widget.set(default_val)
            else:
                widget = ctk.CTkEntry(
                    self.params_frame,
                    width=60,
                    height=28
                )
                widget.insert(0, default_val)

            widget.pack(side="left", padx=(0, 10))
            self.param_widgets[key] = widget

    def _update_model_ui_state(self):
        """Frissiti a UI allapotat az aktualis modell alapjan."""
        model_name = self.model_var.get()
        if not model_name:
            return

        # GPU switch allapot - csak ha a modell tamogatja
        if supports_gpu(model_name):
            self.switch_gpu.configure(state="normal")
        else:
            self.var_gpu.set(False)
            self.switch_gpu.configure(state="disabled")

        # Batch mode gomb - csak ha a modell tamogatja
        if supports_batch(model_name):
            self.btn_batch_mode.configure(state="normal", fg_color="#5d5d5d")
        else:
            self.btn_batch_mode.configure(state="disabled", fg_color="#444444")

        # Panel mode checkbox - csak ha a modell tamogatja
        if supports_panel_mode(model_name):
            self.chk_panel_mode.configure(state="normal")
        else:
            self.var_panel_mode.set(False)
            self.chk_panel_mode.configure(state="disabled")

        # Dual model checkbox - csak ha a modell tamogatja
        if supports_dual_mode(model_name):
            self.chk_dual_model.configure(state="normal")
        else:
            self.var_dual_model.set(False)
            self.chk_dual_model.configure(state="disabled")

    # === Analysis tab eseménykezelők ===

    def _on_category_change(self, category: str):
        """Kategória váltás."""
        self.sound.play_button_click()
        models = get_models_in_category(category)
        self.model_combo.configure(values=models)
        if models:
            self.model_var.set(models[0])
            self._on_model_change(models[0])

    def _on_model_change(self, _model_name: str):
        """Modell váltás."""
        self.sound.play_button_click()
        self._update_param_ui()
        self._update_model_ui_state()
        self._update_model_documentation()

    def _update_model_documentation(self):
        """Frissíti a model dokumentációt."""
        model_name = self.model_var.get()
        if not model_name:
            return

        from models import get_model_info
        info = get_model_info(model_name)

        if info:
            doc_text = f"=== {info.name} ===\n\n"
            doc_text += f"Category: {info.category}\n"
            doc_text += f"GPU Support: {'Yes' if info.supports_gpu else 'No'}\n"
            doc_text += f"Batch Mode: {'Yes' if info.supports_batch else 'No'}\n\n"

            defaults = get_param_defaults(model_name)
            if defaults:
                doc_text += "Default Parameters:\n"
                for k, v in defaults.items():
                    doc_text += f"  - {k}: {v}\n"

            self.model_doc_text.configure(state="normal")
            self.model_doc_text.delete("1.0", "end")
            self.model_doc_text.insert("1.0", doc_text)
            self.model_doc_text.configure(state="disabled")

    def _on_auto_click(self):
        """Auto Execution ablak megnyitása."""
        self.sound.play_button_click()
        self._log("Auto Execution window - coming soon...")

    def _on_run_analysis(self):
        """Elemzes inditasa."""
        self.sound.play_button_click()

        if self.processed_data is None or self.processed_data.empty:
            self._log(tr("Please load data first!"), "warning")
            return

        model_name = self.model_var.get()

        # Feature mode kompatibilitas ellenorzes
        feature_mode = self.feature_var.get() if hasattr(self, 'feature_var') else "Original"
        if not self._check_feature_mode_compatibility(model_name, feature_mode):
            return  # Figyelmeztetett es visszaterunk

        self._log(f"Starting analysis with {model_name}...")

        # UI allapot frissitese
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_pause.configure(state="normal")

        # Parameterek osszegyujtese
        params = self._collect_params()
        horizon = int(self.horizon_slider.get())
        use_gpu = self.var_gpu.get() if hasattr(self, 'var_gpu') else False
        use_batch = False  # Egyelore nem batch mod

        # Analysis engine letrehozasa
        from analysis.engine import AnalysisEngine, AnalysisContext
        import threading

        self._analysis_engine = AnalysisEngine()
        self._analysis_engine.set_progress_callback(self._on_analysis_progress)

        context = AnalysisContext(
            model_name=model_name,
            params=params,
            forecast_horizon=horizon,
            use_gpu=use_gpu,
            use_batch=use_batch
        )

        # Hatterszalon inditasa
        self._analysis_start_time = __import__('time').time()
        self._analysis_thread = threading.Thread(
            target=self._run_analysis_thread,
            args=(context,),
            daemon=True
        )
        self._analysis_thread.start()

        # Timer inditasa az idokijelzeshez
        self._start_time_timer()

    def _collect_params(self) -> dict:
        """Osszegyujti a parametereket a GUI-bol."""
        params = {}
        for param_name, widget in self.param_widgets.items():
            if hasattr(widget, 'get'):
                params[param_name] = widget.get()
        return params

    def _run_analysis_thread(self, context):
        """Hatterszalon az elemzeshez."""
        try:
            with self.data_lock:
                data = self.processed_data.copy() if self.processed_data is not None else None

            if data is None:
                self._log("No data available!", "error")
                return

            results = self._analysis_engine.run(data, context)

            # Eredmenyek feldolgozasa a fo szalon
            self.after(0, lambda: self._on_analysis_complete(results))

        except Exception as e:
            self._log(f"Analysis error: {e}", "error")
            import traceback
            traceback.print_exc()
            self.after(0, self._reset_analysis_ui)

    def _on_analysis_progress(self, progress):
        """Progress callback - fo szalra atadashoz."""
        # A progress callback mas szalrol jon, atiranyitjuk a fo szalra
        try:
            self.after(0, lambda p=progress: self._update_analysis_progress(p))
        except Exception:
            pass  # GUI mar bezarodhatott

    def _update_analysis_progress(self, progress):
        """Progress frissites a fo szalon."""
        if progress.total_strategies > 0:
            pct = progress.completed_strategies / progress.total_strategies

            # Progress bar frissitese (0.0 - 1.0)
            self.set_progress(pct)

            # Debug level - csak file-ba kerul, GUI-ban nem jelenik meg
            self._log(
                f"Progress: {progress.completed_strategies}/{progress.total_strategies} "
                f"({pct*100:.1f}%) - {progress.current_strategy}",
                "debug"
            )

    def _on_analysis_complete(self, results):
        """Elemzes befejezese."""
        self._stop_time_timer()

        success_count = sum(1 for r in results.values() if r.success)
        total_count = len(results)

        self._log(f"Analysis complete: {success_count}/{total_count} strategies succeeded")

        if results:
            # Eredmenyek tarolasa a Results tab-hez
            self._analysis_results = results
            self._log("Results available in Results tab")

            # Eredmenyek konvertalasa Results tab formatumba
            self._convert_results_to_results_tab(results)

        self._reset_analysis_ui()
        self.sound.play_model_complete()

    def _convert_results_to_results_tab(self, results):
        """
        Konvertalja az AnalysisResult dict-et a Results tab altal vart formatumba.
        Aktivalja a Results tab gombjait.
        """
        import pandas as pd
        import time
        import os

        # Sikeres eredmenyek kiszurese
        successful_results = {k: v for k, v in results.items() if v.success}

        if not successful_results:
            self._log("No successful results to display.", "warning")
            return

        # DataFrame keszitese az eredmenyekbol
        rows = []
        for strat_id, result in successful_results.items():
            forecasts = result.forecasts if result.forecasts else []

            # Forecast osszegzesek szamitasa (kumulativ)
            forecast_1w = sum(forecasts[:1]) if len(forecasts) >= 1 else 0
            forecast_1m = sum(forecasts[:4]) if len(forecasts) >= 4 else sum(forecasts)
            forecast_3m = sum(forecasts[:13]) if len(forecasts) >= 13 else sum(forecasts)
            forecast_6m = sum(forecasts[:26]) if len(forecasts) >= 26 else sum(forecasts)
            forecast_12m = sum(forecasts[:52]) if len(forecasts) >= 52 else sum(forecasts)

            rows.append({
                "No.": int(strat_id) if str(strat_id).isdigit() else strat_id,
                "Forecast_1W": round(forecast_1w, 2),
                "Forecast_1M": round(forecast_1m, 2),
                "Forecast_3M": round(forecast_3m, 2),
                "Forecast_6M": round(forecast_6m, 2),
                "Forecast_12M": round(forecast_12m, 2),
                "Method": self.model_var.get(),
                "Forecasts": forecasts,
                "Elapsed_ms": result.elapsed_ms
            })

        results_df = pd.DataFrame(rows)

        # Rendezes Forecast_1M szerint (csokkeno)
        results_df = results_df.sort_values("Forecast_1M", ascending=False).reset_index(drop=True)

        # Legjobb strategia azonositasa
        best_strat_id = results_df.iloc[0]["No."] if not results_df.empty else None
        best_forecasts = results_df.iloc[0]["Forecasts"] if not results_df.empty else None

        # Best strategy data keszitese (torteneti adat)
        best_strat_data = None
        all_strategies_data = {}

        if self.processed_data is not None and best_strat_id is not None:
            try:
                # Legjobb strategia torteneti adatainak kinyerese
                strat_data = self.processed_data[
                    self.processed_data["No."] == best_strat_id
                ].copy()
                if not strat_data.empty:
                    best_strat_data = strat_data

                # OSSZES strategia torteneti adatainak mentese (nem csak top 20!)
                # Ez szukseges a teljes .pkl fajlhoz
                for strategy_id in results_df["No."].values:
                    s_data = self.processed_data[
                        self.processed_data["No."] == strategy_id
                    ].copy()
                    if not s_data.empty:
                        all_strategies_data[strategy_id] = s_data
            except (KeyError, ValueError) as e:
                self._log(f"Warning: Could not extract strategy data: {e}")

        # Filename base keszitese
        filename_base = "Unknown"
        if self.processed_data is not None and "SourceFile" in self.processed_data.columns:
            source_files = self.processed_data["SourceFile"].unique()
            if len(source_files) > 0:
                # Elso fajlnev alapjan (kiterjesztes nelkul)
                first_file = os.path.basename(str(source_files[0]))
                filename_base = os.path.splitext(first_file)[0]

        # Execution time szamitasa
        elapsed = time.time() - getattr(self, '_analysis_start_time', time.time())
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        execution_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # last_results dict letrehozasa
        self.last_results = {
            "method": self.model_var.get(),
            "filename_base": filename_base,
            "best_strat_id": best_strat_id,
            "best_strat_data": best_strat_data,
            "results": results_df,
            "forecast_values": best_forecasts,
            "all_strategies_data": all_strategies_data,
            "params": self._collect_params(),
            "ranking_mode": "forecast",
            "execution_time": execution_time,
            "version": "1.0"
        }

        # Results tab allapot frissitese
        self.results_df = results_df
        self.current_ranking_mode = "forecast"
        self.results_with_stability = None

        # Results tab megjelenites frissitese
        if hasattr(self, '_update_results_display'):
            self._update_results_display(results_df, "Forecast_1M")

        # Results tab gombok aktivalasa
        if hasattr(self, 'btn_generate_report'):
            self.btn_generate_report.configure(state="normal")
        if hasattr(self, 'btn_export_csv'):
            self.btn_export_csv.configure(state="normal")
        if hasattr(self, '_enable_ranking_controls'):
            self._enable_ranking_controls()

        # Info label frissitese
        if hasattr(self, 'results_info_label'):
            self.results_info_label.configure(
                text=f"Analysis complete: {len(results_df)} strategies | Model: {self.model_var.get()}",
                text_color="#2ecc71"
            )

        self._log(f"Results ready: {len(results_df)} strategies. Buttons activated in Results tab.")

        # Automatikus pickle mentes a Data_raw konyvtarba
        self._save_analysis_state_pickle(filename_base)

    def _save_analysis_state_pickle(self, filename_base: str):
        """
        Automatikusan menti az analysis state-et pickle fajlba a Data_raw konyvtarba.
        A fajlnev formatum: {timestamp}_{method}_{currency_pair}.pkl
        Pelda: 20260113_145629_GradientBoosting_EURJPY.pkl
        """
        import pickle
        import os
        from datetime import datetime

        if not hasattr(self, 'last_results') or not self.last_results:
            return

        try:
            # Data_raw konyvtar a working directory-ban (fokonyvtar)
            # Ugyanugy mint a regi kodban: os.getcwd() + "Data_raw"
            data_raw_dir = os.path.join(os.getcwd(), "Data_raw")

            # Ha nincs, letrehozzuk
            if not os.path.exists(data_raw_dir):
                os.makedirs(data_raw_dir)
                self._log(f"Created Data_raw directory: {data_raw_dir}")

            # Fajlnev generalasa - regi formatum: {timestamp}_{method}_{currency_pair}.pkl
            method = self.last_results.get("method", "Unknown")
            # Currency pair kinyerese a filename_base-bol (pl. "EURJPY_D1" -> "EURJPY")
            currency_pair = filename_base.split("_")[0] if "_" in filename_base else filename_base
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pkl_filename = f"{timestamp}_{method}_{currency_pair}.pkl"
            pkl_path = os.path.join(data_raw_dir, pkl_filename)

            # Mentes
            with open(pkl_path, 'wb') as f:
                pickle.dump(self.last_results, f)

            # Fajlmeret kijelzese
            file_size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
            self._log(f"Analysis state saved: Data_raw/{pkl_filename} ({file_size_mb:.2f} MB)")

        except (OSError, IOError, pickle.PicklingError) as e:
            self._log(f"Warning: Could not save analysis state: {e}", "warning")

    def _reset_analysis_ui(self):
        """UI visszaallitasa elemzes utan."""
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_pause.configure(state="disabled")
        self.btn_pause.configure(text="Pause", fg_color="#f39c12")
        # Progress bar nullazasa
        self.set_progress(0)

    def _start_time_timer(self):
        """Idozito inditasa."""
        self._time_timer_running = True
        self._update_time_display()

    def _stop_time_timer(self):
        """Idozito leallitasa."""
        self._time_timer_running = False

    def _update_time_display(self):
        """Ido kijelzes frissitese."""
        if not getattr(self, '_time_timer_running', False):
            return

        import time
        elapsed = time.time() - getattr(self, '_analysis_start_time', time.time())
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        self.time_label.configure(text=f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}")

        # Kovetkezo frissites 1 mp mulva
        self.after(1000, self._update_time_display)

    def _check_feature_mode_compatibility(self, model_name: str, feature_mode: str) -> bool:
        """
        Ellenorzi a feature mode kompatibilitast es figyelmeztet ha szukseges.

        Args:
            model_name: Modell neve
            feature_mode: Feature mod (Original, Forward Calc, Rolling Window)

        Returns:
            True ha kompatibilis vagy a felhasznalo elfogadja, False ha visszavon
        """
        # Ellenorizzuk, hogy a modell tamogatja-e a feature modot
        if feature_mode == "Forward Calc" and not supports_forward_calc(model_name):
            warning_msg = (
                f"WARNING: {model_name} does not support Forward Calc mode!\n\n"
                f"This model works best with raw (Original) data.\n"
                f"Forward Calc features may distort the forecast results.\n\n"
                f"Please switch to 'Original' mode in the Data Loading tab\n"
                f"for optimal results."
            )
            self._log(warning_msg, "warning")
            self._show_feature_warning_popup(model_name, feature_mode, warning_msg)
            return False

        if feature_mode == "Rolling Window" and not supports_rolling_window(model_name):
            warning_msg = (
                f"WARNING: {model_name} does not support Rolling Window mode!\n\n"
                f"This model works best with raw (Original) data.\n"
                f"Rolling Window features may hide important patterns.\n\n"
                f"Please switch to 'Original' mode in the Data Loading tab\n"
                f"for optimal results."
            )
            self._log(warning_msg, "warning")
            self._show_feature_warning_popup(model_name, feature_mode, warning_msg)
            return False

        return True

    def _show_feature_warning_popup(self, model_name: str, feature_mode: str, message: str):
        """Feature mode figyelmeztetest megjelenito popup."""
        popup = ctk.CTkToplevel(self)
        popup.title(f"{model_name} - Feature Mode Warning")
        popup.geometry("500x300")
        popup.transient(self)
        popup.grab_set()

        # Warning icon and title
        title_frame = ctk.CTkFrame(popup, fg_color="transparent")
        title_frame.pack(fill="x", padx=20, pady=(20, 10))

        ctk.CTkLabel(
            title_frame,
            text="Feature Mode Incompatibility",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#e74c3c"
        ).pack()

        # Message
        text = ctk.CTkTextbox(popup, font=ctk.CTkFont(size=12), height=150)
        text.pack(fill="both", expand=True, padx=20, pady=10)
        text.insert("1.0", message)
        text.configure(state="disabled")

        # Buttons
        btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(10, 20))

        ctk.CTkButton(
            btn_frame,
            text="OK - I understand",
            width=150,
            fg_color="#3498db",
            command=popup.destroy
        ).pack(side="right", padx=5)

    def _on_stop_analysis(self):
        """Elemzés leállítása."""
        self.sound.play_button_click()
        self._log("Stopping analysis...")

        # Engine leallitasa
        if hasattr(self, '_analysis_engine') and self._analysis_engine:
            self._analysis_engine.cancel()

        self._stop_time_timer()
        self._reset_analysis_ui()

    def _on_pause_analysis(self):
        """Elemzés szüneteltetése/folytatása."""
        self.sound.play_button_click()
        if self.btn_pause.cget("text") == "Pause":
            # Pause
            if hasattr(self, '_analysis_engine') and self._analysis_engine:
                self._analysis_engine.pause()

            self.btn_pause.configure(
                text="Resume", fg_color="#27ae60", hover_color="#2ecc71"
            )
            self._log("Analysis paused.")
        else:
            # Resume
            if hasattr(self, '_analysis_engine') and self._analysis_engine:
                self._analysis_engine.resume()

            self.btn_pause.configure(
                text="Pause", fg_color="#f39c12", hover_color="#d35400"
            )
            self._log("Analysis resumed.")

    def _on_cpu_change(self, value: float):
        """CPU slider változás - szinkronizálás ResourceManager-rel."""
        res_mgr = get_resource_manager()
        pct = int(value)

        # ResourceManager frissítése
        res_mgr.set_cpu_percentage(pct)

        # UI frissítése
        used_cores = res_mgr.get_n_jobs()
        self.cpu_label.configure(text=f"{pct}% ({used_cores} cores)")

    def _on_gpu_toggle(self):
        """GPU toggle - szinkronizálás ResourceManager-rel."""
        self.sound.play_toggle_switch()

        res_mgr = get_resource_manager()
        enabled = self.var_gpu.get()

        # ResourceManager frissítése
        res_mgr.set_gpu_enabled(enabled)

        # Státusz log
        if enabled and not res_mgr.gpu_available:
            self._log("GPU not available on this system", "warning")
            self.var_gpu.set(False)
            self.switch_gpu.deselect()
        else:
            state = "enabled" if res_mgr.gpu_enabled else "disabled"
            gpu_info = f" ({res_mgr.gpu_name})" if res_mgr.gpu_name else ""
            self._log(f"GPU {state}{gpu_info}")

    def _on_panel_mode_change(self):
        """Panel mode váltás."""
        if self.var_panel_mode.get():
            self.sound.play_checkbox_on()
            if self.var_dual_model.get():
                self.var_dual_model.set(False)

    def _on_dual_model_change(self):
        """Dual model váltás."""
        if self.var_dual_model.get():
            self.sound.play_checkbox_on()
            if self.var_panel_mode.get():
                self.var_panel_mode.set(False)

    def _on_horizon_change(self, value: float):
        """Forecast horizon változás."""
        self.horizon_label.configure(text=str(int(value)))

    def _show_model_help(self):
        """Model help popup."""
        self.sound.play_button_click()
        model_name = self.model_var.get()
        self._log(f"Help for {model_name} - coming soon...")

    def _show_panel_help(self):
        """Panel Mode help popup."""
        self.sound.play_button_click()
        popup = ctk.CTkToplevel(self)
        popup.title("Panel Mode - Help")
        popup.geometry("500x300")
        popup.transient(self)
        popup.grab_set()

        text = ctk.CTkTextbox(popup, font=ctk.CTkFont(size=12))
        text.pack(fill="both", expand=True, padx=15, pady=15)
        text.insert("1.0", """PANEL MODE

Panel mode trains a SINGLE model on ALL strategies at once,
instead of training separate models for each strategy.

Benefits:
- 5-10x faster execution
- Good for quick prototyping
- Works well when strategies are similar

Supported models:
- Random Forest
- XGBoost
- LightGBM
- Gradient Boosting
- KNN Regressor

Note: Panel and Dual modes are mutually exclusive.""")
        text.configure(state="disabled")

        ctk.CTkButton(popup, text="Close", command=popup.destroy).pack(pady=10)

    def _show_dual_help(self):
        """Dual Model help popup."""
        self.sound.play_button_click()
        popup = ctk.CTkToplevel(self)
        popup.title("Dual Model - Help")
        popup.geometry("500x350")
        popup.transient(self)
        popup.grab_set()

        text = ctk.CTkTextbox(popup, font=ctk.CTkFont(size=12))
        text.pack(fill="both", expand=True, padx=15, pady=15)
        text.insert("1.0", """DUAL MODEL MODE

Dual model trains TWO separate models:
1. Activity Model - Predicts trading activity (classification)
2. Profit Model - Predicts profit for active weeks (regression)

Final forecast = Activity × Weeks × Profit_per_active_week

Benefits:
- Better handling of intermittent strategies
- Explicit activity prediction
- More detailed analysis

Supported models:
- Random Forest
- XGBoost
- LightGBM
- Gradient Boosting
- KNN Regressor

Note: Panel and Dual modes are mutually exclusive.""")
        text.configure(state="disabled")

        ctk.CTkButton(popup, text="Close", command=popup.destroy).pack(pady=10)
