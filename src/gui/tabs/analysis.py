"""
Analysis Tab - MBO Trading Strategy Analyzer
Mixin osztály az Analysis tab funkcionalitásához.

Multiprocessing architecture for GUI responsiveness:
- Analysis runs in separate process (worker.py)
- GUI polls results via Queue (no GIL blocking)
- Complete separation of computation and display
"""
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=too-many-statements,import-outside-toplevel

import logging
import customtkinter as ctk

from gui.translate import tr
from analysis.engine import get_resource_manager
from analysis.worker import AnalysisWorkerManager, WorkerProgress, WorkerResult
from models import (
    get_all_models,
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

        # BATCH MODE toggle gomb
        self.var_batch_mode = False  # Batch mode allapot
        self.btn_batch_mode = ctk.CTkButton(
            row1,
            text="BATCH MODE",
            width=95,
            height=28,
            fg_color="#444444",
            hover_color="#555555",
            font=ctk.CTkFont(size=10, weight="bold"),
            state="disabled",
            command=self._on_batch_mode_toggle
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

        # Shutdown checkbox - aktuális model után
        self.var_shutdown = ctk.BooleanVar(value=False)
        self.chk_shutdown = ctk.CTkCheckBox(
            right_controls,
            text="Shutdown",
            variable=self.var_shutdown,
            font=ctk.CTkFont(size=11),
            text_color="#e74c3c",
            command=self._on_shutdown_checkbox_change
        )
        self.chk_shutdown.pack(side="right", padx=(0, 15))

        # Shutdown állapotok inicializálása
        self._shutdown_immediate = False  # Aktuális model után
        self._shutdown_after_all = False  # Teljes queue után (auto)

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
            self.btn_batch_mode.configure(state="normal")
            # Ha mar aktiv volt, megtartjuk a zold szint
            if getattr(self, 'var_batch_mode', False):
                self.btn_batch_mode.configure(fg_color="#27ae60", text="BATCH: ON")
            else:
                self.btn_batch_mode.configure(fg_color="#5d5d5d", text="BATCH MODE")
        else:
            # Nem tamogatott - kikapcsoljuk es letiltjuk
            self.var_batch_mode = False
            self.btn_batch_mode.configure(
                state="disabled",
                fg_color="#444444",
                text="BATCH MODE"
            )

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
        self._open_auto_window()

    def _open_auto_window(self):
        """Auto Execution Window megnyitása vagy előtérbe hozása."""
        from gui.auto_window import AutoExecutionWindow

        # Ha már létezik az ablak, csak megjelenítjük
        if hasattr(self, '_auto_window') and self._auto_window is not None:
            try:
                if self._auto_window.winfo_exists():
                    self._auto_window.show()
                    return
            except Exception:
                pass  # Ablak megsemmisült, újat hozunk létre

        # Új ablak létrehozása
        self._auto_window = AutoExecutionWindow(self)
        self._log("Auto Execution Manager opened")

    def _on_run_analysis(self):
        """Elemzes inditasa - MULTIPROCESSING alapú implementáció."""
        self.sound.play_button_click()

        if self.processed_data is None or self.processed_data.empty:
            self._log(tr("Please load data first!"), "warning")
            return

        model_name = self.model_var.get()

        # Feature mode kompatibilitas ellenorzes
        feature_mode = self.feature_var.get() if hasattr(self, 'feature_var') else "Original"
        if not self._check_feature_mode_compatibility(model_name, feature_mode):
            return  # Figyelmeztetett es visszaterunk

        # Batch mode allapot log
        batch_status = "BATCH MODE" if getattr(self, 'var_batch_mode', False) else "Single Mode"
        self._log(f"Starting analysis with {model_name} ({batch_status})...", "success")

        # UI allapot frissitese
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_pause.configure(state="disabled")  # Pause nem támogatott multiprocessing módban

        # Parameterek osszegyujtese
        params = self._collect_params()
        horizon = int(self.horizon_slider.get())
        use_gpu = self.var_gpu.get() if hasattr(self, 'var_gpu') else False
        use_batch = getattr(self, 'var_batch_mode', False)  # Batch mode a gombbol
        use_panel = self.var_panel_mode.get() if hasattr(self, 'var_panel_mode') else False
        use_dual = self.var_dual_model.get() if hasattr(self, 'var_dual_model') else False

        # Context dict a worker számára (pickle-elhető)
        context_dict = {
            'model_name': model_name,
            'params': params,
            'forecast_horizon': horizon,
            'use_gpu': use_gpu,
            'use_batch': use_batch,
            'panel_mode': use_panel,
            'dual_model': use_dual
        }

        # Environment variables a worker számára
        res_mgr = get_resource_manager()
        env_vars = res_mgr.get_env_vars()

        # Log throttling reset
        self._last_logged_percent = -10
        self._analysis_running = True

        # Worker manager létrehozása és indítása
        if not hasattr(self, '_worker_manager') or self._worker_manager is None:
            self._worker_manager = AnalysisWorkerManager()

        # Data másolat a worker-nek
        with self.data_lock:
            data_copy = self.processed_data.copy()

        self._analysis_start_time = __import__('time').time()
        self._worker_manager.start(data_copy, context_dict, env_vars)

        # Timer inditasa az idokijelzeshez
        self._start_time_timer()

        # GUI progress poller indítása (poll-olja a Queue-t)
        self._start_progress_poller()

    def _collect_params(self) -> dict:
        """Osszegyujti a parametereket a GUI-bol."""
        params = {}
        for param_name, widget in self.param_widgets.items():
            if hasattr(widget, 'get'):
                params[param_name] = widget.get()
        return params

    def _start_progress_poller(self):
        """GUI progress poller indítása - ez fut a fő szálon, periodikusan."""
        self._progress_poller_running = True
        self._poll_progress()

    def _stop_progress_poller(self):
        """GUI progress poller leállítása."""
        self._progress_poller_running = False

    def _poll_progress(self):
        """
        Progress lekérdezése a worker Queue-ból - FŐ SZÁLON fut periodikusan.

        MULTIPROCESSING ARCHITEKTÚRA:
        - A worker process külön fut, teljesen független a GUI-tól
        - A progress a Queue-n keresztül érkezik (nem blocking)
        - A GUI szál SOHA nem vár a worker-re
        """
        if not getattr(self, '_progress_poller_running', False):
            return

        worker = getattr(self, '_worker_manager', None)
        if worker is None:
            return

        # 1. Ellenőrizzük, van-e result (befejezés)
        result = worker.get_result()
        if result is not None:
            self._on_worker_complete(result)
            return

        # 2. Progress lekérdezése (non-blocking)
        progress = worker.get_progress()

        if progress is not None and progress.total_strategies > 0:
            pct = progress.completed_strategies / progress.total_strategies

            # Progress bar frissítése (0.0 - 1.0)
            self.set_progress(pct)

            # Log CSAK 10%-onként
            pct_int = int(pct * 100)
            if not hasattr(self, '_last_logged_percent'):
                self._last_logged_percent = -10

            if pct_int >= self._last_logged_percent + 10:
                self._last_logged_percent = pct_int
                self._log(
                    f"Progress: {progress.completed_strategies}/{progress.total_strategies} "
                    f"({pct_int}%)",
                    "success"
                )

        # 3. Ellenőrizzük, fut-e még a worker
        if not worker.is_running and getattr(self, '_analysis_running', False):
            # Worker leállt result nélkül - hiba
            self._log("Worker process terminated unexpectedly", "error")
            self._reset_analysis_ui()
            return

        # 4. Következő poll ütemezése (250ms interval)
        if getattr(self, '_analysis_running', False):
            try:
                self.after(250, self._poll_progress)
            except Exception:
                pass  # GUI már bezáródhatott

    def _on_worker_complete(self, result: WorkerResult):
        """Worker befejezésekor hívódik - eredmények feldolgozása."""
        from analysis.engine import AnalysisResult

        self._stop_time_timer()
        self._stop_progress_poller()
        self._analysis_running = False

        if not result.success:
            self._log(f"Analysis error: {result.error}", "error")
            self._reset_analysis_ui()
            return

        # Konvertálás AnalysisResult formátumba (kompatibilitás)
        results = {}
        for strategy_id, data in result.results.items():
            results[strategy_id] = AnalysisResult(
                strategy_id=strategy_id,
                forecasts=data['forecasts'],
                elapsed_ms=data['elapsed_ms'],
                success=data['success'],
                error=data.get('error')
            )

        # Eredmények feldolgozása (ugyanaz mint korábban)
        self._on_analysis_complete(results)

    def _on_analysis_complete(self, results):
        """Elemzes befejezese."""
        self._stop_time_timer()
        self._stop_progress_poller()
        self._analysis_running = False

        success_count = sum(1 for r in results.values() if r.success)
        total_count = len(results)

        self._log(f"Analysis complete: {success_count}/{total_count} strategies succeeded")

        if results:
            # Eredmenyek tarolasa a Results tab-hez
            self._analysis_results = results
            self._log("Results available in Results tab")

            # Eredmenyek konvertalasa Results tab formatumba
            self._convert_results_to_results_tab(results)

        # Auto mode ellenőrzés - ha auto futás, következő modell
        if getattr(self, '_is_auto_running', False):
            self.sound.play_model_complete()
            # Immediate shutdown ellenőrzés auto módban is
            if getattr(self, '_shutdown_immediate', False):
                self._log("Immediate shutdown requested - stopping auto execution", "critical")
                self._is_auto_running = False
                self._finish_auto_sequence()
                self._execute_shutdown()
                return
            self._on_auto_model_complete()
            return  # Ne állítsuk vissza a UI-t, mert folytatódik

        self._reset_analysis_ui()
        self.sound.play_model_complete()

        # Immediate shutdown ellenőrzés normál módban
        if getattr(self, '_shutdown_immediate', False):
            self._execute_shutdown()

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
        # Progress poller leállítása
        self._stop_progress_poller()
        self._analysis_running = False

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

        # GUI események feldolgozása
        try:
            self.update()
        except Exception:
            pass

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

        # Auto mode leállítása
        if getattr(self, '_is_auto_running', False):
            self.stop_auto_execution()
            return

        self._log("Stopping analysis...")

        # Worker process leállítása
        if hasattr(self, '_worker_manager') and self._worker_manager:
            self._worker_manager.stop()

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

    def _on_batch_mode_toggle(self):
        """Batch mode toggle - gomb allapotanak valtoztatasa."""
        self.sound.play_button_click()

        # Allapot valtas
        self.var_batch_mode = not self.var_batch_mode

        # Vizualis visszajelzes a gomb szinevel
        if self.var_batch_mode:
            # Aktiv batch mode - zold szin
            self.btn_batch_mode.configure(
                fg_color="#27ae60",
                hover_color="#2ecc71",
                text="BATCH: ON"
            )
            self._log("Batch mode ENABLED - will process strategies in parallel")
        else:
            # Inaktiv batch mode - sotet szurke
            self.btn_batch_mode.configure(
                fg_color="#5d5d5d",
                hover_color="#6d6d6d",
                text="BATCH MODE"
            )
            self._log("Batch mode DISABLED - will process strategies sequentially")

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
        popup.geometry("500x350")
        popup.transient(self)
        popup.grab_set()

        # Dinamikusan lekérjük a Panel módot támogató modelleket
        panel_models = [m for m in get_all_models() if supports_panel_mode(m)]
        model_list = "\n".join(f"- {m}" for m in panel_models) if panel_models else "- None"

        text = ctk.CTkTextbox(popup, font=ctk.CTkFont(size=12))
        text.pack(fill="both", expand=True, padx=15, pady=15)
        text.insert("1.0", f"""PANEL MODE

Panel mode trains a SINGLE model on ALL strategies at once,
instead of training separate models for each strategy.

Benefits:
- 5-10x faster execution
- Good for quick prototyping
- Works well when strategies are similar

Supported models:
{model_list}

Note: Panel and Dual modes are mutually exclusive.""")
        text.configure(state="disabled")

        ctk.CTkButton(popup, text="Close", command=popup.destroy).pack(pady=10)

    def _show_dual_help(self):
        """Dual Model help popup."""
        self.sound.play_button_click()
        popup = ctk.CTkToplevel(self)
        popup.title("Dual Model - Help")
        popup.geometry("500x400")
        popup.transient(self)
        popup.grab_set()

        # Dinamikusan lekérjük a Dual módot támogató modelleket
        dual_models = [m for m in get_all_models() if supports_dual_mode(m)]
        model_list = "\n".join(f"- {m}" for m in dual_models) if dual_models else "- None"

        text = ctk.CTkTextbox(popup, font=ctk.CTkFont(size=12))
        text.pack(fill="both", expand=True, padx=15, pady=15)
        text.insert("1.0", f"""DUAL MODEL MODE

Dual model trains TWO separate models:
1. Activity Model - Predicts trading activity (classification)
2. Profit Model - Predicts profit for active weeks (regression)

Final forecast = Activity × Weeks × Profit_per_active_week

Benefits:
- Better handling of intermittent strategies
- Explicit activity prediction
- More detailed analysis

Supported models:
{model_list}

Note: Panel and Dual modes are mutually exclusive.""")
        text.configure(state="disabled")

        ctk.CTkButton(popup, text="Close", command=popup.destroy).pack(pady=10)

    # === AUTO EXECUTION METHODS ===

    def run_auto_sequence(self, execution_list: list, shutdown_after_all: bool = False):
        """
        Auto execution sequence indítása.

        Args:
            execution_list: Futtatandó modellek listája
            shutdown_after_all: Shutdown a teljes queue után (deprecated - use checkbox)
        """
        if not execution_list:
            self._log("Auto execution: Empty queue!", "warning")
            return

        if self.processed_data is None or self.processed_data.empty:
            self._log("Auto execution: No data loaded!", "warning")
            return

        # Ha a paraméterben True, akkor azt használjuk (backward compatibility)
        # Egyébként az internal state-et (_shutdown_after_all) a checkbox már beállította
        if shutdown_after_all:
            self._shutdown_after_all = True

        self._log(f"Starting Auto Execution: {len(execution_list)} models in queue...")

        # Auto execution állapot inicializálás
        self._auto_execution_list = list(execution_list)  # Másolat
        self._auto_current_index = 0
        self._is_auto_running = True

        # UI állapot
        self.btn_run.configure(state="disabled")
        self.btn_auto.configure(state="normal")  # Auto gomb marad aktív
        self.btn_stop.configure(state="normal")
        self.btn_pause.configure(state="normal")

        # Első modell indítása
        self._run_next_auto_model()

    def _run_next_auto_model(self):
        """Következő modell futtatása az auto queue-ból."""
        if not getattr(self, '_is_auto_running', False):
            return

        # GC cleanup a modellek között - memória felszabadítás
        self._auto_cleanup()

        # Ellenőrzés: van-e még futtatandó modell
        if self._auto_current_index >= len(self._auto_execution_list):
            self._finish_auto_sequence()
            return

        item = self._auto_execution_list[self._auto_current_index]
        total = len(self._auto_execution_list)

        data_mode = item.get("data_mode", "Original")
        self._log(
            f"Auto Run [{self._auto_current_index + 1}/{total}]: "
            f"{item['model']} ({item['category']}) [Mode: {data_mode}]"
        )

        # Data mode váltás kezelése
        current_mode = self.feature_var.get() if hasattr(self, 'feature_var') else "Original"

        if data_mode != current_mode:
            self._log(f"Switching data mode from '{current_mode}' to '{data_mode}'...")
            # Feature mode váltás és újraszámítás
            if hasattr(self, 'feature_var'):
                self.feature_var.set(data_mode)

            # Async recalculation - majd _continue_auto_run hívódik
            self._auto_pending_item = item
            self._trigger_auto_feature_recalc(data_mode)
        else:
            # Nincs mode váltás, folytatás közvetlenül
            self._continue_auto_run(item)

    def _trigger_auto_feature_recalc(self, data_mode: str):
        """Feature újraszámítás indítása auto futtatáshoz."""
        import threading

        def recalc_thread():
            try:
                # A _recalculate_features metódus a data_loading mixin-ben van
                if hasattr(self, '_recalculate_features'):
                    self._recalculate_features()
                # Sikeres - folytatás a fő szálon
                self.after(0, lambda: self._continue_auto_run(self._auto_pending_item))
            except Exception as e:
                logging.error(f"Auto feature recalc error: {e}")
                # Hiba esetén is folytatjuk
                self.after(0, lambda: self._continue_auto_run(self._auto_pending_item))

        threading.Thread(target=recalc_thread, daemon=True).start()

    def _continue_auto_run(self, item: dict):
        """Auto futtatás folytatása a feature recalc után."""
        if not getattr(self, '_is_auto_running', False):
            return

        # UI beállítása az item alapján
        self._apply_auto_item_settings(item)

        # Futtatás indítása
        self._run_analysis_with_item(item)

    def _apply_auto_item_settings(self, item: dict):
        """Auto queue item beállításainak alkalmazása a UI-ra."""
        # Category és Model beállítása
        self.category_var.set(item["category"])
        self._on_category_change(item["category"])
        self.model_var.set(item["model"])
        self._on_model_change(item["model"])

        # Paraméterek beállítása
        for key, value in item.get("params", {}).items():
            if key in self.param_widgets:
                widget = self.param_widgets[key]
                if hasattr(widget, 'set'):
                    widget.set(str(value))
                elif hasattr(widget, 'delete') and hasattr(widget, 'insert'):
                    widget.delete(0, 'end')
                    widget.insert(0, str(value))

        # Horizon
        if "horizon" in item:
            self.horizon_slider.set(item["horizon"])
            self.horizon_label.configure(text=str(item["horizon"]))

        # GPU
        if hasattr(self, 'var_gpu'):
            self.var_gpu.set(item.get("use_gpu", False))

        # Batch mode
        self.var_batch_mode = item.get("batch_mode", False)
        if self.var_batch_mode:
            self.btn_batch_mode.configure(fg_color="#27ae60", text="BATCH: ON")
        else:
            self.btn_batch_mode.configure(fg_color="#5d5d5d", text="BATCH MODE")

        # Panel mode
        if hasattr(self, 'var_panel_mode'):
            self.var_panel_mode.set(item.get("panel_mode", False))

        # Dual mode
        if hasattr(self, 'var_dual_model'):
            self.var_dual_model.set(item.get("dual_model", False))

    def _run_analysis_with_item(self, item: dict):
        """Elemzés futtatása egy auto queue item alapján - MULTIPROCESSING."""
        model_name = item["model"]
        params = item.get("params", {})
        horizon = item.get("horizon", 52)
        use_gpu = item.get("use_gpu", False)
        use_batch = item.get("batch_mode", False)
        use_panel = item.get("panel_mode", False)
        use_dual = item.get("dual_model", False)

        # Store output folder for report generation
        self._auto_current_output_folder = item.get("output_folder", "")

        # Context dict a worker számára (pickle-elhető)
        context_dict = {
            'model_name': model_name,
            'params': params,
            'forecast_horizon': horizon,
            'use_gpu': use_gpu,
            'use_batch': use_batch,
            'panel_mode': use_panel,
            'dual_model': use_dual
        }

        # Environment variables a worker számára
        res_mgr = get_resource_manager()
        env_vars = res_mgr.get_env_vars()

        self._last_logged_percent = -10
        self._analysis_running = True
        self._analysis_start_time = __import__('time').time()

        # Worker manager létrehozása és indítása
        if not hasattr(self, '_worker_manager') or self._worker_manager is None:
            self._worker_manager = AnalysisWorkerManager()

        # Data másolat a worker-nek
        with self.data_lock:
            data_copy = self.processed_data.copy()

        self._worker_manager.start(data_copy, context_dict, env_vars)
        self._start_time_timer()

        # GUI progress poller indítása (poll-olja a Queue-t)
        self._start_progress_poller()

    def _finish_auto_sequence(self):
        """Auto execution sequence befejezése."""
        self._log("Auto Execution Complete!")

        self._is_auto_running = False
        self._auto_execution_list = []
        self._auto_current_index = 0

        # UI visszaállítása
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_pause.configure(state="disabled")

        # Auto window frissítése
        if hasattr(self, '_auto_window') and self._auto_window:
            try:
                if self._auto_window.winfo_exists():
                    self._auto_window.update_start_button_state()
                    self._auto_window.progress_bar.set(0)  # Reset progress
            except Exception:
                pass

        self.sound.play_model_complete()

        # Shutdown kezelés - after all
        if getattr(self, '_shutdown_after_all', False):
            self._execute_shutdown()

    def _on_auto_model_complete(self):
        """Egy auto modell befejezése után - következő indítása."""
        if not getattr(self, '_is_auto_running', False):
            return

        # Automatikus report generálás az output folder-be
        output_folder = getattr(self, '_auto_current_output_folder', '')
        if output_folder and hasattr(self, '_on_generate_report'):
            # Az aktuális item lekérése a report flagekhez
            item = self._auto_execution_list[self._auto_current_index] if self._auto_current_index < len(self._auto_execution_list) else {}
            auto_stability = item.get("auto_stability", False)
            auto_risk = item.get("auto_risk", False)

            self._generate_auto_reports(output_folder, auto_stability, auto_risk)

        self._auto_current_index += 1

        # Progress frissítése az auto window-ban
        if hasattr(self, '_auto_window') and self._auto_window:
            try:
                if self._auto_window.winfo_exists():
                    progress = self._auto_current_index / len(self._auto_execution_list)
                    self._auto_window.progress_bar.set(progress)
            except Exception:
                pass

        # Kis késleltetés a következő modell előtt (GC, stb.)
        self.after(500, self._run_next_auto_model)

    def stop_auto_execution(self):
        """Auto execution leállítása."""
        if getattr(self, '_is_auto_running', False):
            self._log("Auto execution stopped by user.", "warning")
            self._is_auto_running = False

            # Worker process leállítása
            if hasattr(self, '_worker_manager') and self._worker_manager:
                self._worker_manager.stop()

            self._finish_auto_sequence()

    def _generate_auto_reports(self, output_folder: str, auto_stability: bool, auto_risk: bool):
        """Auto execution report generálás - Standard + opcionális Stability/Risk."""
        import os
        import copy

        try:
            # 1. Standard Report generálás
            self._log(f"Generating Standard report to: {output_folder}")
            report_path = self._on_generate_report(auto_mode=True, base_dir=output_folder)

            # Suffix kinyerése a mappa névből (pl. _A01)
            suffix = ""
            if report_path and os.path.isdir(report_path):
                dirname = os.path.basename(report_path)
                if "_A" in dirname:
                    parts = dirname.split("_A")
                    if len(parts) > 1 and parts[-1].isdigit():
                        suffix = f"_A{parts[-1]}"

            if not suffix:
                logging.warning("Could not extract suffix from report path")
                return

            # Eredeti eredmények mentése
            original_results = copy.deepcopy(self.last_results) if hasattr(self, 'last_results') else None

            # 2. Stability Report (opcionális)
            if auto_stability:
                try:
                    self._log("Generating Stability Weighted report...")
                    self._apply_ranking_for_auto("stability")
                    self._on_generate_report(auto_mode=True, base_dir=output_folder, forced_suffix=suffix)
                except Exception as e:
                    self._log(f"Stability report error: {e}", "warning")

            # 3. Risk Adjusted Report (opcionális)
            if auto_risk:
                try:
                    self._log("Generating Risk Adjusted report...")
                    self._apply_ranking_for_auto("risk_adjusted")
                    self._on_generate_report(auto_mode=True, base_dir=output_folder, forced_suffix=suffix)
                except Exception as e:
                    self._log(f"Risk report error: {e}", "warning")

            # Eredeti állapot visszaállítása
            if original_results:
                self.last_results = original_results
                self._apply_ranking_for_auto("forecast")

        except Exception as e:
            self._log(f"Report generation error: {e}", "warning")
            logging.error(f"Auto report generation failed: {e}")

    def _apply_ranking_for_auto(self, mode: str):
        """Ranking alkalmazása auto report generáláshoz."""
        from data.processor import DataProcessor
        import pandas as pd

        # Stability metrikák kiszámítása ha szükséges
        if mode in ["stability", "risk_adjusted"]:
            results_with_stab = getattr(self, 'results_with_stability', None)
            if results_with_stab is None or (isinstance(results_with_stab, pd.DataFrame) and results_with_stab.empty):
                self._log("Calculating stability metrics...")

                # Raw data lekérése - explicit None/empty ellenőrzéssel
                raw_data = getattr(self, 'raw_data', None)
                if raw_data is None or (isinstance(raw_data, pd.DataFrame) and raw_data.empty):
                    raw_data = getattr(self, 'processed_data', None)

                results_df = getattr(self, 'results_df', None)

                # Ellenőrzés: mindkét DataFrame elérhető és nem üres
                raw_ok = raw_data is not None and isinstance(raw_data, pd.DataFrame) and not raw_data.empty
                results_ok = results_df is not None and isinstance(results_df, pd.DataFrame) and not results_df.empty

                if raw_ok and results_ok:
                    self.results_with_stability = DataProcessor.calculate_stability_metrics(
                        raw_data, results_df
                    )
                else:
                    logging.warning(f"Cannot calculate stability: raw_ok={raw_ok}, results_ok={results_ok}")
                    return

        # Base DataFrame kiválasztása
        base_df = getattr(self, 'results_with_stability', None)
        if base_df is None or (isinstance(base_df, pd.DataFrame) and base_df.empty):
            base_df = getattr(self, 'results_df', None)

        if base_df is None or (isinstance(base_df, pd.DataFrame) and base_df.empty):
            logging.warning("No results DataFrame for ranking")
            return

        # Ranking alkalmazása
        ranked_df, _ = DataProcessor.apply_ranking(base_df, ranking_mode=mode, sort_column="Forecast_1M")

        # Eredmények frissítése
        if hasattr(self, 'last_results') and self.last_results:
            self.last_results["results"] = ranked_df
            self.last_results["ranking_mode"] = mode
        self.results_df = ranked_df

    def _auto_cleanup(self):
        """Memória tisztítás auto futtatások között."""
        import gc
        from analysis.engine import AnalysisEngine

        try:
            # Engine shutdown - worker processek leállítása
            AnalysisEngine.shutdown()

            # GC futtatás többször a ciklikus referenciák miatt
            gc.collect()
            gc.collect()

            logging.debug("Auto Exec: Cleanup performed between model runs")
        except Exception as e:
            logging.debug(f"Auto Exec: Cleanup warning: {e}")

    # === SHUTDOWN METHODS ===

    def _on_shutdown_checkbox_change(self):
        """Analysis tab shutdown checkbox változás - immediate shutdown."""
        if self.var_shutdown.get():
            # Bekapcsolva: immediate shutdown, auto window kikapcsolása
            self._shutdown_immediate = True
            self._shutdown_after_all = False
            self._sync_auto_window_shutdown(False)
            self._log("Shutdown enabled: after current model", "warning")
        else:
            # Kikapcsolva
            self._shutdown_immediate = False
            self._log("Shutdown disabled")

    def set_shutdown_after_all(self, enabled: bool):
        """Auto window hívja - shutdown after all beállítása."""
        if enabled:
            # Bekapcsolva: after all, analysis tab kikapcsolása
            self._shutdown_after_all = True
            self._shutdown_immediate = False
            self.var_shutdown.set(False)
            self._log("Shutdown enabled: after ALL models", "warning")
        else:
            # Kikapcsolva
            self._shutdown_after_all = False

    def _sync_auto_window_shutdown(self, enabled: bool):
        """Auto window shutdown checkbox szinkronizálása."""
        if hasattr(self, '_auto_window') and self._auto_window:
            try:
                if self._auto_window.winfo_exists():
                    self._auto_window.var_shutdown.set(enabled)
            except Exception:
                pass

    def _execute_shutdown(self):
        """Számítógép leállítása."""
        import subprocess
        import sys

        self._log("SHUTDOWN: Initiating system shutdown...", "critical")

        try:
            if sys.platform == "win32":
                # Windows shutdown - 60 másodperc késleltetéssel
                subprocess.run(["shutdown", "/s", "/t", "60"], check=True)
                self._log("SHUTDOWN: System will shut down in 60 seconds", "critical")
            else:
                # Linux/Mac
                subprocess.run(["shutdown", "-h", "+1"], check=True)
                self._log("SHUTDOWN: System will shut down in 1 minute", "critical")
        except Exception as e:
            self._log(f"SHUTDOWN ERROR: {e}", "error")
