"""
Analysis Tab - MBO Trading Strategy Analyzer
Mixin osztály az Analysis tab funkcionalitásához.
"""
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=too-many-statements,import-outside-toplevel

import customtkinter as ctk

from gui.translate import tr
from models import (
    get_categories,
    get_models_in_category,
    get_param_defaults,
    get_param_options,
    supports_gpu,
    supports_batch,
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

        # CPU Power slider
        ctk.CTkLabel(
            row2,
            text="CPU Power:",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(0, 5))

        self.cpu_slider = ctk.CTkSlider(
            row2,
            from_=10,
            to=100,
            number_of_steps=18,
            width=150,
            command=self._on_cpu_change
        )
        self.cpu_slider.set(85)
        self.cpu_slider.pack(side="left", padx=(0, 5))

        import multiprocessing
        cores = multiprocessing.cpu_count()
        self.cpu_label = ctk.CTkLabel(
            row2,
            text=f"85% ({int(cores * 0.85)} cores)",
            font=ctk.CTkFont(size=11),
            width=100
        )
        self.cpu_label.pack(side="left", padx=(0, 20))

        # Use GPU switch
        self.var_gpu = ctk.BooleanVar(value=False)
        self.switch_gpu = ctk.CTkSwitch(
            row2,
            text="Use GPU",
            variable=self.var_gpu,
            font=ctk.CTkFont(size=11),
            command=self._on_gpu_toggle
        )
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
        """Frissíti a UI állapotát az aktuális modell alapján."""
        model_name = self.model_var.get()
        if not model_name:
            return

        # GPU switch állapot
        if supports_gpu(model_name):
            self.switch_gpu.configure(state="normal")
        else:
            self.var_gpu.set(False)
            self.switch_gpu.configure(state="disabled")

        # Batch mode gomb
        if supports_batch(model_name):
            self.btn_batch_mode.configure(state="normal", fg_color="#5d5d5d")
        else:
            self.btn_batch_mode.configure(state="disabled", fg_color="#444444")

        # Panel és Dual mode checkbox-ok
        # (mindkettőhöz kell, hogy a modell támogassa a batch-et)
        if supports_batch(model_name):
            self.chk_panel_mode.configure(state="normal")
            self.chk_dual_model.configure(state="normal")
        else:
            self.var_panel_mode.set(False)
            self.var_dual_model.set(False)
            self.chk_panel_mode.configure(state="disabled")
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
        """Elemzés indítása."""
        self.sound.play_button_click()

        if self.processed_data is None or self.processed_data.empty:
            self._log(tr("Please load data first!"), "warning")
            return

        model_name = self.model_var.get()
        self._log(f"Starting analysis with {model_name}...")

        # UI állapot frissítése
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_pause.configure(state="normal")

        # TODO: Tényleges elemzés indítása háttérszálon

    def _on_stop_analysis(self):
        """Elemzés leállítása."""
        self.sound.play_button_click()
        self._log("Stopping analysis...")
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_pause.configure(state="disabled")

    def _on_pause_analysis(self):
        """Elemzés szüneteltetése/folytatása."""
        self.sound.play_button_click()
        if self.btn_pause.cget("text") == "Pause":
            # Resume = zöld (mint az eredetiben)
            self.btn_pause.configure(
                text="Resume", fg_color="#27ae60", hover_color="#2ecc71"
            )
            self._log("Analysis paused.")
        else:
            # Pause = sárga (mint az eredetiben)
            self.btn_pause.configure(
                text="Pause", fg_color="#f39c12", hover_color="#d35400"
            )
            self._log("Analysis resumed.")

    def _on_cpu_change(self, value: float):
        """CPU slider változás."""
        import multiprocessing
        cores = multiprocessing.cpu_count()
        pct = int(value)
        used_cores = int(cores * pct / 100)
        self.cpu_label.configure(text=f"{pct}% ({used_cores} cores)")

    def _on_gpu_toggle(self):
        """GPU toggle."""
        self.sound.play_toggle_switch()
        state = "enabled" if self.var_gpu.get() else "disabled"
        self._log(f"GPU {state}")

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
