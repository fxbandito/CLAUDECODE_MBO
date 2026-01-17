"""
Auto Execution Window - MBO Trading Strategy Analyzer

Queue-alapú automatikus modell futtatás kezelése.
Fázis 4: Perzisztencia (auto.txt)
"""

import json
import logging
import os
from tkinter import TclError, filedialog

import customtkinter as ctk

from gui.translate import tr
from gui.settings import SettingsManager
from models import (
    get_categories,
    get_models_in_category,
    get_param_defaults,
    get_param_options,
    supports_gpu,
    supports_batch,
    supports_panel_mode,
    supports_dual_mode,
)


class AutoExecutionWindow(ctk.CTkToplevel):
    """
    Auto Execution Manager ablak.
    Lehetővé teszi modellek sorba állítását és automatikus futtatását.
    """

    def __init__(self, parent):
        """
        Inicializálás.

        Args:
            parent: A szülő ablak (MainWindow) referencia
        """
        super().__init__(parent)
        self.parent = parent

        # Settings manager
        self.settings = SettingsManager()

        # Ablak beállítások
        self.title(tr("Auto Execution Manager"))
        self._load_geometry()

        # Modal-szerű viselkedés
        self.transient(parent)
        self.lift()
        self.focus_force()

        # Ablak bezárás kezelése
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Adatstruktúrák
        self.execution_list = []  # A futtatandó modellek listája
        self.param_widgets = {}   # Paraméter widgetek
        self.selected_folder = "" # Reports folder

        # Auto.txt fájl útvonala (src mappában - egy szinttel feljebb a gui-tól)
        src_dir = os.path.dirname(os.path.dirname(__file__))
        self.auto_file = os.path.join(src_dir, "auto.txt")

        # UI felépítése
        self._setup_ui()

        # Kezdeti model betöltés
        self._init_model_selection()

        # Queue betöltése fájlból
        self._load_queue()

        # Auto exec beállítások betöltése
        self._load_auto_exec_settings()

        logging.info("AutoExecutionWindow initialized")

    def _load_geometry(self):
        """Ablak pozíció és méret betöltése a beállításokból."""
        geo = self.settings.get_auto_window_geometry()
        width = geo.get("width", 1200)
        height = geo.get("height", 700)
        x = geo.get("x")
        y = geo.get("y")

        if x is not None and y is not None:
            # Van mentett pozíció
            self.geometry(f"{width}x{height}+{x}+{y}")
        else:
            # Középre igazítás
            self.geometry(f"{width}x{height}")

        self.minsize(900, 500)

    def _save_geometry(self):
        """Ablak pozíció és méret mentése."""
        try:
            # Geometry string parse: "WxH+X+Y" or "WxH-X-Y" (negative coords)
            geo = self.geometry()
            # Formátum: "1200x700+100+50" vagy "1200x700-100-50"
            # Először a '+' vagy '-' előtti rész (WxH)
            if '+' in geo:
                size_part = geo.split('+')[0]
                coords = geo.split('+')[1:]
                x = int(coords[0])
                y = int(coords[1]) if len(coords) > 1 else 0
            elif '-' in geo and 'x' in geo:
                # Negatív koordináták kezelése
                size_part = geo.split('-')[0]
                # Ez bonyolultabb, mert a '-' lehet x-ben is
                parts = geo.replace('x', '+').split('+')
                size_part = f"{parts[0]}x{parts[1]}" if len(parts) > 1 else geo
                x, y = 0, 0  # Alapértelmezett
            else:
                size_part = geo
                x, y = 0, 0

            width, height = size_part.split('x')

            self.settings.set_auto_window_geometry(
                int(width), int(height), int(x), int(y)
            )
            self.settings.save()
        except (ValueError, IndexError, AttributeError) as e:
            logging.debug(f"Could not parse geometry: {e}")

    def _load_auto_exec_settings(self):
        """Auto exec beállítások betöltése (checkbox, slider értékek)."""
        settings = self.settings.get_auto_exec_settings()

        # Checkbox-ok betöltése
        self.var_gpu.set(settings.get("use_gpu", False))
        self.slider_horizon.set(settings.get("horizon", 52))
        self.lbl_horizon.configure(text=str(settings.get("horizon", 52)))
        self.combo_data_mode.set(settings.get("data_mode", "Original"))
        self.var_batch.set(settings.get("batch_mode", False))
        self.var_panel.set(settings.get("panel_mode", False))
        self.var_dual.set(settings.get("dual_model", False))
        self.var_stability.set(settings.get("stability_report", False))
        self.var_risk.set(settings.get("risk_report", False))
        self.var_shutdown.set(settings.get("shutdown_after_all", False))

    def _save_auto_exec_settings(self):
        """Auto exec beállítások mentése."""
        settings_dict = {
            "use_gpu": self.var_gpu.get(),
            "horizon": int(self.slider_horizon.get()),
            "data_mode": self.combo_data_mode.get(),
            "batch_mode": self.var_batch.get(),
            "panel_mode": self.var_panel.get(),
            "dual_model": self.var_dual.get(),
            "stability_report": self.var_stability.get(),
            "risk_report": self.var_risk.get(),
            "shutdown_after_all": self.var_shutdown.get(),
        }
        self.settings.set_auto_exec_settings(settings_dict)
        self.settings.save()

    def _setup_ui(self):
        """Fő UI struktúra felépítése."""
        # === FELSŐ SZEKCIÓ: Konfiguráció ===
        self.frame_config = ctk.CTkFrame(self)
        self.frame_config.pack(fill="x", padx=10, pady=10)

        # Row 1: Category és Model választás
        self._setup_selection_row()

        # Row 2: Globális beállítások
        self._setup_settings_row()

        # Row 3: Paraméterek (dinamikus)
        self.frame_params = ctk.CTkFrame(self.frame_config, fg_color="transparent")
        # Nem packeljük most, csak ha vannak paraméterek

        # Add gomb
        self.btn_add = ctk.CTkButton(
            self.frame_config,
            text=tr("+ Add to Queue"),
            width=150,
            height=32,
            fg_color="#27ae60",
            hover_color="#2ecc71",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_add_item,
            state="disabled"  # Később engedélyezzük folder kiválasztás után
        )
        self.btn_add.pack(pady=10)

        # === KÖZÉPSŐ SZEKCIÓ: Execution Queue ===
        self.frame_queue = ctk.CTkScrollableFrame(
            self,
            label_text=tr("Execution Queue")
        )
        self.frame_queue.pack(fill="both", expand=True, padx=10, pady=5)

        # Placeholder szöveg üres queue-hoz
        self.lbl_empty_queue = ctk.CTkLabel(
            self.frame_queue,
            text=tr("Queue is empty. Select a Reports Folder and add models."),
            text_color="gray",
            font=ctk.CTkFont(size=12, slant="italic")
        )
        self.lbl_empty_queue.pack(pady=50)

        # === ALSÓ SZEKCIÓ: Vezérlők ===
        self.frame_controls = ctk.CTkFrame(self)
        self.frame_controls.pack(fill="x", padx=10, pady=10)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.frame_controls, width=400)
        self.progress_bar.pack(side="left", padx=10, fill="x", expand=True)
        self.progress_bar.set(0)

        # Task counter
        self.lbl_tasks = ctk.CTkLabel(
            self.frame_controls,
            text=tr("Tasks: 0"),
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.lbl_tasks.pack(side="left", padx=20)

        # Shutdown checkbox - teljes queue után
        self.var_shutdown = ctk.BooleanVar(value=False)
        self.chk_shutdown = ctk.CTkCheckBox(
            self.frame_controls,
            text=tr("Shutdown after ALL"),
            variable=self.var_shutdown,
            text_color="#e74c3c",
            font=ctk.CTkFont(size=11),
            command=self._on_shutdown_change
        )
        self.chk_shutdown.pack(side="right", padx=10)

        # Start gomb
        self.btn_start = ctk.CTkButton(
            self.frame_controls,
            text=tr("Start Auto Execution"),
            width=180,
            height=36,
            fg_color="#2980b9",
            hover_color="#3498db",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._on_start,
            state="disabled"  # Nincs elem a queue-ban
        )
        self.btn_start.pack(side="right", padx=10)

    def _setup_selection_row(self):
        """Category és Model választás sor."""
        row = ctk.CTkFrame(self.frame_config, fg_color="transparent")
        row.pack(fill="x", padx=5, pady=5)

        # Category
        ctk.CTkLabel(
            row,
            text=tr("Category:"),
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=5)

        categories = get_categories()
        self.combo_category = ctk.CTkComboBox(
            row,
            values=categories if categories else ["--"],
            width=280,
            command=self._on_category_change
        )
        self.combo_category.pack(side="left", padx=5)

        # Model
        ctk.CTkLabel(
            row,
            text=tr("Model:"),
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(20, 5))

        self.combo_model = ctk.CTkComboBox(
            row,
            values=["--"],
            width=200,
            command=self._on_model_change
        )
        self.combo_model.pack(side="left", padx=5)

        # Help gomb
        self.btn_help = ctk.CTkButton(
            row,
            text="?",
            width=28,
            height=28,
            corner_radius=14,
            fg_color="#3d3d5c",
            hover_color="#4d4d6c",
            command=self._on_help
        )
        self.btn_help.pack(side="left", padx=5)

        # Jobb oldalon: Reports Folder
        self.btn_folder = ctk.CTkButton(
            row,
            text=tr("Reports Folder"),
            width=120,
            fg_color="#8e44ad",
            hover_color="#9b59b6",
            command=self._on_select_folder
        )
        self.btn_folder.pack(side="right", padx=5)

        # Folder path label
        self.lbl_folder = ctk.CTkLabel(
            row,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.lbl_folder.pack(side="right", padx=5)

        # Save/Load gombok
        self.btn_save = ctk.CTkButton(
            row,
            text=tr("Save"),
            width=70,
            fg_color="#e67e22",
            hover_color="#d35400",
            command=self._on_save_config
        )
        self.btn_save.pack(side="right", padx=5)

        self.btn_load = ctk.CTkButton(
            row,
            text=tr("Load"),
            width=70,
            fg_color="#3498db",
            hover_color="#2980b9",
            command=self._on_load_config
        )
        self.btn_load.pack(side="right", padx=5)

    def _setup_settings_row(self):
        """Globális beállítások sor (GPU, Horizon, Data Mode, stb.)."""
        row = ctk.CTkFrame(self.frame_config, fg_color="transparent")
        row.pack(fill="x", padx=5, pady=5)

        # GPU switch
        self.var_gpu = ctk.BooleanVar(value=False)
        self.switch_gpu = ctk.CTkSwitch(
            row,
            text=tr("Use GPU"),
            variable=self.var_gpu,
            font=ctk.CTkFont(size=11)
        )
        self.switch_gpu.pack(side="left", padx=10)

        # Forecast Horizon
        ctk.CTkLabel(
            row,
            text=tr("Horizon:"),
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(20, 5))

        self.slider_horizon = ctk.CTkSlider(
            row,
            from_=1,
            to=104,
            number_of_steps=103,
            width=120
        )
        self.slider_horizon.set(52)
        self.slider_horizon.pack(side="left", padx=5)

        self.lbl_horizon = ctk.CTkLabel(
            row,
            text="52",
            font=ctk.CTkFont(size=11, weight="bold"),
            width=30
        )
        self.lbl_horizon.pack(side="left")
        self.slider_horizon.configure(
            command=lambda v: self.lbl_horizon.configure(text=str(int(v)))
        )

        # Data Mode
        ctk.CTkLabel(
            row,
            text=tr("Data Mode:"),
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(20, 5))

        self.combo_data_mode = ctk.CTkComboBox(
            row,
            values=["Original", "Forward Calc", "Rolling Window"],
            width=120,
            state="readonly"
        )
        self.combo_data_mode.set("Original")
        self.combo_data_mode.pack(side="left", padx=5)

        # Batch Mode checkbox
        self.var_batch = ctk.BooleanVar(value=False)
        self.chk_batch = ctk.CTkCheckBox(
            row,
            text=tr("Batch"),
            variable=self.var_batch,
            font=ctk.CTkFont(size=11),
            width=60,
            command=self._on_batch_change
        )
        self.chk_batch.pack(side="left", padx=10)

        # Panel Mode checkbox
        self.var_panel = ctk.BooleanVar(value=False)
        self.chk_panel = ctk.CTkCheckBox(
            row,
            text=tr("Panel"),
            variable=self.var_panel,
            font=ctk.CTkFont(size=11),
            width=60,
            command=self._on_panel_change
        )
        self.chk_panel.pack(side="left", padx=5)

        # Dual Model checkbox
        self.var_dual = ctk.BooleanVar(value=False)
        self.chk_dual = ctk.CTkCheckBox(
            row,
            text=tr("Dual"),
            variable=self.var_dual,
            font=ctk.CTkFont(size=11),
            width=60,
            command=self._on_dual_change
        )
        self.chk_dual.pack(side="left", padx=5)

        # Reports separator
        ctk.CTkLabel(
            row,
            text="|",
            font=ctk.CTkFont(size=11),
            text_color="#666666"
        ).pack(side="left", padx=10)

        ctk.CTkLabel(
            row,
            text=tr("Reports:"),
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(0, 5))

        # Stability Report checkbox
        self.var_stability = ctk.BooleanVar(value=False)
        self.chk_stability = ctk.CTkCheckBox(
            row,
            text=tr("Stability"),
            variable=self.var_stability,
            font=ctk.CTkFont(size=11),
            width=70
        )
        self.chk_stability.pack(side="left", padx=5)

        # Risk Report checkbox
        self.var_risk = ctk.BooleanVar(value=False)
        self.chk_risk = ctk.CTkCheckBox(
            row,
            text=tr("Risk"),
            variable=self.var_risk,
            font=ctk.CTkFont(size=11),
            width=50
        )
        self.chk_risk.pack(side="left", padx=5)

    def _init_model_selection(self):
        """Kezdeti model selection inicializálás."""
        categories = get_categories()
        if categories:
            first_cat = categories[0]
            self.combo_category.set(first_cat)
            self._on_category_change(first_cat)

    # === Event Handlers ===

    def _on_category_change(self, category: str):
        """Kategória váltás - modellek frissítése."""
        models = get_models_in_category(category)
        self.combo_model.configure(values=models if models else ["--"])

        if models:
            self.combo_model.set(models[0])
            self._on_model_change(models[0])
        else:
            self.combo_model.set("--")
            self._clear_params()

        # Data mode engedélyezés (csak Classical ML-nél)
        if "Classical" in category or "Machine Learning" in category:
            self.combo_data_mode.configure(state="readonly")
        else:
            self.combo_data_mode.set("Original")
            self.combo_data_mode.configure(state="disabled")

    def _on_model_change(self, model: str):
        """Modell váltás - paraméterek és UI frissítése."""
        if model == "--":
            self._clear_params()
            return

        # Paraméter UI frissítése
        self._update_param_ui(model)

        # UI állapot frissítése a modell képességei alapján
        self._update_model_ui_state(model)

    def _update_param_ui(self, model_name: str):
        """Paraméter UI generálása a kiválasztott modellhez."""
        # Régi widgetek törlése
        for widget in self.frame_params.winfo_children():
            widget.destroy()
        self.param_widgets.clear()

        defaults = get_param_defaults(model_name)
        options = get_param_options(model_name)

        if not defaults:
            self.frame_params.pack_forget()
            return

        # Frame megjelenítése
        self.frame_params.pack(fill="x", padx=5, pady=5, before=self.btn_add)

        # Paraméterek megjelenítése (vízszintesen, mint az Analysis tab-on)
        for key, default_val in defaults.items():
            # Label
            label_text = key.replace("_", " ").title() + ":"
            ctk.CTkLabel(
                self.frame_params,
                text=label_text,
                font=ctk.CTkFont(size=10)
            ).pack(side="left", padx=(10, 3))

            # Widget típus: ComboBox ha van options, különben Entry
            if key in options:
                widget = ctk.CTkComboBox(
                    self.frame_params,
                    values=[str(v) for v in options[key]],
                    width=80,
                    height=28
                )
                widget.set(str(default_val))
            else:
                widget = ctk.CTkEntry(
                    self.frame_params,
                    width=60,
                    height=28
                )
                widget.insert(0, str(default_val))

            widget.pack(side="left", padx=(0, 10))
            self.param_widgets[key] = widget

    def _update_model_ui_state(self, model_name: str):
        """UI állapot frissítése a modell képességei alapján."""
        # GPU
        if supports_gpu(model_name):
            self.switch_gpu.configure(state="normal")
        else:
            self.var_gpu.set(False)
            self.switch_gpu.configure(state="disabled")

        # Batch
        if supports_batch(model_name):
            self.chk_batch.configure(state="normal")
        else:
            self.var_batch.set(False)
            self.chk_batch.configure(state="disabled")

        # Panel
        if supports_panel_mode(model_name):
            self.chk_panel.configure(state="normal")
        else:
            self.var_panel.set(False)
            self.chk_panel.configure(state="disabled")

        # Dual
        if supports_dual_mode(model_name):
            self.chk_dual.configure(state="normal")
        else:
            self.var_dual.set(False)
            self.chk_dual.configure(state="disabled")

    def _clear_params(self):
        """Paraméter UI törlése."""
        for widget in self.frame_params.winfo_children():
            widget.destroy()
        self.param_widgets.clear()
        self.frame_params.pack_forget()

    def _on_batch_change(self):
        """Batch mode - mutual exclusion."""
        if self.var_batch.get():
            self.var_panel.set(False)
            self.var_dual.set(False)

    def _on_panel_change(self):
        """Panel mode - mutual exclusion."""
        if self.var_panel.get():
            self.var_batch.set(False)
            self.var_dual.set(False)

    def _on_dual_change(self):
        """Dual mode - mutual exclusion."""
        if self.var_dual.get():
            self.var_batch.set(False)
            self.var_panel.set(False)

    def _on_shutdown_change(self):
        """Shutdown after ALL checkbox változás - szinkronizálás a parent-tel."""
        enabled = self.var_shutdown.get()
        # Parent (Analysis tab) értesítése
        if hasattr(self.parent, 'set_shutdown_after_all'):
            self.parent.set_shutdown_after_all(enabled)

    def _on_help(self):
        """Help gomb kezelése."""
        model = self.combo_model.get()
        if model and model != "--":
            # Ha a parent-nek van help metódusa, használjuk
            if hasattr(self.parent, '_show_model_help'):
                self.parent._show_model_help()
            else:
                logging.info(f"Help requested for model: {model}")

    def _on_select_folder(self):
        """Reports folder kiválasztás."""
        # Utolsó folder betöltése
        initial_dir = self.settings.get_last_auto_reports_folder() or None

        folder = filedialog.askdirectory(
            parent=self,
            title=tr("Select Reports Output Folder"),
            initialdir=initial_dir
        )

        if folder:
            self.selected_folder = folder
            self.settings.set_last_auto_reports_folder(folder)
            self.settings.save()

            # Folder label frissítése (rövidített path)
            display_path = folder if len(folder) < 40 else "..." + folder[-37:]
            self.lbl_folder.configure(text=display_path)

            # Add gomb engedélyezése
            self.btn_add.configure(state="normal")

            logging.info(f"Reports folder selected: {folder}")

    def _save_queue(self):
        """Queue mentése auto.txt fájlba."""
        try:
            with open(self.auto_file, "w", encoding="utf-8") as f:
                json.dump(self.execution_list, f, indent=2, ensure_ascii=False)
            logging.debug(f"Queue saved to {self.auto_file}")
        except (IOError, TypeError) as e:
            logging.error(f"Failed to save queue: {e}")

    def _load_queue(self):
        """Queue betöltése auto.txt fájlból."""
        if os.path.exists(self.auto_file):
            try:
                with open(self.auto_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        self.execution_list = json.loads(content)
                        # Ha van elem, frissítsük a folder-t is
                        if self.execution_list:
                            first_folder = self.execution_list[0].get("output_folder", "")
                            if first_folder:
                                self.selected_folder = first_folder
                                display = first_folder if len(first_folder) < 40 else "..." + first_folder[-37:]
                                self.lbl_folder.configure(text=display)
                                self.btn_add.configure(state="normal")
                        self._refresh_queue_ui()
                        self.update_task_count()
                        logging.info(f"Queue loaded: {len(self.execution_list)} items")
            except (IOError, json.JSONDecodeError) as e:
                logging.warning(f"Failed to load queue: {e}")
                self.execution_list = []

    def _on_save_config(self):
        """Konfiguráció mentése külső fájlba."""
        if not self.execution_list:
            logging.warning("Nothing to save - queue is empty")
            return

        filename = filedialog.asksaveasfilename(
            parent=self,
            title=tr("Save Queue Configuration"),
            defaultextension=".txt",
            initialfile="auto_config.txt",
            filetypes=[("Text Files", "*.txt"), ("JSON Files", "*.json"), ("All Files", "*.*")]
        )

        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(self.execution_list, f, indent=2, ensure_ascii=False)
                logging.info(f"Queue exported to: {filename}")
            except (IOError, TypeError) as e:
                logging.error(f"Failed to export queue: {e}")

    def _on_load_config(self):
        """Konfiguráció betöltése külső fájlból."""
        filename = filedialog.askopenfilename(
            parent=self,
            title=tr("Load Queue Configuration"),
            filetypes=[("Text Files", "*.txt"), ("JSON Files", "*.json"), ("All Files", "*.*")]
        )

        if filename:
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    loaded = json.load(f)

                # Validáció
                if not isinstance(loaded, list):
                    raise ValueError("Invalid format: expected list")

                for item in loaded:
                    if not all(k in item for k in ("category", "model", "params")):
                        raise ValueError("Invalid item: missing required fields")
                    # Folder felülírása a jelenlegivel, ha van
                    if self.selected_folder:
                        item["output_folder"] = self.selected_folder

                self.execution_list = loaded
                self._save_queue()  # Mentés auto.txt-be
                self._refresh_queue_ui()
                self.update_task_count()
                logging.info(f"Queue imported: {len(loaded)} items from {filename}")

            except (IOError, json.JSONDecodeError, ValueError) as e:
                logging.error(f"Failed to import queue: {e}")

    def _on_add_item(self):
        """Elem hozzáadása a queue-hoz."""
        model = self.combo_model.get()
        category = self.combo_category.get()

        if model == "--" or not model:
            logging.warning("No model selected")
            return

        if not self.selected_folder:
            logging.warning("No reports folder selected")
            return

        # Paraméterek összegyűjtése
        params = {}
        for key, widget in self.param_widgets.items():
            if hasattr(widget, 'get'):
                params[key] = widget.get()

        # Execution item létrehozása
        item = {
            "category": category,
            "model": model,
            "params": params,
            "output_folder": self.selected_folder,
            "use_gpu": self.var_gpu.get(),
            "horizon": int(self.slider_horizon.get()),
            "data_mode": self.combo_data_mode.get(),
            "batch_mode": self.var_batch.get(),
            "panel_mode": self.var_panel.get(),
            "dual_model": self.var_dual.get(),
            "auto_stability": self.var_stability.get(),
            "auto_risk": self.var_risk.get(),
        }

        self.execution_list.append(item)
        self._save_queue()  # Mentés minden hozzáadás után
        self._refresh_queue_ui()
        self.update_task_count()

        logging.info(f"Added to queue: {model} ({category})")

    def _remove_item(self, index: int):
        """Elem törlése a queue-ból."""
        if 0 <= index < len(self.execution_list):
            removed = self.execution_list.pop(index)
            self._save_queue()  # Mentés törlés után
            self._refresh_queue_ui()
            self.update_task_count()
            logging.info(f"Removed from queue: {removed['model']}")

    def _move_item_up(self, index: int):
        """Elem mozgatása felfelé."""
        if index > 0:
            self.execution_list[index], self.execution_list[index - 1] = \
                self.execution_list[index - 1], self.execution_list[index]
            self._save_queue()  # Mentés mozgatás után
            self._refresh_queue_ui()

    def _move_item_down(self, index: int):
        """Elem mozgatása lefelé."""
        if index < len(self.execution_list) - 1:
            self.execution_list[index], self.execution_list[index + 1] = \
                self.execution_list[index + 1], self.execution_list[index]
            self._save_queue()  # Mentés mozgatás után
            self._refresh_queue_ui()

    def _refresh_queue_ui(self):
        """Queue lista UI frissítése."""
        # Régi elemek törlése
        for widget in self.frame_queue.winfo_children():
            widget.destroy()

        if not self.execution_list:
            # Üres queue placeholder
            self.lbl_empty_queue = ctk.CTkLabel(
                self.frame_queue,
                text=tr("Queue is empty. Select a Reports Folder and add models."),
                text_color="gray",
                font=ctk.CTkFont(size=12, slant="italic")
            )
            self.lbl_empty_queue.pack(pady=50)
            return

        # Elemek megjelenítése
        for i, item in enumerate(self.execution_list):
            self._create_queue_item_row(i, item)

    def _create_queue_item_row(self, index: int, item: dict):
        """Egy queue elem sor létrehozása."""
        row = ctk.CTkFrame(self.frame_queue)
        row.pack(fill="x", pady=2, padx=5)

        # Gombok frame (bal oldal)
        btn_frame = ctk.CTkFrame(row, fg_color="transparent")
        btn_frame.pack(side="left", padx=5)

        # Remove gomb (piros)
        btn_remove = ctk.CTkButton(
            btn_frame,
            text="✕",
            width=28,
            height=28,
            fg_color="#c0392b",
            hover_color="#e74c3c",
            command=lambda idx=index: self._remove_item(idx)
        )
        btn_remove.pack(side="left", padx=1)

        # Move Up gomb
        btn_up = ctk.CTkButton(
            btn_frame,
            text="▲",
            width=28,
            height=28,
            fg_color="#3498db",
            hover_color="#2980b9",
            command=lambda idx=index: self._move_item_up(idx),
            state="normal" if index > 0 else "disabled"
        )
        btn_up.pack(side="left", padx=1)

        # Move Down gomb
        btn_down = ctk.CTkButton(
            btn_frame,
            text="▼",
            width=28,
            height=28,
            fg_color="#3498db",
            hover_color="#2980b9",
            command=lambda idx=index: self._move_item_down(idx),
            state="normal" if index < len(self.execution_list) - 1 else "disabled"
        )
        btn_down.pack(side="left", padx=1)

        # Item info (középen)
        # Formátum: [Mode] Model (Category) | param1: val1, param2: val2
        mode_parts = []
        if item.get("batch_mode"):
            mode_parts.append("Batch")
        if item.get("panel_mode"):
            mode_parts.append("Panel")
        if item.get("dual_model"):
            mode_parts.append("Dual")
        if item.get("data_mode", "Original") != "Original":
            mode_parts.append(item["data_mode"])

        # Report típusok
        report_parts = []
        if item.get("auto_stability"):
            report_parts.append("Stab")
        if item.get("auto_risk"):
            report_parts.append("Risk")

        mode_str = f"[{'/'.join(mode_parts)}] " if mode_parts else ""
        report_str = f" +{','.join(report_parts)}" if report_parts else ""

        params_str = ", ".join([f"{k}: {v}" for k, v in item["params"].items()])
        if len(params_str) > 60:
            params_str = params_str[:57] + "..."

        display_text = f"{mode_str}{item['model']} ({item['category']}){report_str}"
        if params_str:
            display_text += f" | {params_str}"

        lbl_info = ctk.CTkLabel(
            row,
            text=display_text,
            font=ctk.CTkFont(size=11),
            anchor="w"
        )
        lbl_info.pack(side="left", fill="x", expand=True, padx=10)

        # Horizon és GPU info (jobb oldal)
        info_text = f"H:{item['horizon']}"
        if item.get("use_gpu"):
            info_text += " | GPU"

        lbl_settings = ctk.CTkLabel(
            row,
            text=info_text,
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        lbl_settings.pack(side="right", padx=10)

    def _on_start(self):
        """Auto execution indítása."""
        if not self.execution_list:
            logging.warning("Cannot start: queue is empty")
            return

        # Ellenőrzés: van-e adat betöltve
        if not hasattr(self.parent, 'processed_data') or self.parent.processed_data is None:
            logging.warning("Cannot start: no data loaded")
            return

        logging.info(f"Starting auto execution with {len(self.execution_list)} items")

        # Beállítások mentése indítás előtt
        self._save_auto_exec_settings()

        # UI állapot - Start gomb letiltása
        self.btn_start.configure(state="disabled", text=tr("Running..."))
        self.progress_bar.set(0)

        # Parent run_auto_sequence meghívása
        shutdown = self.var_shutdown.get()
        self.parent.run_auto_sequence(self.execution_list, shutdown)

        # Ablak elrejtése futás közben
        self.withdraw()

    def update_start_button_state(self):
        """Start gomb állapotának frissítése (futás befejezése után)."""
        if self.execution_list:
            self.btn_start.configure(state="normal", text=tr("Start Auto Execution"))
        else:
            self.btn_start.configure(state="disabled", text=tr("Start Auto Execution"))
        self.progress_bar.set(0)

    def _on_close(self):
        """Ablak bezárás kezelése - geometry és beállítások mentés, majd elrejtés."""
        logging.debug("AutoExecutionWindow closing (withdraw)")

        # Geometry mentése
        self._save_geometry()

        # Auto exec beállítások mentése
        self._save_auto_exec_settings()

        # Grab elengedése
        try:
            self.grab_release()
        except TclError:
            pass

        # Elrejtés (nem destroy!)
        self.withdraw()

    def show(self):
        """Ablak megjelenítése (ha el volt rejtve)."""
        self.deiconify()
        self.lift()
        self.focus_force()

    def update_task_count(self):
        """Task counter frissítése."""
        count = len(self.execution_list)
        self.lbl_tasks.configure(text=f"{tr('Tasks:')} {count}")

        # Start gomb állapota
        if count > 0:
            self.btn_start.configure(state="normal")
        else:
            self.btn_start.configure(state="disabled")
