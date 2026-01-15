"""
MBO Trading Strategy Analyzer - Fő Alkalmazás Osztály
7 tab: Data Loading, Analysis, Results, Comparison, Inspection, Performance, Optuna
"""
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=too-many-statements,broad-exception-caught,unnecessary-lambda

import gc
import os
import threading
from datetime import datetime
from tkinter import filedialog
from typing import Dict, Optional

import customtkinter as ctk
import pandas as pd
from PIL import Image

from data.loader import DataLoader
from data.processor import DataProcessor
from gui.settings import SettingsManager
from gui.sorrend_data import get_settings
from gui.sound_manager import get_sound_manager
from gui.translate import get_translator, tr
from models import (
    get_registry,
    get_categories,
    get_models_in_category,
    get_param_defaults,
    get_param_options,
    supports_gpu,
    supports_batch,
)


def get_version() -> str:
    """Verzió beolvasása a version.txt fájlból."""
    version_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "version.txt")
    try:
        with open(version_file, "r", encoding="utf-8") as f:
            return f.readline().strip()
    except (FileNotFoundError, IOError):
        return "v5.0.0"


class MBOApp(ctk.CTk):
    """Fő alkalmazás ablak - régi stílus alapján."""

    # Ablak konstansok
    DEFAULT_WIDTH = 1600
    DEFAULT_HEIGHT = 1080
    MIN_WIDTH = 1200
    MIN_HEIGHT = 700

    # Tab nevek
    TAB_NAMES = [
        "Data Loading", "Analysis", "Results", "Comparison",
        "Inspection", "Performance", "Optuna"
    ]

    def __init__(self):
        super().__init__()

        # Settings manager
        self.settings = SettingsManager()

        # Sound manager
        self.sound = get_sound_manager()

        # Translator
        self.translator = get_translator()

        # Verzió
        self.version = get_version()

        # Ablak beállítások
        self.title(f"MBO Trading Strategy Analyzer {self.version}")
        self._restore_window_geometry()
        self.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)

        # Bezárás kezelése
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Állapot változók
        self.current_language = self.settings.get_language()
        self.is_dark_mode = self.settings.get_dark_mode()
        self.is_muted = self.settings.get_muted()
        self.current_tab = "Data Loading"

        # Set language in translator
        self.translator.set_language(self.current_language)

        # Set sound enabled based on muted state
        self.sound.set_enabled(not self.is_muted)

        # Adatok
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.data_lock = threading.Lock()

        # Tab és gomb referenciák
        self.tabs: Dict[str, ctk.CTkFrame] = {}
        self.tab_buttons: Dict[str, ctk.CTkButton] = {}

        # Ablak ikon beállítása
        self._setup_icon()

        # UI felépítése
        self._create_layout()
        self._create_header()
        self._create_progress_bar()
        self._create_content_area()
        self._create_log_panel()

        # Beállítások alkalmazása
        self._apply_saved_settings()

        # Első tab megjelenítése
        self._show_tab("Data Loading")
        self._log(tr("Application started. Ready."))

        # App start sound
        self.sound.play_app_start()

    def tr(self, text: str) -> str:
        """Translate text to current language."""
        return self.translator.tr(text)

    def _restore_window_geometry(self):
        """Ablak méret és pozíció visszaállítása."""
        geom = self.settings.get_window_geometry()
        width = geom.get("width", self.DEFAULT_WIDTH)
        height = geom.get("height", self.DEFAULT_HEIGHT)
        x = geom.get("x")
        y = geom.get("y")

        if x is not None and y is not None:
            self.geometry(f"{width}x{height}+{x}+{y}")
        else:
            self.geometry(f"{width}x{height}")

    def _apply_saved_settings(self):
        """Mentett beállítások alkalmazása."""
        # Dark mode
        if self.is_dark_mode:
            self.switch_theme.select()
            ctk.set_appearance_mode("dark")
        else:
            self.switch_theme.deselect()
            ctk.set_appearance_mode("light")

        # Language
        if self.current_language == "EN":
            self.switch_lang.select()
            self.label_en.configure(text_color="white")
            self.label_hu.configure(text_color="gray")
        else:
            self.switch_lang.deselect()
            self.label_en.configure(text_color="gray")
            self.label_hu.configure(text_color="white")

        # Mute
        self.var_muted.set(self.is_muted)

        # Feature mode
        saved_mode = self.settings.get_feature_mode()
        self.feature_var.set(saved_mode)
        self._update_feature_info()

    def _save_settings_now(self):
        """Save settings immediately (crash protection)."""
        self.settings.set_window_geometry(
            self.winfo_width(),
            self.winfo_height(),
            self.winfo_x(),
            self.winfo_y()
        )
        self.settings.set_language(self.current_language)
        self.settings.set_dark_mode(self.is_dark_mode)
        self.settings.set_muted(self.is_muted)
        self.settings.set_feature_mode(self.feature_var.get())
        self.settings.save()

    def _on_close(self):
        """Ablak bezárásakor beállítások mentése."""
        self.sound.play_app_close()
        self._save_settings_now()
        self._log(tr("Settings saved."))
        self.sound.shutdown()
        self.destroy()

    def _setup_icon(self):
        """Ablak ikon beállítása."""
        try:
            assets_dir = os.path.join(os.path.dirname(__file__), "assets")
            ico_path = os.path.join(assets_dir, "app_icon.ico")
            if os.path.exists(ico_path):
                self.iconbitmap(ico_path)
        except Exception:
            pass

    def _create_layout(self):
        """Fő layout grid konfiguráció."""
        self.grid_rowconfigure(0, weight=0)  # Header - fix
        self.grid_rowconfigure(1, weight=0)  # Progress bar - fix
        self.grid_rowconfigure(2, weight=1)  # Content - növekszik
        self.grid_rowconfigure(3, weight=0)  # Log panel - fix
        self.grid_columnconfigure(0, weight=1)

    def _create_header(self):
        """Header: logo, verzió, sorrend, tab gombok, kontrollok."""
        self.header_frame = ctk.CTkFrame(self, height=50, corner_radius=0)
        self.header_frame.grid(row=0, column=0, sticky="ew")
        self.header_frame.grid_propagate(False)

        # === LOGO ===
        self.logo_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.logo_frame.pack(side="left", padx=(10, 5))

        try:
            assets_dir = os.path.join(os.path.dirname(__file__), "assets")
            dark_logo_path = os.path.join(assets_dir, "dark_logo.png")
            light_logo_path = os.path.join(assets_dir, "light_logo.png")

            if os.path.exists(dark_logo_path) and os.path.exists(light_logo_path):
                with Image.open(dark_logo_path) as dark_img:
                    aspect = dark_img.width / dark_img.height
                    h = 40
                    w = int(h * aspect)
                    dark_copy = dark_img.copy()

                with Image.open(light_logo_path) as light_img:
                    light_copy = light_img.copy()

                self.logo_image = ctk.CTkImage(
                    light_image=light_copy,
                    dark_image=dark_copy,
                    size=(w, h),
                )
                self.logo_label = ctk.CTkLabel(self.logo_frame, text="", image=self.logo_image)
                self.logo_label.pack(side="left")
        except Exception:
            self.logo_label = ctk.CTkLabel(
                self.logo_frame,
                text="MBO",
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color="#4a90d9"
            )
            self.logo_label.pack(side="left")

        # === H-K / SZ-P SORREND ===
        self.sorrend_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.sorrend_frame.pack(side="left", padx=15)

        self.label_hk = ctk.CTkLabel(
            self.sorrend_frame,
            text="H-K",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="white"
        )
        self.label_hk.pack(side="left", padx=(0, 3))

        self.switch_sorrend_mode = ctk.CTkSwitch(
            self.sorrend_frame,
            text="",
            width=40,
            command=self._on_sorrend_mode_change
        )
        self.switch_sorrend_mode.pack(side="left", padx=(0, 3))

        self.label_szp = ctk.CTkLabel(
            self.sorrend_frame,
            text="SZ-P",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray"
        )
        self.label_szp.pack(side="left", padx=(0, 8))

        self.entry_sorrend_no = ctk.CTkEntry(
            self.sorrend_frame,
            width=55,
            height=28,
            placeholder_text="0000",
            font=ctk.CTkFont(size=12)
        )
        self.entry_sorrend_no.pack(side="left", padx=(0, 8))
        self.entry_sorrend_no.bind("<KeyRelease>", self._on_sorrend_input_change)

        self.lbl_sorrend_result = ctk.CTkLabel(
            self.sorrend_frame,
            text="(....)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#2ecc71",
            fg_color="#1a1a2e",
            corner_radius=5,
            padx=8,
            pady=3,
        )
        self.lbl_sorrend_result.pack(side="left")

        # === TAB GOMBOK ===
        self.tab_buttons_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.tab_buttons_frame.pack(side="left", padx=20)

        for tab_name in self.TAB_NAMES:
            btn = ctk.CTkButton(
                self.tab_buttons_frame,
                text=tab_name,
                font=ctk.CTkFont(size=11, weight="bold"),
                height=32,
                width=105,
                corner_radius=6,
                fg_color="#2a2a3e",
                text_color="#888888",
                hover_color="#3d3d5c",
                command=lambda t=tab_name: self._show_tab(t),
            )
            btn.pack(side="left", padx=2)
            self.tab_buttons[tab_name] = btn

        # === JOBB OLDALI KONTROLLOK ===
        self.controls_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.controls_frame.pack(side="right", padx=15)

        self.switch_theme = ctk.CTkSwitch(
            self.controls_frame,
            text=tr("Dark Mode"),
            font=ctk.CTkFont(size=11),
            command=self._on_dark_mode_toggle
        )
        self.switch_theme.select()
        self.switch_theme.pack(side="right", padx=(15, 0))

        self.label_en = ctk.CTkLabel(
            self.controls_frame,
            text="EN",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="white"
        )
        self.label_en.pack(side="right", padx=(0, 5))

        self.switch_lang = ctk.CTkSwitch(
            self.controls_frame,
            text="",
            width=40,
            command=self._on_language_change
        )
        self.switch_lang.select()
        self.switch_lang.pack(side="right", padx=(0, 3))

        self.label_hu = ctk.CTkLabel(
            self.controls_frame,
            text="HU",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray"
        )
        self.label_hu.pack(side="right", padx=(0, 3))

        self.var_muted = ctk.BooleanVar(value=True)
        self.chk_mute = ctk.CTkCheckBox(
            self.controls_frame,
            text=tr("Mute"),
            variable=self.var_muted,
            font=ctk.CTkFont(size=11),
            command=self._on_mute_toggle,
            width=20
        )
        self.chk_mute.pack(side="right", padx=(0, 15))

    def _create_progress_bar(self):
        """Progress bar a header alatt."""
        self.progress_frame = ctk.CTkFrame(self, height=18, corner_radius=0, fg_color="transparent")
        self.progress_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(2, 0))
        self.progress_frame.grid_propagate(False)
        self.progress_frame.grid_columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, height=8, corner_radius=2)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="0%",
            font=ctk.CTkFont(size=10),
            width=40
        )
        self.progress_label.grid(row=0, column=1)

    def _create_content_area(self):
        """Fő tartalom terület."""
        self.content_frame = ctk.CTkFrame(self, corner_radius=0)
        self.content_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(5, 0))
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)

        for tab_name in self.TAB_NAMES:
            if tab_name == "Data Loading":
                tab_frame = self._create_data_tab()
            elif tab_name == "Analysis":
                tab_frame = self._create_analysis_tab()
            elif tab_name == "Results":
                tab_frame = self._create_results_tab()
            else:
                tab_frame = self._create_placeholder_tab(tab_name)
            tab_frame.grid(row=0, column=0, sticky="nsew")
            self.tabs[tab_name] = tab_frame

    def _create_data_tab(self) -> ctk.CTkFrame:
        """Data Loading tab - teljes implementáció."""
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")

        # Felső sor: folder info + gombok
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
            width=150,
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
            fg_color="#555555",
            hover_color="#777777",
            command=self._show_feature_help
        ).pack(side="left", padx=10)

        self.feature_info_label = ctk.CTkLabel(
            feature_frame,
            text=tr("No additional features - raw data only"),
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.feature_info_label.pack(side="left", padx=20)

        # Fő terület: bal - fájl lista, jobb - adat előnézet
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
            font=ctk.CTkFont(family="Consolas", size=10)
        )
        self.files_textbox.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        # Jobb: Adat előnézet
        preview_frame = ctk.CTkFrame(main_frame, corner_radius=5)
        preview_frame.grid(row=0, column=1, sticky="nsew")

        self.data_preview = ctk.CTkTextbox(
            preview_frame,
            font=ctk.CTkFont(family="Consolas", size=10)
        )
        self.data_preview.pack(fill="both", expand=True, padx=5, pady=5)

        return frame

    def _create_analysis_tab(self) -> ctk.CTkFrame:
        """Analysis tab - teljes implementáció a screenshot alapján."""
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")

        # === ROW 1: Fő vezérlők ===
        row1 = ctk.CTkFrame(frame, fg_color="transparent", height=40)
        row1.pack(fill="x", pady=(5, 3))

        # Auto gomb - cián
        self.btn_auto = ctk.CTkButton(
            row1,
            text="Auto",
            width=60,
            height=28,
            fg_color="#17a2b8",
            hover_color="#138496",
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

        # Help gomb (?)
        self.btn_model_help = ctk.CTkButton(
            row1,
            text="?",
            width=26,
            height=26,
            corner_radius=13,
            fg_color="#555555",
            hover_color="#666666",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._show_model_help
        )
        self.btn_model_help.pack(side="left", padx=(0, 15))

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

        # Run Analysis gomb - zöld
        self.btn_run = ctk.CTkButton(
            right_controls,
            text="Run Analysis",
            width=110,
            height=28,
            fg_color="#27ae60",
            hover_color="#229954",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_run_analysis
        )
        self.btn_run.pack(side="right", padx=(8, 0))

        # Pause gomb - sárga/narancs
        self.btn_pause = ctk.CTkButton(
            right_controls,
            text="Pause",
            width=70,
            height=28,
            fg_color="#f39c12",
            hover_color="#d68910",
            font=ctk.CTkFont(size=11, weight="bold"),
            state="disabled",
            command=self._on_pause_analysis
        )
        self.btn_pause.pack(side="right", padx=(8, 0))

        # Stop gomb - piros
        self.btn_stop = ctk.CTkButton(
            right_controls,
            text="Stop",
            width=70,
            height=28,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            font=ctk.CTkFont(size=11, weight="bold"),
            state="disabled",
            command=self._on_stop_analysis
        )
        self.btn_stop.pack(side="right", padx=(8, 0))

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
            row2, text="?", width=22, height=22, corner_radius=11,
            fg_color="#555555", hover_color="#777777",
            font=ctk.CTkFont(size=10, weight="bold"),
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
        self.chk_dual_model.pack(side="left", padx=(0, 3))

        ctk.CTkButton(
            row2, text="?", width=22, height=22, corner_radius=11,
            fg_color="#555555", hover_color="#777777",
            font=ctk.CTkFont(size=10, weight="bold"),
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

    def _on_model_change(self, model_name: str):
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
            self.btn_pause.configure(text="Resume", fg_color="#27ae60")
            self._log("Analysis paused.")
        else:
            self.btn_pause.configure(text="Pause", fg_color="#f39c12")
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
        self.sound.play_checkbox_on() if self.var_panel_mode.get() else None
        if self.var_panel_mode.get() and self.var_dual_model.get():
            self.var_dual_model.set(False)

    def _on_dual_model_change(self):
        """Dual model váltás."""
        self.sound.play_checkbox_on() if self.var_dual_model.get() else None
        if self.var_dual_model.get() and self.var_panel_mode.get():
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

        # Help button
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

        # Generate Report button - piros/narancs
        self.btn_generate_report = ctk.CTkButton(
            row2,
            text="Generate Report",
            width=115,
            height=28,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_generate_report
        )
        self.btn_generate_report.pack(side="left", padx=(0, 8))

        # Export CSV button - kék
        self.btn_export_csv = ctk.CTkButton(
            row2,
            text="Export CSV",
            width=85,
            height=28,
            fg_color="#3498db",
            hover_color="#2980b9",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_export_csv
        )
        self.btn_export_csv.pack(side="left", padx=(0, 8))

        # All Results button - narancs
        self.btn_all_results = ctk.CTkButton(
            row2,
            text="All Results",
            width=85,
            height=28,
            fg_color="#e67e22",
            hover_color="#d35400",
            font=ctk.CTkFont(size=11, weight="bold"),
            state="disabled",
            command=self._on_show_all_results
        )
        self.btn_all_results.pack(side="left", padx=(0, 8))

        # Monthly Results button - lila
        self.btn_monthly_results = ctk.CTkButton(
            row2,
            text="Monthly Results",
            width=105,
            height=28,
            fg_color="#9b59b6",
            hover_color="#8e44ad",
            font=ctk.CTkFont(size=11, weight="bold"),
            state="disabled",
            command=self._on_show_monthly_results
        )
        self.btn_monthly_results.pack(side="left", padx=(0, 8))

        # Load Analysis State button - cián
        self.btn_load_state = ctk.CTkButton(
            row2,
            text="Load Analysis State",
            width=130,
            height=28,
            fg_color="#17a2b8",
            hover_color="#138496",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_load_analysis_state
        )
        self.btn_load_state.pack(side="left", padx=(0, 8))

        # Show Params button - sötétkék
        self.btn_show_params = ctk.CTkButton(
            row2,
            text="Show Params",
            width=95,
            height=28,
            fg_color="#2c3e50",
            hover_color="#34495e",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_show_params
        )
        self.btn_show_params.pack(side="left", padx=(0, 5))

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

    def _create_placeholder_tab(self, tab_name: str) -> ctk.CTkFrame:
        """Placeholder tab."""
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")

        ctk.CTkLabel(
            frame,
            text=tab_name,
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(expand=True)

        ctk.CTkLabel(
            frame,
            text="(Coming soon...)",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        ).pack()

        return frame

    def _create_log_panel(self):
        """Application Log panel."""
        self.log_frame = ctk.CTkFrame(self, height=200, corner_radius=0)
        self.log_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(10, 20))
        self.log_frame.grid_propagate(False)

        ctk.CTkLabel(
            self.log_frame,
            text=tr("Application Log"),
            font=ctk.CTkFont(size=11, weight="bold"),
            anchor="w"
        ).pack(anchor="w", padx=10, pady=5)

        self.log_textbox = ctk.CTkTextbox(
            self.log_frame,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.log_textbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # === TAB KEZELÉS ===

    def _show_tab(self, tab_name: str):
        """Tab megjelenítése."""
        self.sound.play_tab_switch()
        self.current_tab = tab_name

        for name, tab in self.tabs.items():
            tab.grid_remove()

        if tab_name in self.tabs:
            self.tabs[tab_name].grid()

        for name, btn in self.tab_buttons.items():
            if name == tab_name:
                btn.configure(fg_color="#4a4a6a", text_color="#FFFFFF")
            else:
                btn.configure(fg_color="#2a2a3e", text_color="#888888")

    # === SORREND KEZELÉS ===

    def _on_sorrend_mode_change(self):
        """H-K / SZ-P mód váltás."""
        self.sound.play_toggle_switch()
        if self.switch_sorrend_mode.get():
            self.label_szp.configure(text_color="white")
            self.label_hk.configure(text_color="gray")
        else:
            self.label_hk.configure(text_color="white")
            self.label_szp.configure(text_color="gray")
        self._update_sorrend_result()

    def _on_sorrend_input_change(self, _event=None):
        """Szám input változás - szűrés és validálás."""
        current_text = self.entry_sorrend_no.get()
        filtered = "".join(c for c in current_text if c.isdigit())

        if len(filtered) > 4:
            filtered = filtered[:4]

        if filtered:
            num = int(filtered)
            if num > 6143:
                filtered = "6143"

        if filtered != current_text:
            self.entry_sorrend_no.delete(0, "end")
            self.entry_sorrend_no.insert(0, filtered)

        self._update_sorrend_result()

    def _update_sorrend_result(self):
        """Sorrend eredmény frissítése."""
        try:
            text = self.entry_sorrend_no.get()
            if not text:
                self.lbl_sorrend_result.configure(text="(....)")
                return

            no = int(text)
            if no < 0 or no > 6143:
                self.lbl_sorrend_result.configure(text="(....)")
                return

            mode = "SZP" if self.switch_sorrend_mode.get() else "HK"
            result = get_settings(no, mode)

            if result:
                self.lbl_sorrend_result.configure(text=result)
            else:
                self.lbl_sorrend_result.configure(text="(....)")
        except ValueError:
            self.lbl_sorrend_result.configure(text="(....)")

    # === KONTROLLOK ===

    def _on_language_change(self):
        """Nyelv váltás."""
        self.sound.play_toggle_switch()
        if self.switch_lang.get():
            self.current_language = "EN"
            self.label_en.configure(text_color="white")
            self.label_hu.configure(text_color="gray")
        else:
            self.current_language = "HU"
            self.label_en.configure(text_color="gray")
            self.label_hu.configure(text_color="white")

        self.translator.set_language(self.current_language)
        self._log(f"Language: {self.current_language}")
        self._save_settings_now()

    def _on_dark_mode_toggle(self):
        """Dark mode váltás."""
        self.sound.play_toggle_switch()
        self.is_dark_mode = self.switch_theme.get()
        mode = "dark" if self.is_dark_mode else "light"
        ctk.set_appearance_mode(mode)
        self._save_settings_now()

    def _on_mute_toggle(self):
        """Mute váltás."""
        self.is_muted = self.var_muted.get()
        self.sound.set_enabled(not self.is_muted)
        if not self.is_muted:
            self.sound.play_checkbox_on()
        self._save_settings_now()

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
        self._log(f"Feature mode: {mode}")
        self._save_settings_now()

        # Ha már van adat, újraszámoljuk a feature-öket
        if self.raw_data is not None and not self.raw_data.empty:
            threading.Thread(target=self._recalculate_features, daemon=True).start()

    def _recalculate_features(self):
        """Feature-ök újraszámolása háttérszálon."""
        try:
            self.after(0, lambda: self._log(tr("Recalculating features...")))

            # Memory cleanup before recalculation
            with self.data_lock:
                if self.processed_data is not None:
                    del self.processed_data
                    self.processed_data = None
            gc.collect()

            processed = DataProcessor.clean_data(self.raw_data.copy())
            processed = self._apply_feature_mode(processed)

            with self.data_lock:
                self.processed_data = processed

            # Log feature count
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
            self._save_settings_now()  # Immediate save
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
                rows = summary['rows']
                strats = summary['strategies']
                files = summary['files']
                loaded = tr('Loaded:')
                r_txt, s_txt, f_txt = tr('rows'), tr('strategies'), tr('files')
                msg = f"{loaded} {rows} {r_txt}, {strats} {s_txt}, {files} {f_txt}"
                if summary['features'] > 0:
                    msg += f", {summary['features']} features"
                self.after(0, lambda m=msg: self._log(m))
                self.after(0, lambda: self.sound.play_model_complete())
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
            self._save_settings_now()  # Immediate save
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
                rows = summary['rows']
                strats = summary['strategies']
                msg = f"{tr('Loaded:')} {rows} {tr('rows')}, {strats} {tr('strategies')}"
                if summary['features'] > 0:
                    msg += f", {summary['features']} features"
                self.after(0, lambda m=msg: self._log(m))
                self.after(0, lambda: self.sound.play_model_complete())
            else:
                no_data_msg = tr("No data in parquet files.")
                self.after(0, lambda: self._log(no_data_msg, "warning"))

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
            self._save_settings_now()  # Immediate save
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
            self.after(0, lambda: self.sound.play_model_complete())

            self.after(500, lambda: self.set_progress(0))

        except Exception as e:
            self.after(0, lambda: self._log(f"{tr('Conversion error:')} {e}", "critical"))
            self.after(0, lambda: self.set_progress(0))

    def _update_data_ui(self):
        """Data tab UI frissítése betöltött adatokkal."""
        if self.processed_data is None or self.processed_data.empty:
            return

        # Files lista frissítése
        self.files_textbox.delete("1.0", "end")
        if "SourceFile" in self.processed_data.columns:
            files = self.processed_data["SourceFile"].unique()
            for f in sorted(files):
                self.files_textbox.insert("end", f"- {f}\n")

        # Adat előnézet (első 50 sor)
        self.data_preview.delete("1.0", "end")

        # Oszlopok
        cols = list(self.processed_data.columns)
        header = " | ".join(str(c)[:12].ljust(12) for c in cols[:10])
        if len(cols) > 10:
            header += " | ..."
        self.data_preview.insert("end", header + "\n")
        self.data_preview.insert("end", "-" * len(header) + "\n")

        # Adatok
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

        help_text = f"""{tr("FEATURE MODES")}

1. {tr("ORIGINAL")}
   {tr("No additional features - uses only raw data columns.")}
   {tr("Best for: Quick analysis, baseline testing.")}
   {tr("Speed: Fastest")}

2. {tr("FORWARD CALC (Expanding Window)")}
   {tr("Features calculated from ENTIRE history up to each point.")}
   Features: weeks_count, active_ratio, profit_consistency,
   total_profit, cumulative_trades, volatility, sharpe_ratio,
   max_drawdown
   {tr("Best for: Strategy profiling, long-term patterns.")}
   {tr("Speed: Medium")}

3. {tr("ROLLING WINDOW (13 weeks)")}
   {tr("Features calculated from sliding 13-week window.")}
   Features: rolling_active_ratio, rolling_profit_consistency,
   rolling_sharpe, rolling_volatility, rolling_avg_profit,
   rolling_momentum_4w, rolling_momentum_13w, rolling_max_dd
   {tr("Best for: Capturing recent trends, time-series ML.")}
   {tr("Speed: Slowest")}
"""
        text.insert("1.0", help_text)
        text.configure(state="disabled")

        ctk.CTkButton(
            popup,
            text=tr("Close"),
            command=popup.destroy
        ).pack(pady=10)

    # === LOG ===

    def _log(self, message: str, level: str = "normal"):
        """Log üzenet timestamp-tel."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if level == "critical":
            prefix = "[ERROR] "
        elif level == "warning":
            prefix = "[WARN] "
        else:
            prefix = ""

        formatted = f"[{timestamp}] {prefix}{message}\n"
        self.log_textbox.insert("end", formatted)
        self.log_textbox.see("end")

    def set_progress(self, value: float):
        """Progress bar beállítása (0.0 - 1.0)."""
        self.progress_bar.set(value)
        self.progress_label.configure(text=f"{int(value * 100)}%")


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = MBOApp()
    app.mainloop()
