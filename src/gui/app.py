"""
MBO Trading Strategy Analyzer - Fő Alkalmazás Osztály
7 tab: Data Loading, Analysis, Results, Comparison, Inspection, Performance, Optuna
"""
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=too-many-statements,broad-exception-caught,unnecessary-lambda

import os
import threading
from datetime import datetime
from typing import Dict, Optional

import customtkinter as ctk
import pandas as pd
from PIL import Image

from gui.settings import SettingsManager
from gui.sorrend_data import get_settings
from gui.sound_manager import get_sound_manager
from gui.translate import get_translator, tr

from gui.tabs.data_loading import DataLoadingMixin
from gui.tabs.analysis import AnalysisMixin
from gui.tabs.results import ResultsMixin
from gui.tabs.comparison import ComparisonMixin
from gui.tabs.inspection import InspectionTabMixin

from analysis.engine import get_resource_manager


def get_version() -> str:
    """Verzió beolvasása a version.txt fájlból."""
    version_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "version.txt")
    try:
        with open(version_file, "r", encoding="utf-8") as f:
            return f.readline().strip()
    except (FileNotFoundError, IOError):
        return "v5.0.0"


class MBOApp(DataLoadingMixin, AnalysisMixin, ResultsMixin, ComparisonMixin, InspectionTabMixin, ctk.CTk):
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

        # Resource manager (CPU/GPU control)
        self.resource_manager = get_resource_manager()
        self._init_resource_manager()

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

        # Mappa útvonalak betöltése ELŐBB (mielőtt a tabok létrejönnek)
        self._load_folder_paths()

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

    def _load_folder_paths(self):
        """Load saved folder paths from settings."""
        self.last_open_folder = self.settings.get_last_folder()
        self.last_parquet_folder = self.settings.get_last_parquet_folder()
        self.last_convert_folder = self.settings.get_last_convert_folder()
        self.last_results_output_folder = self.settings.get_last_results_output_folder()
        self.last_analysis_state_folder = self.settings.get_last_analysis_state_folder()
        self.last_csv_export_folder = self.settings.get_last_csv_export_folder()

    def save_window_state(self):
        """Save folder paths to settings (called from tabs)."""
        self.settings.set_last_folder(getattr(self, "last_open_folder", ""))
        self.settings.set_last_parquet_folder(getattr(self, "last_parquet_folder", ""))
        self.settings.set_last_convert_folder(getattr(self, "last_convert_folder", ""))
        self.settings.set_last_results_output_folder(
            getattr(self, "last_results_output_folder", "")
        )
        self.settings.set_last_analysis_state_folder(
            getattr(self, "last_analysis_state_folder", "")
        )
        self.settings.set_last_csv_export_folder(
            getattr(self, "last_csv_export_folder", "")
        )
        self.settings.save()

    def _init_resource_manager(self):
        """Initialize ResourceManager with saved settings."""
        cpu_pct = self.settings.get_cpu_percentage()
        gpu_enabled = self.settings.get_gpu_enabled()

        self.resource_manager.set_cpu_percentage(cpu_pct)
        self.resource_manager.set_gpu_enabled(gpu_enabled)

    def _save_resource_settings(self):
        """Save current resource settings."""
        self.settings.set_cpu_percentage(self.resource_manager.cpu_percentage)
        self.settings.set_gpu_enabled(self.resource_manager.gpu_enabled)

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
        # Save resource settings
        self._save_resource_settings()
        # Also save folder paths
        self.save_window_state()

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
        self.grid_rowconfigure(2, weight=2)  # Content - növekszik (kevésbé)
        self.grid_rowconfigure(3, weight=1)  # Log panel - növekszik
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
            elif tab_name == "Comparison":
                tab_frame = self._create_comparison_tab()
            elif tab_name == "Inspection":
                tab_frame = self._create_inspection_tab()
            else:
                tab_frame = self._create_placeholder_tab(tab_name)
            tab_frame.grid(row=0, column=0, sticky="nsew")
            self.tabs[tab_name] = tab_frame

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
        self.log_frame = ctk.CTkFrame(self, corner_radius=0)
        self.log_frame.grid(row=3, column=0, sticky="nsew", padx=20, pady=(10, 20))

        ctk.CTkLabel(
            self.log_frame,
            text=tr("Application Log"),
            font=ctk.CTkFont(size=11, weight="bold"),
            anchor="w"
        ).pack(anchor="w", padx=10, pady=5)

        self.log_textbox = ctk.CTkTextbox(
            self.log_frame,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#1a1a2e"
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
