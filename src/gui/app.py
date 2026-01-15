"""
MBO Trading Strategy Analyzer - Fő Alkalmazás Osztály
7 tab: Data Loading, Analysis, Results, Comparison, Inspection, Performance, Optuna
"""

import os
import customtkinter as ctk
from typing import Dict
from datetime import datetime
from PIL import Image

from gui.sorrend_data import get_settings


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
    TAB_NAMES = ["Data Loading", "Analysis", "Results", "Comparison", "Inspection", "Performance", "Optuna"]

    def __init__(self):
        super().__init__()

        # Verzió
        self.version = get_version()

        # Ablak beállítások
        self.title(f"MBO Trading Strategy Analyzer {self.version}")
        self.geometry(f"{self.DEFAULT_WIDTH}x{self.DEFAULT_HEIGHT}")
        self.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)

        # Állapot változók
        self.current_language = "EN"
        self.is_dark_mode = True
        self.is_muted = True
        self.current_tab = "Data Loading"

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

        # Első tab megjelenítése
        self._show_tab("Data Loading")
        self._log("Application started. Ready.")

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
                pil_img = Image.open(dark_logo_path)
                aspect = pil_img.width / pil_img.height
                h = 40
                w = int(h * aspect)

                self.logo_image = ctk.CTkImage(
                    light_image=Image.open(light_logo_path),
                    dark_image=Image.open(dark_logo_path),
                    size=(w, h),
                )
                self.logo_label = ctk.CTkLabel(self.logo_frame, text="", image=self.logo_image)
                self.logo_label.pack(side="left")
        except Exception:
            # Fallback szöveg logo
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

        # H-K label
        self.label_hk = ctk.CTkLabel(
            self.sorrend_frame,
            text="H-K",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="white"
        )
        self.label_hk.pack(side="left", padx=(0, 3))

        # Switch
        self.switch_sorrend_mode = ctk.CTkSwitch(
            self.sorrend_frame,
            text="",
            width=40,
            command=self._on_sorrend_mode_change
        )
        self.switch_sorrend_mode.pack(side="left", padx=(0, 3))

        # SZ-P label
        self.label_szp = ctk.CTkLabel(
            self.sorrend_frame,
            text="SZ-P",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray"
        )
        self.label_szp.pack(side="left", padx=(0, 8))

        # Szám input (0-6143, max 4 digit)
        self.entry_sorrend_no = ctk.CTkEntry(
            self.sorrend_frame,
            width=55,
            height=28,
            placeholder_text="0000",
            font=ctk.CTkFont(size=12)
        )
        self.entry_sorrend_no.pack(side="left", padx=(0, 8))
        self.entry_sorrend_no.bind("<KeyRelease>", self._on_sorrend_input_change)

        # Eredmény label (zöld, zárójelben)
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

        # Dark Mode toggle
        self.switch_theme = ctk.CTkSwitch(
            self.controls_frame,
            text="Dark Mode",
            font=ctk.CTkFont(size=11),
            command=self._on_dark_mode_toggle
        )
        self.switch_theme.select()
        self.switch_theme.pack(side="right", padx=(15, 0))

        # EN label
        self.label_en = ctk.CTkLabel(
            self.controls_frame,
            text="EN",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="white"
        )
        self.label_en.pack(side="right", padx=(0, 5))

        # Nyelv switch
        self.switch_lang = ctk.CTkSwitch(
            self.controls_frame,
            text="",
            width=40,
            command=self._on_language_change
        )
        self.switch_lang.select()  # EN alapértelmezett
        self.switch_lang.pack(side="right", padx=(0, 3))

        # HU label
        self.label_hu = ctk.CTkLabel(
            self.controls_frame,
            text="HU",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray"
        )
        self.label_hu.pack(side="right", padx=(0, 3))

        # Mute checkbox
        self.var_muted = ctk.BooleanVar(value=True)
        self.chk_mute = ctk.CTkCheckBox(
            self.controls_frame,
            text="Mute",
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

        # Tab frame-ek létrehozása
        for tab_name in self.TAB_NAMES:
            if tab_name == "Data Loading":
                tab_frame = self._create_data_tab()
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
            text="No folder selected",
            font=ctk.CTkFont(size=12),
            anchor="w"
        )
        self.folder_label.pack(side="left")

        # Gombok jobb oldalon
        self.btn_open_folder = ctk.CTkButton(
            top_frame,
            text="Open Folder",
            width=110,
            height=32,
            fg_color="#3498db",
            hover_color="#2980b9",
            command=self._on_open_folder
        )
        self.btn_open_folder.pack(side="right", padx=5)

        self.btn_open_parquet = ctk.CTkButton(
            top_frame,
            text="Open Parquet",
            width=110,
            height=32,
            fg_color="#3498db",
            hover_color="#2980b9",
            command=self._on_open_parquet
        )
        self.btn_open_parquet.pack(side="right", padx=5)

        self.btn_convert = ctk.CTkButton(
            top_frame,
            text="Convert Excel to Parquet",
            width=160,
            height=32,
            fg_color="#f39c12",
            hover_color="#d68910",
            text_color="black",
            command=self._on_convert_excel
        )
        self.btn_convert.pack(side="right", padx=5)

        # Feature Mode sor
        feature_frame = ctk.CTkFrame(frame, fg_color="transparent", height=40)
        feature_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(
            feature_frame,
            text="Feature Mode:",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(0, 10))

        self.feature_var = ctk.StringVar(value="Original")

        self.radio_original = ctk.CTkRadioButton(
            feature_frame,
            text="Original",
            variable=self.feature_var,
            value="Original",
            command=self._on_feature_mode_change
        )
        self.radio_original.pack(side="left", padx=10)

        self.radio_forward = ctk.CTkRadioButton(
            feature_frame,
            text="Forward Calc",
            variable=self.feature_var,
            value="Forward Calc",
            command=self._on_feature_mode_change
        )
        self.radio_forward.pack(side="left", padx=10)

        self.radio_rolling = ctk.CTkRadioButton(
            feature_frame,
            text="Rolling Window",
            variable=self.feature_var,
            value="Rolling Window",
            command=self._on_feature_mode_change
        )
        self.radio_rolling.pack(side="left", padx=10)

        # ? gomb
        ctk.CTkButton(
            feature_frame,
            text="?",
            width=25,
            height=25,
            corner_radius=12,
            fg_color="#555555",
            hover_color="#777777"
        ).pack(side="left", padx=10)

        # Feature info
        self.feature_info_label = ctk.CTkLabel(
            feature_frame,
            text="No additional features - raw data only",
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
            text="Files:",
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
            text="Application Log",
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
        self.current_tab = tab_name

        for name, tab in self.tabs.items():
            tab.grid_remove()

        if tab_name in self.tabs:
            self.tabs[tab_name].grid()

        # Gomb stílusok
        for name, btn in self.tab_buttons.items():
            if name == tab_name:
                btn.configure(fg_color="#4a4a6a", text_color="#FFFFFF")
            else:
                btn.configure(fg_color="#2a2a3e", text_color="#888888")

    # === SORREND KEZELÉS ===

    def _on_sorrend_mode_change(self):
        """H-K / SZ-P mód váltás."""
        if self.switch_sorrend_mode.get():
            # SZ-P aktív
            self.label_szp.configure(text_color="white")
            self.label_hk.configure(text_color="gray")
        else:
            # H-K aktív
            self.label_hk.configure(text_color="white")
            self.label_szp.configure(text_color="gray")
        self._update_sorrend_result()

    def _on_sorrend_input_change(self, _event=None):
        """Szám input változás - szűrés és validálás."""
        current_text = self.entry_sorrend_no.get()

        # Csak számok
        filtered = "".join(c for c in current_text if c.isdigit())

        # Max 4 karakter
        if len(filtered) > 4:
            filtered = filtered[:4]

        # 0-6143 tartomány
        if filtered:
            num = int(filtered)
            if num > 6143:
                filtered = "6143"

        # Frissítés ha változott
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
        if self.switch_lang.get():
            self.current_language = "EN"
            self.label_en.configure(text_color="white")
            self.label_hu.configure(text_color="gray")
        else:
            self.current_language = "HU"
            self.label_en.configure(text_color="gray")
            self.label_hu.configure(text_color="white")
        self._log(f"Language: {self.current_language}")

    def _on_dark_mode_toggle(self):
        """Dark mode váltás."""
        self.is_dark_mode = self.switch_theme.get()
        mode = "dark" if self.is_dark_mode else "light"
        ctk.set_appearance_mode(mode)

    def _on_mute_toggle(self):
        """Mute váltás."""
        self.is_muted = self.var_muted.get()

    # === DATA TAB ESEMÉNYEK ===

    def _on_feature_mode_change(self):
        """Feature mode váltás."""
        mode = self.feature_var.get()
        info_texts = {
            "Original": "No additional features - raw data only",
            "Forward Calc": "Forward calculation features enabled",
            "Rolling Window": "Rolling window aggregation enabled"
        }
        self.feature_info_label.configure(text=info_texts.get(mode, ""))
        self._log(f"Feature mode: {mode}")

    def _on_open_folder(self):
        """Mappa megnyitása."""
        from tkinter import filedialog
        folder = filedialog.askdirectory(title="Select Data Folder")
        if folder:
            self.folder_label.configure(text=folder)
            self._log(f"Folder: {folder}")
            self._load_folder_files(folder)

    def _on_open_parquet(self):
        """Parquet fájl megnyitása."""
        from tkinter import filedialog
        file = filedialog.askopenfilename(
            title="Select Parquet File",
            filetypes=[("Parquet files", "*.parquet"), ("All files", "*.*")]
        )
        if file:
            self._log(f"Parquet: {file}")

    def _on_convert_excel(self):
        """Excel → Parquet konverzió."""
        self._log("Convert Excel to Parquet clicked")

    def _load_folder_files(self, folder: str):
        """Mappa fájlok betöltése."""
        from pathlib import Path
        self.files_textbox.delete("1.0", "end")
        folder_path = Path(folder)
        files = list(folder_path.glob("*.xlsx")) + list(folder_path.glob("*.csv")) + list(folder_path.glob("*.parquet"))
        for f in sorted(files):
            self.files_textbox.insert("end", f"{f.name}\n")

    # === LOG ===

    def _log(self, message: str, level: str = "normal"):
        """Log üzenet timestamp-tel."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Szín beállítása szint alapján
        if level == "critical":
            color = "#e74c3c"  # Piros
        elif level == "warning":
            color = "#f39c12"  # Narancs
        else:
            color = None  # Alapértelmezett

        formatted = f"[{timestamp}] {message}\n"
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
