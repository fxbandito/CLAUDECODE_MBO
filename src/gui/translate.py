"""
Translation system for MBO Trading Strategy Analyzer v5.
Simple and extensible HU/EN translation support.

Usage:
    from gui.translate import Translator, tr

    # Create translator instance
    translator = Translator()

    # Get translation
    text = translator.tr("Loading files...")

    # Change language
    translator.set_language("HU")

    # Or use the global function after setting language
    text = tr("Loading files...")
"""

from typing import Dict, Optional


class Translator:
    """Translation manager with HU/EN support."""

    # Translation dictionary: {"EN text": "HU text"}
    TRANSLATIONS: Dict[str, str] = {
        # === Data Loading Tab ===
        "No folder selected": "Nincs mappa kiválasztva",
        "Open Folder": "Mappa megnyitása",
        "Open Parquet": "Parquet megnyitása",
        "Convert Excel to Parquet": "Excel konvertálása Parquet-ba",
        "Feature Mode:": "Feature mód:",
        "Original": "Eredeti",
        "Forward Calc": "Előre számolt",
        "Rolling Window": "Gördülő ablak",
        "No additional features - raw data only": "Nincs extra feature - csak nyers adat",
        "Features from entire history (expanding window)": "Feature-ök a teljes előzményből (bővülő ablak)",
        "Features from rolling 13-week window (dynamic)": "Feature-ök 13 hetes gördülő ablakból (dinamikus)",
        "Files:": "Fájlok:",
        "Loading folder:": "Mappa betöltése:",
        "Loading files...": "Fájlok betöltése...",
        "Loaded:": "Betöltve:",
        "rows": "sor",
        "strategies": "stratégia",
        "files": "fájl",
        "No data found in folder.": "Nincs adat a mappában.",
        "No data in parquet files.": "Nincs adat a parquet fájlokban.",
        "parquet file(s) selected": "parquet fájl kiválasztva",
        "Converting Excel files in:": "Excel fájlok konvertálása:",
        "Conversion complete! Saved:": "Konverzió kész! Mentve:",
        "Conversion error:": "Konverziós hiba:",
        "Error loading folder:": "Hiba a mappa betöltésekor:",
        "Error loading parquet:": "Hiba a parquet betöltésekor:",
        "Recalculating features...": "Feature-ök újraszámolása...",
        "Feature calculation error:": "Feature számítási hiba:",
        "Forward features added:": "Előre számolt feature-ök hozzáadva:",
        "Rolling features added:": "Gördülő feature-ök hozzáadva:",
        "new columns": "új oszlop",

        # === Header / Controls ===
        "Language:": "Nyelv:",
        "Dark Mode": "Sötét mód",
        "Mute": "Némítás",

        # === Tabs ===
        "Data Loading": "Adat betöltés",
        "Analysis": "Elemzés",
        "Results": "Eredmények",
        "Comparison": "Összehasonlítás",
        "Inspection": "Vizsgálat",
        "Performance": "Teljesítmény",
        "Optuna": "Optuna",

        # === General ===
        "Application started. Ready.": "Alkalmazás elindult. Kész.",
        "Settings saved.": "Beállítások mentve.",
        "Close": "Bezárás",
        "Help": "Súgó",
        "Error": "Hiba",
        "Warning": "Figyelmeztetés",
        "Success": "Sikeres",
        "Cancel": "Mégse",
        "OK": "OK",
        "Yes": "Igen",
        "No": "Nem",

        # === Feature Mode Help ===
        "Feature Mode - Help": "Feature mód - Súgó",
        "FEATURE MODES": "FEATURE MÓDOK",
        "ORIGINAL": "EREDETI",
        "No additional features - uses only raw data columns.": "Nincs extra feature - csak a nyers adat oszlopok.",
        "Best for: Quick analysis, baseline testing.": "Legjobb: Gyors elemzés, alapvonal tesztelés.",
        "Speed: Fastest": "Sebesség: Leggyorsabb",
        "FORWARD CALC (Expanding Window)": "ELŐRE SZÁMOLT (Bővülő ablak)",
        "Features calculated from ENTIRE history up to each point.": "Feature-ök a TELJES előzményből minden pontig.",
        "Best for: Strategy profiling, long-term patterns.": "Legjobb: Stratégia profilozás, hosszú távú minták.",
        "Speed: Medium": "Sebesség: Közepes",
        "ROLLING WINDOW (13 weeks)": "GÖRDÜLŐ ABLAK (13 hét)",
        "Features calculated from sliding 13-week window.": "Feature-ök 13 hetes csúszó ablakból.",
        "Best for: Capturing recent trends, time-series ML.": "Legjobb: Friss trendek, idősor ML.",
        "Speed: Slowest": "Sebesség: Leglassabb",

        # === Log messages ===
        "Application Log": "Alkalmazás napló",
    }

    _instance: Optional["Translator"] = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.language = "EN"

    def set_language(self, lang: str):
        """Set current language ('EN' or 'HU')."""
        self.language = lang.upper()

    def get_language(self) -> str:
        """Get current language."""
        return self.language

    def tr(self, text: str) -> str:
        """
        Translate text to current language.

        Args:
            text: English text to translate

        Returns:
            Translated text (or original if no translation found)
        """
        if self.language == "EN":
            return text

        # Hungarian - look up translation
        return self.TRANSLATIONS.get(text, text)

    def add_translation(self, en_text: str, hu_text: str):
        """Add a new translation dynamically."""
        self.TRANSLATIONS[en_text] = hu_text


# Global translator instance
_translator: Optional[Translator] = None


def get_translator() -> Translator:
    """Get the global Translator instance."""
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator


def tr(text: str) -> str:
    """
    Global translation function.

    Args:
        text: English text to translate

    Returns:
        Translated text based on current language setting
    """
    return get_translator().tr(text)


def set_language(lang: str):
    """Set the global language."""
    get_translator().set_language(lang)
