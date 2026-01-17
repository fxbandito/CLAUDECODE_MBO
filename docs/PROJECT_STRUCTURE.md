# MBO Trading Strategy Analyzer v5 - Projekt Dokumentáció

> **Verzió:** v5.5.2
> **Utolsó frissítés:** 2026-01-17
> **Státusz:** Kiindulási dokumentum - folyamatosan bővül

---

## Tartalomjegyzék

1. [Összefoglaló](#összefoglaló)
2. [Projekt Struktúra](#projekt-struktúra)
3. [Belépési Pontok](#1-belépési-pontok)
4. [Data Réteg](#2-data---adat-réteg)
5. [GUI Réteg](#3-gui---felhasználói-felület)
6. [Analysis Réteg](#4-analysis---elemzési-motor)
7. [Models Réteg](#5-models---modellek-könyvtára)
8. [Reporting Réteg](#6-reporting---riport-generálás)
9. [Utils Réteg](#7-utils---segédeszközök)
10. [Fájlok Fontossági Sorrendje](#fájlok-fontossági-sorrendje)
11. [Adatfolyam Diagram](#adatfolyam-diagram)
12. [Architektúra Minták](#architektúra-minták)

---

## Összefoglaló

A projekt egy **kereskedési stratégia elemző alkalmazás**, amely előrejelzési modellekkel elemzi a stratégiák profitabilitását. Az architektúra modern, többrétegű felépítésű:

- **152 Python fájl** 8 fő könyvtárban
- **GUI Réteg**: CustomTkinter desktop felület 7 tabbal
- **Data Réteg**: Excel/Parquet betöltés és feature engineering
- **Analysis Réteg**: Multiprocessing motor rekurzív előrejelzéssel
- **Models Réteg**: 80+ előrejelzési modell 13 kategóriában
- **Reporting Réteg**: HTML/Markdown riport generálás
- **Utils Réteg**: Logging, erőforrás kezelés, fordítás

Az alkalmazás **multiprocessing architektúrával** biztosítja, hogy a GUI soha ne fagyjon le az elemzés során.

---

## Projekt Struktúra

```
src/
├── main.py              ← Fő belépési pont
├── main_debug.py        ← Debug módú indítás
├── data/                ← Adat betöltés és feldolgozás
│   ├── __init__.py
│   ├── loader.py        ← DataLoader osztály
│   └── processor.py     ← DataProcessor osztály
├── gui/                 ← Felhasználói felület (7 tab)
│   ├── __init__.py
│   ├── app.py           ← MBOApp fő ablak
│   ├── settings.py      ← SettingsManager
│   ├── auto_window.py   ← AutoExecManager
│   ├── sound_manager.py ← SoundManager
│   ├── translate.py     ← Translator
│   ├── sorrend_data.py  ← Globális beállítások
│   └── tabs/            ← Tab-specifikus mixinek
│       ├── data_loading.py
│       ├── analysis.py
│       ├── results.py
│       ├── comparison.py
│       └── inspection.py
├── analysis/            ← Elemzési motor és worker-ek
│   ├── __init__.py
│   ├── engine.py        ← ResourceManager, AnalysisEngine
│   ├── worker.py        ← AnalysisWorkerManager
│   ├── process_utils.py ← Worker segédfüggvények
│   ├── dual_executor.py ← Dual-model végrehajtó
│   ├── dual_task.py     ← Rekurzív előrejelzés
│   ├── panel_executor.py← Panel mód végrehajtó
│   ├── inspection.py    ← InspectionEngine
│   ├── metrics.py       ← FinancialMetrics
│   └── comparator/      ← Riport összehasonlító modulok
│       ├── base.py
│       ├── horizon.py
│       ├── main_data.py
│       ├── main_data_ar.py
│       └── main_data_mr.py
├── models/              ← 80+ előrejelzési modell
│   ├── __init__.py      ← Model registry
│   ├── base.py          ← BaseModel absztrakt osztály
│   ├── utils/
│   │   └── postprocessing.py
│   └── [13 kategória mappa...]
├── reporting/           ← Riport generálás (MD/HTML)
│   ├── __init__.py
│   ├── exporter.py      ← ReportExporter
│   └── visualizer.py    ← Visualizer
└── utils/               ← Segédeszközök
    ├── __init__.py
    └── logging_utils.py ← Logging rendszer
```

---

## 1. BELÉPÉSI PONTOK

### main.py
**Útvonal:** `src/main.py`
**Szerep:** Fő alkalmazás indító

**Funkciók:**
- Környezeti változók beállítása (PyTorch, Julia)
- Debug logging konfigurálás (`MBO_DEBUG_MODE`)
- CustomTkinter megjelenés beállítása
- MBOApp ablak létrehozása és futtatása
- Multiprocessing freeze support (Windows exe-hez)

**Kód struktúra:**
```python
if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Környezeti változók
    # Debug logging
    # ctk.set_appearance_mode()
    app = MBOApp()
    app.mainloop()
```

---

### main_debug.py
**Útvonal:** `src/main_debug.py`
**Szerep:** Debug módú indítás

**Funkciók:**
- `MBO_DEBUG_MODE=1` környezeti változó beállítása
- Teljes log kimenet a `Log/` mappába
- Részletes hibakeresési információk
- Ugyanazt a main.py-t hívja, csak debug módban

---

## 2. DATA/ - Adat Réteg

### data/loader.py
**Útvonal:** `src/data/loader.py`
**Osztály:** `DataLoader` (statikus metódusok)

| Metódus | Paraméterek | Visszatérés | Funkció |
|---------|-------------|-------------|---------|
| `load_file()` | `filepath: str` | `pd.DataFrame` | Excel/Parquet fájl betöltés speciális blokk struktúrával |
| `load_folder()` | `folder_path: str` | `pd.DataFrame` | Párhuzamos mappa betöltés (joblib, max 8 worker) |
| `load_parquet_files()` | `file_list: list` | `pd.DataFrame` | Több parquet fájl összefűzése |
| `convert_excel_to_parquet()` | `excel_path: str` | `str` | Excel → Parquet konverzió |
| `get_file_list()` | `folder: str` | `list` | Kompatibilis fájlok listázása (xlsx, parquet, csv) |

**Excel formátum:**
- 0-1. sor: fejlécek és dátumok
- 2+ sor: adat blokkok

---

### data/processor.py
**Útvonal:** `src/data/processor.py`
**Osztály:** `DataProcessor` (statikus metódusok)

| Metódus | Funkció | Kimenet |
|---------|---------|---------|
| `clean_data()` | Adat tisztítás (%, negatív számok, NaN) | Tisztított DataFrame |
| `prepare_for_analysis()` | Rendezés stratégia ID és dátum szerint | Rendezett DataFrame |
| `add_features_forward()` | Expanding window feature-ök | +8 oszlop |
| `add_features_rolling()` | Rolling 13-hetes feature-ök | +9 oszlop |
| `calculate_stability_metrics()` | Történeti stabilitás pontszám | +4 metrika |
| `apply_ranking()` | Stratégia rangsorolás | Rangsorolt DataFrame |
| `group_strategies()` | O(1) dict lookup optimalizáció | Dict[str, DataFrame] |
| `detect_data_mode()` | Adat mód detektálás | "rolling"/"forward"/"original" |

**Feature oszlopok (forward mód):**
- `feat_weeks_count`, `feat_active_ratio`, `feat_profit_consistency`
- `feat_total_profit`, `feat_cumulative_trades`, `feat_volatility`
- `feat_sharpe_ratio`, `feat_max_drawdown`

**Rangsorolási módok:**
1. `forecast` - Előrejelzés alapú
2. `stability_weighted` - Stabilitás súlyozott
3. `risk_adjusted` - Kockázat korrigált

---

## 3. GUI/ - Felhasználói Felület

### gui/app.py ⭐ KULCSFONTOSSÁGÚ
**Útvonal:** `src/gui/app.py`
**Osztály:** `MBOApp(DataLoadingMixin, AnalysisMixin, ResultsMixin, ComparisonMixin, InspectionTabMixin, ctk.CTk)`

**Architektúra:** Mixin-alapú öröklődés 5 tab mixin-ből

**7 Tab:**
1. Data Loading - Adat betöltés
2. Analysis - Elemzés futtatás
3. Results - Eredmények megjelenítés
4. Comparison - Riport összehasonlítás
5. Inspection - Előrejelzés validálás
6. Performance - Teljesítmény metrikák
7. Optuna - Hiperparaméter optimalizálás

**Ablak tulajdonságok:**
- Méret: 1600x1080
- Minimum: 1200x700

**Integrált komponensek:**
- `SettingsManager` - Beállítások
- `SoundManager` - Hangok
- `Translator` - Fordítás
- `ResourceManager` - Erőforrások

---

### gui/settings.py
**Útvonal:** `src/gui/settings.py`
**Osztály:** `SettingsManager`

**Perzisztencia:** JSON (`gui/window_config.json`)

**Tárolt beállítások:**
- Ablak geometria (pozíció, méret)
- Auto execution beállítások
- Utolsó használt útvonalak
- Felhasználói preferenciák
- Erőforrás beállítások

---

### gui/auto_window.py
**Útvonal:** `src/gui/auto_window.py`
**Osztály:** `AutoExecManager`

**Funkció:** Ütemezett elemzések automatikus futtatása

---

### gui/sound_manager.py
**Útvonal:** `src/gui/sound_manager.py`
**Osztály:** `SoundManager` (Singleton)

**Hangeffektek (9 db):**
- `app_start`, `app_close`
- `tab_switch`, `button_click`
- `analysis_start`, `analysis_complete`, `analysis_error`
- `export_complete`, `notification`

**Backend-ek:**
- Windows: `winsound`
- Cross-platform: `pygame`

**Lejátszás:** `ThreadPoolExecutor` háttérszálban

---

### gui/translate.py
**Útvonal:** `src/gui/translate.py`
**Osztály:** `Translator`

**Funkció:** HU/EN szótár alapú fordítás

**Használat:**
```python
from gui.translate import tr
label = tr("Betöltés")  # Returns "Loading" if EN mode
```

---

### GUI/TABS/ - Tab Mixinek

| Fájl | Mixin Osztály | Tab | Fő funkciók |
|------|---------------|-----|-------------|
| `data_loading.py` | `DataLoadingMixin` | Data Loading | Fájl kiválasztás, betöltés, feature mód választás |
| `analysis.py` | `AnalysisMixin` | Analysis | Modell kategória/választás, paraméterek, futtatás |
| `results.py` | `ResultsMixin` | Results | Eredmény táblázat, rangsorolás, export |
| `comparison.py` | `ComparisonMixin` | Comparison | Riport összehasonlítás horizont szerint |
| `inspection.py` | `InspectionTabMixin` | Inspection | Előrejelzés vs benchmark validálás |

---

## 4. ANALYSIS/ - Elemzési Motor

### analysis/engine.py ⭐ KULCSFONTOSSÁGÚ
**Útvonal:** `src/analysis/engine.py`

#### ResourceManager (Singleton)
**Funkció:** Központi erőforrás kezelés

| Attribútum | Típus | Leírás |
|------------|-------|--------|
| `physical_cores` | int | Fizikai CPU magok száma |
| `logical_cores` | int | Logikai CPU magok száma |
| `cpu_percentage` | float | Használandó CPU % |
| `gpu_available` | bool | GPU elérhetőség |
| `gpu_devices` | list | Elérhető GPU eszközök |

#### AnalysisEngine
**Funkció:** Fő előrejelzési orchestrátor

| Metódus | Funkció |
|---------|---------|
| `run_analysis()` | Teljes elemzés futtatás |
| `run_single_model()` | Egy modell futtatása |
| `run_dual_model()` | Dual-model futtatás |
| `detect_data_mode()` | Adat mód detektálás |

---

### analysis/worker.py ⭐ KULCSFONTOSSÁGÚ
**Útvonal:** `src/analysis/worker.py`
**Osztály:** `AnalysisWorkerManager`

**Architektúra:** Külön process az elemzéshez

**Kommunikáció:**
```
GUI Process          Worker Process
    │                      │
    ├──[start]──────────>  │
    │                [analysis runs]
    │  <──[progress_queue]─┤
    │  <──[result_queue]───┤
    │                [cleanup CUDA]
```

**Dataclass-ok:**
```python
@dataclass
class WorkerProgress:
    total_strategies: int
    completed_strategies: int
    current_strategy: str
    is_running: bool
    is_paused: bool
    is_cancelled: bool
    error: Optional[str]

@dataclass
class WorkerResult:
    success: bool
    results: Dict[str, Any]
    elapsed_seconds: float
    error: Optional[str]
```

---

### analysis/dual_task.py ⭐ KULCSFONTOSSÁGÚ
**Útvonal:** `src/analysis/dual_task.py`

**Dual Model Architektúra:**
```
┌─────────────────────────────────────────────┐
│  Activity Model → Aktivitás valószínűség    │
│                   (0.0 - 1.0)               │
├─────────────────────────────────────────────┤
│  Profit Model → Profit per aktív hét        │
├─────────────────────────────────────────────┤
│  Végső = Activity × Expected_Weeks × Profit │
└─────────────────────────────────────────────┘
```

**Rekurzív előrejelzés:**
```python
def apply_recursive_forecasting(model, initial_features, horizon):
    history_buffer = [...]  # 30 elem (lag_26-hoz elég)
    forecasts = []

    for step in range(horizon):
        # 1. Előrejelzés aktuális feature-ökkel
        pred = model.predict(features)
        forecasts.append(pred)

        # 2. Lag feature-ök frissítése
        update_lag_features(features, history_buffer, pred)

    return forecasts
```

**Lag oszlopok:** `[lag_1, lag_2, lag_4, lag_8, lag_13, lag_26]`

---

### analysis/panel_executor.py
**Útvonal:** `src/analysis/panel_executor.py`

**Panel mód:** Egy modell az összes stratégiára (gyorsabb ML modellekhez)

**Előnyök:**
- Sokkal gyorsabb ML modellekhez
- Közös minták tanulása
- Kevesebb memória használat

---

### analysis/inspection.py
**Útvonal:** `src/analysis/inspection.py`
**Osztály:** `InspectionEngine`

**Funkció:** Előrejelzés pontosság validálás

**Dataclass-ok:**
```python
@dataclass
class ForecastRecord:
    model: str
    forecast_year: int
    horizon: str
    predicted_rank: int
    strategy_no: int
    predicted_profit: float
    currency_pair: str
    training_years: str

@dataclass
class BenchmarkRecord:
    strategy_no: int
    actual_profit: float
    actual_rank: int

@dataclass
class ComparisonRecord:
    forecast: ForecastRecord
    benchmark: BenchmarkRecord
    rank_difference: int
    profit_difference: float
```

---

### analysis/metrics.py
**Útvonal:** `src/analysis/metrics.py`
**Osztály:** `FinancialMetrics`

| Metrika | Számítás |
|---------|----------|
| Total Profit | Összesített profit |
| Win Rate | Nyerő hetek / összes hét |
| Profit Factor | Gross profit / Gross loss |
| Average Trade | Átlagos heti profit |
| Max Drawdown | Maximális visszaesés |
| Sharpe Ratio | (Return - Rf) / StdDev (évesített) |
| Sortino Ratio | Return / Downside StdDev |
| Recovery Factor | Total Profit / Max Drawdown |
| Calmar Ratio | Annual Return / Max Drawdown |

---

### ANALYSIS/COMPARATOR/ - Riport Összehasonlító

| Fájl | Funkció |
|------|---------|
| `base.py` | `scan_reports()`, `is_aggregate_report()`, `parse_report()` |
| `horizon.py` | Horizont alapú összehasonlítás (1W, 1M, 3M, 6M, 1Y) |
| `main_data.py` | Main Data riport aggregálás |
| `main_data_ar.py` | All Results (AR_) heti bontás összehasonlítás |
| `main_data_mr.py` | Monthly Results (MR_) összehasonlítás |

---

## 5. MODELS/ - Modellek Könyvtára

### models/base.py
**Útvonal:** `src/models/base.py`

**BaseModel absztrakt osztály:**
```python
class BaseModel(ABC):
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Modell tanítása"""
        pass

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Előrejelzés generálása"""
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Modell metaadatok"""
        pass

    # Opcionális
    def create_dual_regressor(self) -> Any:
        """Dual-model módhoz"""
        pass

    def create_panel_regressor(self) -> Any:
        """Panel módhoz"""
        pass
```

**ModelInfo dataclass:**
```python
@dataclass
class ModelInfo:
    name: str
    category: str
    supports_gpu: bool
    supports_batch: bool
    description: str
```

---

### models/__init__.py
**Útvonal:** `src/models/__init__.py`

**Model Registry:**
- Dinamikus modul felfedezés a models/ almappákból
- `MODEL_INFO` kinyerése minden modellből
- `PARAM_DEFAULTS` és `PARAM_OPTIONS` GUI konfigurációhoz
- `CATEGORY_ORDER` és `MODEL_ORDER` konzisztens UI sorrendhez

---

### Modell Kategóriák (13 kategória, 80+ modell)

| # | Kategória | Modellek | Példák |
|---|-----------|----------|--------|
| 1 | Statistical | 13 | ARIMA, SARIMA, VAR, GAM, GARCH, OGARCH |
| 2 | Smoothing & Decomposition | 5 | ETS, STL, MSTL, Theta |
| 3 | Classical ML | 6 | XGBoost, LightGBM, Random Forest, SVR |
| 4 | Deep Learning - RNN | 6 | LSTM, GRU, DeepAR, Seq2Seq |
| 5 | Deep Learning - CNN/Hybrid | 6 | N-BEATS, N-HiTS, TCN, TiDE, TimesNet |
| 6 | Deep Learning - Transformer | 8 | TFT, PatchTST, Informer, Autoformer |
| 7 | Graph & Specialized | 12 | Neural ODE, KAN, StemGNN, MTGNN |
| 8 | Meta-Learning & AutoML | 7 | DARTS, FFORMA, NAS, MoE |
| 9 | Bayesian & Probabilistic | 5 | Prophet, BSTS, Gaussian Process |
| 10 | Frequency Domain | 7 | FFT, Wavelet, SSA, Spectral |
| 11 | Distance & Similarity | 4 | DTW, k-NN, k-Shape, Matrix Profile |
| 12 | State Space | 4 | Kalman Filter, State Space Model |
| 13 | Symbolic Regression | 3 | GPlearn, PySR, PySindy |

---

## 6. REPORTING/ - Riport Generálás

### reporting/exporter.py
**Útvonal:** `src/reporting/exporter.py`
**Osztály:** `ReportExporter`

**Funkciók:**
- Markdown riport generálás
- HTML riport generálás beágyazott CSS-sel
- Composite best számítás (4-4-5 naptár minta)
- Horizont aggregálás (1W, 1M, 3M, 6M, 12M)

---

### reporting/visualizer.py
**Útvonal:** `src/reporting/visualizer.py`
**Osztály:** `Visualizer`

| Metódus | Kimenet | Funkció |
|---------|---------|---------|
| `plot_forecast()` | PNG | Történeti + előrejelzett profit grafikon |
| `plot_comparison()` | PNG | Top 10 stratégia oszlopdiagram |

**Backend:** Matplotlib "Agg" (thread-safe)

---

## 7. UTILS/ - Segédeszközök

### utils/logging_utils.py
**Útvonal:** `src/utils/logging_utils.py`

**LogCategory enum:**
- `APP`, `UI`, `DATA`, `ANALYSIS`, `MODEL`, `REPORT`, `SYSTEM`, `CONFIG`, `ERROR`

**LogLevel szintek:**
- `info`, `debug`, `warning`, `error`, `critical`, `success`, `highlight`
- GUI-ban látható: `info`, `warning`, `error`, `critical`, `success`, `highlight`

**Handler-ek:**
- `GuiLogHandler` - GUI log box-hoz
- `SafeQueueHandler` - Multiprocessing queue-hoz
- `StreamToLogger` - stdout/stderr átirányítás

**Debug mód aktiválás:**
```bash
set MBO_DEBUG_MODE=1
python src/main.py
# vagy
python src/main_debug.py
```

**Log fájlok:** `Log/mbo_debug_YYYY-MM-DD_HH-MM-SS.log`

---

## Fájlok Fontossági Sorrendje

### 🔴 Kritikus (program nem indul nélkülük)

| # | Fájl | Szerep |
|---|------|--------|
| 1 | `src/main.py` | Belépési pont |
| 2 | `src/gui/app.py` | Fő ablak |
| 3 | `src/data/loader.py` | Adat betöltés |
| 4 | `src/analysis/engine.py` | Elemzési motor |
| 5 | `src/analysis/worker.py` | Multiprocessing worker |

### 🟠 Alapvető elemzéshez

| # | Fájl | Szerep |
|---|------|--------|
| 6 | `src/analysis/dual_task.py` | Rekurzív előrejelzés |
| 7 | `src/models/__init__.py` | Modell registry |
| 8 | `src/models/base.py` | Modell alap osztály |

### 🟡 Teljes funkcionalitáshoz

| # | Fájl | Szerep |
|---|------|--------|
| 9 | `src/reporting/exporter.py` | Riport generálás |
| 10 | `src/analysis/inspection.py` | Validálás |
| 11 | `src/gui/tabs/comparison.py` | Összehasonlítás |

---

## Adatfolyam Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  FELHASZNÁLÓ: Excel/Parquet fájl kiválasztás                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  DATA RÉTEG                                                 │
│  ├─ DataLoader.load_file() → Nyers adat                     │
│  ├─ DataProcessor.clean_data() → Tisztított adat            │
│  └─ DataProcessor.add_features_*() → Feature-ök hozzáadva   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  GUI RÉTEG: Modell és paraméter kiválasztás                 │
│  Analysis Tab → Run gomb → AnalysisWorkerManager            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  WORKER PROCESS (külön folyamat - GUI nem fagy)             │
│  ├─ AnalysisEngine orchestrál                               │
│  ├─ Model.fit() → Model.predict()                           │
│  └─ Rekurzív előrejelzés (dual_task.py)                     │
│                                                             │
│  ←── progress_queue (státusz frissítések)                   │
│  ←── result_queue (végeredmények)                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  RESULTS TAB: Eredmények megjelenítése                      │
│  ├─ DataProcessor.apply_ranking() → Rangsorolás             │
│  └─ ReportExporter.generate() → MD/HTML export              │
└─────────────────────────────────────────────────────────────┘
```

---

## Architektúra Minták

| Minta | Hol használják | Előny |
|-------|----------------|-------|
| **Mixin** | GUI tabok (5 mixin) | Moduláris, független tabok, könnyű tesztelés |
| **Singleton** | ResourceManager, SoundManager, Translator | Központi vezérlés, egy példány |
| **Multiprocessing** | AnalysisWorker | GUI nem fagy, párhuzamos feldolgozás |
| **Registry** | Models/__init__.py | Dinamikus modell felfedezés |
| **Queue** | Worker ↔ GUI kommunikáció | Biztonságos process kommunikáció |
| **Abstract Factory** | BaseModel | Egységes modell interfész |
| **Observer** | ResourceManager callbacks | GUI frissítések |

---

## Verzió Történet

| Verzió | Dátum | Változások |
|--------|-------|------------|
| v5.5.2 | 2026-01 | Multiprocessing architektúra, GUI soha nem fagy |
| v5.5.1 | 2026-01 | Auto exec beállítások perzisztencia, GUI optimalizálás, Log színek |
| v5.5.0 | 2026-01 | Auto Execution Manager implementálás |
| v5.4.8 | 2026-01 | ARIMA modell, GUI Batch Mode toggle |
| v5.4.7 | 2026-01 | Riport generálás javítások, ADIDA modell fejlesztések |

---

## TODO - Bővítendő szekciók

- [ ] Részletes modell dokumentáció minden kategóriához
- [ ] API referencia minden publikus metódushoz
- [ ] Konfigurációs opciók teljes listája
- [ ] Hibakezelési stratégiák
- [ ] Teljesítmény optimalizálási tippek
- [ ] Tesztelési útmutató

---

> **Megjegyzés:** Ez a dokumentum folyamatosan bővül. Kérdések esetén lásd a forráskódot vagy kérdezz!
