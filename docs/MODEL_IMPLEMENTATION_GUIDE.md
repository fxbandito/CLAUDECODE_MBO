# Model Implementation Guide
**MBO Trading Strategy Analyzer - Standard Model Implementation Checklist**

Ez a dokumentum tartalmazza az osszes lepest, amit minden uj modell implementalasanal el kell vegezni.
A cél: csak a modell nevet kell megadni, es az implementacio automatikusan, hiba nelkul elkeszul.

---

## Quick Start - Automatikus Implementacio

Amikor egy uj modellt kell implementalni, a kovetkezo informaciokat kell megadni:

```
MODEL_NAME: [pl. ARIMA]
CATEGORY: [pl. Statistical Models]
SRC_OLD_PATH: [pl. src_old/analysis/models/statistical_models/arima.py]
```

---

## 1. Elokeszites

### 1.1 Forraskod Beolvasasa
- [ ] Olvasd be a regi implementaciot: `src_old/analysis/models/{category}/{model}.py`
- [ ] Jegyezd fel a meglevo parametreket es metodusokat
- [ ] Azonositsd a fo algoritmus(oka)t

### 1.2 Internet Verifikacio
Kereses: `"{MODEL_NAME} algorithm" OR "{MODEL_NAME} time series forecasting"`

Ellenorizendo szempontok:
- [ ] Matematikai keplet(ek) helyessege
- [ ] Parameter ertelmezesek
- [ ] Alternativ implementaciok osszehasonlitasa
- [ ] Legujabb best practices

Hasznos forrasok:
- Nixtla dokumentacio: https://nixtlaverse.nixtla.io/
- statsmodels: https://www.statsmodels.org/
- scikit-learn: https://scikit-learn.org/
- PyTorch: https://pytorch.org/docs/
- Akademiai cikkek (Google Scholar)

---

## 2. Hiba Ellenorzes

### 2.1 Logikai Hibak
- [ ] Algoritmus helyessege (keplet, levezetés)
- [ ] Edge case-ek kezelese (ures adat, NaN, negatív ertekek)
- [ ] Parametervalidacio (ertektartomany, tipusok)
- [ ] Loop invariansok, off-by-one hibak

### 2.2 Matematikai Hibak
- [ ] Kepletek verifikacioja irodalommal
- [ ] Numerikus stabilitás (overflow, underflow, division by zero)
- [ ] Varhato ertektartomanyok ellenorzese
- [ ] Specialis esetek (konstans idosor, trendek)

### 2.3 Szintaktikai Hibak
- [ ] Python szintaxis helyesseg
- [ ] Import-ok ellenorzese
- [ ] Tipusannotaciok (ahol szukseges)
- [ ] Pylint/flake8 figyelmeztetesek

---

## 3. Implementacio

### 3.1 Fajl Letrehozasa
Uj fajl: `src/models/{subcategory}/{model_name}.py`

Kategoriak es mappanevek:
| Kategoria | Mappa |
|-----------|-------|
| Statistical Models | `statistical/` |
| Smoothing & Decomposition | `smoothing/` |
| Classical Machine Learning | `classical_ml/` |
| Deep Learning - RNN-based | `dl_rnn/` |
| Deep Learning - CNN & Hybrid | `dl_cnn_hybrid/` |
| Deep Learning - Transformer-based | `dl_transformer/` |
| Deep Learning - Graph & Specialized | `dl_graph_specialized/` |
| Meta-Learning & AutoML | `meta_learning/` |
| Bayesian & Probabilistic Methods | `bayesian/` |
| Frequency Domain & Signal Processing | `frequency_domain/` |
| Distance & Similarity-based | `distance_similarity/` |
| State Space & Other | `state_space_other/` |
| Symbolic Regression | `symbolic_regression/` |

**FONTOS:** Minden modellhez mar letezik stub fajl! Csak az implementaciot kell kitolteni.

### 3.2 Kod Struktura

```python
"""
{MODEL_NAME} Model Implementation
MBO Trading Strategy Analyzer

{Rovid leiras a modellrol}
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from models.base import BaseModel, ModelInfo


class {ModelName}Model(BaseModel):
    """
    {MODEL_NAME} - {Teljes leiras}

    Referencia:
    - {Cikk vagy dokumentacio URL}
    """

    # ===== KOTELEZO DEFINICIOK =====

    MODEL_INFO = ModelInfo(
        name="{MODEL_NAME}",
        category="{CATEGORY}",
        supports_gpu=False,         # GPU tamogatas
        supports_batch=True,        # Batch mod
        gpu_threshold=1000,         # Min adat GPU-hoz
        supports_forward_calc=True, # Forward Calc mod
        supports_rolling_window=True,  # Rolling Window mod
        supports_panel_mode=False,  # Panel Mode
        supports_dual_mode=False,   # Dual Mode
    )

    PARAM_DEFAULTS = {
        "param1": "default_value",
        "param2": "default_value",
    }

    PARAM_OPTIONS = {
        "param1": ["option1", "option2", "option3"],
        "param2": ["option1", "option2"],
    }

    # ===== FO METODUSOK =====

    def forecast(
        self,
        data: List[float],
        steps: int,
        params: Dict[str, Any]
    ) -> List[float]:
        """
        Elorejelzes keszitese.

        Args:
            data: Bemeneti idosor
            steps: Elorejelzesi horizont
            params: Parameterek

        Returns:
            Elorejelzett ertekek listaja
        """
        # Parameterek kinyerese
        merged_params = self.get_params_with_defaults(params)

        # Adat validacio
        clean_data = self._validate_and_clean(data)
        if clean_data is None or len(clean_data) < self._get_min_data_length():
            return [np.nan] * steps

        try:
            # Fo algoritmus
            result = self._compute_forecast(clean_data, steps, merged_params)
            return result
        except Exception as e:
            print(f"{self.MODEL_INFO.name} forecast error: {e}")
            return [np.nan] * steps

    def forecast_batch(
        self,
        all_data: Dict[str, List[float]],
        steps: int,
        params: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """
        Batch elorejelzes tobb strategiara.
        """
        from joblib import Parallel, delayed
        import os

        def _process_single(name: str, data: List[float]):
            return name, self.forecast(data, steps, params)

        n_jobs = min(os.cpu_count() or 4, len(all_data), 8)
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_process_single)(name, data)
            for name, data in all_data.items()
        )
        return dict(results)

    # ===== SEGED METODUSOK =====

    def _validate_and_clean(self, data: List[float]) -> Optional[np.ndarray]:
        """Adat validacio es tisztitas."""
        if data is None or len(data) == 0:
            return None

        arr = np.array(data, dtype=np.float64)

        # NaN interpolacio
        if np.any(np.isnan(arr)):
            arr = self._interpolate_nan(arr)

        return arr

    def _interpolate_nan(self, arr: np.ndarray) -> np.ndarray:
        """NaN ertekek linearis interpolacioja."""
        nan_mask = np.isnan(arr)
        if not np.any(nan_mask):
            return arr

        if np.all(nan_mask):
            return arr  # Mind NaN - nem lehet interpolalni

        x = np.arange(len(arr))
        arr[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], arr[~nan_mask])
        return arr

    def _get_min_data_length(self) -> int:
        """Minimum szukseges adathossz."""
        return 10

    def _compute_forecast(
        self,
        data: np.ndarray,
        steps: int,
        params: Dict[str, Any]
    ) -> List[float]:
        """
        Fo elorejelzesi algoritmus.
        Ezt a metodust kell testreszabni minden modellnel!
        """
        raise NotImplementedError("Subclass must implement _compute_forecast")

    # ===== FEATURE MODE KOMPATIBILITAS =====

    @staticmethod
    def check_feature_mode_compatibility(feature_mode: str) -> Tuple[bool, str]:
        """
        Ellenorzi, hogy a Feature Mode kompatibilis-e a modellel.

        Args:
            feature_mode: "Original", "Forward Calc", vagy "Rolling Window"

        Returns:
            (kompatibilis, figyelmezteto_uzenet)
        """
        if feature_mode == "Original":
            return True, ""

        if feature_mode == "Forward Calc":
            # Ha a modell NEM tamogatja a Forward Calc-ot:
            return False, (
                "{MODEL_NAME} WARNING: Forward Calc mode is not recommended! "
                "[Indoklas, miert nem jo ez a mod ehhez a modellhez]"
            )

        if feature_mode == "Rolling Window":
            # Ha a modell NEM tamogatja a Rolling Window-t:
            return False, (
                "{MODEL_NAME} WARNING: Rolling Window mode is not recommended! "
                "[Indoklas, miert nem jo ez a mod ehhez a modellhez]"
            )

        return True, ""
```

### 3.3 ModelInfo Mezo Magyarazatok

| Mezo | Tipus | Leiras | Mikor True? |
|------|-------|--------|-------------|
| `supports_gpu` | bool | GPU gyorsitas | Deep Learning modellek, matrix muveletek |
| `supports_batch` | bool | Tobb strategia egyszerre | Ha van batch implementacio |
| `gpu_threshold` | int | Min adatmeret GPU-hoz | 1000-5000 tipikusan |
| `supports_forward_calc` | bool | Forward Calc mod | Ha a modell jol kezeli az ujra-trenezes nelkuli predikciokat |
| `supports_rolling_window` | bool | Rolling Window mod | Ha a modell jol kezeli a csusko ablakot |
| `supports_panel_mode` | bool | Panel Mode | Ha egy modell hasznalhato az osszes strategiara |
| `supports_dual_mode` | bool | Dual Mode | Ha kulon activity es profit modell hasznalhato |

### 3.4 Batch Mode Implementáció

> **FONTOS:** A Batch Mode az új architektúrában **a modell osztályban van**!
>
> A régi `src_old/analysis/engine.py`-ban 30+ `_run_*_batch()` metódus volt,
> ami delegált külön `*_batch.py` fájlokba. Ez **NEM KELL** az új rendszerben!
>
> **Új architektúra:**
> - A `BaseModel.forecast_batch()` metódus a default batch implementáció
> - Minden modell felülírhatja saját optimalizált batch verzióval
> - A batch fájlok (`src_old/analysis/models/*_batch.py`) referenciának használhatók
>
> **Default batch implementáció (BaseModel-ben):**
> ```python
> def forecast_batch(
>     self,
>     all_data: Dict[str, List[float]],
>     steps: int,
>     params: Dict[str, Any]
> ) -> Dict[str, List[float]]:
>     """Batch előrejelzés - párhuzamos feldolgozás."""
>     from joblib import Parallel, delayed
>
>     def _process_single(name: str, data: List[float]):
>         return name, self.forecast(data, steps, params)
>
>     n_jobs = min(os.cpu_count() or 4, len(all_data), 8)
>     results = Parallel(n_jobs=n_jobs, prefer="threads")(
>         delayed(_process_single)(name, data)
>         for name, data in all_data.items()
>     )
>     return dict(results)
> ```
>
> **Egyedi batch implementáció (ha szükséges):**
> ```python
> class LSTMModel(BaseModel):
>     MODEL_INFO = ModelInfo(
>         name="LSTM",
>         supports_batch=True,  # Engedélyezve
>         ...
>     )
>
>     def forecast_batch(self, all_data, steps, params):
>         """
>         Egyedi batch implementáció GPU-optimalizált verzióval.
>         Referencia: src_old/analysis/models/dl_rnn/lstm_batch.py
>         """
>         # GPU batch processing ha sok stratégia van
>         if len(all_data) > 100 and self._use_gpu(params):
>             return self._gpu_batch_forecast(all_data, steps, params)
>
>         # Egyébként default batch
>         return super().forecast_batch(all_data, steps, params)
> ```
>
> **Régi batch fájlok (referencia):**
> - `src_old/analysis/models/dl_rnn/lstm_batch.py`
> - `src_old/analysis/models/dl_transformer/autoformer_batch.py`
> - `src_old/analysis/models/dl_cnn/timesnet_batch.py`
> - stb.

### 3.5 Dual Mode Implementacio

> **FONTOS:** A Dual Mode **100% modell-fuggetlen** architekturara epul!
> Az infrastruktura fajlok NEM TARTALMAZNAK semmilyen modell-specifikus kodot.
>
> **Dual Mode koncepcio:**
> - **Activity Model**: Elorejezi, hogy a strategia aktiv lesz-e (0.0-1.0 valoszinuseg)
> - **Profit Model**: Elorejezi a profitot aktiv hetekre (regresszio)
> - **Vegso Forecast** = Activity_Ratio x Expected_Active_Weeks x Profit_Per_Active_Week
>
> **Architektura:**
> - `src/analysis/dual_task.py` - 100% modell-fuggetlen worker infrastruktura
> - `src/analysis/dual_executor.py` - Multiprocessing executor (nincs modell kod!)
> - `src/models/base.py` - `create_dual_regressor()` absztrakt metodus
>
> **KOTELEZO lepesek uj Dual Mode modellhez:**
>
> 1. Allitsd be `supports_dual_mode=True` a MODEL_INFO-ban
> 2. **KOTELEZO** implementalni a `create_dual_regressor()` metodust
>    - NINCS FALLBACK - ha nem implementalod, hiba lesz!
>    - A modell SAJAT MAGA donti el a parametereket
>    - A modell SAJAT MAGA hozza letre a regresszort
>
> **Pelda implementacio:**
> ```python
> class LightGBMModel(BaseModel):
>     MODEL_INFO = ModelInfo(
>         name="LightGBM",
>         category="Classical Machine Learning",
>         supports_dual_mode=True,  # <-- KOTELEZO bekapcsolni
>         ...
>     )
>
>     def create_dual_regressor(self, params: Dict[str, Any], n_jobs: int = 1) -> Any:
>         """
>         Regresszor letrehozasa dual mode-hoz.
>
>         FONTOS: A MODELL donti el, hogyan ertelmezi a parametereket!
>         Az infrastruktura csak atadja a params dict-et.
>         """
>         import lightgbm as lgb
>
>         # A modell SAJAT MAGA parse-olja a parametereket
>         n_estimators = int(params.get("n_estimators", self.PARAM_DEFAULTS.get("n_estimators", "100")))
>         max_depth_str = params.get("max_depth", self.PARAM_DEFAULTS.get("max_depth", "-1"))
>         max_depth = int(max_depth_str) if max_depth_str not in ("None", "-1") else -1
>         learning_rate = float(params.get("learning_rate", self.PARAM_DEFAULTS.get("learning_rate", "0.1")))
>
>         return lgb.LGBMRegressor(
>             n_estimators=n_estimators,
>             max_depth=max_depth,
>             learning_rate=learning_rate,
>             device="cpu",  # Worker-ben MINDIG CPU!
>             n_jobs=n_jobs,
>             verbose=-1,
>         )
> ```
>
> **Hasznalat:**
> ```python
> from models import supports_dual_mode
> from analysis import run_dual_model_mode
>
> if supports_dual_mode("LightGBM"):
>     results_df, best_id, best_data, filename, all_data = run_dual_model_mode(
>         data=my_dataframe,
>         method_name="LightGBM",
>         params={"n_estimators": "200"},  # A MODELL ertelmezi!
>     )
> ```
>
> **Rekurziv elorejelzes:**
> A `dual_task.apply_recursive_forecasting()` fuggveny MODELL-FUGGETLEN.
> Barmely sklearn-kompatibilis regresszorral mukodik (fit/predict metodusok).
> Aktivalas: `recursive_horizon` parameter a params-ban.

### 3.6 Panel Mode Implementacio

> **FONTOS:** A Panel Mode **100% modell-fuggetlen** architekturara epul!
> Az infrastruktura fajlok NEM TARTALMAZNAK semmilyen modell-specifikus kodot.
>
> **Panel Mode koncepcio:**
> - **Egy modell** tanit az **osszes strategiara** egyszerre (panel adatstruktura)
> - Sokkal **gyorsabb** ML-alapu modelleknel, mint az egyenkenti strategia-feldolgozas
> - **Lag feature-ok** az idosoros jelleg megorzesehez
> - **Time-aware split** a data leakage elkerulesere
>
> **Architektura:**
> - `src/analysis/panel_executor.py` - 100% modell-fuggetlen panel mode infrastruktura
> - `src/analysis/dual_task.py` - Worker task (KOZOS a Dual Mode-dal!)
> - `src/models/base.py` - `create_dual_regressor()` UJRAHASZNALVA (ugyanaz a metodus!)
>
> **KOTELEZO lepesek uj Panel Mode modellhez:**
>
> 1. Allitsd be `supports_panel_mode=True` a MODEL_INFO-ban
> 2. **KOTELEZO** implementalni a `create_dual_regressor()` metodust
>    - UGYANAZ a metodus mint a Dual Mode-nal!
>    - Ha mar implementaltad Dual Mode-hoz, Panel Mode automatikusan mukodik
>    - NINCS FALLBACK - ha nem implementalod, hiba lesz!
>
> **Pelda implementacio:**
> ```python
> class XGBoostModel(BaseModel):
>     MODEL_INFO = ModelInfo(
>         name="XGBoost",
>         category="Classical Machine Learning",
>         supports_panel_mode=True,  # <-- Panel Mode BEKAPCSOLVA
>         supports_dual_mode=True,   # <-- Dual Mode is tamogatott
>         ...
>     )
>
>     def create_dual_regressor(self, params: Dict[str, Any], n_jobs: int = 1) -> Any:
>         """
>         Regresszor letrehozasa - KOZOS a Panel es Dual mode-hoz!
>
>         FONTOS: A MODELL donti el, hogyan ertelmezi a parametereket!
>         """
>         import xgboost as xgb
>
>         n_estimators = int(params.get("n_estimators", self.PARAM_DEFAULTS.get("n_estimators", "100")))
>         max_depth_str = params.get("max_depth", self.PARAM_DEFAULTS.get("max_depth", "6"))
>         max_depth = int(max_depth_str) if max_depth_str not in ("None", "0") else None
>         learning_rate = float(params.get("learning_rate", self.PARAM_DEFAULTS.get("learning_rate", "0.1")))
>
>         return xgb.XGBRegressor(
>             n_estimators=n_estimators,
>             max_depth=max_depth,
>             learning_rate=learning_rate,
>             tree_method="hist",  # CPU-optimalizalt
>             n_jobs=n_jobs,
>             verbosity=0,
>         )
> ```
>
> **Hasznalat:**
> ```python
> from models import supports_panel_mode
> from analysis import run_panel_mode
>
> if supports_panel_mode("XGBoost"):
>     results_df, best_id, best_data, filename, all_data = run_panel_mode(
>         data=my_dataframe,
>         method_name="XGBoost",
>         params={"n_estimators": "200"},  # A MODELL ertelmezi!
>     )
> ```
>
> **Panel Mode vs Dual Mode kulonbsegek:**
>
> | Jellemzo | Panel Mode | Dual Mode |
> |----------|------------|-----------|
> | Modell szam | 1 modell / horizon | 2 modell / horizon (activity + profit) |
> | Cel | Gyors batch predikció | Finomhangolt forecasting |
> | Adat struktura | Panel (osszes strategia egyutt) | Strategiankent kulon |
> | Tipikus hasznalat | Sok strategia gyors elemzese | Reszletes elorejelzes |
> | Implementacio | `create_dual_regressor()` | `create_dual_regressor()` |
>
> **FONTOS:** Ha egy modell mindket mod-ot tamogatja, EGYETLEN `create_dual_regressor()`
> implementacio eleg - az infrastruktura automatikusan kezeli!
>
> **Elerheto utility fuggvenyek (panel_executor.py):**
> ```python
> from analysis.panel_executor import (
>     run_panel_mode,              # Fo belepes pont
>     prepare_panel_data,          # Panel adat elokeszites lag feature-okkel
>     time_aware_split,            # Data leakage-mentes split
>     get_panel_mode_models,       # Tamogatott modellek listaja
>     PANEL_MODE_SUPPORTED_MODELS, # Legacy alias
> )
> ```

---

## 4. Parameter Kezeles

### 4.1 PARAM_DEFAULTS
Az osszes parameter alapertelmezett erteke STRING formatumban:

```python
PARAM_DEFAULTS = {
    "param_name": "default_value",  # String!
}
```

### 4.2 PARAM_OPTIONS
A GUI dropdown listat generalja ebbol. Minden opcio STRING:

```python
PARAM_OPTIONS = {
    "param_name": ["option1", "option2", "option3"],
}
```

### 4.3 Parameter Konverzio
A `forecast()` metodusban konvertalod a stringeket a megfelelo tipusra:

```python
alpha = float(params.get("alpha", "0.1"))
p = int(params.get("p", "1"))
use_trend = params.get("use_trend", "True").lower() == "true"
method = params.get("method", "default")  # String marad
```

---

## 5. CPU/GPU Kezeles

> **FONTOS REFERENCIA:** A CPU/GPU kezelés moduláris felépítésű.
>
> **Új architektúra fájlok:**
> - `src/analysis/engine.py` - `ResourceManager` singleton (GUI globális beállítások)
> - `src/analysis/process_utils.py` - **KÖZPONTI** process management utility függvények
> - `src/analysis/dual_task.py` - Dual mode algoritmusok (re-exportálja process_utils-t)
> - `src/data/processor.py` - `DataProcessor.detect_data_mode()` adat mód detektálás
>
> **Elérhető utility függvények (`src/analysis/process_utils.py`):**
> ```python
> from analysis.process_utils import (
>     cleanup_cuda_context,        # CUDA memória cleanup
>     force_kill_child_processes,  # Worker processek leállítása
>     init_worker_environment,     # Worker thread/env beállítás (KÖZPONTI!)
>     set_process_priority,        # Process prioritás (Windows/Unix)
> )
> ```
>
> ### Worker Inicializálás - KÖZPONTI FÜGGVÉNY
>
> Az `init_worker_environment()` a **KÖZPONTI** függvény minden multiprocessing worker-hez.
> **FONTOS:** Nincs duplikáció - minden modul innen importálja!
>
> ```python
> def init_worker_environment(n_threads: int = 1) -> None:
>     """
>     Worker process környezet inicializálása.
>
>     Miért fontos:
>     - Megakadályozza a "halálspirált" (túl sok thread)
>     - Elkerüli a CUDA context konfliktusokat
>     - Megelőzi a FAISS AVX2/CUDA ütközéseket
>     """
>     # 1. GPU LETILTÁS - Worker-ek NEM használhatnak GPU-t
>     os.environ["CUDA_VISIBLE_DEVICES"] = ""
>     os.environ["FAISS_OPT_LEVEL"] = "generic"
>
>     # 2. THREAD LIMITEK - Megelőzi a túlterhelést
>     # 12 worker × 12 thread = 144 thread → rendszer lefagyás
>     # 12 worker × 1 thread = 12 thread → stabil futás
>     os.environ["OMP_NUM_THREADS"] = str(n_threads)
>     os.environ["MKL_NUM_THREADS"] = str(n_threads)
>     os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
>     os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
>     os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
>
>     # 3. WORKER JELÖLÉS
>     os.environ["MBO_MP_WORKER"] = "1"
>
>     # 4. PYTORCH KONFIGURÁCIÓ
>     torch.set_num_threads(n_threads)
>     torch.set_num_interop_threads(n_threads)
>
>     # 5. WARNING ELNYOMÁS
>     warnings.filterwarnings("ignore", message="Loky-backed parallel loops...")
> ```
>
> **Használat multiprocessing.Pool-ban:**
> ```python
> from analysis.process_utils import init_worker_environment
>
> pool = multiprocessing.Pool(
>     processes=optimal_workers,
>     initializer=init_worker_environment  # KÖZPONTI függvény!
> )
> ```
>
> **Import struktúra (nincs duplikáció):**
> ```
> process_utils.py          ◀── EGYETLEN KÖZPONTI HELY
>        │
>        ├── dual_task.py        ──re-export──▶  (backward compat)
>        ├── dual_executor.py    ──imports──────▶
>        └── __init__.py         ──exports──────▶
> ```
>
> **Adat mód detektálás (`src/data/processor.py`):**
> ```python
> from data.processor import DataProcessor
>
> data_mode = DataProcessor.detect_data_mode(df)
> # Visszatérési értékek: "original", "forward", "rolling"
> # Rolling mode = több memória, kevesebb worker ajánlott
> ```
>
> **Stratégia előcsoportosítás - O(1) lookup (`src/data/processor.py`):**
> ```python
> from data.processor import DataProcessor
>
> # Egyszer csoportosít, utána O(1) lookup
> groups = DataProcessor.group_strategies(df, strategy_col="No.")
> for strategy_id in strategy_ids:
>     strat_data = groups[strategy_id]  # GYORS!
> ```
>
> **AnalysisEngine shutdown (`src/analysis/engine.py`):**
> ```python
> from analysis.engine import AnalysisEngine
>
> # Cleanup hívás (auto-execution végén, hiba után, app bezáráskor)
> AnalysisEngine.shutdown(protected_pids={12345})  # Védett PID-ek megadhatók
> ```
>
> **Régi referencia (csak olvasásra):** `src_old/analysis/cpu_manager.py`

### 5.1 GPU Tamogatas Dontes

A modell GPU-t hasznal, ha:
- **Matrix muveletek**: nagy meretu szorzasok, inverzek
- **Neural Network**: barmelyik deep learning modell
- **Batch processing**: tobb adat parhuzamos feldolgozasa

A modell NEM hasznal GPU-t, ha:
- **Szekvencialis algoritmus**: pl. exponential smoothing
- **Kicsi adatok**: kevesebb mint 1000 pont
- **Nincs tensor muvelet**: pl. statisztikai modszerek

### 5.2 GPU Implementacio (ha szukseges)

```python
def forecast(self, data, steps, params):
    device = self.get_device(len(data), params.get("use_gpu", True))

    if device != "cpu":
        import torch
        tensor = torch.tensor(data, dtype=torch.float32, device=device)
        # GPU muveletek
        result = self._gpu_compute(tensor, steps, params)
        return result.cpu().numpy().tolist()
    else:
        # CPU muveletek
        return self._cpu_compute(data, steps, params)
```

### 5.3 GPU Optimalizacios Szempontok

Ellenorizendo:
- [ ] CUDA Graphs - statikus input shape eseten (advanced)
- [ ] Non-blocking data transfer - `non_blocking=True`
- [ ] GPU memory management - `torch.cuda.empty_cache()`
- [ ] Mixed precision - `torch.cuda.amp` (advanced)

### 5.4 Opcionális Optimalizációk (Régi engine.py Referencia)

> **FONTOS:** Az alábbi optimalizációk a `src_old/analysis/engine.py`-ból származnak.
> Ezek NEM kötelezők - minden modell SAJÁT MAGA dönti el, hogy implementálja-e őket!
> Csak akkor szükségesek, ha a modell speciális kezelést igényel.

#### 5.4.1 GPU Auto-Optimization (Automatikus CPU-ra Váltás)

Ha a modell sok kis stratégiával (~200 sample/stratégia) fut, a GPU overhead
nagyobb lehet, mint a haszon. Ilyenkor CPU parallel gyorsabb.

```python
# Referencia: src_old/analysis/engine.py L683-706
def forecast_batch(self, all_data, steps, params):
    """Batch feldolgozás GPU auto-optimization-nel."""
    strategy_count = len(all_data)
    use_gpu = params.get("use_gpu", True)

    # Automatikus CPU-ra váltás sok stratégia esetén
    # GPU overhead > benefit kis adatoknál (~200 sample/stratégia)
    if (use_gpu
        and strategy_count > 50
        and self._avg_samples_per_strategy(all_data) < 500):
        logger.info(
            "AUTO-OPTIMIZATION: GPU disabled for %s with %d strategies. "
            "CPU multiprocessing is 2-3x faster for small per-strategy data.",
            self.MODEL_INFO.name, strategy_count
        )
        params = params.copy()
        params["use_gpu"] = False

    return super().forecast_batch(all_data, steps, params)
```

**Modellek ahol ez hasznos lehet:**
- Deep Learning modellek (LSTM, GRU, Transformer, stb.)
- GFM, Meta-learning (benchmark: CPU 1.7-2.3x gyorsabb kis adatoknál)

#### 5.4.2 Worker Limit Optimization (Erőforrás-Igényes Modellek)

Bizonyos modellek olyan erőforrás-igényesek, hogy szekvenciálisan kell futniuk.

```python
# Referencia: src_old/analysis/engine.py L752-789
class MoEModel(BaseModel):
    MODEL_INFO = ModelInfo(
        name="MoE",
        supports_batch=True,
        # Új attribútumok - opcionálisak:
        # max_parallel_workers=1,  # Erőforrás-igényes
        # requires_sequential=True,  # Kötelezően szekvenciális
    )

    def forecast_batch(self, all_data, steps, params):
        """Szekvenciális batch - magas RAM/VRAM használat miatt."""
        # Kötelezően szekvenciális feldolgozás
        results = {}
        for name, data in all_data.items():
            results[name] = self.forecast(data, steps, params)
            self.cleanup_after_batch()  # Memória felszabadítás stratégiánként!
        return results
```

**Modellek ahol ez szükséges lehet:**
- MoE (Mixture of Experts) - magas paraméter szám
- KAN - magas számítási költség
- MTGNN, StemGNN - GPU megosztott state

#### 5.4.3 Julia-Based Models Speciális Kezelése

Julia backend-et használó modellek (pl. PySR) nem futhatnak párhuzamosan
a Julia lock konfliktusok miatt.

```python
# Referencia: src_old/analysis/engine.py L759-789
class PySRModel(BaseModel):
    MODEL_INFO = ModelInfo(
        name="PySR",
        supports_batch=True,
        # julia_based=True,  # Opcionális jelölés
    )

    def forecast_batch(self, all_data, steps, params):
        """Szekvenciális batch - Julia lock miatt."""
        # Julia nem tud párhuzamosan inicializálódni
        # SIGABRT ha több worker próbálja egyszerre
        results = {}
        for name, data in all_data.items():
            results[name] = self.forecast(data, steps, params)
        return results
```

**Érintett modellek:** PySR, GPlearn (ha Julia backend-et használ)

#### 5.4.4 Rolling Mode Worker Reduction

Rolling feature mode több memóriát használ. Ilyenkor kevesebb worker ajánlott.

```python
# Referencia: src_old/analysis/engine.py L840-849
def forecast_batch(self, all_data, steps, params):
    """Batch feldolgozás rolling mode optimalizációval."""
    from data.processor import DataProcessor

    # Ha rolling mode, csökkentjük a worker számot
    data_mode = DataProcessor.detect_data_mode(self._get_sample_df())

    if data_mode == "rolling":
        n_jobs = min(self.get_n_jobs(), 4)  # Max 4 worker rolling mode-ban
        logger.info("Rolling mode detected: limiting workers to %d", n_jobs)
    else:
        n_jobs = self.get_n_jobs()

    # ... batch feldolgozás n_jobs-szal
```

#### 5.4.5 Optimal Worker Calculation

A régi engine.py tartalmazott egy `calculate_optimal_workers()` függvényt.
Az új architektúrában ezt a modell saját maga kezeli a `get_n_jobs()` metódussal.

```python
# BaseModel-ben már elérhető:
n_jobs = self.get_n_jobs(max_limit=8)  # ResourceManager alapján

# Ha egyedi logika kell:
def _calculate_optimal_workers(self, strategy_count: int) -> int:
    """Optimális worker szám stratégia számtól függően."""
    base_jobs = self.get_n_jobs()

    # Kevés stratégia = kevesebb worker (overhead elkerülése)
    if strategy_count <= 10:
        return min(base_jobs, strategy_count)

    # Sok stratégia = több worker, de max 8
    return min(base_jobs, 8)
```

---

## 6. GUI Integracio

### 6.1 Automatikus Regisztracio

A modell automatikusan regisztralodik, ha:
1. A fajl a `src/models/{subcategory}/` mappaban van
2. Az osztaly orokol a `BaseModel`-tol
3. Definialva van: `MODEL_INFO`, `PARAM_DEFAULTS`, `PARAM_OPTIONS`

### 6.2 GUI Kontrollok

A GUI automatikusan:
- Letiltja a GPU switch-et ha `supports_gpu=False`
- Letiltja a Batch gombot ha `supports_batch=False`
- Letiltja a Panel Mode-ot ha `supports_panel_mode=False`
- Letiltja a Dual Mode-ot ha `supports_dual_mode=False`
- Figyelmeztetest mutat ha a Feature Mode nem kompatibilis

### 6.3 Ellenorzes

Futtasd a tesztet:
```bash
python -c "from models import get_param_defaults, get_param_options; print(get_param_defaults('{MODEL_NAME}')); print(get_param_options('{MODEL_NAME}'))"
```

---

## 7. Teszteles

### 7.1 Teszt Fajl Letrehozasa

Uj fajl: `tests/test_{model_name}_model.py`

```python
"""
{MODEL_NAME} Model Validation Test
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import time

def test_{model_name}_model():
    """Alapveto funkcionalitas teszt."""
    from models.{subcategory}.{model_name} import {ModelName}Model

    model = {ModelName}Model()

    # Test data
    test_data = list(np.random.randn(100) * 100 + 1000)

    # Test forecast
    result = model.forecast(test_data, steps=12, params={})

    assert len(result) == 12, "Forecast length mismatch"
    assert not all(np.isnan(result)), "All forecasts are NaN"

    print("[OK] Basic forecast test passed")

def test_parameter_loading():
    """GUI parameter teszt."""
    from models import get_param_defaults, get_param_options, get_model_info

    defaults = get_param_defaults("{MODEL_NAME}")
    options = get_param_options("{MODEL_NAME}")
    info = get_model_info("{MODEL_NAME}")

    assert defaults is not None, "No defaults found"
    assert options is not None, "No options found"
    assert info is not None, "No model info found"

    print(f"[OK] Parameters loaded: {list(defaults.keys())}")

def test_edge_cases():
    """Edge case teszt."""
    from models.{subcategory}.{model_name} import {ModelName}Model

    model = {ModelName}Model()

    # Empty data
    result = model.forecast([], steps=5, params={})
    assert len(result) == 5, "Empty data handling failed"

    # NaN data
    result = model.forecast([np.nan] * 10, steps=5, params={})
    assert len(result) == 5, "NaN data handling failed"

    # Single value
    result = model.forecast([100.0], steps=5, params={})
    assert len(result) == 5, "Single value handling failed"

    print("[OK] Edge case tests passed")

if __name__ == "__main__":
    test_parameter_loading()
    test_{model_name}_model()
    test_edge_cases()
    print("\\n[OK] ALL TESTS PASSED")
```

### 7.2 AUDJPY Teszt

```python
def test_on_audjpy():
    """Teszt az AUDJPY adatokon."""
    from data.loader import DataLoader
    from data.processor import DataProcessor
    from models.{subcategory}.{model_name} import {ModelName}Model

    # Load data
    parquet_path = Path(__file__).parent.parent / "testdata" / "AUDJPY_2020_2021_2022_2023_Weekly_Fix.parquet"
    raw_data = DataLoader.load_parquet_files([str(parquet_path)])
    processed = DataProcessor.clean_data(raw_data)

    # Extract test data
    if "Profit" in processed.columns:
        test_data = processed["Profit"].dropna().values.tolist()
    else:
        numeric_cols = processed.select_dtypes(include=[np.number]).columns
        test_data = processed[numeric_cols[0]].dropna().values.tolist()

    # Test model
    model = {ModelName}Model()

    start = time.perf_counter()
    result = model.forecast(test_data, steps=12, params={})
    elapsed = (time.perf_counter() - start) * 1000

    print(f"[OK] AUDJPY test: {elapsed:.2f}ms")
    print(f"     Forecast: {result[:5]}")
```

---

## 8. Riport Keszites

### 8.1 Riport Template

Fajl: `docs/{MODEL_NAME}_MODEL_VALIDATION_REPORT.md`

```markdown
# {MODEL_NAME} Model Validation Report
**Date:** {DATE}
**Model:** {MODEL_NAME} ({Full Name})
**Test Data:** AUDJPY_2020_2021_2022_2023_Weekly_Fix.parquet

---

## 1. Executive Summary

| Teszt | Eredmeny | Megjegyzes |
|-------|----------|------------|
| Parameter betoltes | PASS/FAIL | |
| Forecast muvelet | PASS/FAIL | X.XX ms |
| Edge case kezeles | PASS/FAIL | |
| Batch mod | PASS/FAIL | |
| CPU/GPU kezeles | PASS/FAIL | |
| Feature Mode kompatibilitas | PASS/FAIL | |
| Model Capabilities | PASS/FAIL | |

---

## 2. Algoritmikus Elemzes

### 2.1 Elmeleti Hatter
{Leiras a modell mukodeserol}

### 2.2 Referencia
- [{Paper/Docs neve}]({URL})

---

## 3. Azonositott es Javitott Problemak

| # | Hiba | Sulyossag | Javitas |
|---|------|-----------|---------|
| 1 | {Problema} | {Alacsony/Kozepes/Magas} | {Megoldas} |

---

## 4. Parameterek

### 4.1 Alapertelmezett Parameterek
{PARAM_DEFAULTS lista}

### 4.2 Opcio Lista
{PARAM_OPTIONS lista}

---

## 5. Teljesitmeny Eredmenyek

| Metrika | Ertek |
|---------|-------|
| Atlagos futasi ido | X.XX ms |
| Memory hasznalat | X MB |
| GPU kihasznaltsag | N/A / X% |

---

## 6. Konkluzio

{Osszefoglalas, keszenleti allapot}

---

**Keszitette:** Claude Opus 4.5
**Verzio:** 1.0
**Utolso frissites:** {DATE}
```

---

## 9. Checklist - Teljes Implementacio

### Elokeszites
- [ ] Regi kod beolvasva: `src_old/...`
- [ ] Internet verifikacio elvegezve
- [ ] Referencia dokumentacio megtalava

### Hibakeresés
- [ ] Logikai hibak ellenorizve
- [ ] Matematikai hibak javitva
- [ ] Szintaxis helyes

### Implementacio
- [ ] Fajl letrehozva: `src/models/{subcat}/{model}.py`
- [ ] MODEL_INFO definiava (minden mezo!)
- [ ] PARAM_DEFAULTS definiava
- [ ] PARAM_OPTIONS definiava
- [ ] `forecast()` implementalva
- [ ] `forecast_batch()` implementalva (ha supports_batch=True)
- [ ] `check_feature_mode_compatibility()` implementalva
- [ ] Edge case-ek kezelve
- [ ] NaN handling implementalva

### Teszteles
- [ ] Teszt fajl letrehozva
- [ ] Alapveto teszt sikeres
- [ ] Edge case teszt sikeres
- [ ] AUDJPY teszt sikeres
- [ ] Parameter loading teszt sikeres
- [ ] Batch mode teszt sikeres (ha tamogatott)

### GUI
- [ ] Model megjelenik a kategoriaban
- [ ] Parameterek betoltodnek
- [ ] GPU switch helyesen mukodik
- [ ] Feature mode figyelmeztetés mukodik

### Dokumentacio
- [ ] Validation report elkeszult

---

## 10. Gyakori Hibak

### 10.1 Model nem jelenik meg a GUI-ban
- Ellenorizd, hogy a fajl a megfelelo mappaban van
- Ellenorizd, hogy az osztaly orokol BaseModel-tol
- Ellenorizd, hogy MODEL_INFO nem None

### 10.2 Parameterek nem toltodnek be
- Ellenorizd, hogy PARAM_DEFAULTS es PARAM_OPTIONS string ertekeket tartalmaz
- Ellenorizd, hogy minden PARAM_OPTIONS opcio lista

### 10.3 Forecast NaN-okat ad vissza
- Ellenorizd a bemeno adatokat
- Ellenorizd a numerikus stabilitas problemakat
- Adj hozza reszletes logging-ot

### 10.4 Unicode Error Windows-on
- Hasznalj ASCII karaktereket: [OK], [FAIL], [PASS], [WARN]
- Kereld az emojik hasznalatat

---

**Ez a dokumentum az automatikus modell implementacio alapja.**
**Frissites szukseges, ha uj funkcionalitas kerul a rendszerbe.**
