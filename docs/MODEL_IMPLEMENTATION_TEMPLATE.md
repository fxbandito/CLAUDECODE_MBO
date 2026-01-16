# Model Implementation Template
**Hasznalat: Masold be ezt a prompt-ot es add meg a modell nevet**

---

## Prompt az Automatikus Implementaciohoz

```
Implementald a kovetkezo modellt a MODEL_IMPLEMENTATION_GUIDE.md alapjan:

MODEL_NAME: [MODEL NEV]
CATEGORY: [KATEGORIA]

Lepesek:
1. Olvasd be a regi implementaciot: src_old/analysis/models/{megfelelő mappa}/{model}.py
2. Ellenorizd az algoritmust az interneten (akademiai forrasok, dokumentacio)
3. Azonositsd es javitsd a logikai, matematikai es szintaktikai hibakat
4. Hozd letre az uj implementaciot: src/models/{subcategory}/{model}.py
5. Definald helyesen: MODEL_INFO, PARAM_DEFAULTS, PARAM_OPTIONS
6. Implementald a forecast() es forecast_batch() metodusokat
7. Add hozza a check_feature_mode_compatibility() metodust
8. Hozd letre a teszt fajlt: tests/test_{model}_model.py
9. Futtasd a tesztet az AUDJPY adatokon
10. Keszitsd el a validation report-ot: docs/{MODEL}_MODEL_VALIDATION_REPORT.md

Fontos:
- Minden teszt sikeres legyen
- A GUI-ban helyesen jelenjen meg
- Az Optuna integraciot most hagyjuk ki
```

---

## Peldak

### Pelda 1: ARIMA
```
Implementald a kovetkezo modellt a MODEL_IMPLEMENTATION_GUIDE.md alapjan:

MODEL_NAME: ARIMA
CATEGORY: Statistical Models

[... lepesek ...]
```

### Pelda 2: LSTM
```
Implementald a kovetkezo modellt a MODEL_IMPLEMENTATION_GUIDE.md alapjan:

MODEL_NAME: LSTM
CATEGORY: Deep Learning - RNN-based

[... lepesek ...]
```

### Pelda 3: XGBoost
```
Implementald a kovetkezo modellt a MODEL_IMPLEMENTATION_GUIDE.md alapjan:

MODEL_NAME: XGBoost
CATEGORY: Classical Machine Learning

[... lepesek ...]
```

---

## Kategoriak es Mappanevek

| Kategoria | Mappa neve | Megjegyzes |
|-----------|------------|------------|
| Statistical Models | `statistical/` | ADIDA kesz, tobbi stub |
| Smoothing & Decomposition | `smoothing/` | stub fajlok |
| Classical Machine Learning | `classical_ml/` | stub fajlok |
| Deep Learning - RNN-based | `dl_rnn/` | stub fajlok |
| Deep Learning - CNN & Hybrid Architectures | `dl_cnn_hybrid/` | stub fajlok |
| Deep Learning - Transformer-based | `dl_transformer/` | stub fajlok |
| Deep Learning - Graph & Specialized Neural Networks | `dl_graph_specialized/` | stub fajlok |
| Meta-Learning & AutoML | `meta_learning/` | stub fajlok |
| Bayesian & Probabilistic Methods | `bayesian/` | stub fajlok |
| Frequency Domain & Signal Processing | `frequency_domain/` | stub fajlok |
| Distance & Similarity-based | `distance_similarity/` | stub fajlok |
| State Space & Other | `state_space_other/` | stub fajlok |
| Symbolic Regression | `symbolic_regression/` | stub fajlok |

**FONTOS:** A stub fajlok mar leteznek minden modellhez! Csak az implementaciot kell kitolteni.

---

## MODEL_INFO Gyorsreferencia

```python
MODEL_INFO = ModelInfo(
    name="MODEL_NAME",              # Pontosan a regisztralt nev
    category="CATEGORY",            # Pontosan a kategoria nev
    supports_gpu=False,             # True: DL modellek
    supports_batch=True,            # True: ha van parhuzamos feldolgozas
    gpu_threshold=1000,             # Tipikus: 1000-5000
    supports_forward_calc=True,     # False: ha ujra kell trenelni
    supports_rolling_window=True,   # False: ha nem kezeli jol a csusko ablakot
    supports_panel_mode=False,      # True: ha egy modell tobb strategiara
    supports_dual_mode=False,       # True: ha activity+profit kulon
)
```

---

## CPU/GPU Kezeles Referencia

> **FONTOS:** Minden modell egyedi CPU/GPU kezelést kaphat!
>
> **Új architektúra fájlok:**
> - `src/analysis/engine.py` - `ResourceManager` (GUI globális beállítások)
> - `src/analysis/process_utils.py` - Process management utilities
> - `src/data/processor.py` - `DataProcessor.detect_data_mode()`
>
> **Elérhető utility függvények:**
> ```python
> from analysis.process_utils import (
>     cleanup_cuda_context,        # CUDA memória cleanup
>     force_kill_child_processes,  # Worker leállítás
>     init_worker_environment,     # Worker thread/env config
>     set_process_priority,        # Process prioritás
> )
>
> from data.processor import DataProcessor
> data_mode = DataProcessor.detect_data_mode(df)  # "original"/"forward"/"rolling"
> groups = DataProcessor.group_strategies(df)     # O(1) stratégia lookup
>
> from analysis.engine import AnalysisEngine
> AnalysisEngine.shutdown()  # Cleanup (workers + CUDA)
> ```
>
> **A modellek SAJÁT MAGUK döntik el**, hogyan használják ezeket!
>
> **Régi referencia (csak olvasásra):** `src_old/analysis/cpu_manager.py`

---

## Batch Mode Referencia

> **Batch Mode = Több stratégia egyidejű feldolgozása**
>
> **Új architektúra:** A batch a `BaseModel.forecast_batch()` metódusban van!
> - Default: `joblib.Parallel` párhuzamos feldolgozás
> - Egyedi: Felülírható modell szinten GPU-optimalizált verzióval
>
> **Ha a modell támogatja** (`supports_batch=True`):
> 1. Default batch automatikusan elérhető
> 2. Egyedi batch: override `forecast_batch()` metódust
>
> **Régi batch fájlok (referencia optimalizált implementációkhoz):**
> - `src_old/analysis/models/dl_rnn/lstm_batch.py`
> - `src_old/analysis/models/dl_transformer/autoformer_batch.py`
> - `src_old/analysis/models/dl_cnn/timesnet_batch.py`
> - stb. (30+ batch fájl az src_old/analysis/models/ mappában)
>
> **FONTOS:** A régi `engine.py`-ban lévő `_run_*_batch()` metódusok
> az új architektúrában **NEM KELLENEK** - a batch a modellben van!

---

## Dual Mode Referencia

> **Ha a modell támogatja a Dual Mode-ot** (`supports_dual_mode=True`):
>
> **Dual Mode = Activity Model + Profit Model külön**
> - Activity: Előrejelzi az aktivitás valószínűségét (0.0-1.0)
> - Profit: Előrejelzi a profitot aktív hetekre
> - Végső = Activity × Weeks × Profit
>
> **100% Modell-független architektúra:**
> - `src/analysis/dual_task.py` - NINCS modell kód!
> - `src/analysis/dual_executor.py` - NINCS modell kód!
> - A modellek SAJÁT MAGUK implementálják a `create_dual_regressor()` metódust
>
> **KÖTELEZŐ lépések új modellhez:**
> 1. Állítsd `supports_dual_mode=True` a MODEL_INFO-ban
> 2. **KÖTELEZŐ** implementálni a `create_dual_regressor()` metódust
>    - NINCS FALLBACK - hiba lesz, ha hiányzik!
>
> ```python
> def create_dual_regressor(self, params: Dict[str, Any], n_jobs: int = 1) -> Any:
>     """
>     Regresszor létrehozása dual mode-hoz.
>     A MODELL SAJÁT MAGA értelmezi a paramétereket!
>     """
>     # Saját paraméter feldolgozás
>     n_estimators = int(params.get("n_estimators", self.PARAM_DEFAULTS.get("n_estimators", "100")))
>
>     import lightgbm as lgb
>     return lgb.LGBMRegressor(
>         n_estimators=n_estimators,
>         device="cpu",  # Worker-ben MINDIG CPU!
>         n_jobs=n_jobs,
>         verbose=-1,
>     )
> ```
>
> **Használat:**
> ```python
> from models import supports_dual_mode
> from analysis import run_dual_model_mode
>
> if supports_dual_mode(model_name):
>     results = run_dual_model_mode(data, model_name, params)
> ```

---

## Opcionális Optimalizációk Referencia

> **FONTOS:** Ezek az optimalizációk a `src_old/analysis/engine.py`-ból származnak.
> NEM kötelezők - csak akkor implementáld, ha a modell speciális kezelést igényel!
> Részletes leírás: `MODEL_IMPLEMENTATION_GUIDE.md` szekció 5.4
>
> | Optimalizáció | Mikor kell? | Referencia |
> |---------------|-------------|------------|
> | GPU Auto-Optimization | DL modellek, sok kis stratégia (>50, <500 sample) | 5.4.1 |
> | Worker Limit | MoE, KAN, MTGNN - magas RAM/VRAM | 5.4.2 |
> | Julia-based Sequential | PySR, GPlearn - Julia lock | 5.4.3 |
> | Rolling Mode Reduction | Ha `detect_data_mode()` = "rolling" | 5.4.4 |
> | Optimal Workers | Ha egyedi worker számítás kell | 5.4.5 |
>
> **Gyors példa - Erőforrás-igényes modell:**
> ```python
> def forecast_batch(self, all_data, steps, params):
>     """Szekvenciális batch - magas memória használat miatt."""
>     results = {}
>     for name, data in all_data.items():
>         results[name] = self.forecast(data, steps, params)
>         self.cleanup_after_batch()  # FONTOS: memória felszabadítás!
>     return results
> ```
>
> **Gyors példa - GPU auto-disable:**
> ```python
> def forecast_batch(self, all_data, steps, params):
>     if len(all_data) > 50 and params.get("use_gpu"):
>         params = {**params, "use_gpu": False}
>         logger.info("Auto-switched to CPU (many small strategies)")
>     return super().forecast_batch(all_data, steps, params)
> ```

---

## Kovetkezo Modell Sorrend

A `src/models/__init__.py` MODEL_ORDER alapjan:

### Statistical Models
1. ~~ADIDA~~ [KESZ]
2. ARIMA
3. ARIMAX
4. Auto-ARIMA
5. CES
6. Change Point Detection
7. GAM
8. GARCH Family
9. OGARCH
10. Quantile Regression
11. SARIMA
12. VAR
13. VECM

### Smoothing & Decomposition
1. ETS
2. Exponential Smoothing
3. MSTL
4. STL
5. Theta

### Classical Machine Learning
1. Gradient Boosting
2. KNN Regressor
3. LightGBM
4. Random Forest
5. SVR
6. XGBoost

### Deep Learning - RNN-based
1. DeepAR
2. ES-RNN
3. GRU
4. LSTM
5. MQRNN
6. Seq2Seq

### Deep Learning - CNN & Hybrid
1. DLinear
2. N-BEATS
3. N-HiTS
4. TCN
5. TiDE
6. TimesNet

### Deep Learning - Transformer-based
1. Autoformer
2. FEDFormer
3. FiTS
4. Informer
5. PatchTST
6. TFT
7. Transformer
8. iTransformer

### Deep Learning - Graph & Specialized
1. Diffusion
2. KAN
3. MTGNN
4. Neural ARIMA
5. Neural Basis Functions
6. Neural GAM
7. Neural ODE
8. Neural Quantile Regression
9. Neural VAR
10. Neural Volatility
11. Spiking Neural Networks
12. StemGNN

### Meta-Learning & AutoML
1. DARTS
2. FFORMA
3. GFM
4. Meta-learning
5. MoE
6. Multi-task Learning
7. NAS

### Bayesian & Probabilistic
1. BSTS
2. Conformal Prediction
3. Gaussian Process
4. Monte Carlo
5. Prophet

### Frequency Domain & Signal Processing
1. DFT
2. FFT
3. Periodogram
4. Spectral Analysis
5. SSA
6. Wavelet Analysis
7. Welchs Method

### Distance & Similarity-based
1. DTW
2. k-NN
3. k-Shape
4. Matrix Profile

### State Space & Other
1. Kalman Filter
2. State Space Model
3. TDA
4. Time Series Ensemble

### Symbolic Regression
1. GPLearn
2. PySR
3. PySindy

---

**Osszes modell szama: ~70**
**Implementalt: 1 (ADIDA)**
**Hatralevo: ~69**
