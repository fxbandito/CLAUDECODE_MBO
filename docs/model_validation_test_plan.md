# Modell Validációs Teszt Terv
## Minden Modell Alapvető Működésének Ellenőrzése

**Készült:** 2025-12-29
**Cél:** Minden modell (100+) egyszeri futtatása gyors, validációs beállításokkal

---

## Tartalomjegyzék

1. [Célok és Ellenőrzési Szempontok](#1-célok-és-ellenőrzési-szempontok)
2. [Validációs Paraméterek Filozófiája](#2-validációs-paraméterek-filozófiája)
3. [Modellenkénti Validációs Beállítások](#3-modellenkénti-validációs-beállítások)
4. [Ellenőrzési Checklist](#4-ellenőrzési-checklist)
5. [Hibakezelési Útmutató](#5-hibakezelési-útmutató)

---

## 1. Célok és Ellenőrzési Szempontok

### Mit ellenőrzünk?

| # | Szempont | Leírás |
|---|----------|--------|
| 1 | **Futás sikeressége** | A modell hiba nélkül lefut-e? |
| 2 | **Eredmény generálás** | Készít-e 52 hetes előrejelzést? |
| 3 | **CPU/GPU kezelés** | GPU-s modellek használják-e a GPU-t? |
| 4 | **Memória kezelés** | Nem fut-e ki a memóriából? |
| 5 | **Időzítés** | Elfogadható időn belül fut-e? |
| 6 | **Kimenet validitás** | Az előrejelzés értelmesnek tűnik-e? (nem NaN, nem végtelenség) |

### Sikerkritériumok

```
✅ PASS feltételei:
├── Nincs Python hiba/exception
├── 52 hetes forecast generálódik
├── Forecast értékek nem NaN és nem Inf
├── Futási idő < 5 perc (CPU) vagy < 2 perc (GPU)
└── Memória használat stabil (nem nő folyamatosan)

❌ FAIL feltételei:
├── Exception/hiba a futás során
├── Hiányzó vagy csonka forecast
├── NaN vagy Inf értékek a kimenetben
├── Timeout (> 10 perc)
└── Out of Memory hiba
```

---

## 2. Validációs Paraméterek Filozófiája

### Gyors Futás Elvei

A validációs teszteknél a cél a **gyors, megbízható futás**, nem a pontosság:

| Paraméter típus | Validációs érték | Indoklás |
|-----------------|------------------|----------|
| `epochs` | **5-10** | Elég a működés ellenőrzéséhez |
| `batch_size` | **128-256** | Nagyobb batch = gyorsabb futás |
| `hidden_size` | **16-32** | Kisebb modell = gyorsabb |
| `num_layers` | **1** | Minimális komplexitás |
| `look_back` | **10-12** | Rövidebb input = gyorsabb |
| `patience` | **2-3** | Gyors early stopping |
| `n_trials` (NAS) | **3** | Minimális keresés |

### CPU vs GPU Modellek

```
CPU MODELLEK (48 db):
├── Statistical (13): ARIMA, SARIMA, stb.
├── Smoothing (5): ETS, Holt-Winters, stb.
├── Probabilistic (5): Prophet, GP, stb.
├── Spectral (7): FFT, Wavelet, stb.
├── Similarity (4): DTW, k-NN, stb.
├── State Space (2): Kalman, stb.
└── Topological (1): TDA

GPU MODELLEK (48 db):
├── Classical ML (4): XGBoost, LightGBM, RF, Conformal
├── DL-RNN (6): LSTM, GRU, DeepAR, stb.
├── DL-CNN (7): N-BEATS, TCN, TimesNet, stb.
├── DL-Transformer (9): Informer, PatchTST, stb.
├── DL-Graph (13): MTGNN, Neural ODE, stb.
├── Meta-Learning (7): NAS, DARTS, stb.
└── Ensemble (1): Time Series Ensemble
```

---

## 3. Modellenkénti Validációs Beállítások

### 3.1 Statistical Models (13 modell)

| # | Modell | Validációs Paraméterek | Becsült idő |
|---|--------|------------------------|-------------|
| 1 | **ARIMA** | `p=1, d=1, q=1` | ~5 sec |
| 2 | **SARIMA** | `p=1, d=1, q=1, P=1, D=0, Q=1, s=12` | ~10 sec |
| 3 | **Auto-ARIMA** | `max_p=2, max_d=1, max_q=2, stepwise=True` | ~30 sec |
| 4 | **ARIMAX** | `p=1, d=1, q=1, add_trend=True` | ~10 sec |
| 5 | **VAR** | `maxlags=5, ic=aic` | ~10 sec |
| 6 | **VECM** | `k_ar_diff=1, coint_rank=1` | ~15 sec |
| 7 | **GARCH Family** | `p=1, q=1, variant=Standard` | ~10 sec |
| 8 | **OGARCH** | `p=1, q=1, n_components=0.95` | ~15 sec |
| 9 | **Quantile Regression** | `quantile=0.5, lags=3` | ~5 sec |
| 10 | **GAM** | `n_knots=10, degree=3` | ~10 sec |
| 11 | **Change Point Detection** | `method=PELT, penalty=10` | ~5 sec |
| 12 | **ADIDA** | `method=standard, aggregation_level=4` | ~10 sec |
| 13 | **CES** | `season_length=12, model_type=Z` | ~5 sec |

**Összesen: ~13 modell, ~2-3 perc**

---

### 3.2 Smoothing & Decomposition (5 modell)

| # | Modell | Validációs Paraméterek | Becsült idő |
|---|--------|------------------------|-------------|
| 1 | **Exponential Smoothing (Holt-Winters)** | `seasonal_periods=12, trend=add, seasonal=add` | ~5 sec |
| 2 | **ETS** | `error=add, trend=add, seasonal=add` | ~10 sec |
| 3 | **Theta** | `period=12, deseasonalize=True` | ~5 sec |
| 4 | **STL** | `period=12, seasonal=7, arima_p=1, arima_d=1, arima_q=1` | ~10 sec |
| 5 | **MSTL** | `periods=4,12, iterate=2` | ~10 sec |

**Összesen: ~5 modell, ~1 perc**

---

### 3.3 Classical Machine Learning (6 modell)

| # | Modell | Validációs Paraméterek | GPU | Becsült idő |
|---|--------|------------------------|-----|-------------|
| 1 | **Random Forest** | `n_estimators=50, max_depth=5, lags=5` | ✅ | ~10 sec |
| 2 | **XGBoost** | `n_estimators=50, learning_rate=0.1, max_depth=5, lags=5, early_stopping=True, early_stopping_rounds=5` | ✅ | ~10 sec |
| 3 | **LightGBM** | `n_estimators=50, learning_rate=0.1, max_depth=5, lags=5` | ✅ | ~10 sec |
| 4 | **Gradient Boosting** | `n_estimators=50, learning_rate=0.1, max_depth=5, lags=5` | ❌ | ~15 sec |
| 5 | **SVR** | `kernel=rbf, C=1.0, lags=5, auto_tune=false` | ❌ | ~10 sec |
| 6 | **KNN Regressor** | `n_neighbors=5, lags=5, auto_tune=false` | ❌ | ~5 sec |

**Összesen: ~6 modell, ~1 perc**

---

### 3.4 Deep Learning - RNN-based (6 modell)

| # | Modell | Validációs Paraméterek | GPU | Becsült idő |
|---|--------|------------------------|-----|-------------|
| 1 | **LSTM** | `epochs=5, batch_size=128, look_back=10, hidden_size=16, num_layers=1, patience=2` | ✅ | ~30 sec |
| 2 | **GRU** | `epochs=5, batch_size=128, look_back=10, hidden_size=16, num_layers=1, patience=2` | ✅ | ~30 sec |
| 3 | **DeepAR** | `epochs=5, batch_size=128, look_back=10, hidden_size=16, num_layers=1, patience=2` | ✅ | ~30 sec |
| 4 | **ES-RNN** | `epochs=5, batch_size=128, look_back=10, hidden_size=16, patience=2` | ✅ | ~30 sec |
| 5 | **MQRNN** | `epochs=5, batch_size=128, look_back=10, hidden_size=16, patience=2` | ✅ | ~30 sec |
| 6 | **Seq2Seq** | `epochs=5, batch_size=128, look_back=10, hidden_size=16, num_layers=1, patience=2` | ✅ | ~30 sec |

**Összesen: ~6 modell, ~3 perc (GPU)**

---

### 3.5 Deep Learning - CNN & Hybrid (7 modell)

| # | Modell | Validációs Paraméterek | GPU | Becsült idő |
|---|--------|------------------------|-----|-------------|
| 1 | **TCN** | `epochs=5, batch_size=128, look_back=10, num_channels=16, num_layers=2, patience=2` | ✅ | ~30 sec |
| 2 | **N-BEATS** | `epochs=5, batch_size=128, look_back=10, hidden_dim=16, num_stacks=1, num_blocks=1, patience=2` | ✅ | ~30 sec |
| 3 | **N-HiTS** | `epochs=5, batch_size=128, look_back=10, hidden_dim=16, num_stacks=1, num_blocks=1, patience=2` | ✅ | ~30 sec |
| 4 | **DLinear** | `epochs=5, batch_size=128, look_back=12, moving_avg=7, patience=2` | ✅ | ~20 sec |
| 5 | **TiDE** | `epochs=5, batch_size=128, look_back=12, hidden_dim=32, patience=2` | ✅ | ~30 sec |
| 6 | **TimesNet** | `epochs=5, batch_size=256, look_back=12, d_model=16, d_ff=16, num_layers=1, patience=2` | ✅ | ~45 sec |
| 7 | **TimesNet Batch** | `epochs=5, batch_size=256, look_back=12, d_model=16, d_ff=16, num_layers=1, patience=2` | ✅ | ~60 sec |

**Összesen: ~7 modell, ~4 perc (GPU)**

---

### 3.6 Deep Learning - Transformer-based (9 modell)

| # | Modell | Validációs Paraméterek | GPU | Becsült idő |
|---|--------|------------------------|-----|-------------|
| 1 | **Transformer (Attention Mechanism)** | `epochs=5, batch_size=128, look_back=10, d_model=16, n_heads=2, num_layers=1, patience=2` | ✅ | ~30 sec |
| 2 | **Informer** | `epochs=5, batch_size=128, look_back=12, d_model=16, n_heads=2, num_layers=1, patience=2` | ✅ | ~45 sec |
| 3 | **Autoformer** | `epochs=5, batch_size=128, look_back=12, d_model=16, n_heads=2, n_layers=1, patience=2` | ✅ | ~45 sec |
| 4 | **FEDformer** | `epochs=5, batch_size=128, look_back=12, d_model=16, n_heads=2, n_layers=1, patience=2` | ✅ | ~45 sec |
| 5 | **PatchTST** | `epochs=5, batch_size=128, look_back=12, patch_len=4, d_model=16, n_heads=2, num_layers=1, patience=2` | ✅ | ~45 sec |
| 6 | **TFT** | `epochs=5, batch_size=128, look_back=10, hidden_size=16, lstm_layers=1, num_heads=2, patience=2` | ✅ | ~45 sec |
| 7 | **iTransformer** | `epochs=5, batch_size=128, look_back=10, d_model=16, n_heads=2, e_layers=1, patience=2` | ✅ | ~30 sec |
| 8 | **FITS** | `epochs=5, batch_size=128, look_back=12, cut_freq=0, patience=2` | ✅ | ~20 sec |
| 9 | **Channel Independence** | `epochs=5, batch_size=128, look_back=10, hidden_dim=16, num_layers=1, patience=2` | ✅ | ~30 sec |

**Összesen: ~9 modell, ~5-6 perc (GPU)**

---

### 3.7 Deep Learning - Graph & Specialized (13 modell)

| # | Modell | Validációs Paraméterek | GPU | Becsült idő |
|---|--------|------------------------|-----|-------------|
| 1 | **MTGNN** | `epochs=10, batch_size=64, look_back=24, layers=2, patience=3` | ✅ | ~60 sec |
| 2 | **MTGNN Batch** | `epochs=10, batch_size=128, look_back=12, layers=2, patience=3` | ✅ | ~90 sec |
| 3 | **StemGNN** | `epochs=10, batch_size=64, look_back=12, hidden_dim=16, patience=3` | ✅ | ~60 sec |
| 4 | **Neural ODE** | `epochs=5, batch_size=128, look_back=10, hidden_size=16, solver_steps=5, patience=2` | ✅ | ~45 sec |
| 5 | **Spiking Neural Network** | `epochs=5, batch_size=128, look_back=10, hidden_size=16, num_layers=1, patience=2` | ✅ | ~30 sec |
| 6 | **KAN** | `epochs=10, batch_size=64, look_back=12, hidden_size=32, num_layers=1, grid_size=4, patience=3` | ✅ | ~60 sec |
| 7 | **Diffusion** | `epochs=10, batch_size=64, look_back=12, diffusion_steps=50, hidden_dim=32, patience=3, num_samples=3` | ✅ | ~90 sec |
| 8 | **Neural ARIMA** | `epochs=5, batch_size=128, look_back=10, hidden_size=16, p=3, d=1, q=3, patience=2` | ✅ | ~30 sec |
| 9 | **Neural VAR** | `epochs=5, batch_size=128, look_back=10, hidden_size=16, num_layers=1, patience=2` | ✅ | ~30 sec |
| 10 | **Neural GAM** | `epochs=5, batch_size=128, look_back=10, hidden_size=16, num_layers=1, patience=2` | ✅ | ~30 sec |
| 11 | **Neural Basis Functions** | `epochs=5, batch_size=128, look_back=10, num_centers=20, patience=2` | ✅ | ~20 sec |
| 12 | **Neural Quantile Regression** | `epochs=10, batch_size=64, look_back=10, hidden_size=32, num_layers=1, patience=3` | ✅ | ~45 sec |
| 13 | **Neural Volatility** | `epochs=5, batch_size=128, look_back=10, hidden_size=16, num_layers=1, patience=2` | ✅ | ~30 sec |

**Összesen: ~13 modell, ~10-12 perc (GPU)**

---

### 3.8 Meta-Learning & AutoML (7 modell)

| # | Modell | Validációs Paraméterek | GPU | Becsült idő |
|---|--------|------------------------|-----|-------------|
| 1 | **FFORMA** | `n_windows=10, look_back=26, period=52` | ❌ | ~60 sec |
| 2 | **NAS** | `n_trials=3, epochs=5, batch_size=64, look_back=12, patience=2` | ✅ | ~120 sec |
| 3 | **DARTS** | `epochs=5, batch_size=128, look_back=10, hidden_size=16, num_cells=1, patience=2` | ✅ | ~60 sec |
| 4 | **Meta-learning** | `meta_epochs=5, meta_lr=0.001, inner_lr=0.01, adaptation_steps=2, look_back=10, hidden_dim=16` | ✅ | ~60 sec |
| 5 | **Mixture of Experts** | `epochs=5, batch_size=128, look_back=10, num_experts=2, expert_hidden_size=16, patience=2` | ✅ | ~45 sec |
| 6 | **Multi-task Learning** | `epochs=10, batch_size=64, look_back=10, hidden_size=32, patience=3` | ✅ | ~60 sec |
| 7 | **GFM** | `epochs=5, batch_size=128, look_back=12, hidden_size=32, num_layers=1, patience=2` | ✅ | ~45 sec |

**Összesen: ~7 modell, ~8-10 perc**

---

### 3.9 Bayesian & Probabilistic Methods (5 modell)

| # | Modell | Validációs Paraméterek | GPU | Becsült idő |
|---|--------|------------------------|-----|-------------|
| 1 | **BSTS** | `seasonal=52, trend=local level, ar_order=0` | ❌ | ~30 sec |
| 2 | **Prophet** | `growth=linear, changepoint_prior_scale=0.05, n_changepoints=15` | ❌ | ~15 sec |
| 3 | **Gaussian Process** | `kernel=RBF, n_restarts_optimizer=1, periodicity=52` | ❌ | ~30 sec |
| 4 | **Conformal Prediction** | `significance=0.1, calibration_size=0.2, backend=auto` | ✅ | ~20 sec |
| 5 | **Monte Carlo Simulation** | `simulations=500, random_seed=42` | ❌ | ~10 sec |

**Összesen: ~5 modell, ~2 perc**

---

### 3.10 Frequency Domain & Signal Processing (7 modell)

| # | Modell | Validációs Paraméterek | GPU | Becsült idő |
|---|--------|------------------------|-----|-------------|
| 1 | **FFT** | `n_harmonics=5, trend_poly_order=1` | ❌ | ~5 sec |
| 2 | **DFT** | `n_harmonics=5, trend_poly_order=1` | ❌ | ~5 sec |
| 3 | **Wavelet Analysis** | `wavelet=db4, level=2` | ❌ | ~10 sec |
| 4 | **Spectral Analysis** | `n_harmonics=5, trend_poly_order=1` | ❌ | ~5 sec |
| 5 | **Periodogram** | `n_harmonics=5, trend_poly_order=1` | ❌ | ~5 sec |
| 6 | **Welch's Method** | `n_harmonics=5, trend_poly_order=1` | ❌ | ~5 sec |
| 7 | **SSA** | `window_size=0, n_components=0, damping=0.98` (auto) | ❌ | ~15 sec |

**Összesen: ~7 modell, ~1 perc**

---

### 3.11 Distance & Similarity-based (4 modell)

| # | Modell | Validációs Paraméterek | GPU | Becsült idő |
|---|--------|------------------------|-----|-------------|
| 1 | **DTW** | `look_back=26, top_k=3, window=auto` | ❌ | ~30 sec |
| 2 | **k-NN** | `look_back=26, top_k=3, metric=euclidean` | ❌ | ~15 sec |
| 3 | **Matrix Profile** | `look_back=26, top_k=3` | ❌ | ~30 sec |
| 4 | **k-Shape** | `look_back=12, n_clusters=3, max_iter=50, top_k=3` | ❌ | ~45 sec |

**Összesen: ~4 modell, ~2 perc**

---

### 3.12 State Space & Filtering (2 modell)

| # | Modell | Validációs Paraméterek | GPU | Becsült idő |
|---|--------|------------------------|-----|-------------|
| 1 | **Kalman Filter** | `level=True, trend=True, seasonal=12, ar_order=0` | ❌ | ~15 sec |
| 2 | **State Space Models** | `level=True, trend=True, seasonal=12, ar_order=0` | ❌ | ~15 sec |

**Összesen: ~2 modell, ~30 sec**

---

### 3.13 Topological Methods (1 modell)

| # | Modell | Validációs Paraméterek | GPU | Becsült idő |
|---|--------|------------------------|-----|-------------|
| 1 | **TDA** | `embedding_dim=3, time_delay=1, window_size=20` | ❌ | ~30 sec |

**Összesen: ~1 modell, ~30 sec**

---

### 3.14 Ensemble Methods (1 modell)

| # | Modell | Validációs Paraméterek | GPU | Becsült idő |
|---|--------|------------------------|-----|-------------|
| 1 | **Time Series Ensemble** | `models=ARIMA,ETS, combination=Mean` | ✅ | ~30 sec |

**Összesen: ~1 modell, ~30 sec**

---

## Összesítés

| Kategória | Modellek | Becsült idő |
|-----------|----------|-------------|
| Statistical Models | 13 | ~3 perc |
| Smoothing & Decomposition | 5 | ~1 perc |
| Classical Machine Learning | 6 | ~1 perc |
| Deep Learning - RNN | 6 | ~3 perc |
| Deep Learning - CNN | 7 | ~4 perc |
| Deep Learning - Transformer | 9 | ~6 perc |
| Deep Learning - Graph | 13 | ~12 perc |
| Meta-Learning & AutoML | 7 | ~10 perc |
| Bayesian & Probabilistic | 5 | ~2 perc |
| Frequency Domain | 7 | ~1 perc |
| Distance & Similarity | 4 | ~2 perc |
| State Space | 2 | ~0.5 perc |
| Topological | 1 | ~0.5 perc |
| Ensemble | 1 | ~0.5 perc |
| **ÖSSZESEN** | **86 modell** | **~45-50 perc** |

**Megjegyzés:** Néhány modell alias (pl. Holt-Winters = Exponential Smoothing), ezért 86 egyedi modell van ~100 névvel.

---

## 4. Ellenőrzési Checklist

### Minden Modellnél Ellenőrizd

```
□ 1. FUTÁS
   □ Modell indul hiba nélkül
   □ Progress bar/log látható
   □ Nincs exception/error üzenet
   □ Futás befejeződik

□ 2. EREDMÉNY
   □ 52 hetes forecast generálódott
   □ Értékek megjelennek az UI-ban
   □ Inspection-be menthető

□ 3. GPU (ha GPU-s modell)
   □ GPU kihasználtság látható (nvidia-smi)
   □ CUDA memória foglalódik
   □ GPU memória felszabadul futás után

□ 4. KIMENET MINŐSÉGE
   □ Forecast értékek számok (nem NaN)
   □ Nincs Inf érték
   □ Értékek "értelmesnek" tűnnek (nem 0, nem extrém)
   □ Grafikon renderelődik

□ 5. MEMÓRIA
   □ Python memória stabil
   □ GPU memória nem nő folyamatosan
   □ Nincs memória leak jel
```

### Eredmény Dokumentálás

Minden modellnél:

```
Modell: [NÉV]
Dátum: [DÁTUM]
Státusz: ✅ PASS / ❌ FAIL / ⚠️ WARNING
Futási idő: [X sec]
GPU használat: Igen / Nem
Megjegyzés: [Bármilyen észrevétel]
```

---

## 5. Hibakezelési Útmutató

### Gyakori Hibák és Megoldások

| Hiba | Lehetséges ok | Megoldás |
|------|---------------|----------|
| `CUDA out of memory` | Túl nagy batch_size vagy modell | Csökkentsd batch_size-t (64 vagy 32) |
| `NaN loss` | Túl nagy learning_rate | Csökkentsd learning_rate-et (0.0001) |
| `Dimension mismatch` | Hibás look_back/input méret | Ellenőrizd az adathosszt |
| `Module not found` | Hiányzó dependency | `pip install [csomag]` |
| `Timeout` | Túl lassú modell | Csökkentsd epochs/hidden_size-t |
| `Empty forecast` | Modell nem konvergált | Növeld epochs-t vagy változtass paramétereket |

### GPU Specifikus Hibák

| Hiba | Megoldás |
|------|----------|
| `CUDA not available` | Ellenőrizd: `torch.cuda.is_available()` |
| `GPU memory fragmentation` | Restart Python kernel |
| `cuDNN error` | Frissítsd cuDNN-t vagy csökkentsd batch_size-t |

---

## 6. Végrehajtási Terv

### Javasolt Sorrend

1. **Először: CPU modellek** (gyorsak, egyszerűek)
   - Statistical Models
   - Smoothing & Decomposition
   - Frequency Domain
   - Similarity-based

2. **Másodszor: Egyszerű GPU modellek**
   - Classical ML (XGBoost, LightGBM)
   - Egyszerű DL (LSTM, GRU, TCN)

3. **Harmadszor: Komplex GPU modellek**
   - Transformer-ek
   - Graph Neural Networks
   - Meta-Learning

### Időbeosztás

```
Session 1 (~15 perc): CPU modellek (27 db)
├── Statistical (13)
├── Smoothing (5)
├── Frequency (7)
└── Similarity + State Space + Topological (7)

Session 2 (~15 perc): Egyszerű GPU modellek (19 db)
├── Classical ML (6)
├── RNN (6)
└── CNN & Hybrid (7)

Session 3 (~20 perc): Komplex GPU modellek (22 db)
├── Transformer (9)
├── Graph & Specialized (13)

Session 4 (~10 perc): Meta & Ensemble (8 db)
├── Meta-Learning & AutoML (7)
└── Ensemble (1)
```

---

## Következő Lépések

1. **Döntés**: Manuálisan vagy automatizáltan futtassuk a teszteket?
2. **Ha manuális**: Használd ezt a dokumentumot checklistként
3. **Ha automatizált**: Készíthetek egy scriptet ami végigmegy mindenen
4. **Eredmények gyűjtése**: Készíts egy Excel/CSV táblázatot az eredményekről

---

*Dokumentum vége*
