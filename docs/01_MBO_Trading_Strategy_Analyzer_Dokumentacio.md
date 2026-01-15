# MBO Trading Strategy Analyzer - Program Dokumentáció

## Tartalomjegyzék

- [01.00 Program alapvető működése](#0100-program-alapvető-működése)
- [01.01 Modellek kategóriái (fa szerkezet)](#0101-modellek-kategóriái-fa-szerkezet)
- [01.02 Technikai háttér és statisztika](#0102-technikai-háttér-és-statisztika)
- [01.03 PC konfiguráció](#0103-pc-konfiguráció)

---

# 01.00 Program alapvető működése

## Mi az MBO Trading Strategy Analyzer?

Az **MBO Trading Strategy Analyzer** egy professzionális idősor-előrejelző alkalmazás, amelyet kereskedési stratégiák elemzésére és előrejelzésére fejlesztettek ki. A program **100+ különböző előrejelző modellt** tartalmaz **14 kategóriában**, és képes több devizapár stratégiáit párhuzamosan elemezni.

## Fő funkciók

### 1. Adatbetöltés és feldolgozás
- **Excel fájlok betöltése**: Speciális horizontális időblokk struktúrájú Excel fájlok feldolgozása
- **Parquet támogatás**: Nagyméretű adathalmazok gyors betöltése Parquet formátumban
- **Kötegelt betöltés**: Teljes mappák automatikus feldolgozása egyszerre
- **Előfeldolgozás**: Normalizálás, hiányzó értékek kezelése, idősor ablakozás

### 2. Modell kiválasztás és konfigurálás
A felhasználó a következő kategóriákból választhat:
- Statisztikai modellek (ARIMA, SARIMA, GARCH, stb.)
- Simítási és dekompozíciós módszerek (ETS, Holt-Winters, STL)
- Klasszikus gépi tanulás (Random Forest, XGBoost, LightGBM)
- Mélytanulás - RNN alapú (LSTM, GRU, DeepAR)
- Mélytanulás - CNN és hibrid (TCN, N-BEATS, TimesNet)
- Mélytanulás - Transformer alapú (TFT, Informer, PatchTST)
- És még 8 további kategória...

Minden modellhez **egyedi paraméterbeállítások** érhetők el.

### 3. Végrehajtási módok

#### Független mód (Per-Strategy)
- **Alapértelmezett működés**
- Minden stratégiát külön-külön elemez
- Legjobb pontosságot biztosít
- Legtöbb időt igényli

#### Panel mód
- **Egyetlen modell az összes stratégiára**
- 5-10x gyorsabb végrehajtás
- Csak Multi-output ML modellek támogatják
- Támogatott: Random Forest, XGBoost, LightGBM, Gradient Boosting, KNN Regressor

#### Dual Model mód
- **Két külön modell**: Activity (aktivitás) és Profit (nyereség)
- Activity Model: Bináris klasszifikáció - kereskedünk-e
- Profit Model: Regresszió - várható profit
- Külön hiperparaméter-hangolás mindkét modellhez

### 4. Párhuzamos végrehajtás
- **Multiprocessing**: Konfigurálható worker pool
- **CPU Manager**: Intelligens erőforrás-allokáció (alapértelmezett: 75%)
- **GPU támogatás**: 47 modell GPU-gyorsítással (CUDA)
- **Memória kezelés**: Pool újraindítás minden 500 stratégia után

### 5. Eredmények megjelenítése és exportálás
- **Előrejelzés vizualizáció**: Interaktív grafikonok
- **Stratégia rangsor**: Horizont-specifikus rangsorolás (1 hét, 1 hónap, 3 hónap, 6 hónap, 12 hónap)
- **Teljesítmény metrikák**: MAE, RMSE, MAPE, R², és pénzügyi mutatók
- **Export formátumok**: Markdown, HTML, Excel

### 6. Összehasonlítás és ellenőrzés
- **Modell összehasonlítás**: Több modell eredményeinek összevetése
- **Benchmark összehasonlítás**: Előrejelzés vs. valós adatok
- **AR-specifikus elemzés**: Autoregresszív modellek külön elemzése

## Hogyan indítsd el a programot?

### Asztali alkalmazás (GUI)
```bash
python src/main.py
```

### Web felület (Streamlit)
```bash
streamlit run src/web_ui.py
```

### Docker konténer
```bash
docker build -t mbo-analyzer .
docker run -p 8501:8501 mbo-analyzer
```

## Felhasználói felület felépítése

### Fülek (Tabs)
1. **Data (Adatok)** - Fájlok betöltése és előnézet
2. **Analysis (Elemzés)** - Modell kiválasztás és futtatás
3. **Results (Eredmények)** - Előrejelzések és rangsorok
4. **Compare (Összehasonlítás)** - Modellek összevetése
5. **Inspection (Ellenőrzés)** - Riportok elemzése
6. **Performance (Teljesítmény)** - Rendszer erőforrások monitorozása

## Tipikus munkamenet

1. **Adatok betöltése**: Data fülön Excel/Parquet fájl kiválasztása
2. **Modell kiválasztás**: Analysis fülön kategória és modell választás
3. **Paraméterek beállítása**: Model-specifikus beállítások konfigurálása
4. **Futtatás**: Start Analysis gomb megnyomása
5. **Eredmények megtekintése**: Results fülön előrejelzések elemzése
6. **Export**: Riport mentése Markdown/HTML formátumban

## Speciális funkciók

### Auto Execution (Automatikus végrehajtás)
- Több elemzés sorba állítása
- Szekvenciális vagy párhuzamos futtatás
- Automatikus leállítás befejezés után

### Hang visszajelzések
- Elemzés indítása/befejezése hangjelzés
- Tab váltás hangok
- Bekapcsolható/kikapcsolható

### Többnyelvű támogatás
- Magyar (HU)
- Angol (EN)

## Fontos jellemzők

| Jellemző | Érték |
|----------|-------|
| Verzió | v3.60.0 Stable |
| Modellek száma | 100+ |
| Kategóriák száma | 14 |
| GPU támogatott modellek | 47 |
| Ensemble-képes modellek | 11 |
| Panel mód támogatás | 5 modell |
| Dual model támogatás | 5 modell |

---

# 01.01 Modellek kategóriái (fa szerkezet)

```
MBO Trading Strategy Analyzer - Modellek
│
├── 01. Statistical Models (Statisztikai modellek) [13 modell]
│   ├── ADIDA
│   ├── ARIMA
│   ├── ARIMAX
│   ├── Auto-ARIMA
│   ├── CES (Complex Exponential Smoothing)
│   ├── Change Point Detection
│   ├── GAM (Generalized Additive Model)
│   ├── GARCH Family
│   ├── OGARCH (Orthogonal GARCH)
│   ├── Quantile Regression
│   ├── SARIMA
│   ├── VAR (Vector AutoRegression)
│   └── VECM (Vector Error Correction Model)
│
├── 02. Smoothing & Decomposition (Simítás és dekompozíció) [5 modell]
│   ├── ETS (Error-Trend-Seasonal)
│   ├── Exponential Smoothing (Holt-Winters)
│   ├── MSTL (Multiple STL)
│   ├── STL (Seasonal-Trend Decomposition using LOESS)
│   └── Theta
│
├── 03. Classical Machine Learning (Klasszikus ML) [6 modell]
│   ├── Gradient Boosting
│   ├── KNN Regressor
│   ├── LightGBM
│   ├── Random Forest
│   ├── SVR (Support Vector Regressor)
│   └── XGBoost
│
├── 04. Deep Learning - RNN-based (RNN alapú) [6 modell]
│   ├── DeepAR
│   ├── ES-RNN
│   ├── GRU
│   ├── LSTM
│   ├── MQRNN (Multi-Quantile RNN)
│   └── Seq2Seq
│
├── 05. Deep Learning - CNN & Hybrid Architectures [7 modell]
│   ├── DLinear
│   ├── N-BEATS
│   ├── N-HiTS
│   ├── TCN (Temporal Convolutional Network)
│   ├── TiDE
│   ├── TimesNet
│   └── TimesNet Batch (GPU-optimalizált batch mód)
│
├── 06. Deep Learning - Transformer-based [9 modell]
│   ├── Autoformer
│   ├── Channel Independence
│   ├── FEDformer
│   ├── FITS
│   ├── Informer
│   ├── iTransformer
│   ├── PatchTST
│   ├── TFT (Temporal Fusion Transformer)
│   └── Transformer (Attention Mechanism)
│
├── 07. Deep Learning - Graph & Specialized Neural Networks [12 modell]
│   ├── Diffusion
│   ├── KAN (Kolmogorov-Arnold Network)
│   ├── MTGNN (Multivariate Time-series Graph NN)
│   ├── Neural ARIMA
│   ├── Neural Basis Functions
│   ├── Neural GAM
│   ├── Neural ODE
│   ├── Neural Quantile Regression
│   ├── Neural VAR
│   ├── Neural Volatility
│   ├── Spiking Neural Network
│   └── StemGNN
│
├── 08. Meta-Learning & AutoML [7 modell]
│   ├── DARTS (Differentiable Architecture Search)
│   ├── FFORMA (Feature-based Forecast Model Averaging)
│   ├── GFM (Generalist Foundation Model)
│   ├── Meta-learning
│   ├── Mixture of Experts
│   ├── Multi-task Learning
│   └── NAS (Neural Architecture Search)
│
├── 09. Bayesian & Probabilistic Methods [5 modell]
│   ├── BSTS (Bayesian Structural Time Series)
│   ├── Conformal Prediction
│   ├── Gaussian Process
│   ├── Monte Carlo Simulation
│   └── Prophet
│
├── 10. Frequency Domain & Signal Processing [7 modell]
│   ├── DFT (Discrete Fourier Transform)
│   ├── FFT (Fast Fourier Transform)
│   ├── Periodogram
│   ├── Spectral Analysis
│   ├── SSA (Singular Spectrum Analysis)
│   ├── Wavelet Analysis
│   └── Welch's Method
│
├── 11. Distance & Similarity-based [4 modell]
│   ├── DTW (Dynamic Time Warping)
│   ├── k-NN
│   ├── k-Shape
│   └── Matrix Profile
│
├── 12. State Space & Filtering [2 modell]
│   ├── Kalman Filter
│   └── State Space Models
│
├── 13. Topological Methods [1 modell]
│   └── TDA (Topological Data Analysis)
│
└── 14. Ensemble Methods [1 modell]
    └── Time Series Ensemble
```

### GPU támogatott modellek (47 db)

```
GPU Supported Models
├── Attention Mechanism / Transformer
├── Autoformer
├── Channel Independence
├── Conformal Prediction (XGBoost backend)
├── DARTS
├── DeepAR
├── Diffusion
├── DLinear
├── ES-RNN
├── FEDformer
├── FITS
├── GFM
├── GRU
├── Informer
├── iTransformer
├── KAN
├── LightGBM
├── LSTM
├── Meta-learning
├── Mixture of Experts
├── MQRNN
├── MTGNN
├── Multi-task Learning
├── N-BEATS
├── N-HiTS
├── NAS
├── Neural ARIMA
├── Neural Basis Functions
├── Neural GAM
├── Neural ODE
├── Neural Quantile Regression
├── Neural VAR
├── Neural Volatility
├── PatchTST
├── Random Forest (Hummingbird/PyTorch backend)
├── Seq2Seq
├── Spiking Neural Network
├── StemGNN
├── TCN
├── TFT
├── TiDE
├── Time Series Ensemble
├── TimesNet
├── TimesNet Batch
└── XGBoost
```

### Panel mód támogatott modellek (5 db)

```
Panel Mode Models (egy modell - összes stratégia)
├── Gradient Boosting
├── KNN Regressor
├── LightGBM
├── Random Forest
└── XGBoost
```

### Dual Model mód támogatott modellek (5 db)

```
Dual Model Mode (Activity + Profit külön modell)
├── Gradient Boosting
├── KNN Regressor
├── LightGBM
├── Random Forest
└── XGBoost
```

---

# 01.02 Technikai háttér és statisztika

## Futtatási környezet

### Alkalmazás típusok

| Felület | Technológia | Belépési pont |
|---------|-------------|---------------|
| Asztali GUI | CustomTkinter | `src/main.py` |
| Web UI | Streamlit | `src/web_ui.py` |
| Docker | Konténerizált | `Dockerfile` / `docker-compose.yml` |

### Függőségek (főbb könyvtárak)

| Kategória | Könyvtárak |
|-----------|-----------|
| Adatfeldolgozás | pandas, numpy, scipy |
| Klasszikus ML | scikit-learn, xgboost, lightgbm |
| Mélytanulás | PyTorch, TensorFlow |
| Statisztika | statsmodels, pmdarima, arch |
| Idősor specifikus | prophet |
| GUI | customtkinter, PIL |
| Web | streamlit, plotly |
| Párhuzamosítás | joblib, multiprocessing |
| Rendszer monitoring | psutil, GPUtil |

## Fájl statisztika

| Metrika | Érték |
|---------|-------|
| **Összes fájl a src mappában** | 349 |
| **Python fájlok száma** | 158 |
| **Python kód sorok (összesen)** | 55,142 |
| **Átlagos sorok/fájl** | ~349 |

### Mappa struktúra

```
src/
├── main.py                    # Fő belépési pont (GUI)
├── main_debug.py              # Debug mód
├── web_ui.py                  # Streamlit web felület
├── requirements.txt           # Python függőségek
│
├── analysis/                  # Elemzés motor
│   ├── engine.py              # Fő orchestrátor
│   ├── strategy_analyzer.py   # Modell router
│   ├── cpu_manager.py         # CPU/GPU erőforrás kezelés
│   ├── performance.py         # Teljesítmény monitor
│   ├── metrics.py             # Pénzügyi metrikák
│   ├── inspection.py          # Riport ellenőrzés
│   ├── panel_executor.py      # Panel mód végrehajtó
│   ├── dual_executor.py       # Dual model végrehajtó
│   ├── comparator/            # Összehasonlító modul (5 fájl)
│   └── models/                # 89 modell implementáció
│       ├── classical_ml/      # 8 fájl
│       ├── dl_cnn/            # 7 fájl
│       ├── dl_rnn/            # 6 fájl
│       ├── dl_transformer/    # 9 fájl
│       ├── dl_graph_specialized/ # 12 fájl
│       ├── ensemble/          # 1 fájl
│       ├── meta_learning/     # 7 fájl
│       ├── probabilistic/     # 5 fájl
│       ├── similarity/        # 4 fájl
│       ├── smoothing_and_decomposition/ # 5 fájl
│       ├── spectral/          # 7 fájl
│       ├── state_space/       # 2 fájl
│       ├── statistical_models/ # 13 fájl
│       └── topological/       # 1 fájl
│
├── data/                      # Adat betöltés és feldolgozás
│   ├── loader.py              # DataLoader (Excel/Parquet)
│   └── processor.py           # DataProcessor (normalizálás)
│
├── config/                    # Központi konfiguráció
│   ├── models.py              # Modell kategóriák és listák
│   ├── parameters.py          # Modell alapértelmezett paraméterek
│   └── __init__.py
│
├── gui/                       # CustomTkinter GUI
│   ├── main_window.py         # Főablak (mixin pattern)
│   ├── auto_execution_mixin.py
│   ├── sorrend_data.py        # Beállítások perzisztencia
│   ├── sound_manager.py       # Hang visszajelzések
│   ├── help_parser.py         # Súgó elemző
│   ├── translations.py        # Többnyelvű szövegek
│   ├── tabs/                  # 8 tab modul
│   ├── widgets/               # Egyedi UI komponensek
│   ├── utils/                 # GUI segédeszközök
│   ├── assets/                # Képek, ikonok
│   └── sounds/                # Hangfájlok
│
├── web/                       # Streamlit web felület
│   ├── tabs/                  # 6 web tab modul
│   ├── components/            # Újrafelhasználható komponensek
│   └── utils/                 # Web segédeszközök
│
├── reporting/                 # Riport generálás
│   ├── exporter.py            # Markdown/HTML export
│   └── visualizer.py          # Vizualizációk
│
├── utils/                     # Közös segédeszközök
│   └── optional_imports.py    # Opcionális import kezelés
│
└── help/                      # Súgó dokumentáció fájlok
```

### Kulcs fájlok mérete

| Fájl | Funkció | Sor (kb.) |
|------|---------|-----------|
| `config/parameters.py` | Modell paraméterek | ~1,564 |
| `gui/tabs/performance_tab.py` | Teljesítmény UI | ~68KB |
| `gui/tabs/analysis_tab.py` | Elemzés UI | ~53KB |
| `gui/tabs/results_tab.py` | Eredmények UI | ~44KB |
| `analysis/engine.py` | Fő motor | ~27KB |
| `analysis/strategy_analyzer.py` | Model router | ~24KB |
| `analysis/cpu_manager.py` | CPU kezelés | ~17KB |

---

# 01.03 PC konfiguráció

## Fejlesztői/Futtatási környezet

### Processzor (CPU)

| Tulajdonság | Érték |
|-------------|-------|
| **Modell** | Intel Core i9-14900HX |
| **Fizikai magok** | 24 |
| **Logikai szálak** | 32 |
| **Architektúra** | Hibrid (P-core + E-core) |
| **Generáció** | 14. generáció (Raptor Lake Refresh) |

### Grafikus kártya (GPU)

| GPU | Típus | Memória |
|-----|-------|---------|
| **NVIDIA GeForce RTX 4060 Laptop** | Dedikált | ~4 GB GDDR6 |
| **Intel UHD Graphics** | Integrált | ~2 GB (megosztott) |

**Megjegyzés**: A program a dedikált NVIDIA GPU-t használja CUDA-val a mélytanulási modellekhez.

### Memória (RAM)

| Tulajdonság | Érték |
|-------------|-------|
| **Összkapacitás** | 64 GB |
| **Típus** | DDR5 (valószínűleg) |

### Operációs rendszer

| Tulajdonság | Érték |
|-------------|-------|
| **OS** | Microsoft Windows 11 Pro |
| **Verzió** | 10.0.26200 |
| **Build** | Insider/Dev Channel (magas build szám) |

## Erőforrás-kezelés a programban

### CPU Manager
- Alapértelmezett CPU allokáció: **75%**
- Automatikus worker pool méretezés a CPU magok alapján
- CPU-intenzív modellek korlátozása (SVR, KNN Regressor)
- Pool újraindítás minden 500 stratégia után (memória optimalizálás)

### GPU támogatás
- PyTorch CUDA backend
- TensorFlow GPU támogatás
- 47 modell használhatja a GPU-t
- Automatikus GPU detektálás és inicializálás

### Memória kezelés
- Worker pool újraindítás a memória felszabadításához
- Garbage collection stratégiai pontokon
- Gyermek folyamatok figyelése és lezárása

---

*Dokumentum készítése: 2024.12.29*
*Program verzió: v3.60.0 Stable*
*Dokumentáció verzió: 1.0*
