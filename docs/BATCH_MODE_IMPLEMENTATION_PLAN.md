# Batch Mode Implementation Plan

## Overview

A Batch Mode modellekben **egyetlen modellt** tanítunk az **összes stratégiára** egyszerre, szemben a Per-Strategy móddal, ahol minden stratégiához külön modell készül. Ez 10-50x gyorsítást eredményez nagy számú stratégia (6000+) esetén.

---

## Már Implementált Batch Modellek

| Modell | Fájl | Státusz |
|--------|------|---------|
| TimesNet | `src/analysis/models/dl_cnn/timesnet_batch.py` | ✅ Kész |
| MTGNN | `src/analysis/models/dl_graph_specialized/mtgnn_batch.py` | ✅ Kész |
| Autoformer | `src/analysis/models/dl_transformer/autoformer_batch.py` | ✅ Kész |
| **Informer** | `src/analysis/models/dl_transformer/informer_batch.py` | ✅ Kész |
| **PatchTST** | `src/analysis/models/dl_transformer/patchtst_batch.py` | ✅ Kész |
| **TFT** | `src/analysis/models/dl_transformer/tft_batch.py` | ✅ Kész |
| **Transformer** | `src/analysis/models/dl_transformer/transformer_batch.py` | ✅ Kész |
| **iTransformer** | `src/analysis/models/dl_transformer/itransformer_batch.py` | ✅ Kész |
| DeepAR | `src/analysis/models/dl_rnn/deepar_batch.py` | ✅ Kész |
| ES-RNN | `src/analysis/models/dl_rnn/es_rnn_batch.py` | ✅ Kész |
| GRU | `src/analysis/models/dl_rnn/gru_batch.py` | ✅ Kész |
| LSTM | `src/analysis/models/dl_rnn/lstm_batch.py` | ✅ Kész |
| MQRNN | `src/analysis/models/dl_rnn/mqrnn_batch.py` | ✅ Kész |
| Seq2Seq | `src/analysis/models/dl_rnn/seq2seq_batch.py` | ✅ Kész |
| DLinear | `src/analysis/models/dl_cnn/dlinear_batch.py` | ✅ Kész |
| N-BEATS | `src/analysis/models/dl_cnn/nbeats_batch.py` | ✅ Kész |
| N-HiTS | `src/analysis/models/dl_cnn/nhits_batch.py` | ✅ Kész |
| TCN | `src/analysis/models/dl_cnn/tcn_batch.py` | ✅ Kész |
| TiDE | `src/analysis/models/dl_cnn/tide_batch.py` | ✅ Kész |
| FEDFormer | `src/analysis/models/dl_transformer/fedformer_batch.py` | ✅ Kész |
| FiTS | `src/analysis/models/dl_transformer/fits_batch.py` | ✅ Kész |
| **Diffusion** | `src/analysis/models/dl_graph_specialized/diffusion_batch.py` | ✅ Kész |
| **KAN** | `src/analysis/models/dl_graph_specialized/kan_batch.py` | ✅ Kész |
| **Neural ARIMA** | `src/analysis/models/dl_graph_specialized/neural_arima_batch.py` | ✅ Kész |
| **Neural Basis Functions** | `src/analysis/models/dl_graph_specialized/rbf_batch.py` | ✅ Kész |
| **Neural GAM** | `src/analysis/models/dl_graph_specialized/neural_gam_batch.py` | ✅ Kész |
| **Neural ODE** | `src/analysis/models/dl_graph_specialized/neural_ode_batch.py` | ✅ Kész |
| **Neural VAR** | `src/analysis/models/dl_graph_specialized/neural_var_batch.py` | ✅ Kész |
| **Neural Volatility** | `src/analysis/models/dl_graph_specialized/neural_volatility_batch.py` | ✅ Kész |
| **Spiking Neural Networks** | `src/analysis/models/dl_graph_specialized/snn_batch.py` | ✅ Kész |
| **StemGNN** | `src/analysis/models/dl_graph_specialized/stemgnn_batch.py` | ✅ Kész |

---

## Hiányzó Batch Implementációk

### CNN Modellek (`src/analysis/models/dl_cnn/`)

| Modell | Prioritás | Komplexitás | Megjegyzés |
|--------|-----------|-------------|------------|
| ~~**N-BEATS**~~ | ~~Magas~~ | ~~Közepes~~ | ✅ Implementálva |
| ~~**N-HiTS**~~ | ~~Magas~~ | ~~Közepes~~ | ✅ Implementálva |
| ~~**TCN**~~ | ~~Közepes~~ | ~~Közepes~~ | ✅ Implementálva |
| ~~**DLinear**~~ | ~~Alacsony~~ | ~~Alacsony~~ | ✅ Implementálva |
| ~~**TiDE**~~ | ~~Közepes~~ | ~~Közepes~~ | ✅ Implementálva |

### Transformer Modellek (`src/analysis/models/dl_transformer/`)

| Modell | Prioritás | Komplexitás | Megjegyzés |
|--------|-----------|-------------|------------|
| ~~**Informer**~~ | ~~Magas~~ | ~~Magas~~ | ✅ Implementálva |
| ~~**TFT**~~ | ~~Magas~~ | ~~Magas~~ | ✅ Implementálva |
| ~~**PatchTST**~~ | ~~Magas~~ | ~~Közepes~~ | ✅ Implementálva |
| ~~**iTransformer**~~ | ~~Közepes~~ | ~~Közepes~~ | ✅ Implementálva |
| ~~**FEDformer**~~ | ~~Közepes~~ | ~~Magas~~ | ✅ Implementálva |
| ~~**FITS**~~ | ~~Alacsony~~ | ~~Alacsony~~ | ✅ Implementálva |
| ~~**Transformer**~~ | ~~Közepes~~ | ~~Közepes~~ | ✅ Implementálva |

---

## Implementációs Lépések (Minden Modellhez)

### 1. FÁZIS: Batch Fájl Létrehozása

```
Fájl elnevezés: {model_name}_batch.py
Helye: Ugyanaz a mappa mint az eredeti modell
```

**Kötelező komponensek:**

#### 1.1 TradingRobustScaler osztály
```python
class TradingRobustScaler:
    """
    Trading adatokhoz optimalizált scaler.
    - Kezeli a ritka profit eloszlásokat (sok 0 érték)
    - Non-zero értékekből számol statisztikákat
    - Fallback mean/std-re ha IQR túl kicsi
    """
    def __init__(self):
        self.center_ = None
        self.scale_ = None
        self._fitted = False

    def fit(self, x_input): ...
    def transform(self, x_input): ...
    def fit_transform(self, x_input): ...
    def inverse_transform(self, x_input): ...
```

#### 1.2 Batch Network osztály
```python
class {Model}BatchNetwork(nn.Module):
    """
    PyTorch network a batch tanításhoz.
    - Összes stratégia egyszerre batch-ben
    - Közös súlyok minden stratégiára
    """
    def __init__(self, input_size, hidden_size, output_size, ...):
        super().__init__()
        # Rétegek definíciója

    def forward(self, x):
        # Forward pass
        return output
```

#### 1.3 Batch Model osztály
```python
class {Model}BatchModel:
    """
    Fő batch modell osztály.
    """
    def __init__(self, all_data, max_horizon, ...):
        self.all_data = all_data
        self.max_horizon = max_horizon
        self.scalers = {}  # KRITIKUS: Strategy-specifikus scalerek

    def _create_training_data(self):
        """
        Összes stratégia adatainak összegyűjtése.
        FONTOS: Itt kell inicializálni a scalereket!
        """
        for strat_id in self.all_data["No."].unique():
            scaler = TradingRobustScaler()
            scaled = scaler.fit_transform(strat_data)
            self.scalers[strat_id] = scaler  # Mentés későbbi használatra

    def _prepare_sequences(self, look_back):
        """
        FONTOS: REUSE scalereket _create_training_data()-ból!
        NE hozz létre új scalereket!
        """
        for strat_id in self.all_data["No."].unique():
            scaler = self.scalers[strat_id]  # Már létező scaler
            scaled = scaler.transform(data)  # NEM fit_transform!

    def _train_model(self, device):
        """Modell tanítás early stopping-gal."""

    def _predict_all_strategies(self, device):
        """
        Batch predikció + post-processing.
        FONTOS: Mean reversion alkalmazása!
        """

    def run(self, use_gpu=False, progress_callback=None, stop_callback=None):
        """Fő belépési pont."""
```

### 2. FÁZIS: Engine.py Integrálás

**Fájl:** `src/analysis/engine.py`

#### 2.1 Dispatch hozzáadása (run() metódusban, ~130-160 sor környékén)
```python
if method_name == "{ModelName}":
    logger.info("Batch Mode enabled for {ModelName} -> Using {MODELNAME} BATCH")
    return self._run_{model_name}_batch(
        params, use_gpu, max_horizon, progress_callback, stop_callback
    )
```

#### 2.2 Legacy név támogatás (~200-230 sor)
```python
if method_name == "{ModelName} Batch":
    logger.info("Using {MODELNAME} BATCH MODE (Legacy Name)")
    return self._run_{model_name}_batch(
        params, use_gpu, max_horizon, progress_callback, stop_callback
    )
```

#### 2.3 Új metódus implementálása (fájl végéhez)
```python
def _run_{model_name}_batch(
    self, params, use_gpu, max_horizon, progress_callback=None, stop_callback=None
):
    """Run {ModelName} in batch mode."""
    from analysis.models.{category}/{model_name}_batch import {Model}BatchModel

    if progress_callback:
        progress_callback(0, 1, "{ModelName} Batch: Initializing...")

    model = {Model}BatchModel(
        all_data=self.data,
        max_horizon=max_horizon,
        # További paraméterek
    )

    results_df, all_strats_data = model.run(
        use_gpu=use_gpu,
        progress_callback=progress_callback,
        stop_callback=stop_callback,
    )

    # Best strategy kiválasztása
    if not results_df.empty:
        best_idx = results_df["Avg Profit"].idxmax()
        best_row = results_df.loc[best_idx]
        best_strat_id = int(best_row["No."])
        best_strat_data = all_strats_data.get(best_strat_id, {})
    else:
        best_strat_id = None
        best_strat_data = {}

    filename_base = f"{ModelName}_Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return results_df, best_strat_id, best_strat_data, filename_base, all_strats_data
```

### 3. FÁZIS: GPU Threshold Beállítás

**Fájl:** `src/analysis/models/dl_utils/utils.py`

```python
GPU_SAMPLE_THRESHOLDS = {
    # ... existing entries ...
    "{ModelName} Batch": 50,  # Batch models always prefer GPU
}
```

### 4. FÁZIS: Tesztelés

#### 4.1 Test Script Készítése
```python
# test_{model_name}_batch.py
import pandas as pd
from analysis.models.{category}.{model_name}_batch import {Model}BatchModel

# Adatok betöltése
df = pd.read_parquet("testdata/AUDJPY_2020_2021_2022_2023_Weekly_Fix.parquet")

# Modell tesztelése
model = {Model}BatchModel(all_data=df, max_horizon=52)
results_df, all_strats = model.run(use_gpu=True)

print(f"Strategies: {len(results_df)}")
print(results_df.head(20))
```

#### 4.2 Ellenőrzési Pontok
- [ ] Modell hiba nélkül fut
- [ ] Scalerek konzisztensek (train vs predict)
- [ ] Predikciók reális tartományban
- [ ] Mean reversion működik
- [ ] GPU/CPU váltás működik
- [ ] Progress callback működik
- [ ] Stop callback működik
- [ ] Eredmények formátuma helyes

#### 4.3 Pylint Futtatás
```bash
pylint src/analysis/models/{category}/{model_name}_batch.py
```

---

## Auto Execution Integráció

Az Auto Execution automatikusan kezeli a Batch Mode-ot, ha a checkbox be van jelölve.

**Fájl:** `src/gui/auto_execution_mixin.py`

A `_start_analysis()` metódus már tartalmazza a szükséges logikát:
```python
use_batch = self.check_batch.isChecked() if hasattr(self, 'check_batch') else False
```

Nincs szükség további módosításra - az engine.py dispatch-eli a megfelelő batch implementációt.

---

## Gomb Működése (GUI)

**Fájl:** `src/gui/tabs/analysis_tab.py`

### Batch Mode Toggle
```python
def toggle_batch_mode(self, state):
    """Toggle batch/global mode for models."""
    if state == Qt.Checked:
        # Deactivate mutually exclusive modes
        if hasattr(self, 'check_panel_mode'):
            self.check_panel_mode.setChecked(False)
        if hasattr(self, 'check_dual_model'):
            self.check_dual_model.setChecked(False)
```

### Mutual Exclusion
- Batch Mode ↔ Panel Mode ↔ Dual Model Mode
- Egyszerre csak egy lehet aktív
- A toggle függvények automatikusan kezelik

---

## GPU Optimalizáció: CUDA Graphs

### Mi a CUDA Graphs?

A CUDA Graphs egy GPU optimalizációs technika, amely **rögzíti (capture)** a GPU műveleteket és **újrajátssza (replay)** őket minimális CPU overhead-del. Az inference loop-ban ez 10-30% gyorsulást eredményez.

### Működési elv

```
Hagyományos inference:
  CPU → GPU kernel launch → GPU compute → CPU → GPU kernel launch → ...
                 ↑ CPU overhead minden lépésnél

CUDA Graphs:
  Capture phase: CPU → GPU kernel launch → GPU compute → ...
  Replay phase: GPU compute → GPU compute → GPU compute → ...
                 ↑ Nincs CPU overhead a replay alatt!
```

### Implementáció Pattern

Minden Batch modell inference loop-jához hozzá kell adni a következő kódot:

```python
# Import a utils-ból
from analysis.models.dl_utils.utils import create_cuda_graph_for_inference

# ... a run metódusban, az inference előtt ...

with torch.inference_mode():
    # GPU OPTIMIZATION: Use CUDA Graphs for inference if on GPU
    cuda_graph = None
    use_cuda_graph = (
        device.type == "cuda"
        and steps >= 10
        and hasattr(torch.cuda, "CUDAGraph")
    )

    if use_cuda_graph:
        try:
            cuda_graph = create_cuda_graph_for_inference(
                model, current_batch, device, warmup=3
            )
            if not cuda_graph.is_enabled:
                cuda_graph = None
                logger.debug("Model: CUDA Graph not enabled, using standard inference")
        except Exception as cuda_err:
            logger.debug("Model: CUDA Graph creation failed: %s", cuda_err)
            cuda_graph = None

    for step in range(steps):
        # Check stop callback
        if stop_callback and stop_callback():
            if cuda_graph is not None:
                cuda_graph.cleanup()
            return {}

        try:
            # Use CUDA Graph if available
            if cuda_graph is not None:
                preds = cuda_graph.replay(current_batch)
            else:
                preds = model(current_batch)

            # ... process predictions ...

        except RuntimeError as e:
            if "CUDA" in str(e):
                if cuda_graph is not None:
                    cuda_graph.cleanup()
                    cuda_graph = None
                continue
            raise

    # Cleanup CUDA Graph
    if cuda_graph is not None:
        cuda_graph.cleanup()
```

### CUDA Graphs Implementációs Státusz

#### Batch Modellek
| Modell | CUDA Graphs | Megjegyzés |
|--------|-------------|------------|
| LSTM Batch | ✅ Kész | Statikus input shape |
| GRU Batch | ✅ Kész | Statikus input shape |
| ES-RNN Batch | ✅ Kész | Statikus input shape |
| MQRNN Batch | ✅ Kész | Statikus input shape |
| Autoformer Batch | ✅ Kész | Statikus input shape |
| MTGNN Batch | ✅ Kész | Statikus input shape |
| TimesNet Batch | ✅ Kész | Statikus input shape |
| DeepAR Batch | ⚠️ Speciális | Hidden state kezelés miatt bonyolultabb |

#### Per-Strategy (Nem-batch) Modellek - CUDA Graphs (2026-01-05)

**dl_graph_specialized (5+1 modell)**
| Modell | CUDA Graphs | Megjegyzés |
|--------|-------------|------------|
| neural_var.py | ✅ Kész | Statikus input shape |
| neural_arima.py | ✅ Kész | Statikus input shape |
| neural_quantile_regression.py | ✅ Kész | Statikus input shape |
| neural_volatility.py | ✅ Kész | Tuple output kezelés |
| rbf.py | ✅ Kész | Statikus input shape |
| neural_gam.py | ⏭️ Kihagyva | Két input tensor - komplex |

**dl_cnn (6 modell)**
| Modell | CUDA Graphs | Megjegyzés |
|--------|-------------|------------|
| dlinear.py | ✅ Kész | Statikus input shape |
| tcn.py | ✅ Kész | Statikus input shape |
| tide.py | ✅ Kész | Statikus input shape |
| nbeats.py | ✅ Kész | 2D input_batch átváltás |
| nhits.py | ✅ Kész | 2D input_batch átváltás |
| timesnet.py | ✅ Kész | inference_mode() támogatás |

**dl_rnn (4 modell)**
| Modell | CUDA Graphs | Megjegyzés |
|--------|-------------|------------|
| lstm.py | ✅ Kész | Statikus input shape |
| gru.py | ✅ Kész | Statikus input shape |
| es_rnn.py | ✅ Kész | return_components kezelés |
| mqrnn.py | ✅ Kész | Statikus input shape |
| seq2seq.py | ⏭️ Nem alkalmas | Egyszeri hívás, nincs loop |
| deepar.py | ⏭️ Nem alkalmas | Hidden state átadás |

**dl_transformer (7 modell)**
| Modell | CUDA Graphs | Megjegyzés |
|--------|-------------|------------|
| autoformer.py | ✅ Kész | Statikus input shape |
| informer.py | ✅ Kész | Statikus input shape |
| fedformer.py | ✅ Kész | Statikus input shape |
| patchtst.py | ✅ Kész | Statikus input shape |
| itransformer.py | ✅ Kész | Multivariate támogatás |
| fits.py | ✅ Kész | Statikus input shape |
| transformer.py | ✅ Kész | Statikus input shape |

**Összesen: 22 per-strategy modell kapott CUDA Graphs támogatást.**

### Mikor NEM használható CUDA Graphs?

1. **Dinamikus input shape** - Ha az input mérete változik futás közben
2. **Control flow az inference-ben** - If/else a modellben
3. **Hidden state átadás** - DeepAR-nál a hidden state lépésről lépésre változik
4. **CPU-GPU sync** - Ha gyakran kell szinkronizálni

### Várható gyorsulás

| Modell | Steps | Gyorsulás |
|--------|-------|-----------|
| Kis modell (LSTM/GRU) | 52 | 15-25% |
| Közepes modell (ES-RNN) | 52 | 20-30% |
| Nagy modell (Autoformer) | 52 | 10-20% |
| MTGNN (sok stratégia) | 52 | 15-25% |

### Checklist Új Batch Modellhez

- [ ] Import hozzáadása: `from analysis.models.dl_utils.utils import create_cuda_graph_for_inference`
- [ ] CUDA Graph inicializálás az inference loop előtt
- [ ] `cuda_graph.replay()` használata ha elérhető
- [ ] Error handling és cleanup minden return előtt
- [ ] Cleanup az inference loop végén

---

## Kritikus Hibák és Megoldások

### 1. Scaler Inkonzisztencia Bug
**Probléma:** `_create_training_data()` és `_prepare_sequences()` különböző scalereket használ.

**Megoldás:**
```python
def _create_training_data(self):
    self.scalers = {}  # Inicializálás
    for strat_id in ...:
        scaler = TradingRobustScaler()
        scaled = scaler.fit_transform(data)
        self.scalers[strat_id] = scaler  # Mentés

def _prepare_sequences(self, look_back):
    for strat_id in ...:
        scaler = self.scalers[strat_id]  # Reuse!
        scaled = scaler.transform(data)  # NEM fit_transform!
```

### 2. Batch Predikció Eltérés
**Probléma:** Globális modell nem tanulja meg a stratégia-specifikus mintákat.

**Megoldás:** Mean Reversion Post-Processing
```python
# Erősebb mean reversion az első néhány lépésben
if j == 0:
    val = val * 0.7 + recent_mean * 0.3
elif j < 4:
    blend = 0.25 + j * 0.05
    val = val * (1 - blend) + recent_mean * blend
```

### 3. Zero-Heavy Distributions
**Probléma:** Trading adatok sok 0-t tartalmaznak (nincs trade).

**Megoldás:** TradingRobustScaler
- Csak non-zero értékekből számol statisztikákat
- Fallback mean/std-re ha nincs elég non-zero adat

---

## Fájl Struktúra Összefoglaló

```
src/
├── analysis/
│   ├── engine.py                    # Dispatch + _run_xxx_batch() metódusok
│   └── models/
│       ├── dl_rnn/
│       │   ├── lstm.py             # Eredeti
│       │   ├── lstm_batch.py       # ✅ Kész
│       │   ├── gru.py
│       │   ├── gru_batch.py        # ✅ Kész
│       │   ├── seq2seq.py
│       │   ├── seq2seq_batch.py    # ✅ Kész
│       │   ├── mqrnn.py
│       │   ├── mqrnn_batch.py      # ✅ Kész
│       │   ├── deepar.py
│       │   ├── deepar_batch.py     # ✅ Kész
│       │   ├── es_rnn.py
│       │   └── es_rnn_batch.py     # ✅ Kész
│       ├── dl_cnn/
│       │   ├── timesnet.py
│       │   ├── timesnet_batch.py   # ✅ Kész
│       │   ├── nbeats.py
│       │   ├── nbeats_batch.py     # ✅ Kész
│       │   ├── nhits.py
│       │   ├── nhits_batch.py      # ✅ Kész
│       │   ├── tcn.py
│       │   ├── tcn_batch.py        # ✅ Kész
│       │   ├── dlinear.py
│       │   ├── dlinear_batch.py    # ✅ Kész
│       │   ├── tide.py
│       │   └── tide_batch.py       # ✅ Kész
│       ├── dl_transformer/
│       │   ├── autoformer.py
│       │   ├── autoformer_batch.py # ✅ Kész
│       │   ├── informer.py
│       │   ├── informer_batch.py   # ✅ Kész
│       │   ├── tft.py
│       │   ├── tft_batch.py        # ✅ Kész
│       │   ├── patchtst.py
│       │   ├── patchtst_batch.py   # ✅ Kész
│       │   ├── itransformer.py
│       │   ├── itransformer_batch.py # ✅ Kész
│       │   ├── fedformer.py
│       │   ├── fedformer_batch.py  # ✅ Kész
│       │   ├── fits.py
│       │   ├── fits_batch.py       # ✅ Kész
│       │   ├── transformer.py
│       │   └── transformer_batch.py # ✅ Kész
│       ├── dl_graph_specialized/
│       │   ├── mtgnn.py
│       │   ├── mtgnn_batch.py      # ✅ Kész
│       │   ├── diffusion.py
│       │   ├── diffusion_batch.py  # ✅ Kész
│       │   ├── kan.py
│       │   ├── kan_batch.py        # ✅ Kész
│       │   ├── neural_arima.py
│       │   ├── neural_arima_batch.py # ✅ Kész
│       │   ├── rbf.py
│       │   ├── rbf_batch.py        # ✅ Kész
│       │   ├── neural_gam.py
│       │   ├── neural_gam_batch.py # ✅ Kész
│       │   ├── neural_ode.py
│       │   ├── neural_ode_batch.py # ✅ Kész
│       │   ├── neural_volatility.py
│       │   ├── neural_volatility_batch.py # ✅ Kész
│       │   ├── snn.py
│       │   ├── snn_batch.py        # ✅ Kész
│       │   ├── stemgnn.py
│       │   └── stemgnn_batch.py    # ✅ Kész
│       └── dl_utils/
│           └── utils.py            # GPU thresholds
└── gui/
    ├── tabs/
    │   └── analysis_tab.py         # Batch Mode toggle
    └── auto_execution_mixin.py     # Auto Exec integráció
```

---

## Javasolt Implementációs Sorrend

1. ~~**LSTM Batch** - Legegyszerűbb, GRU mintájára~~ ✅ Kész
2. ~~**Seq2Seq Batch** - Encoder-Decoder alapú~~ ✅ Kész
3. ~~**N-BEATS Batch** - Népszerű, block-alapú~~ ✅ Kész
4. ~~**Informer Batch** - Fontos Transformer variáns~~ ✅ Kész
5. ~~**TFT Batch** - Gyakran használt~~ ✅ Kész
6. ~~**N-HiTS Batch** - N-BEATS hierarchikus verziója~~ ✅ Kész
7. ~~**PatchTST Batch** - Modern Transformer~~ ✅ Kész
8. ~~**TCN Batch** - Dilated convolutions~~ ✅ Kész
9. ~~**MQ-RNN Batch** - Multi-quantile~~ ✅ Kész
10. ~~**iTransformer Batch** - Inverted attention~~ ✅ Kész
11. ~~**FEDformer Batch** - Fourier alapú~~ ✅ Kész
12. ~~**TiDE Batch** - Dense encoder~~ ✅ Kész
13. ~~**DLinear Batch** - Egyszerű baseline~~ ✅ Kész
14. ~~**FITS Batch** - Frequency interpolation~~ ✅ Kész

---

## Checklist Minden Modellhez

- [ ] `{model}_batch.py` létrehozva a megfelelő mappában
- [ ] TradingRobustScaler implementálva
- [ ] Batch Network osztály implementálva
- [ ] Batch Model osztály implementálva
- [ ] Scaler konzisztencia biztosítva (fit vs transform)
- [ ] Mean reversion post-processing hozzáadva
- [ ] `engine.py` dispatch hozzáadva
- [ ] `engine.py` legacy név támogatás hozzáadva
- [ ] `engine.py` `_run_xxx_batch()` metódus implementálva
- [ ] `dl_utils/utils.py` GPU threshold hozzáadva
- [ ] `parameters.py` MODEL_DEFAULTS és PARAM_OPTIONS bejegyzés hozzáadva
- [ ] `parameter_spaces.py` Optuna space és regisztráció hozzáadva
- [ ] Test script futtatva
- [ ] Pylint hibák javítva
- [ ] GPU/CPU tesztelve
- [ ] Progress/Stop callback tesztelve

---

## Jövőbeli GPU Optimalizációk

### 1. CUDA Graphs Statikus Input Shape-hez

**Státusz:** ✅ Implementálva 22+ modellben

A CUDA Graphs rögzíti a GPU műveleteket és újrajátssza őket minimális CPU overhead-del.
Lásd: "GPU Optimalizáció: CUDA Graphs" szekció fent.

### 2. Non-Blocking Data Transfer

**Státusz:** ✅ Implementálva minden Batch modellben

```python
# Már implementálva a batch modellekben:
batch_x = batch_x.to(device, non_blocking=True)
batch_y = batch_y.to(device, non_blocking=True)

# DataLoader konfiguráció:
pin_memory = device.type == "cuda"  # Enables pinned memory for faster transfers
```

### 3. GPU Sequential Mode Javítása - Párhuzamosítás

**Státusz:** 🔄 Tervezés alatt

**Probléma:** Jelenleg a per-strategy mode szekvenciálisan dolgozza fel a stratégiákat.

**Megoldási terv:**
```python
# 1. Mini-batch grouping a per-strategy modellekhez
def run_strategies_parallel(strategies, batch_size=8):
    """Process strategies in parallel batches on GPU."""
    for i in range(0, len(strategies), batch_size):
        batch = strategies[i:i+batch_size]
        # Process batch in parallel using torch.vmap or DataParallel
        results = process_batch(batch)
```

**Implementálandó:**
- [ ] Strategy grouping by similar data length
- [ ] torch.vmap for vectorized per-strategy inference
- [ ] Memory-efficient parallel training with gradient accumulation

### 4. DataLoader num_workers Optimalizáció

**Státusz:** ✅ Részben implementálva

**Jelenlegi beállítások:**
```python
# GPU esetén:
num_workers = 2 if os.name == "nt" else 4  # Windows vs Linux

# CPU esetén:
num_workers = 0  # Nincs párhuzamos adat betöltés
```

**Javasolt threshold csökkentés:**
- [x] Alacsonyabb threshold a Batch modellekhez (50 sample)
- [ ] Dinamikus num_workers dataset méret alapján:
  ```python
  # Proposed dynamic configuration
  if dataset_len > 10000:
      num_workers = 4 if os.name == "nt" else 8
  elif dataset_len > 1000:
      num_workers = 2 if os.name == "nt" else 4
  else:
      num_workers = 0  # Small datasets don't benefit from workers
  ```

### 5. Párhuzamos Stratégia Feldolgozás

**Státusz:** 🔄 Tervezés alatt

**Megközelítések:**

#### 5.1 Multi-GPU Támogatás
```python
# torch.nn.DataParallel használata több GPU esetén
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

#### 5.2 Strategy-Level Parallelism
```python
# Több stratégia párhuzamos feldolgozása
from concurrent.futures import ThreadPoolExecutor

def process_strategies_parallel(strategies, model, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_strategy, s, model) for s in strategies]
        results = [f.result() for f in futures]
    return results
```

#### 5.3 CUDA Streams
```python
# Párhuzamos GPU műveletek CUDA streams segítségével
streams = [torch.cuda.Stream() for _ in range(num_strategies)]
for i, (strategy, stream) in enumerate(zip(strategies, streams)):
    with torch.cuda.stream(stream):
        result = model(strategy)
torch.cuda.synchronize()  # Wait for all streams
```

---

## Paraméterezési Feladatok

### Batch Model Paraméter Integráció Checklist

Minden Batch modellhez szükséges paraméter frissítések:

- [x] `parameters.py` → `MODEL_DEFAULTS` - Alapértelmezett paraméterek (batch-re optimalizálva)
- [x] `parameters.py` → `PARAM_OPTIONS` - GUI dropdown opciók
- [x] `parameter_spaces.py` → `get_{model}_batch_space()` - Optuna hyperparameter space
- [x] `parameter_spaces.py` → `MODEL_SPACE_MAP` - Model regisztráció

### Batch-Specifikus Paraméter Ajánlások

| Paraméter | Per-Strategy | Batch Mode | Indoklás |
|-----------|--------------|------------|----------|
| `batch_size` | 32-64 | 128-512 | Több adat = nagyobb batch |
| `epochs` | 25-50 | 10-20 | Konvergál gyorsabban |
| `learning_rate` | 0.001 | 0.001-0.005 | Nagyobb batch = nagyobb LR |
| `patience` | 3-5 | 4-8 | Több idő a konvergálásra |
| `num_workers` | 0-2 | 2-4 | Több adat betöltése |

### Implementált Batch Model Paraméterek

| Modell | MODEL_DEFAULTS | PARAM_OPTIONS | parameter_spaces.py |
|--------|----------------|---------------|---------------------|
| LSTM Batch | ✅ | ✅ | ✅ |
| GRU Batch | ✅ | ✅ | ✅ |
| DeepAR Batch | ✅ | ✅ | ✅ |
| ES-RNN Batch | ✅ | ✅ | ✅ |
| MQRNN Batch | ✅ | ✅ | ✅ |
| Seq2Seq Batch | ✅ | ✅ | ✅ |
| DLinear Batch | ✅ | ✅ | ✅ |
| N-BEATS Batch | ✅ | ✅ | ✅ |
| N-HiTS Batch | ✅ | ✅ | ✅ |
| TCN Batch | ✅ | ✅ | ✅ |
| TiDE Batch | ✅ | ✅ | ✅ |
| TimesNet Batch | ✅ | ✅ | ✅ |
| Autoformer Batch | ✅ | ✅ | ✅ |
| FEDFormer Batch | ✅ | ✅ | ✅ |
| FiTS Batch | ✅ | ✅ | ✅ |
| **Informer Batch** | ✅ | ✅ | ✅ |
| **PatchTST Batch** | ✅ | ✅ | ✅ |
| **TFT Batch** | ✅ | ✅ | ✅ |
| **Transformer Batch** | ✅ | ✅ | ✅ |
| **iTransformer Batch** | ✅ | ✅ | ✅ |
| MTGNN Batch | ✅ | ✅ | ✅ |
| Diffusion Batch | ✅ | ✅ | ✅ |
| **KAN Batch** | ✅ | ✅ | ✅ |
| **Neural ARIMA Batch** | ✅ | ✅ | ✅ |
| **Neural Basis Functions Batch** | ✅ | ✅ | ✅ |
| **Neural GAM Batch** | ✅ | ✅ | ✅ |
| **Neural ODE Batch** | ✅ | ✅ | ✅ |
| **Neural VAR Batch** | ✅ | ✅ | ✅ |
| **Neural Volatility Batch** | ✅ | ✅ | ✅ |
| **Spiking Neural Networks Batch** | ✅ | ✅ | ✅ |
| **StemGNN Batch** | ✅ | ✅ | ✅ |

---

*Készült: 2026-01-03*
*Frissítve: 2026-01-04 - Seq2Seq Batch, DLinear Batch, N-BEATS Batch, N-HiTS Batch, TCN Batch implementálva*
*Frissítve: 2026-01-05 - CUDA Graphs implementálva 22 per-strategy modellben*
*Frissítve: 2026-01-05 - TiDE Batch, FEDFormer Batch, FiTS Batch, Informer Batch, PatchTST Batch, **TFT Batch** implementálva*
*Frissítve: 2026-01-05 - GPU optimalizációs tervek, paraméterezési feladatok dokumentálva*
*Frissítve: 2026-01-06 - **Transformer Batch**, **iTransformer Batch** implementálva*
*Frissítve: 2026-01-07 - **Diffusion Batch**, **KAN Batch**, **Neural ARIMA Batch** implementálva*
*Frissítve: 2026-01-07 - **Neural ODE Batch** implementálva*
*Frissítve: 2026-01-07 - **Neural Basis Functions Batch**, **Neural GAM Batch** implementálva*
*Frissítve: 2026-01-07 - **Neural VAR Batch** implementálva*
*Frissítve: 2026-01-07 - **Neural Volatility Batch** implementálva*
*Frissítve: 2026-01-08 - **Spiking Neural Networks Batch** implementálva*
*Frissítve: 2026-01-08 - **StemGNN Batch** implementálva*
*Verzió: 4.2.17*
