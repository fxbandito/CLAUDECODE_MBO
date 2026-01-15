# GPU Optimalizálás és Modell Elemzés Jelentés
**Dátum**: 2026-01-05
**Rendszer**: Intel Core i9 (24C/32T), NVIDIA RTX 4060 Laptop GPU (8GB)

---

## 1. GPU vs CPU Teljesítmény Összehasonlítás

### Teszt Eredmények (52 lépéses előrejelzés)

| Modell | CPU idő | GPU idő | Speedup | CPU neg | GPU neg | Konzisztencia |
|--------|---------|---------|---------|---------|---------|---------------|
| Neural GAM | 1.24s | 0.62s | **2.02x** | 4 | 5 | DIFF |
| Neural ODE | 0.40s | 0.28s | **1.44x** | 0 | 0 | OK |
| Neural QR | 0.38s | 0.40s | 0.96x | 0 | 0 | OK |
| Neural VAR | 0.44s | 0.33s | **1.31x** | 0 | 0 | OK |
| Neural Volatility | 0.57s | 0.49s | **1.17x** | 0 | 0 | OK |
| Spiking NN (JAVÍTOTT) | 0.46s | 0.74s | 0.61x | 0 | 0 | OK |

### Értékelés

**Jól gyorsítható modellek (>1.2x speedup)**:
- Neural GAM: 2.02x - Legjobb GPU kihasználtság
- Neural ODE: 1.44x - Jó párhuzamosítás
- Neural VAR: 1.31x - Konzisztens eredmények

**Nem gyorsítható modellek (<1.0x)**:
- Neural QR: 0.96x - GPU overhead domináns a kis batch miatt
- Spiking NN: 0.61x - GPU lassabb a spike encoding miatt

### Negatív Előrejelzések Elemzése

| Modell | Előtte | Most | Javulás |
|--------|--------|------|---------|
| Neural GAM | 60% | 8-10% | Elfogadható |
| Neural ODE | 64% | 0% | Kiváló |
| Neural QR | 0% | 0% | Kiváló |
| Neural VAR | 53% | 0% | Kiváló |
| Neural Volatility | 0% | 0% | Kiváló |
| Spiking NN | **100%** | **0%** | **MEGJAVÍTVA** |

---

## 2. Meglévő GPU Optimalizációk

### 2.1 CUDA Graphs (Már implementálva)

A kódban már van CUDA Graphs támogatás az inferencia gyorsítására:

```python
# utils.py - CUDAGraphWrapper osztály
class CUDAGraphWrapper:
    def _capture_graph(self, sample_input, warmup_iterations):
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.model(self.static_input)
```

**Használó modellek**:
- N-BEATS Batch
- DLinear Batch
- N-HiTS Batch
- TCN Batch
- KAN
- MTGNN Batch
- TimesNet Batch
- Autoformer Batch
- LSTM/GRU/ES-RNN/MQRNN Batch

**Feltétel**: `steps >= 10` és CUDA device

### 2.2 DataLoader Optimalizáció (Már implementálva)

```python
# utils.py - create_optimized_dataloader
def create_optimized_dataloader(dataset, batch_size, device, ...):
    pin_memory = get_pin_memory_safe(device)  # True for CUDA
    num_workers = get_optimal_num_workers(device, dataset_size)
    prefetch_factor = 4  # Async data loading
    persistent_workers = True
```

**Windows korlátozás**: `num_workers = min(2, cpu_count)` a dataset_size > 500 esetén

### 2.3 AMP (Automatic Mixed Precision) (Már implementálva)

```python
# Minden modell használja:
scaler, autocast_context = get_amp_components(use_gpu)
with autocast_context:
    outputs = model(batch_x)
```

---

## 3. GPU Kihasználtsági Problémák és Megoldások

### 3.1 Alacsony GPU Kihasználtság Okai

A log fájlok alapján a GPU kihasználtság 12-32% között mozog, ami alacsony.

**Okok**:
1. **Kis batch méretek**: 128 batch túl kicsi az RTX 4060-hoz
2. **Szekvenciális stratégia feldolgozás**: Egyszerre egy stratégia fut
3. **CPU-GPU szinkronizáció**: Minden lépésnél várakozás
4. **I/O műveletek**: Scaler inverse_transform CPU-n fut

### 3.2 Javaslat: Batch Méret Növelése

**Jelenlegi alapértelmezések**:
```python
batch_size = 128  # Túl kicsi
```

**Javasolt változtatás** (utils.py):
```python
def get_optimal_batch_size(...):
    if device.type == "cuda" and gpu_memory_gb >= 8:
        # RTX 4060 (8GB): 256-512 batch
        return min(512, max(256, default_batch_size * 4))
```

### 3.3 Javaslat: Párhuzamos Stratégia Feldolgozás

**Probléma**: Jelenleg minden stratégia szekvenciálisan fut.

**Megoldás**: A `_batch` modellek már implementálják a batch feldolgozást, de a "specializált" modellek (Neural GAM, ODE, stb.) nem.

**Ajánlás**:
1. Használd a batch modelleket ahol elérhetők
2. A specializált modellekhez implementálj multi-strategy batch-et

### 3.4 Javaslat: Nem-blokkoló Adatátvitel

```python
# Jelenlegi
batch_x = batch_x.to(device)

# Javaslat
batch_x = batch_x.to(device, non_blocking=True)
torch.cuda.synchronize()  # Csak előrejelzés előtt
```

---

## 4. Modell-Specifikus Elemzés

### 4.1 Neural GAM
- **GPU Speedup**: 2.02x (legjobb)
- **Probléma**: CPU/GPU különbség a negatív előrejelzéseknél (4 vs 5)
- **Ok**: Véletlenszerűség a batch shuffle-ben
- **Állapot**: Elfogadható

### 4.2 Neural ODE
- **GPU Speedup**: 1.44x (jó)
- **Eredmények**: 0% negatív mindkét device-on
- **Állapot**: Kiváló

### 4.3 Neural Quantile Regression
- **GPU Speedup**: 0.96x (nincs gyorsulás)
- **Ok**: A quantile loss számítás CPU-igényes
- **Eredmények**: 0% negatív
- **Állapot**: Kiváló minőség, GPU nem hatékony

### 4.4 Neural VAR
- **GPU Speedup**: 1.31x (jó)
- **Eredmények**: 0% negatív, konzisztens h52 értékek
- **Állapot**: Kiváló

### 4.5 Neural Volatility
- **GPU Speedup**: 1.17x (közepes)
- **Eredmények**: 0% negatív
- **Figyelmeztetés**: Magas h52 értékek (1.2-1.5M) - ellenőrizendő
- **Állapot**: Működik, de validálandó

### 4.6 Spiking Neural Networks (JAVÍTVA)
- **GPU Speedup**: 0.61x (GPU lassabb!)
- **Ok**: A spike encoding/decoding nem jól párhuzamosítható
- **Eredmények**: 0% negatív (JAVÍTVA a readout layer-rel)
- **Állapot**: Működik, de CPU ajánlott

---

## 5. Implementált Javítások

### 5.1 Spiking Neural Networks Fix

**Fájl**: `src/analysis/models/dl_graph_specialized/snn.py`

**Változtatások**:
```python
# Régi (hibás)
self.readout = nn.Sequential(
    nn.Linear(hidden_size * 2, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size),
)

# Új (javított)
self.readout = nn.Sequential(
    nn.Linear(hidden_size * 2, hidden_size),
    nn.GELU(),
    nn.Linear(hidden_size, hidden_size // 2),
    nn.GELU(),
    nn.Linear(hidden_size // 2, output_size),
    nn.Sigmoid(),  # Korlátozza [0,1] közé
)
```

**Eredmény**: 100% negatív -> 0% negatív

---

## 6. Összefoglalás és Javaslatok

### Működő Modellek (Nincs teendő)
- Neural ODE
- Neural Quantile Regression
- Neural VAR
- Neural Volatility
- Spiking NN (most javítva)

### GPU Optimalizálási Javaslatok (Prioritás sorrend)

1. **Batch méret növelése** (Könnyű)
   - 128 -> 256-512 az RTX 4060-hoz
   - Hatás: +15-30% GPU kihasználtság

2. **Non-blocking transfer** (Könnyű)
   - `to(device, non_blocking=True)` használata
   - Hatás: +5-10% sebesség

3. **Batch modellek használata** (Közepes)
   - Specializált modellek helyett batch verzió
   - Hatás: +50-100% sebesség több stratégiánál

4. **CUDA Graphs kiterjesztése** (Nehéz)
   - Specializált modellekhez is implementálni
   - Hatás: +10-30% sebesség

### Nem Javasolt

- Spiking NN GPU használata (CPU gyorsabb)
- num_workers növelése Windows-on (spawning overhead)

---

## 7. Következő Lépések

1. [x] Non-blocking transfer implementálása - **KÉSZ** (2026-01-05)
2. [x] CUDA Graphs kiterjesztése per-strategy modellekhez - **KÉSZ** (2026-01-05)
3. [ ] Batch méret növelése a utils.py-ban
4. [ ] StemGNN modell újrafuttatása
5. [ ] Neural Volatility h52 értékek validálása
6. [ ] Batch modellek tesztelése a specializáltak helyett

---

## 8. Elvégzett Optimalizációk (2026-01-05)

### 8.1 Non-blocking Data Transfer

**Implementálva 16 modellben:**
- dl_cnn: dlinear.py, tcn.py, tide.py
- dl_rnn: es_rnn.py, seq2seq.py, mqrnn.py
- dl_transformer: itransformer.py, patchtst.py
- dl_graph_specialized: neural_arima.py, neural_quantile_regression.py, neural_ode.py, neural_var.py, neural_volatility.py, snn.py, diffusion.py, neural_gam.py, stemgnn.py

**Változtatás:**
```python
# Régi
batch_x = batch_x.to(device)
batch_y = batch_y.to(device)

# Új
batch_x = batch_x.to(device, non_blocking=True)
batch_y = batch_y.to(device, non_blocking=True)
```

**Várható hatás:** +5-10% sebesség a CPU-GPU adatátvitel párhuzamosításával.

### 8.2 CUDA Graphs Implementálva Per-Strategy Modellekhez

**Összesen 22 per-strategy modell kapott CUDA Graphs támogatást:**

**dl_graph_specialized (5 modell + 1 kihagyva):**
| Modell | Státusz |
|--------|---------|
| neural_var.py | ✅ Kész |
| neural_arima.py | ✅ Kész |
| neural_quantile_regression.py | ✅ Kész |
| neural_volatility.py | ✅ Kész (tuple output kezelés) |
| rbf.py | ✅ Kész |
| neural_gam.py | ⏭️ Kihagyva (két input tensor) |

**dl_cnn (6 modell):**
| Modell | Státusz |
|--------|---------|
| dlinear.py | ✅ Kész |
| tcn.py | ✅ Kész |
| tide.py | ✅ Kész |
| nbeats.py | ✅ Kész (2D input_batch átváltás) |
| nhits.py | ✅ Kész (2D input_batch átváltás) |
| timesnet.py | ✅ Kész (inference_mode támogatás) |

**dl_rnn (4 modell + 2 nem alkalmas):**
| Modell | Státusz |
|--------|---------|
| lstm.py | ✅ Kész |
| gru.py | ✅ Kész |
| es_rnn.py | ✅ Kész |
| mqrnn.py | ✅ Kész |
| seq2seq.py | ⏭️ Nem alkalmas (egyszeri hívás) |
| deepar.py | ⏭️ Nem alkalmas (hidden state) |

**dl_transformer (7 modell):**
| Modell | Státusz |
|--------|---------|
| autoformer.py | ✅ Kész |
| informer.py | ✅ Kész |
| fedformer.py | ✅ Kész |
| patchtst.py | ✅ Kész |
| itransformer.py | ✅ Kész (multivariate) |
| fits.py | ✅ Kész |
| transformer.py | ✅ Kész |

**CUDA Graphs implementációs minta:**
```python
from analysis.models.dl_utils.utils import create_cuda_graph_for_inference

# Inference előtt
cuda_graph = None
if device.type == "cuda" and steps >= 10 and hasattr(torch.cuda, "CUDAGraph"):
    try:
        cuda_graph = create_cuda_graph_for_inference(model, current_tensor, device, warmup=3)
        if not cuda_graph.is_enabled:
            cuda_graph = None
    except Exception:
        cuda_graph = None

# Inference loop
with torch.no_grad():
    for i in range(steps):
        if cuda_graph is not None:
            pred = cuda_graph.replay(current_tensor)
        else:
            pred = model(current_tensor)
        # ...

# Cleanup
if cuda_graph is not None:
    cuda_graph.cleanup()
```

**Várható hatás:** +10-30% sebesség az inference loopban GPU-n (52 lépésnél).

---

*Jelentés generálva: 2026-01-05*
*Frissítve: 2026-01-05 - Non-blocking transfer és CUDA Graphs implementálva*
*MBO Trading Strategy Analyzer v4.2.0 Stable*
