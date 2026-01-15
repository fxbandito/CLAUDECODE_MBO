# 09. Performance Test Tab - Teljesítmény Monitoring

## 09.00. Performance Test Tab Fő Funkciója

A **Performance Test Tab** a rendszer erőforrás-használatának valós idejű monitorozására szolgál. A tab lehetővé teszi:

1. **CPU használat monitorozása** - App, Model és Other (rendszer) bontásban
2. **GPU terhelés követése** - Load és VRAM használat
3. **RAM felhasználás figyelése** - App és Model memória
4. **Historikus grafikon** - 60 másodperces visszatekintés
5. **Teljesítmény naplózás** - Részletes log fájlok generálása
6. **Rendszer kompatibilitás teszt** - PC-Test funkció

### Felhasználói Felület Szerkezete

```
+------------------------------------------------------------------+
| Top Bar                                                           |
| [Performance Monitor] [Debug Log] [Perf Log] | [Auto Log] [Save] |
|                                              | [Reset] [PC-Test]  |
+------------------------------------------------------------------+
| CPU Card (50%)      | GPU Card (25%)      | RAM Card (25%)       |
| [App] [Models] [Other] | [Load] [VRAM]    | [App] [Models]       |
| [Core Heatmap]         |                  | Total: X MB          |
+------------------------------------------------------------------+
|                    History (60s) Chart                           |
| [Legend]  [======== Realtime Line Chart ========]                |
+------------------------------------------------------------------+
```

### Erőforrás-Optimalizálás

A Performance Tab **csak akkor aktív**, amikor látható:

```python
# src/gui/tabs/performance_tab.py:361-376
def _on_tab_change_perf(self):
    """Handle tab change - start/stop monitoring."""
    current_tab = self.tabview.get()

    if current_tab == "Performance Test":
        self._start_perf_monitoring()  # Indítás
    else:
        self._stop_perf_monitoring()   # Leállítás (erőforrás megtakarítás)
        # Fontos: A logging NEM áll le tab váltáskor!
```

---

## 09.01. Performance Monitor Kijelzők

A Performance Monitor három fő kártyából áll, mindegyik speciális mérőműszerekkel.

### CPU Card (50% szélesség)

A CPU kártya három körkörös mérőt és egy hőtérképet tartalmaz:

#### 1. App CPU Gauge

```
+----------+
|   App    |
|   45%    |  <- Zöld/Sárga/Piros szín a terhelés alapján
|  [====]  |
+----------+
```

**Mit mér**: A főalkalmazás (GUI) CPU használata
- Normalizálva 0-100% skálára
- Színek: Zöld (<50%), Sárga (50-70%), Piros (>70%)

```python
# Számítás
app_cpu_raw = self.process.cpu_percent(interval=None)
# Normalizálás: nyers érték / magok száma
app_cpu = min(100.0, app_cpu_raw / num_cores)
```

#### 2. Models CPU Gauge

```
+----------+
|  Models  |
|   78%    |  <- Kék/Sárga/Piros szín
|  [====]  |
+----------+
```

**Mit mér**: A gyermek folyamatok (worker-ek) CPU használata
- Dual Mode: Valódi child process-ek
- Panel Mode: Threaded model detection (speciális logika)

```python
# Dual Mode - Child process-ek
for pid, child in self.child_procs.items():
    model_cpu_raw += child.cpu_percent(interval=None)
model_cpu = min(100.0, model_cpu_raw / num_cores)

# Panel Mode - Thread detection
is_threaded_model_work = (
    self._model_running and
    worker_count == 0 and
    main_thread_count > 4 and
    (app_cpu_raw > 50 or active_cores > 4)
)
```

#### 3. Other CPU Gauge

```
+----------+
|  Other   |
|   15%    |  <- Szürke árnyalatok
|  [====]  |
+----------+
```

**Mit mér**: Rendszer CPU (összes - app - model)

```python
other_cpu = max(0, cpu_total - app_cpu - model_cpu)
```

#### 4. Core Heatmap

```
[|||||||||||||||||||||||||||||||||||||||||||||]
 0  2  4  6  8  10 12 14 16 18 20 22 24 26 28 30
 Core használat %-ban (sötétebb = magasabb)
```

**Mit mér**: Minden logikai mag egyedi terhelése
- Színskála: Sötétkék (0%) → Piros (100%)
- A `psutil.cpu_percent(percpu=True)` alapján

### GPU Card (25% szélesség)

#### 1. Load Gauge

```
+----------+
|   Load   |
|   65%    |  <- Lila/Sárga/Piros szín
|  [====]  |
+----------+
```

**Mit mér**: GPU számítási terhelés %
- GPUtil library (ha elérhető)
- Fallback: PyTorch CUDA

```python
if GPUTIL_AVAILABLE:
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu_load = gpus[0].load * 100
```

#### 2. VRAM Gauge

```
+----------+
|   VRAM   |
| 4096 MB  |  <- Lila/Sárga/Piros szín
|  [====]  |
+----------+
```

**Mit mér**: GPU memória használat MB-ban
- Dinamikus maximum: A tényleges VRAM méret alapján
- Thresholds: 60% (sárga), 85% (piros)

```python
gpu_mem_used = gpus[0].memoryUsed  # MB
gpu_mem_total = gpus[0].memoryTotal  # MB

# Fallback PyTorch-ra
if gpu_mem_total == 0 and torch.cuda.is_available():
    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    gpu_mem_used = torch.cuda.memory_reserved(0) / (1024**2)
```

### RAM Card (25% szélesség)

#### 1. App RAM Gauge

```
+----------+
|   App    |
| 1024 MB  |  <- Zöld/Sárga/Piros szín
|  [====]  |
+----------+
```

**Mit mér**: Főalkalmazás memória használata

```python
app_mem = self.process.memory_info().rss / (1024**2)  # MB
```

#### 2. Models RAM Gauge

```
+----------+
|  Models  |
| 8192 MB  |  <- Kék/Sárga/Piros szín
|  [====]  |
+----------+
```

**Mit mér**: Worker folyamatok összesített memóriája

```python
for pid, child in self.child_procs.items():
    model_mem += child.memory_info().rss / (1024**2)
```

#### 3. Total Label

```
Total: 9.21 GB
```

**Számítás**: `total_mem = app_mem + model_mem`

### History Chart (100% szélesség)

```
100% ┤
 80% ┤    ╭──╮
 60% ┤   ╭╯  ╰╮    ╭──╮
 40% ┤  ╭╯    ╰────╯  ╰────
 20% ┤──╯
  0% ┼────────────────────────────────────────
     0s                                    60s

[●] App CPU  [●] Model CPU  [●] GPU
```

**Jellemzők**:
- 60 másodperces visszatekintés
- 3 adatsor: App CPU, Model CPU, GPU Load
- 500 ms frissítési intervallum
- Interaktív legend

```python
self.history_chart = RealtimeLineChart(
    max_points=60,
    y_max=100,
    series_config={
        "app_cpu": "#00D4AA",
        "model_cpu": "#FF6B6B",
        "gpu": "#9B59B6"
    }
)
```

---

## 09.02. Debug Log Gomb

A **Debug Log** gomb megnyitja a debug napló viewer ablakot.

### Működés

```python
# src/gui/tabs/performance_tab.py:1351-1359
def _open_debug_log_viewer(self):
    """Open the debug log viewer popup."""
    log_file = self._get_latest_log_file("debug")
    if not log_file:
        self._show_no_log_message("Debug")
        return

    self._open_log_viewer_popup(log_file, "Debug Log Viewer")
```

### Debug Log Fájl Keresés

A rendszer a legfrissebb "debug" szót tartalmazó `.log` fájlt keresi:

```python
def _get_latest_log_file(self, pattern: str):
    log_dir = self._get_log_directory()  # {project_root}/Log

    matching_files = []
    for filename in os.listdir(log_dir):
        if pattern.lower() in filename.lower() and filename.endswith(".log"):
            filepath = os.path.join(log_dir, filename)
            matching_files.append((filepath, os.path.getmtime(filepath)))

    # Legfrissebb fájl
    matching_files.sort(key=lambda x: x[1], reverse=True)
    return matching_files[0][0]
```

### Debug Log Aktiválása

A debug logging csak `main_debug.py`-ból indítva aktív:

```python
# main_debug.py
os.environ["MBO_DEBUG_MODE"] = "1"

# Részletes logging beállítás
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler(f"Log/debug_{timestamp}.log"),
        logging.StreamHandler()
    ]
)
```

### Log Viewer Popup Ablak

A megnyíló ablak jellemzői:

| Funkció | Leírás |
|---------|--------|
| **Auto-refresh** | 1 másodperces automatikus frissítés |
| **Pause on click** | Kattintásra megáll a frissítés |
| **Pause on scroll** | Görgetésre megáll a frissítés |
| **Resume** | Kattintás vagy Resume gomb a folytatáshoz |
| **Scroll to Bottom** | Ugrás a log végére |
| **Monospace font** | Consolas 11pt az olvashatóságért |

```
+------------------------------------------------------------------+
| Debug Log Viewer                                            [X]  |
+------------------------------------------------------------------+
| File: debug_20240115_143052.log | Click to pause | Refreshing... |
+------------------------------------------------------------------+
| 2024-01-15 14:30:52.123 | DEBUG    | analysis.engine | Starting...|
| 2024-01-15 14:30:52.456 | INFO     | data.processor | Loaded 500  |
| 2024-01-15 14:30:52.789 | DEBUG    | analysis.engine | Worker #1...|
| ...                                                              |
+------------------------------------------------------------------+
| Lines: 1,234 | Size: 45.2 KB | [Scroll to Bottom] [Resume] [Close]|
+------------------------------------------------------------------+
```

---

## 09.03. Perf Log Gomb

A **Perf Log** gomb megnyitja a teljesítmény napló viewer ablakot.

### Működés

```python
# src/gui/tabs/performance_tab.py:1361-1369
def _open_perf_log_viewer(self):
    """Open the performance log viewer popup."""
    log_file = self._get_latest_log_file("perf")
    if not log_file:
        self._show_no_log_message("Performance")
        return

    self._open_log_viewer_popup(log_file, "Performance Log Viewer")
```

### Performance Log Struktúra

A performance log részletes fejléccel és adatsorokkal rendelkezik:

```
================================================================================
MBO TRADING STRATEGY ANALYZER - PERFORMANCE REPORT
================================================================================

Recording Started: 2024-01-15 14:30:52.123
App Version: 3.20.8

----------------------------------------
MODEL CONFIGURATION
----------------------------------------
Model: LightGBM
Currency Pair: EURUSD
Execution Mode: Panel
CPU Threads (n_jobs): 16
GPU Enabled: False

Model Parameters:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1

----------------------------------------
SYSTEM INFORMATION
----------------------------------------
OS: Windows 11 10.0.22631
Processor: Intel64 Family 6 Model 170
CPU Cores (Physical): 24
CPU Cores (Logical): 32
RAM Total: 64.00 GB
CUDA Available: Yes
GPU Name: NVIDIA GeForce RTX 4060
GPU Memory: 8.00 GB

----------------------------------------
PC-TEST BENCHMARK RESULTS
----------------------------------------
CPU Benchmark:
  Test: Matrix Multiplication (1000x1000)
  Score: 4523
  Duration: 2.2103 s
GPU Benchmark:
  Test: Tensor Multiplication (4000x4000)
  Score: 18234
  Duration: 1.0969 s

----------------------------------------
COMPATIBILITY CHECK
----------------------------------------
[OK] RAM: 64.0 GB (Optimal)
[OK] CPU Cores: 24C/32T (Optimal)
[OK] CUDA GPU: NVIDIA GeForce RTX 4060 (8.00 GB)
[OK] Disk Space: 234.5 GB free
[OK] Python: 3.12.0

----------------------------------------
PERFORMANCE DATA
----------------------------------------
Columns: Timestamp(ms) | Elapsed(s) | App_CPU% | Model_CPU% | Other_CPU% | ...
------------------------------------------------------------------------
     100 |    0.10 |     5.2 |      45.3 |       8.1 | ...
     200 |    0.20 |     4.8 |      52.1 |       7.3 | ...
     300 |    0.30 |     5.1 |      78.5 |       6.2 | ...
     ...

----------------------------------------
RECORDING SUMMARY
----------------------------------------
Recording Ended: 2024-01-15 14:35:22.456
Total Duration: 270.33 seconds
Total Samples: 2703
Sample Rate: 10.0 samples/sec
================================================================================
```

### Adatsorok Oszlopai

| Oszlop | Leírás |
|--------|--------|
| **Timestamp(ms)** | Eltelt idő milliszekundumban |
| **Elapsed(s)** | Eltelt idő másodpercben |
| **App_CPU%** | Alkalmazás CPU használat |
| **Model_CPU%** | Modell/Worker CPU használat |
| **Other_CPU%** | Rendszer többi CPU használat |
| **GPU_Load%** | GPU terhelés |
| **GPU_Mem(MB)** | GPU memória használat |
| **App_RAM(MB)** | Alkalmazás RAM |
| **Model_RAM(MB)** | Modell RAM |
| **Workers** | Aktív worker folyamatok száma |
| **Per-Core CPU%** | Minden mag egyedi terhelése |

---

## 09.04. Auto Log Checkbox

Az **Auto Log** checkbox engedélyezi az automatikus teljesítmény naplózást minden modell futásnál.

### Működés

```python
# src/gui/tabs/performance_tab.py:145-159
self.auto_perf_log_enabled = ctk.BooleanVar(value=False)
self.chk_auto_perf_log = ctk.CTkCheckBox(
    top_bar,
    text=self.tr("Auto Log"),
    variable=self.auto_perf_log_enabled,
    command=self._on_auto_log_toggle,
)
```

### Auto Log Toggle Logika

```python
# src/gui/tabs/performance_tab.py:1263-1283
def _on_auto_log_toggle(self):
    """Handle Auto Log checkbox toggle."""
    if self.auto_perf_log_enabled.get():
        # Csak status frissítés - NEM indít logolást!
        # A logolás az elemzés indításakor kezdődik
        self.lbl_perf_log_status.configure(
            text="Auto-logging enabled",
            text_color="#00D4AA"
        )
    else:
        # Ha ki van kapcsolva és aktív a logolás, leállítja
        if self.perf_logging_active:
            self._stop_performance_logging()
```

### Automatikus Indítás/Leállítás

Az Analysis Tab hívja ezeket a függvényeket:

```python
# Elemzés indulásakor (analysis_tab.py)
if self.is_auto_log_enabled():
    self.start_auto_performance_logging()

# Elemzés befejeződésekor
if self.perf_logging_active and self.is_auto_log_enabled():
    self.stop_auto_performance_logging()
```

### Fontos Viselkedések

| Esemény | Auto Log viselkedés |
|---------|---------------------|
| **Checkbox bekapcsolás** | Csak engedélyezi, NEM indít logolást |
| **Elemzés indítás** | HA engedélyezve: új log fájl indul |
| **Tab váltás** | A logolás FOLYTATÓDIK |
| **Elemzés befejezés** | Log fájl lezárása, összegzés írása |
| **Checkbox kikapcsolás** | Ha aktív logolás van, leállítja |

---

## 09.05. Save Report / Stop Recording Gomb

A **Save Report** gomb manuális teljesítmény naplózást indít/állít meg.

### Állapotok

| Állapot | Gomb szöveg | Szín |
|---------|-------------|------|
| **Inaktív** | "Save Report" | Zöld (#2E7D32) |
| **Aktív** | "Stop Recording" | Piros (#C62828) |

### Működés

```python
# src/gui/tabs/performance_tab.py:854-859
def _toggle_performance_logging(self):
    """Toggle performance logging on/off."""
    if self.perf_logging_active:
        self._stop_performance_logging()
    else:
        self._start_performance_logging()
```

### Logolás Indítása

```python
# src/gui/tabs/performance_tab.py:979-1033
def _start_performance_logging(self):
    """Start logging performance data to file."""
    # 1. Futtatási információk begyűjtése
    run_info = self._get_current_run_info()

    # 2. Log mappa létrehozása
    log_dir = os.path.join(project_root, "Log")
    os.makedirs(log_dir, exist_ok=True)

    # 3. Fájlnév generálás
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{model_name}_{currency}_perf.log"

    # 4. Fejléc írása (PC-Test eredményekkel)
    self._write_log_header(run_info)

    # 5. Háttérszál indítása
    self.perf_log_thread = threading.Thread(
        target=self._performance_logging_loop,
        daemon=True
    )
    self.perf_log_thread.start()
```

### Logolás Ciklus

```python
# src/gui/tabs/performance_tab.py:1152-1198
def _performance_logging_loop(self):
    """Background thread for high-frequency performance logging."""
    interval_sec = PERF_LOG_INTERVAL_MS / 1000.0  # 100ms = 10 minta/sec

    while not self.perf_log_stop_event.is_set():
        load = self.performance_monitor.get_extended_load()

        # Adatsor formázása
        line = (
            f"{timestamp_ms:>8} | "
            f"{elapsed:>7.2f} | "
            f"{load['app_cpu']:>7.1f} | "
            f"{load['model_cpu']:>9.1f} | "
            f"[{per_core_str}]"
        )

        self.perf_log_file.write(line + "\n")

        # 10 mintánként flush
        if self.perf_log_samples % 10 == 0:
            self.perf_log_file.flush()

        time.sleep(interval_sec)
```

### Logolás Leállítása

```python
# src/gui/tabs/performance_tab.py:1209-1261
def _stop_performance_logging(self):
    """Stop performance logging and finalize the log file."""
    self.perf_logging_active = False
    self.perf_log_stop_event.set()

    # Összegzés írása
    summary = [
        "RECORDING SUMMARY",
        f"Recording Ended: {datetime.now()}",
        f"Total Duration: {elapsed:.2f} seconds",
        f"Total Samples: {self.perf_log_samples}",
        f"Sample Rate: {self.perf_log_samples / elapsed:.1f} samples/sec"
    ]

    self.perf_log_file.write("\n".join(summary))
    self.perf_log_file.close()
```

---

## 09.06. PC-Test Gomb

A **PC-Test** gomb egy részletes rendszer-információs és kompatibilitás ellenőrző ablakot nyit meg.

### Működés

```python
# src/gui/tabs/performance_tab.py:506-687
def _show_system_info_popup(self):
    """Show system information in a popup window with compatibility check."""
    full_info = self.performance_monitor.get_system_info()
    gpu_info = self.performance_monitor.get_gpu_info()

    popup = ctk.CTkToplevel(self)
    popup.title("System Information")
    popup.geometry("550x650")
```

### Popup Ablak Felépítése

```
+------------------------------------------------------------------+
| System Information                                          [X]  |
+------------------------------------------------------------------+
| COMPATIBILITY CHECK                                              |
| [✓] RAM: 64.0 GB (Optimal)                                       |
| [✓] CPU Cores: 24C/32T (Optimal)                                 |
| [✓] CUDA GPU: RTX 4060 (8.00 GB)                                 |
| [✓] Disk Space: 234.5 GB free                                    |
| [✓] Python: 3.12.0                                               |
+------------------------------------------------------------------+
| CPU / SYSTEM                                                     |
| OS: Windows 11 10.0.22631                                        |
| Processor: Intel Core i9-14900HX                                 |
| CPU Cores (Physical): 24                                         |
| CPU Cores (Logical): 32                                          |
| RAM Total: 64.00 GB                                              |
+------------------------------------------------------------------+
| GPU                                                              |
| CUDA Available: Yes                                              |
| GPU Name: NVIDIA GeForce RTX 4060                                |
| GPU Memory: 8.00 GB                                              |
| CUDA Version: 12.1                                               |
+------------------------------------------------------------------+
| PYTHON / LIBRARIES                                               |
| Python: 3.12.0                                                   |
| PyTorch: 2.1.0                                                   |
| CUDA (PyTorch): 12.1                                             |
| NumPy: 1.26.0                                                    |
| Pandas: 2.1.0                                                    |
| CustomTkinter: 5.2.0                                             |
| psutil: 5.9.5                                                    |
+------------------------------------------------------------------+
|                         [Close]                                  |
+------------------------------------------------------------------+
```

### Kompatibilitás Ellenőrzés

```python
# src/gui/tabs/performance_tab.py:689-765
def _check_system_requirements(self, sys_info, gpu_info):
    """Check if system meets requirements for the application."""
    requirements = []

    # 1. RAM ellenőrzés
    ram_total = psutil.virtual_memory().total / (1024**3)
    if ram_total >= 16:
        requirements.append(("RAM", "ok", f"{ram_total:.1f} GB (Optimal)"))
    elif ram_total >= 8:
        requirements.append(("RAM", "warning", f"{ram_total:.1f} GB (Minimum)"))
    else:
        requirements.append(("RAM", "error", f"{ram_total:.1f} GB (Too low)"))

    # 2. CPU magok ellenőrzése
    # 3. CUDA/GPU ellenőrzés
    # 4. Szabad lemezterület
    # 5. Python verzió
```

### Követelmények Táblázat

| Követelmény | Optimális | Minimum | Kritikus |
|-------------|-----------|---------|----------|
| **RAM** | 16+ GB | 8 GB | < 8 GB |
| **CPU Cores** | 6+ physical | 4 physical | < 4 |
| **CUDA GPU** | Elérhető | - | Nem elérhető (csak warning) |
| **Disk Space** | 20+ GB | 5 GB | < 5 GB |
| **Python** | 3.10+ | 3.8+ | < 3.8 |

### CPU Benchmark

```python
# src/analysis/performance.py:141-158
def run_cpu_benchmark(self):
    """Runs a CPU benchmark (matrix multiplication)."""
    start_time = time.time()

    # Mátrix szorzás benchmark
    size = 1000
    mat_a = np.random.rand(size, size)
    mat_b = np.random.rand(size, size)
    _ = np.dot(mat_a, mat_b)

    duration = time.time() - start_time
    score = int(10000 / duration)  # Magasabb = gyorsabb

    return {
        "Score": score,
        "Duration": f"{duration:.4f} s",
        "Test": f"Matrix Multiplication ({size}x{size})"
    }
```

### GPU Benchmark

```python
# src/analysis/performance.py:160-185
def run_gpu_benchmark(self):
    """Runs a GPU benchmark (tensor operation)."""
    if not torch.cuda.is_available():
        return {"Error": "CUDA not available"}

    start_time = time.time()

    # Tensor szorzás benchmark
    size = 4000
    device = torch.device("cuda")
    mat_a = torch.rand(size, size, device=device)
    mat_b = torch.rand(size, size, device=device)
    _ = torch.matmul(mat_a, mat_b)
    torch.cuda.synchronize()  # GPU befejezés bevárása

    duration = time.time() - start_time
    score = int(20000 / duration)

    return {
        "Score": score,
        "Duration": f"{duration:.4f} s",
        "Test": f"Tensor Multiplication ({size}x{size})"
    }
```

### Státusz Ikonok

| Ikon | Szín | Jelentés |
|------|------|----------|
| ✓ | Zöld (#00D4AA) | OK - Követelmény teljesítve |
| ⚠ | Sárga (#FFB800) | Warning - Minimum teljesítve |
| ✗ | Piros (#FF4757) | Error - Követelmény nem teljesítve |

---

## Összefoglaló: Performance Test Tab Munkafolyamat

```
                    [Performance Test Tab]
                           │
        ┌──────────────────┴──────────────────┐
        │                                      │
   [Monitoring]                           [Logging]
        │                                      │
        ▼                                      ▼
┌───────────────────┐              ┌───────────────────┐
│  get_extended_    │              │  Auto Log / Save  │
│     load()        │              │     Report        │
│                   │              │                   │
│ - App CPU/RAM     │              │ - Header + PC-Test│
│ - Model CPU/RAM   │              │ - 10 samples/sec  │
│ - GPU Load/VRAM   │              │ - Per-core data   │
│ - Per-core %      │              │ - Summary         │
└─────────┬─────────┘              └─────────┬─────────┘
          │                                  │
          ▼                                  ▼
┌───────────────────┐              ┌───────────────────┐
│   UI Update       │              │   .log file       │
│   (500ms)         │              │   (in Log/)       │
│                   │              │                   │
│ - Circular gauges │              │ - Timestamp       │
│ - Core heatmap    │              │ - All metrics     │
│ - History chart   │              │ - Worker count    │
└───────────────────┘              └───────────────────┘
```

### Frissítési Intervallumok

| Komponens | Intervallum | Megjegyzés |
|-----------|-------------|------------|
| **UI Monitor** | 500 ms | Csak ha a tab aktív |
| **Perf Logging** | 100 ms | 10 minta/másodperc |
| **Log Viewer** | 1000 ms | Auto-refresh popup-ban |

### Tippek a Használathoz

1. **Erőforrás takarékosság**: A tab csak akkor monitoroz, ha látható
2. **Auto Log**: Engedélyezze elemzések előtt a teljes futásidő rögzítéséhez
3. **PC-Test**: Futtassa az alkalmazás első indításakor a kompatibilitás ellenőrzéséhez
4. **Debug Log**: Csak `main_debug.py`-ból indítva elérhető
5. **Panel Mode detection**: A rendszer automatikusan detektálja a threaded workload-ot

---

*Dokumentáció generálva: 2024*
*Forrásfájlok: `src/gui/tabs/performance_tab.py`, `src/analysis/performance.py`*
