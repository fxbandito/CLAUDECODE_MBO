# MBO Trading Strategy Analyzer - Analysis Tab

## Tartalomjegyzék

- [04.00 Analysis Tab fő funkciója](#0400-analysis-tab-fő-funkciója)
- [04.01 Progress bár működése](#0401-progress-bár-működése)
- [04.02 Category és Model választó](#0402-category-és-model-választó)
- [04.03 Auto gomb](#0403-auto-gomb)
- [04.04 Shutdown checkbox](#0404-shutdown-checkbox)
- [04.05 Stop, Pause és Run Analysis gombok](#0405-stop-pause-és-run-analysis-gombok)
- [04.06 CPU Power csúszka](#0406-cpu-power-csúszka)
- [04.07 Use GPU kapcsoló](#0407-use-gpu-kapcsoló)
- [04.08 Panel Mode checkbox](#0408-panel-mode-checkbox)
- [04.09 Dual Mode checkbox](#0409-dual-mode-checkbox)
- [04.10 Panel és Dual Mode összehasonlítása](#0410-panel-és-dual-mode-összehasonlítása)
- [04.11 Forecast Horizon csúszka](#0411-forecast-horizon-csúszka)
- [04.12 Time kiírás](#0412-time-kiírás)
- [04.13 Paraméterek szekció](#0413-paraméterek-szekció)

---

# 04.00 Analysis Tab fő funkciója

## Áttekintés

Az **Analysis Tab** (Elemzés fül) a program szíve, ahol a kereskedési stratégiák előrejelző elemzését konfigurálod és indítod el. Itt választod ki a modellt, állítod be a paramétereket, és figyeled az elemzés folyamatát.

## Fő feladatok

1. **Modell kiválasztás** - 14 kategóriából 100+ modell közül választhatsz
2. **Paraméterezés** - Minden modellhez egyedi beállítások
3. **Végrehajtási mód** - Independent / Panel / Dual Model
4. **Erőforrás-kezelés** - CPU magok és GPU használat
5. **Futtatás vezérlése** - Indítás, megállítás, szüneteltetés
6. **Automatizálás** - Auto Execution több modell egymás utáni futtatásához

## Tab felépítése

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ROW 0: Vezérlő sáv                                                             │
│  ┌──────┐ ┌─────────────────────┐ ┌──────────────────┐ ┌──┐                     │
│  │ Auto │ │ Category: [▼ ____] │ │ Model: [▼ ____]  │ │? │                     │
│  └──────┘ └─────────────────────┘ └──────────────────┘ └──┘                     │
│                                                                                 │
│                        ┌──────────┐ ┌───────┐ ┌──────┐ ┌──────────────────────┐ │
│                        │ Shutdown │ │ Stop  │ │Pause │ │    Run Analysis      │ │
│                        │    [ ]   │ │       │ │      │ │        ▶            │ │
│                        └──────────┘ └───────┘ └──────┘ └──────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ROW 1: Teljesítmény vezérlők                                                   │
│  CPU Power: [═══════○═══] 75% (18 cores)   [✓] Use GPU                         │
│  [ ] Panel Mode [?]   [ ] Dual Model [?]                                        │
│  Forecast Horizon: [═══○═════] 52                         Time: 00:00:00       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ROW 2: Paraméterek                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ P: [1]  D: [1]  Q: [1]  Seasonal: [False ▼]  Period: [12]  ...          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ROW 3: Modell Súgó Panel                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  ## ARIMA Model                                                         │   │
│  │  AutoRegressive Integrated Moving Average for time series...            │   │
│  │  **Parameters:** p=AR order, d=differencing, q=MA order                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Működési folyamat

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ANALYSIS WORKFLOW                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Adat betöltés (Data Tab)                                            │
│           │                                                             │
│           ▼                                                             │
│  2. ┌─────────────┐                                                     │
│     │ Kategória   │ → 14 kategória (Statistical, ML, DL, stb.)         │
│     │ kiválasztás │                                                     │
│     └──────┬──────┘                                                     │
│            │                                                            │
│            ▼                                                            │
│  3. ┌─────────────┐                                                     │
│     │ Modell      │ → 100+ modell a kategóriából                       │
│     │ kiválasztás │                                                     │
│     └──────┬──────┘                                                     │
│            │                                                            │
│            ▼                                                            │
│  4. ┌─────────────┐                                                     │
│     │ Paraméterek │ → Modell-specifikus beállítások                    │
│     │ beállítása  │                                                     │
│     └──────┬──────┘                                                     │
│            │                                                            │
│            ▼                                                            │
│  5. ┌─────────────┐                                                     │
│     │ Módok       │ → Independent / Panel / Dual                       │
│     │ kiválasztás │                                                     │
│     └──────┬──────┘                                                     │
│            │                                                            │
│            ▼                                                            │
│  6. ┌─────────────┐                                                     │
│     │ Run         │ → Háttérszálon indítás                             │
│     │ Analysis    │ → Progress callback                                 │
│     └──────┬──────┘                                                     │
│            │                                                            │
│            ▼                                                            │
│  7. ┌─────────────┐                                                     │
│     │ Eredmények  │ → Results Tab megjelenítés                         │
│     │ megjelenít. │ → Riport generálás                                 │
│     └─────────────┘                                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# 04.01 Progress bár működése

## Áttekintés

A **Progress bár** (haladásjelző) egy globális elem a fejléc alatt, amely vizuálisan mutatja az elemzés előrehaladását.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  [████████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░]  65%          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Komponensek

| Elem | Típus | Funkció |
|------|-------|---------|
| **progress_bar** | CTkProgressBar | Grafikus sáv (0.0 - 1.0) |
| **lbl_progress_pct** | CTkLabel | Százalékos kijelzés ("65%") |

## Működési logika

### Progress callback mechanizmus

```python
def progress_callback(val):
    """Update progress with throttling."""
    current_time = time.time()
    # Throttle: csak 0.1 másodpercenként frissít (10 FPS)
    if val >= 1.0 or (current_time - last_time > 0.1):
        self.after(0, lambda: self.progress_bar.set(val))
        self.after(0, lambda: self.lbl_progress_pct.configure(text=f"{int(val * 100)}%"))
```

### Progress értékek az elemzés során

| Fázis | Progress érték | Leírás |
|-------|----------------|--------|
| **Inicializálás** | 0% | Elemzés indítása |
| **Adat előkészítés** | 5-10% | Feature engineering, split |
| **Modell tanítás** | 10-80% | Stratégiánkénti/batch feldolgozás |
| **Előrejelzés** | 80-95% | Forecast generálás |
| **Eredmények** | 95-100% | DataFrame összeállítás |

### Throttling (Sebesség korlátozás)

A UI blokkolásának elkerülése érdekében:
- Maximum **10 FPS** (100ms közönként)
- **100%**-nál mindig frissít (befejezés)
- `self.after(0, ...)` - főszálra ütemezés

## Matematikai modell

**Stratégiánkénti feldolgozásnál**:

```
progress(i) = 0.1 + 0.7 × (i / N)

ahol:
  i = aktuális stratégia index
  N = összes stratégia száma
```

**Panel/Dual módnál** (target oszloponként):

```
progress(t) = 0.1 + 0.6 × (t / T)

ahol:
  t = aktuális target oszlop index
  T = összes target oszlop (5: 1W, 1M, 3M, 6M, 12M)
```

---

# 04.02 Category és Model választó

## Category ComboBox

```
┌─────────────────────────────────────┐
│ Category: [Statistical Models    ▼] │
└─────────────────────────────────────┘
```

### Elérhető kategóriák (14 db)

| # | Kategória neve | Modellek száma |
|---|----------------|----------------|
| 1 | Statistical Models | 13 |
| 2 | Smoothing & Decomposition | 5 |
| 3 | Classical Machine Learning | 6 |
| 4 | Deep Learning - RNN-based | 6 |
| 5 | Deep Learning - CNN & Hybrid | 7 |
| 6 | Deep Learning - Transformer-based | 9 |
| 7 | Deep Learning - Graph & Specialized | 12 |
| 8 | Meta-Learning & AutoML | 7 |
| 9 | Bayesian & Probabilistic | 5 |
| 10 | Frequency Domain & Signal Processing | 7 |
| 11 | Distance & Similarity-based | 4 |
| 12 | State Space & Filtering | 2 |
| 13 | Topological Methods | 1 |
| 14 | Ensemble Methods | 1 |

### Működés

```python
def update_model_list(self, category):
    """Update model dropdown based on selected category."""
    models = self.model_categories.get(category, [])
    self.model_combo.configure(values=models)
    if models:
        self.model_combo.set(models[0])  # Első modell automatikusan
        self.update_parameters(models[0])  # Paraméterek frissítése
```

## Model ComboBox

```
┌───────────────────────────────┐
│ Model: [ARIMA               ▼] │
└───────────────────────────────┘
```

### Dinamikus frissítés

Kategória váltáskor:
1. Model lista frissül a kategória modelljeire
2. Első modell automatikusan kiválasztódik
3. Paraméterek UI újraépül
4. Súgó panel frissül
5. Panel/Dual Mode checkbox állapot frissül
6. GPU switch állapot frissül

## Help gomb (?)

```
┌────┐
│ ?  │ ← Kör alakú, 28x28 px
└────┘
```

Kattintásra **popup ablak** jelenik meg a kiválasztott modell részletes dokumentációjával:
- Modell leírása
- Matematikai háttér
- Paraméterek magyarázata
- Használati javaslatok

---

# 04.03 Auto gomb

## Megjelenés

```
┌──────────┐
│   Auto   │  ← Lila szín (#8e44ad)
└──────────┘
```

## Funkció (rövid ismertető)

Az **Auto** gomb megnyitja az **Auto Execution Window**-t, amely lehetővé teszi több modell egymás utáni automatikus futtatását.

### Fő jellemzők

| Jellemző | Leírás |
|----------|--------|
| **Queue kezelés** | Modellek sorba állítása |
| **Batch futtatás** | Összes modell automatikus futtatása |
| **Output mappa** | Riportok célmappája |
| **Feature mód** | Original / Forward / Rolling |
| **Riport típusok** | Standard, Stability, Risk |
| **Shutdown opció** | Leállítás minden modell után |

### Működés

```python
def open_auto_window(self):
    """Open/show the auto execution window."""
    if self.auto_window is not None and self.auto_window.winfo_exists():
        # Már létezik - előtérbe hozás
        self.auto_window.lift()
        self.auto_window.focus_force()
    else:
        # Új ablak létrehozása
        self.auto_window = AutoExecutionWindow(self)
        self.auto_window.grab_set()  # Modal ablak
```

**Részletes leírás** egy későbbi fejezetben található (Auto Execution Tab dokumentáció).

---

# 04.04 Shutdown checkbox

## Megjelenés

```
┌──────────────────┐
│  [ ] Shutdown    │  ← Piros szöveg (#e74c3c)
└──────────────────┘
```

## Funkció

A **Shutdown** checkbox bejelölése esetén a számítógép **automatikusan leáll** az elemzés befejezése után.

## Két működési mód

### 1. Analysis Tab Shutdown (Azonnali mód)

| Tulajdonság | Érték |
|-------------|-------|
| **Helye** | Analysis Tab, jobb oldal |
| **Hatás** | Az aktuális modell után leáll |
| **Auto Exec alatt** | Megszakítja a várakozási sort |
| **Színe** | Piros (#e74c3c) |

**Működés Auto Exec alatt**:
```python
# Ha az Analysis Tab checkbox BE van pipálva Auto Exec közben:
if self.var_shutdown_after_run.get():
    if not self.auto_shutdown_after_all:  # Nem "minden után" mód
        # AZONNALI leállás - nem fut a következő modell
        self.auto_shutdown_immediate = True
        self.log("Shutdown checkbox enabled - stopping after current model.")
        self._finish_auto_sequence()
```

### 2. Auto Exec Tab Shutdown ("Minden után" mód)

| Tulajdonság | Érték |
|-------------|-------|
| **Helye** | Auto Execution Window |
| **Hatás** | MINDEN modell lefutása után áll le |
| **Analysis checkbox** | Letiltva (szürke) |
| **Színe** | Szürke (letiltva) |

**Logika**:
```python
def run_auto_sequence(self, execution_list, shutdown_after_all=False):
    if shutdown_after_all:
        # Auto Exec módú shutdown
        self.var_shutdown_after_run.set(False)  # Analysis checkbox kikapcs
        self.chk_shutdown.configure(state="disabled", text_color="gray")
        self.log("Shutdown after ALL models - Analysis tab checkbox locked.")
```

## Összehasonlítás

| Szempont | Analysis Tab | Auto Exec Tab |
|----------|--------------|---------------|
| **Mikor áll le** | Aktuális modell után | Minden modell után |
| **Megszakítja a sort** | Igen | Nem |
| **Checkbox állapot** | Aktív | Az Analysis Tab-ot letiltja |
| **Visszavonható** | Igen (ki-pipálás) | Igen (ki-pipálás) |

## Biztonsági funkció

A shutdown **újra ellenőrzi** a checkbox állapotát közvetlenül leállás előtt:

```python
def _finish_auto_sequence(self):
    # RE-CHECK: User unchecked during execution?
    if self.var_shutdown_after_run.get():  # Még mindig be?
        shutdown_handler.trigger_shutdown_sequence()
    else:
        self.log("Shutdown cancelled - checkbox was unchecked.")
```

**Ez lehetővé teszi**, hogy az elemzés közben meggondold magad és leállítsd a shutdown-t.

---

# 04.05 Stop, Pause és Run Analysis gombok

## Run Analysis gomb

```
┌────────────────────────┐
│     Run Analysis       │  ← Zöld (#008000), 180px széles
│          ▶            │
└────────────────────────┘
```

### Funkció
Elindítja az elemzést a kiválasztott modellel és paraméterekkel.

### Előfeltételek
1. Adat be van töltve (`processed_data` nem None)
2. Modell ki van választva
3. Nincs futó elemzés

### Működés

```python
def run_analysis_thread(self):
    """Start analysis in a new thread."""
    get_sound_manager().play_button_click()
    self.start_analysis()

def start_analysis(self):
    # Validáció
    if self.processed_data is None:
        messagebox.showwarning("Warning", "Please load data first!")
        return

    # UI frissítés
    self.progress_bar.set(0)
    self.btn_stop.configure(state="normal")
    self.btn_pause.configure(state="normal")
    self.btn_run.configure(state="disabled")

    # Timer indítás
    self.start_timer()

    # Háttérszál indítás
    threading.Thread(
        target=self.run_analysis_logic,
        args=(method, params, n_jobs, use_gpu, max_horizon),
        daemon=True
    ).start()
```

## Pause gomb

```
┌────────────┐                    ┌────────────┐
│   Pause    │  ←→ Váltakozik →   │   Resume   │
│            │                    │            │
└────────────┘                    └────────────┘
  Sárga (#f39c12)                   Zöld (#27ae60)
```

### Funkció
Szünetelteti/folytatja a futó elemzést.

### Működés

```python
def toggle_pause(self):
    if self.pause_event.is_set():
        # Futó → Szüneteltetés
        self.pause_event.clear()
        self.btn_pause.configure(text="Resume", fg_color="#27ae60")
        self.timer_running = False  # Timer megállítás
        self.log("Analysis paused. Click Resume to continue.")
    else:
        # Szünetel → Folytatás
        self.pause_event.set()
        self.btn_pause.configure(text="Pause", fg_color="#f39c12")
        self.timer_running = True  # Timer folytatás
        self.log("Analysis resumed.")
```

### Worker szüneteltetés (Dual Mode)

```python
# Worker folyamatok suspend/resume (psutil-lal)
for worker in pool._pool:
    if worker.is_alive():
        proc = psutil.Process(worker.pid)
        proc.suspend()  # Szüneteltetés

# Resume
for pid in suspended_pids:
    proc = psutil.Process(pid)
    proc.resume()  # Folytatás
```

## Stop gomb

```
┌────────────┐
│    Stop    │  ← Piros (#e74c3c), 140px széles
└────────────┘
```

### Funkció
Megszakítja a futó elemzést.

### Működés

```python
def stop_analysis(self):
    get_sound_manager().play_button_click()
    if hasattr(self, "stop_check"):
        self.stop_check.set()  # Jelző flag beállítása

        # Ha szünetel, először folytatás (deadlock elkerülés)
        if not self.pause_event.is_set():
            self.pause_event.set()

        self.log("Stopping analysis... please wait.")
        self.btn_stop.configure(state="disabled")
```

### Részleges eredmények

A Stop nem dobja el az eddigi eredményeket:
- Már kiszámított stratégiák megmaradnak
- "Partial results" megjelenik a Results Tab-on
- Riport generálható az eddigiekből

## Gombok állapot diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    BUTTON STATE DIAGRAM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [IDLE STATE]                                                    │
│  ┌────────┐  ┌────────┐  ┌────────────────────┐                 │
│  │ Stop   │  │ Pause  │  │   Run Analysis     │                 │
│  │DISABLED│  │DISABLED│  │     ENABLED        │                 │
│  └────────┘  └────────┘  └─────────┬──────────┘                 │
│                                    │ Click                       │
│                                    ▼                             │
│  [RUNNING STATE]                                                 │
│  ┌────────┐  ┌────────┐  ┌────────────────────┐                 │
│  │ Stop   │  │ Pause  │  │   Run Analysis     │                 │
│  │ENABLED │  │ENABLED │  │     DISABLED       │                 │
│  └───┬────┘  └───┬────┘  └────────────────────┘                 │
│      │           │                                               │
│      │           ▼                                               │
│      │     [PAUSED STATE]                                        │
│      │     ┌────────┐  ┌────────┐                               │
│      │     │ Stop   │  │ Resume │                               │
│      │     │ENABLED │  │ENABLED │                               │
│      │     └────────┘  └────────┘                               │
│      │                                                           │
│      ▼                                                           │
│  [STOPPING/FINISHED]                                             │
│  → Visszatér IDLE STATE-be                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

# 04.06 CPU Power csúszka

## Megjelenés

```
┌─────────────────────────────────────────────────────────────────┐
│ CPU Power: [══════════════○══════] 75% (18 cores)               │
└─────────────────────────────────────────────────────────────────┘
```

## Specifikáció

| Tulajdonság | Érték |
|-------------|-------|
| **Minimum** | 10% |
| **Maximum** | 100% |
| **Lépésköz** | 5% (18 lépés) |
| **Alapértelmezett** | 75% |
| **Szélesség** | 150px |

## Működés

### CPU Manager (Központi singleton)

```python
# Slider érték változásakor
def update_cpu_slider(value):
    percentage = int(value)
    # Központi CPU manager frissítése
    cpu_manager.set_percentage(percentage)
    # Label frissítése
    self.lbl_cpu_val.configure(text=f"{percentage}% ({cpu_manager.get_n_jobs()} cores)")
```

### Mag számítás

```python
def get_n_jobs(self):
    """Calculate n_jobs based on percentage."""
    n_jobs = max(1, int(self._logical_cores * self.percentage / 100))
    return n_jobs

# Példa: 24 logikai mag, 75%
# n_jobs = max(1, int(24 * 75 / 100)) = max(1, 18) = 18 cores
```

## Auto Exec alatti különbség

### Normál futtatás
- A csúszka **mindig aktív**
- Valós időben változtatható
- Azonnal érvényesül a következő batch-re

### Auto Execution
- A csúszka **változtatható**, de:
- Csak a **következő modell** indulásakor érvényesül
- Az aktuálisan futó modellt nem befolyásolja
- CPU Manager callback értesíti a GUI-t

```python
# Auto Exec közben a CPU manager callback:
def on_cpu_change(new_percentage):
    self.after(0, lambda: self.lbl_cpu_val.configure(
        text=f"{new_percentage}% ({cpu_manager.get_n_jobs()} cores)"
    ))
    self.after(0, lambda: self.slider_cpu.set(new_percentage))
```

## Matematikai háttér

**Worker pool méret számítás**:

```
n_jobs = floor(logical_cores × percentage / 100)
n_jobs = max(1, n_jobs)  # Minimum 1 worker
```

**Példa számítások** (i9-14900HX, 32 logikai mag):

| Percentage | n_jobs |
|------------|--------|
| 10% | 3 |
| 25% | 8 |
| 50% | 16 |
| 75% | 24 |
| 100% | 32 |

---

# 04.07 Use GPU kapcsoló

## Megjelenés

```
┌───────────────────┐
│  [✓] Use GPU      │  ← Switch típus
└───────────────────┘
```

## Funkció

A **Use GPU** kapcsoló engedélyezi a grafikus kártya (NVIDIA CUDA) használatát a támogatott modelleknél.

## GPU támogatott modellek (47 db)

A következő modellek használhatják a GPU-t:

| Kategória | Modellek |
|-----------|----------|
| **Classical ML** | XGBoost, LightGBM, Random Forest (Hummingbird) |
| **RNN-based** | LSTM, GRU, DeepAR, ES-RNN, MQRNN, Seq2Seq |
| **CNN & Hybrid** | TCN, N-BEATS, N-HiTS, DLinear, TiDE, TimesNet |
| **Transformer** | TFT, Informer, Autoformer, FEDformer, PatchTST, iTransformer |
| **Graph & Specialized** | MTGNN, StemGNN, KAN, Neural ODE, Diffusion, SNN |
| **Meta-Learning** | NAS, DARTS, Meta-learning, MoE, MTL, GFM |
| **Ensemble** | Time Series Ensemble |

## Működési logika

### Automatikus állapot kezelés

```python
def update_parameters(self, method):
    # GPU támogatás ellenőrzése
    if method in GPU_SUPPORTED_MODELS:
        self.switch_gpu.configure(state="normal")
    else:
        self.switch_gpu.deselect()  # Kikapcsolás
        self.switch_gpu.configure(state="disabled")  # Letiltás
```

### Elemzés indításkor

```python
use_gpu = self.switch_gpu.get() == 1

# GPU inicializálás a modellben
if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda")
    model = model.to(device)
    data = data.to(device)
```

## Előfeltételek

| Követelmény | Leírás |
|-------------|--------|
| **NVIDIA GPU** | CUDA-kompatibilis kártya |
| **CUDA Toolkit** | 11.8+ vagy 12.x |
| **PyTorch** | CUDA-val fordított verzió |
| **Memória** | Min. 4GB VRAM ajánlott |

---

# 04.08 Panel Mode checkbox

## Megjelenés

```
┌─────────────────────────┐
│  [ ] Panel Mode   [?]   │  ← Checkbox + Help gomb
└─────────────────────────┘
```

## Funkció

A **Panel Mode** egyetlen modellt tanít az **összes stratégiára egyszerre**, ahelyett, hogy stratégiánként külön-külön tanítana.

## Támogatott modellek

A Panel Mode támogatás **dinamikusan** a modell registry-ből generálódik.
Minden modell saját maga dönti el a támogatást a `MODEL_INFO.supports_panel_mode` flag-gel.

**Hogyan támogat egy modell Panel Mode-ot:**
1. Beállítja: `supports_panel_mode=True` a MODEL_INFO-ban
2. Implementálja: `create_dual_regressor()` metódust

**Aktuálisan támogatott modellek lekérdezése:**
```python
from analysis import get_panel_mode_models
print(get_panel_mode_models())  # Dinamikus lista
```

**Tipikus támogatott modellek:** Random Forest, XGBoost, LightGBM, Gradient Boosting, KNN Regressor

## Matematikai háttér

### Hagyományos (Independent) mód

```
Minden S stratégiára külön:
  Model_s = train(X_s, Y_s)
  Prediction_s = Model_s.predict(X_s_last)

Összesen: S darab modell tanítás
```

### Panel mód

```
Egyetlen globális modell:
  X_panel = concat(X_1, X_2, ..., X_S)  # Összes stratégia adata
  Y_panel = concat(Y_1, Y_2, ..., Y_S)

  Model_global = train(X_panel, Y_panel)

  Prediction_s = Model_global.predict(X_s_last)  # Stratégiánként

Összesen: 1 darab modell tanítás
```

## Logikai működés

### Adat előkészítés

```python
from analysis import prepare_panel_data

# Panel adat előkészítés lag feature-ökkel
x_data, y_data, feature_cols = prepare_panel_data(data)

# A függvény automatikusan:
# - Lag feature-öket hoz létre (1, 2, 4, 8, 13, 26 hét)
# - Rolling statisztikákat számol (mean, std)
# - Target oszlopokat generál (1w, 4w, 13w, 26w, 52w)
```

### Time-aware split (Adatszivárgás elkerülése)

```python
from analysis import time_aware_split

# Time-aware split - elkerüli az adatszivárgást
x_train, y_train, x_test, y_test = time_aware_split(
    x_data, y_data,
    train_ratio=0.8  # Konfigurálható arány
)

# A függvény stratégiánként végzi a split-et:
# - Minden stratégia 80%-a tanításhoz (időrendi sorrendben)
# - Minden stratégia 20%-a teszteléshez
```

### Előrejelzés

```python
# Az utolsó ismert hét adatai stratégiánként
last_indices = data.groupby("No.").tail(1).index
X_predict = data.loc[last_indices]

# Egyetlen predict hívás
predictions = model.predict(X_predict)
```

## Sebesség összehasonlítás

| Mód | 6144 stratégia | Relatív sebesség |
|-----|----------------|------------------|
| Independent | ~30-60 perc | 1x |
| Panel | ~3-6 perc | **5-10x gyorsabb** |

## Mikor használd?

| Használati eset | Ajánlás |
|-----------------|---------|
| Gyors prototípus | ✓ Panel |
| Sok stratégia | ✓ Panel |
| Maximális pontosság | ✗ Independent |
| Egyedi stratégia jellemzők | ✗ Independent |

---

# 04.09 Dual Mode checkbox

## Megjelenés

```
┌─────────────────────────┐
│  [ ] Dual Model   [?]   │  ← Checkbox + Help gomb
└─────────────────────────┘
```

## Funkció

A **Dual Model** mód **két külön modellt** tanít:
1. **Activity Model** - Kereskedési aktivitás előrejelzése (klasszifikáció)
2. **Profit Model** - Profit előrejelzése aktív heteknél (regresszió)

## Támogatott modellek

A Dual Mode támogatás **dinamikusan** a modell registry-ből generálódik.
Minden modell saját maga dönti el a támogatást a `MODEL_INFO.supports_dual_mode` flag-gel.

**Hogyan támogat egy modell Dual Mode-ot:**
1. Beállítja: `supports_dual_mode=True` a MODEL_INFO-ban
2. Implementálja: `create_dual_regressor()` metódust

**Aktuálisan támogatott modellek lekérdezése:**
```python
from analysis import get_dual_mode_models
print(get_dual_mode_models())  # Dinamikus lista
```

**Tipikus támogatott modellek:** Random Forest, XGBoost, LightGBM, Gradient Boosting, KNN Regressor

## Matematikai háttér

### Modell definíciók

**Activity Model (Klasszifikáció)**:
```
Activity(t+h) = P(Trades > 0 | X_t)

ahol:
  h = horizont (1, 4, 13, 26, 52 hét)
  X_t = feature-ök t időpontban
  Kimenet: [0, 1] valószínűség
```

**Profit Model (Regresszió)**:
```
Profit_per_active(t+h) = E[Profit | Trades > 0, X_t]

ahol:
  Csak aktív heteken tanítva
  Kimenet: várható profit/aktív hét
```

### Kombinált előrejelzés képlete

```
Final_Forecast(h) = Activity(h) × weeks(h) × Profit_per_active(h)

ahol:
  weeks(h) = horizont hossza hetekben
  Activity(h) = aktív hetek aránya valószínűsége
  Profit_per_active(h) = átlagos profit aktív hetenként

Példa (1 hónap, h=4):
  Activity = 0.75 (75% esély aktivitásra)
  weeks = 4
  Profit_per_active = $50/hét

  Expected_active_weeks = 0.75 × 4 = 3 hét
  Final_Forecast = 3 × $50 = $150
```

### Horizon mapping

```python
horizon_mapping = {
    1:  ("target_activity_1w",  "target_profit_1w",  "Forecast_1W",  1),
    4:  ("target_activity_4w",  "target_profit_4w",  "Forecast_1M",  4),
    13: ("target_activity_13w", "target_profit_13w", "Forecast_3M",  13),
    26: ("target_activity_26w", "target_profit_26w", "Forecast_6M",  26),
    52: ("target_activity_52w", "target_profit_52w", "Forecast_12M", 52),
}
```

## Logikai működés

### Task generálás

```python
tasks = []

# Activity Tasks (5 horizont × klasszifikáció)
for col in y_activity_train.columns:  # 5 horizont
    task = (method, x_train, y_activity_train[col],
            x_predict, params, use_gpu, "activity")
    tasks.append(task)

# Profit Tasks (5 horizont × regresszió)
for col in y_profit_train.columns:  # 5 horizont
    task = (method, x_train, y_profit_train[col],
            x_predict, params, use_gpu, "regression")
    tasks.append(task)

# Összesen: 10 task
```

### Párhuzamos végrehajtás

```python
from analysis.process_utils import init_worker_environment

# Worker pool létrehozás KÖZPONTI inicializálóval
pool = multiprocessing.Pool(
    processes=optimal_workers,
    initializer=init_worker_environment  # GPU letiltás, thread limitek
)

# Aszinkron task küldés
async_results = []
for task_args in tasks:
    ar = pool.apply_async(train_dual_model_task, args=task_args)
    async_results.append(ar)

# Eredmények gyűjtése progress-szel
while completed < total_tasks:
    for ar in async_results:
        if ar.ready():
            result = ar.get()
            # Activity vagy Profit eredmény szétválogatás
```

### Worker inicializálás (process_utils.py)

A `init_worker_environment()` függvény biztosítja a stabil párhuzamos futtatást:
- **GPU letiltás**: `CUDA_VISIBLE_DEVICES=""` - elkerüli a CUDA context konfliktusokat
- **FAISS generic mód**: `FAISS_OPT_LEVEL="generic"` - AVX2/CUDA ütközések megelőzése
- **Thread limitek**: 1 thread/worker - megakadályozza a "halálspirált"
- **PyTorch konfig**: `torch.set_num_threads(1)` - explicit thread kontroll

## Eredmény struktúra

Minden stratégiához:
```python
result = {
    "No.": strategy_id,
    "Method": "XGBoost (Dual)",
    "Forecast_1W": combined_1w,
    "Forecast_1M": combined_1m,
    "Forecast_3M": combined_3m,
    "Forecast_6M": combined_6m,
    "Forecast_12M": combined_12m,

    # Extra mezők elemzéshez
    "Activity_4w": 0.75,
    "Profit_per_active_4w": 50.0,
    "Expected_active_weeks_4w": 3.0,
}
```

---

# 04.10 Panel és Dual Mode összehasonlítása

## Összehasonlító táblázat

| Szempont | Independent | Panel Mode | Dual Mode |
|----------|-------------|------------|-----------|
| **Modellek száma** | N (stratégiánként) | 1 | 2 |
| **Tanítási idő** | Hosszú | Rövid (5-10x) | Közepes |
| **Memória használat** | Alacsony/stratégia | Magas (összes adat) | Közepes |
| **Pontosság** | Magas | Közepes | Magas |
| **Aktivitás kezelés** | Implicit | Implicit | **Explicit** |
| **Párhuzamosítás** | Stratégiánként | Nem | Task-onként |

## Koncepcionális különbségek

### Panel Mode
```
                    ┌─────────────────────┐
                    │   Single Model      │
                    │  (All Strategies)   │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │ Strategy 1   │    │ Strategy 2   │    │ Strategy N   │
    │ Prediction   │    │ Prediction   │    │ Prediction   │
    └──────────────┘    └──────────────┘    └──────────────┘
```

### Dual Mode
```
                    ┌─────────────────────────────────────┐
                    │         Two Separate Models          │
                    │  ┌─────────────┐ ┌─────────────┐    │
                    │  │  Activity   │ │   Profit    │    │
                    │  │   Model     │ │   Model     │    │
                    │  │(Classifier) │ │(Regressor)  │    │
                    │  └──────┬──────┘ └──────┬──────┘    │
                    └─────────┼───────────────┼───────────┘
                              │               │
                              ▼               ▼
                    ┌─────────────────────────────────────┐
                    │         Combined Forecast            │
                    │  Final = Activity × Weeks × Profit  │
                    └─────────────────────────────────────┘
```

## Mikor melyiket használd?

### Panel Mode
| Ideális esetben | Elkerülendő |
|-----------------|-------------|
| Gyors tesztelés | Pontos előrejelzés kell |
| Sok stratégia (6000+) | Egyedi stratégia jellemzők fontosak |
| Hasonló stratégiák | Nagyon eltérő stratégiák |
| Prototípus készítés | Produkciós elemzés |

### Dual Mode
| Ideális esetben | Elkerülendő |
|-----------------|-------------|
| Aktivitás fontos | Csak profit érdekel |
| Változó aktivitású stratégiák | Mindig aktív stratégiák |
| Részletes elemzés | Gyors eredmény kell |
| Kockázat kezelés | Egyszerű rangsorolás |

## Kizáró működés

**Panel és Dual Mode kölcsönösen kizárják egymást**:

```python
def _on_panel_mode_change(self):
    if self.panel_mode.get():
        # Dual kikapcsolása
        if self.dual_model_mode.get():
            self.dual_model_mode.set(False)

def _on_dual_model_change(self):
    if self.dual_model_mode.get():
        # Panel kikapcsolása
        if self.panel_mode.get():
            self.panel_mode.set(False)
```

---

# 04.11 Forecast Horizon csúszka

## Megjelenés

```
┌─────────────────────────────────────────────────────────────┐
│ Forecast Horizon: [═══════════○═══════════════] 52         │
└─────────────────────────────────────────────────────────────┘
```

## Specifikáció

| Tulajdonság | Érték |
|-------------|-------|
| **Minimum** | 1 hét |
| **Maximum** | 104 hét (2 év) |
| **Lépésköz** | 1 hét (103 lépés) |
| **Alapértelmezett** | 52 hét (1 év) |
| **Szélesség** | 150px |

## Funkció

A **Forecast Horizon** meghatározza, hogy **hány hétre előre** készüljön előrejelzés.

## Matematikai háttér

### Horizont értelmezése

```
Horizon = h hét

Az előrejelzés a következőket tartalmazza:
  - Forecast_1W:  Σ Profit(t+1 ... t+1)     [1 hét]
  - Forecast_1M:  Σ Profit(t+1 ... t+4)     [4 hét]
  - Forecast_3M:  Σ Profit(t+1 ... t+13)    [13 hét]
  - Forecast_6M:  Σ Profit(t+1 ... t+26)    [26 hét]
  - Forecast_12M: Σ Profit(t+1 ... t+52)    [52 hét]

ahol t = utolsó ismert időpont
```

### Horizon hatása

| Horizon érték | Számított előrejelzések |
|---------------|------------------------|
| h = 1 | Csak Forecast_1W |
| h = 4 | Forecast_1W, 1M |
| h = 13 | Forecast_1W, 1M, 3M |
| h = 26 | Forecast_1W, 1M, 3M, 6M |
| h ≥ 52 | Mind az 5 horizont |

### Adat validáció

```python
# Ellenőrzés: van-e elég historikus adat
min_required_weeks = horizon + look_back  # pl. 52 + 10 = 62 hét

if len(strategy_data) < min_required_weeks:
    logger.warning(f"Strategy {id} has insufficient data for horizon {horizon}")
```

## Modell-specifikus viselkedés

| Modell típus | Horizon kezelés |
|--------------|-----------------|
| **ARIMA** | Rekurzív h-lépéses előrejelzés |
| **ML** | Külön target minden horizontra |
| **RNN** | Sequence-to-sequence output |
| **Panel** | Multi-output regression |

---

# 04.12 Time kiírás

## Megjelenés

```
┌─────────────────────────┐
│  Time: 01:23:45         │  ← Consolas 12pt, félkövér
└─────────────────────────┘
```

## Funkció

A **Time** kijelzés mutatja az aktuális elemzés **eltelt idejét** HH:MM:SS formátumban.

## Működési logika

### Timer indítása

```python
def start_timer(self):
    """Start the analysis timer."""
    self.start_time = datetime.now()
    self.accumulated_time = timedelta(0)
    self.timer_running = True
    self.update_timer()
```

### Timer frissítése

```python
def update_timer(self):
    """Update timer display every second."""
    if self.timer_running:
        elapsed = datetime.now() - self.start_time + self.accumulated_time

        # Formázás HH:MM:SS
        total_seconds = int(elapsed.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        self.lbl_timer.configure(text=f"Time: {hours:02}:{minutes:02}:{seconds:02}")

        # Következő frissítés 1 mp múlva
        self.after(1000, self.update_timer)
```

### Pause kezelés

```python
def toggle_pause(self):
    if self.pause_event.is_set():
        # Szüneteltetés
        self.timer_running = False
        # Eltelt idő mentése
        self.accumulated_time += datetime.now() - self.start_time
    else:
        # Folytatás
        self.start_time = datetime.now()  # Új kezdőpont
        self.timer_running = True
        self.update_timer()
```

### Timer leállítása

```python
def stop_timer(self):
    """Stop the timer."""
    self.timer_running = False
```

## Elemzés végén

Az execution_time mentésre kerül az eredményekkel:

```python
execution_time = f"{hours:02}:{minutes:02}:{seconds:02}"

self.last_results = {
    ...
    "execution_time": execution_time,  # Riportba kerül
    ...
}
```

---

# 04.13 Paraméterek szekció

## Megjelenés

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  P: [1]  D: [1]  Q: [1]  Seasonal: [False ▼]  Period: [12]  Stepwise: [True]│
│  Max P: [3]  Max D: [1]  Max Q: [3]  Information Criterion: [aic ▼]         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Funkció

A **Paraméterek** szekció dinamikusan mutatja és engedélyezi a kiválasztott modell összes konfigurálható paraméterét.

## Dinamikus UI generálás

```python
def _setup_standard_params(self, params, param_options):
    """Setup parameter UI for standard models."""
    cols = 8  # 8 oszlopos elrendezés

    for i, (key, val) in enumerate(params.items()):
        row = i // cols
        col = i % cols

        # Címke formázása
        pretty_key = key.replace("_", " ").title() + ": "

        # Ha van előre definiált opció lista → ComboBox
        if key in param_options:
            entry = ctk.CTkComboBox(frame, values=param_options[key])
            entry.set(val)
        else:
            # Egyébként szabad beviteli mező → Entry
            entry = ctk.CTkEntry(frame, width=55)
            entry.insert(0, val)

        self.param_entries[key] = entry
```

## Paraméter típusok

### 1. Szabad bevitel (Entry)

```
┌─────────────┐
│ P: [ 1   ]  │  ← Numerikus érték
└─────────────┘
```

Használat: Számok, float értékek

### 2. Legördülő lista (ComboBox)

```
┌────────────────────────────┐
│ Seasonal: [  False    ▼]   │  ← Előre definiált opciók
│             True           │
│             False          │
└────────────────────────────┘
```

Használat: Boolean, enum típusok

### 3. Multi-select (Ensemble)

```
┌─────────────────────────────┐
│ Models:                     │
│  [✓] ARIMA                  │
│  [✓] ETS                    │
│  [ ] Prophet                │
│  [✓] LSTM                   │
│  [ ] XGBoost                │
└─────────────────────────────┘
```

Használat: Time Series Ensemble modell

## Modell-specifikus paraméterek

### Statistical Models (példák)

| Modell | Paraméterek |
|--------|-------------|
| **ARIMA** | p, d, q |
| **SARIMA** | p, d, q, P, D, Q, s |
| **Auto-ARIMA** | max_p, max_d, max_q, seasonal, stepwise, ic |
| **GARCH** | p, q, variant, mean, vol |

### Machine Learning (példák)

| Modell | Paraméterek |
|--------|-------------|
| **Random Forest** | n_estimators, max_depth, lags |
| **XGBoost** | n_estimators, learning_rate, max_depth, reg_alpha, reg_lambda, ... |
| **LightGBM** | n_estimators, learning_rate, num_leaves, max_depth, ... |

### Deep Learning (példák)

| Modell | Paraméterek |
|--------|-------------|
| **LSTM** | epochs, batch_size, learning_rate, look_back, hidden_size, num_layers, dropout |
| **Transformer** | d_model, nhead, num_layers, dim_feedforward, dropout, epochs |

## Paraméter validáció

```python
def convert_param(self, value):
    """Convert parameter string to appropriate type."""
    if value in ["True", "true"]:
        return True
    if value in ["False", "false"]:
        return False
    if value in ["None", "null", ""]:
        return None

    # Numerikus konverzió
    try:
        if "." in str(value):
            return float(value)
        return int(value)
    except ValueError:
        return value  # String marad
```

## Paraméterek mentése riportba

Az összes paraméter mentésre kerül a generált riportban:

```markdown
## Model Settings

| Parameter | Value |
|-----------|-------|
| p | 1 |
| d | 1 |
| q | 1 |
| seasonal | False |
| ...
```

## Közös paraméter opciók

A `config/parameters.py`-ban definiált közös opciók:

```python
COMMON_PARAM_OPTIONS = {
    "trend_poly_order": ["0", "1", "2", "3"],
    "window": ["hann", "hamming", "blackman", "bartlett"],
    "ic": ["aic", "bic", "hqic", "fpe"],
    "variant": ["Standard", "GJR-GARCH", "EGARCH", "TGARCH"],
    "normalize": ["True", "False"],
    "seasonal": ["12", "0", "4", "7", "52"],
    # ...
}
```

---

*Dokumentum készítése: 2024.12.29*
*Program verzió: v3.60.0 Stable*
*Fejezet: 04 - Analysis Tab*
