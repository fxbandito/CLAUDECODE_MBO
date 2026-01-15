# MBO Trading Strategy Analyzer - Auto Execution Window

## Tartalomjegyzék

- [05.00 Auto gomb fő funkciója](#0500-auto-gomb-fő-funkciója)
- [05.01 Category és Model választás](#0501-category-és-model-választás)
- [05.02 Beállítások és összevetés az Analysis Tab-bal](#0502-beállítások-és-összevetés-az-analysis-tab-bal)
- [05.03 Task kiírás](#0503-task-kiírás)
- [05.04 Load List gomb](#0504-load-list-gomb)
- [05.05 Save gomb](#0505-save-gomb)
- [05.06 Reports Folder](#0506-reports-folder)
- [05.07 +Add to List gomb](#0507-add-to-list-gomb)
- [05.08 Auto.txt automatikus létrehozása](#0508-autotxt-automatikus-létrehozása)
- [05.09 Lista elemek kezelése](#0509-lista-elemek-kezelése)
- [05.10 Shutdown PC after ALL checkbox](#0510-shutdown-pc-after-all-checkbox)
- [05.11 Start Auto Execution gomb](#0511-start-auto-execution-gomb)

---

# 05.00 Auto gomb fő funkciója

## Áttekintés

Az **Auto Execution Window** (Automatikus Végrehajtás Ablak) egy önálló, modális ablak, amely lehetővé teszi **több modell egymás utáni automatikus futtatását**. Ahelyett, hogy minden modellt kézzel indítanál és várnál a befejezésére, összeállíthatsz egy várakozási listát és a program automatikusan végigfuttatja az összeset.

## Az Auto gomb megnyitja az ablakot

```
┌──────────┐
│   Auto   │  ← Kattintásra megnyílik az Auto Execution Window
└──────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Auto Execution Manager                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Category: [▼ ____]  Model: [▼ ____]  [?]     [Load] [Save] [Folder]   │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │ CPU: 75% (18)  [✓]GPU  Horizon: [═══○═══] 52  Data: [original ▼]     │  │
│  │ [ ]Panel  [ ]Dual  [ ]Stability  [ ]Risk              Tasks: 5       │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │ P: [1]  D: [1]  Q: [1]  Seasonal: [False ▼]                           │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │                     [+ Add to List]                                   │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │ Execution Queue                                                       │  │
│  │ ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │ │ [-][▲][▼] [original] ARIMA (Statistical) | P: 1 | D: 1 | Q: 1   │   │  │
│  │ │ [-][▲][▼] [forward/Panel] XGBoost (ML) | n_est: 100 | lr: 0.1   │   │  │
│  │ │ [-][▲][▼] [rolling/Dual] LightGBM (ML) | n_est: 200 | lr: 0.05  │   │  │
│  │ └─────────────────────────────────────────────────────────────────┘   │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │ CPU Power (from Analysis Tab): 75% (18 cores)                         │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │ [═══════════════════░░░░░░]  [ ]Shutdown PC after ALL  [Start Auto]  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Fő funkciók

| Funkció | Leírás |
|---------|--------|
| **Queue építés** | Modellek sorba állítása egyedi beállításokkal |
| **Paraméterezés** | Modell-specifikus paraméterek beállítása |
| **Batch futtatás** | Összes modell automatikus egymás utáni futtatása |
| **Riport generálás** | Standard, Stability, Risk riportok automatikus létrehozása |
| **Mentés/Betöltés** | Queue konfigurációk mentése és visszatöltése |
| **Shutdown** | Számítógép leállítása minden modell után |

## Ablak viselkedés

| Tulajdonság | Érték |
|-------------|-------|
| **Típus** | CTkToplevel (modális) |
| **Alapméret** | 1400x960 px |
| **Bezárás** | Elrejti (withdraw), nem zárja be |
| **Újra megnyitás** | Auto gomb újra előhozza |
| **Geometria mentés** | Automatikus pozíció/méret mentés |

```python
def on_close(self):
    """Handle window closing - hides instead of destroying."""
    self.parent.auto_window_geometry = self.geometry()
    self.parent.save_window_state()
    self.grab_release()
    self.withdraw()  # Elrejti, nem zárja be
```

---

# 05.01 Category és Model választás

## Megjelenés

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Category: [Statistical Models         ▼]  Model: [ARIMA              ▼] │
└──────────────────────────────────────────────────────────────────────────┘
```

## Működés

### Category választás
- Azonos kategóriák mint az Analysis Tab-on (14 kategória)
- Dinamikus szélesség a leghosszabb kategórianév alapján
- Kategória váltáskor a modell lista automatikusan frissül

```python
def on_category_change(self, category):
    models = self.parent.model_categories.get(category, [])
    self.combo_model.configure(values=models)
    if models:
        self.combo_model.set(models[0])
        self.on_model_change(models[0])

    # Data Mode kezelés - csak ML modellek használhatják a forward/rolling módot
    if category == "Classical Machine Learning":
        self.combo_data_mode.configure(state="readonly")
    else:
        self.combo_data_mode.set("original")
        self.combo_data_mode.configure(state="disabled")
```

### Model választás
- A kiválasztott kategória modelljeinek listája
- 260px szélesség
- Modell váltáskor a paraméterek automatikusan frissülnek

### Szinkronizáció az Analysis Tab-bal

```python
def on_model_change(self, model):
    # Ha ugyanaz a modell van kiválasztva az Analysis Tab-on,
    # átveszi az ottani paramétereket
    if self.parent.model_combo.get() == model:
        for key in params.keys():
            if key in self.parent.param_entries:
                params[key] = self.parent.param_entries[key].get()
```

### Help gomb (?)

```
┌────┐
│ ?  │  ← Modell súgó popup megnyitása
└────┘
```

---

# 05.02 Beállítások és összevetés az Analysis Tab-bal

## Auto Execution Window beállítások

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CPU: 75% (18)  [✓]GPU  Horizon: [═══○═══] 52  Data: [original ▼]           │
│ [ ]Panel  [ ]Dual  [ ]Stability  [ ]Risk                     Tasks: 5      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Összehasonlító táblázat

| Beállítás | Analysis Tab | Auto Exec Window | Különbség |
|-----------|--------------|------------------|-----------|
| **CPU Power** | Csúszka (10-100%) | Csak kijelzés | Auto: **csak olvasható**, az Analysis Tab vezérli |
| **Use GPU** | Switch | Switch | Azonos - item-enként menthető |
| **Forecast Horizon** | Csúszka (1-104) | Csúszka (1-104) | Azonos - item-enként menthető |
| **Data Mode** | RadioButton | ComboBox | Auto: **original/forward/rolling**, csak ML-nél |
| **Panel Mode** | Checkbox | Checkbox | Azonos - item-enként menthető |
| **Dual Model** | Checkbox | Checkbox | Azonos - item-enként menthető |
| **Stability** | Results Tab-on | Checkbox | Auto: **riport generálás beállítás** |
| **Risk** | Results Tab-on | Checkbox | Auto: **riport generálás beállítás** |

## Részletes különbségek

### CPU Power

**Analysis Tab**:
```
CPU Power: [═══════════○═══════] 75% (18 cores)
                      ↑
              Állítható csúszka
```

**Auto Exec Window**:
```
CPU Power (controlled from Analysis Tab): 75% (18 cores)
                                          ↑
                                  Csak kijelzés
```

**Logika**: A CPU Manager egy központi singleton, amelyet az Analysis Tab vezérel. Az Auto Exec Window callback-kel figyeli a változásokat:

```python
def on_cpu_change(new_percentage):
    # Automatikusan frissül, ha az Analysis Tab-on változik
    self.lbl_cpu_val.configure(
        text=f"{new_percentage}% ({cpu_manager.get_n_jobs()} cores)"
    )

cpu_manager.add_callback(on_cpu_change)
```

### Data Mode

**Analysis Tab**: Feature Mode RadioButton (a Data Loading Tab-on)
- Original / Forward Calc / Rolling Window

**Auto Exec Window**: ComboBox - modell-specifikusan állítható
- Csak **Classical Machine Learning** kategóriánál engedélyezett
- Más kategóriáknál automatikusan "original" és letiltott

```python
if category == "Classical Machine Learning":
    self.combo_data_mode.configure(state="readonly")  # Engedélyezve
else:
    self.combo_data_mode.set("original")
    self.combo_data_mode.configure(state="disabled")  # Letiltva
```

### Stability és Risk

**Analysis Tab**: A Results Tab-on a Ranking Mode választóval érhető el
- Manuálisan kell váltani és riportot generálni

**Auto Exec Window**: Checkboxok minden modellhez
- Bejelölve → automatikusan generál **extra riportot** az adott rankinggal
- Egy modellből akár 3 riport is készülhet: Standard, Stability, Risk

---

# 05.03 Task kiírás

## Megjelenés

```
┌──────────────────────────────────────────┐
│ Tasks: 5                                 │  ← Félkövér, jobb oldalon
└──────────────────────────────────────────┘
```

## Funkció

A **Tasks** kijelzés mutatja a végrehajtási sorban lévő modellek számát.

## Működés

```python
def refresh_list_ui(self):
    # Lista újraépítése után...
    self.lbl_count.configure(text=f"Tasks: {len(self.execution_list)}")
```

## Értelmezés

| Tasks érték | Jelentés |
|-------------|----------|
| **Tasks: 0** | Üres lista, Start gomb letiltva |
| **Tasks: N** | N modell vár végrehajtásra |

---

# 05.04 Load List gomb

## Megjelenés

```
┌────────────────┐
│   Load List    │  ← Kék (#3498db), 120px
└────────────────┘
```

## Funkció

A **Load List** gomb lehetővé teszi egy korábban elmentett konfigurációs fájl betöltését.

## Működési folyamat

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOAD LIST WORKFLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. [Load List] gomb kattintás                                  │
│           │                                                     │
│           ▼                                                     │
│  2. ┌─────────────────────────────┐                             │
│     │ Fájl választó dialógus      │                             │
│     │ Szűrő: *.txt                │                             │
│     └──────────────┬──────────────┘                             │
│                    │                                            │
│                    ▼                                            │
│  3. ┌─────────────────────────────┐                             │
│     │ JSON formátum validálás     │                             │
│     │ - Lista típus?              │                             │
│     │ - Van category, model,      │                             │
│     │   params minden elemben?    │                             │
│     └──────────────┬──────────────┘                             │
│                    │                                            │
│           ┌───────┴───────┐                                     │
│           │               │                                     │
│        VALID          INVALID                                   │
│           │               │                                     │
│           ▼               ▼                                     │
│  4a. output_folder     4b. Hibaüzenet                           │
│      felülírása            a status bar-on                      │
│      (ha van kiválasztva)                                       │
│           │                                                     │
│           ▼                                                     │
│  5. execution_list = betöltött adat                             │
│           │                                                     │
│           ▼                                                     │
│  6. save_list() → auto.txt felülírása                           │
│           │                                                     │
│           ▼                                                     │
│  7. refresh_list_ui() → megjelenítés frissítése                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Előfeltétel

**FONTOS**: A Load List gomb **inaktív** amíg nincs Reports Folder kiválasztva!

```python
self.btn_load = ctk.CTkButton(
    ...
    state="disabled",  # Alapból letiltva
)

def select_folder(self):
    ...
    if folder:
        self.btn_load.configure(state="normal")  # Csak mappa választás után aktív
```

## Validálás

```python
def load_config_file(self):
    # JSON parse
    data = json.loads(content)

    # Típus ellenőrzés
    if not isinstance(data, list):
        raise ValueError("Content is not a list.")

    # Elem validálás
    for item in data:
        if not all(k in item for k in ("category", "model", "params")):
            raise ValueError("Item missing required keys.")

        # Output folder felülírása a jelenlegivel
        if self.selected_folder:
            item["output_folder"] = self.selected_folder
```

---

# 05.05 Save gomb

## Megjelenés

```
┌────────────────┐
│   Save List    │  ← Narancssárga (#e67e22), 80px
└────────────────┘
```

## Funkció

A **Save List** gomb elmenti az aktuális végrehajtási listát egy tetszőleges nevű fájlba.

## Működés

```python
def save_config_file(self):
    # Mentés dialógus
    filename = filedialog.asksaveasfilename(
        defaultextension=".txt",
        initialfile="auto_export.txt",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
    )

    if filename:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.execution_list, f, indent=2)
        self.lbl_status.configure(text=f"Saved: {os.path.basename(filename)}")
```

## Különbség az auto.txt-től

| Szempont | auto.txt | Save List |
|----------|----------|-----------|
| **Automatikus** | Igen, minden változáskor | Nem, kézi mentés |
| **Helye** | src/ mappa (fix) | Tetszőleges |
| **Név** | Mindig auto.txt | Egyedi |
| **Cél** | Session perzisztencia | Archiválás, megosztás |

---

# 05.06 Reports Folder

## Megjelenés

```
┌──────────────────┐
│  Reports Folder  │  ← Alapszín, jobb oldalon
└──────────────────┘
```

## Funkció

A **Reports Folder** gomb lehetővé teszi a riportok célmappájának kiválasztását.

## KRITIKUS SZABÁLY

**A Reports Folder kiválasztása KÖTELEZŐ a lista bővítése előtt!**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    REPORTS FOLDER - BLOKKOLÓ LOGIKA                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Program induláskor:                                                   │
│   ┌───────────────────────────────────────────────────────────────┐    │
│   │  self.selected_folder = ""   ← Üres string                    │    │
│   │  self.btn_add.configure(state="disabled")  ← + Add letiltva   │    │
│   │  self.btn_load.configure(state="disabled") ← Load letiltva    │    │
│   └───────────────────────────────────────────────────────────────┘    │
│                                                                         │
│   [Reports Folder] gomb kattintás után:                                 │
│   ┌───────────────────────────────────────────────────────────────┐    │
│   │  Ha mappa kiválasztva:                                         │    │
│   │    self.selected_folder = folder                               │    │
│   │    self.btn_add.configure(state="normal")   ← + Add engedélyez │    │
│   │    self.btn_load.configure(state="normal")  ← Load engedélyez  │    │
│   └───────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Működés

```python
def select_folder(self):
    get_sound_manager().play_button_click()
    folder = filedialog.askdirectory(initialdir=initial_dir)

    if folder:
        self.selected_folder = folder
        self.parent.last_auto_reports_folder = folder
        self.parent.save_window_state()  # Mentés azonnal

        # + Add gomb engedélyezése
        self.btn_add.configure(state="normal")

        # Ha már vannak elemek, azok output_folder-ét is frissíti
        if self.execution_list:
            for item in self.execution_list:
                item["output_folder"] = self.selected_folder
            self.save_list()
            self.refresh_list_ui()

        # Load gomb engedélyezése
        self.btn_load.configure(state="normal")
```

## Mappa struktúra

Kiválasztott mappa alatt a riportok almappákba kerülnek:

```
Reports Folder (kiválasztott)
└── ModelName_CurrencyPair_RankingMode_Axx/
    ├── report.md
    ├── report.html
    └── charts/
        ├── top10_performance.png
        └── horizon_comparison.png
```

## Perzisztencia

A kiválasztott mappa mentésre kerül a `window_config.json`-ban:

```json
{
    "last_auto_reports_folder": "C:/Reports/AutoExec"
}
```

---

# 05.07 +Add to List gomb

## Megjelenés

### Aktív állapot (Reports Folder kiválasztva)
```
┌──────────────────┐
│  + Add to List   │  ← Zöld, aktív
└──────────────────┘
```

### Inaktív állapot (nincs Reports Folder)
```
┌──────────────────┐
│  + Add to List   │  ← Szürke, inaktív (disabled)
└──────────────────┘
```

## Funkció

A **+ Add to List** gomb hozzáadja az aktuálisan konfigurált modellt a végrehajtási sorhoz.

## Előfeltétel

| Feltétel | Ellenőrzés |
|----------|------------|
| Reports Folder kiválasztva | `self.selected_folder != ""` |

## Működés

```python
def add_item(self):
    get_sound_manager().play_button_click()

    # Adatok összegyűjtése
    item = {
        "category": self.combo_category.get(),
        "model": self.combo_model.get(),
        "params": {key: entry.get() for key, entry in self.param_entries.items()},
        "output_folder": self.selected_folder,
        "use_gpu": self.switch_gpu.get() == 1,
        "horizon": int(self.slider_horizon.get()),
        "data_mode": self.combo_data_mode.get(),
        "panel_mode": self.panel_mode_var.get(),
        "dual_model": self.dual_model_var.get(),
        "auto_stability": self.auto_stability_var.get(),
        "auto_risk": self.auto_risk_var.get(),
    }

    # Listához adás
    self.execution_list.append(item)

    # Automatikus mentés
    self.save_list()

    # UI frissítés
    self.refresh_list_ui()
    self.update_start_button_state()
```

## Inaktív állapot oka

```python
# __init__-ben:
self.btn_add = ctk.CTkButton(
    ...
    state="disabled"  # ALAPBÓL LETILTVA
)

# select_folder()-ben aktiválódik:
if folder:
    self.btn_add.configure(state="normal")
```

---

# 05.08 Auto.txt automatikus létrehozása

## Fájl helye

```
C:\Users\bandi\ClaudeCode\src\auto.txt
```

## Funkció

Az **auto.txt** fájl automatikusan tárolja a végrehajtási listát a program session-ök között.

## Automatikus létrehozás folyamata

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AUTO.TXT LIFECYCLE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. AUTO EXECUTION WINDOW MEGNYITÁSA                                    │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │  def __init__(self, parent):                                  │  │
│     │      self.auto_file = os.path.join(                           │  │
│     │          os.path.dirname(os.path.dirname(__file__)),          │  │
│     │          "auto.txt"                                           │  │
│     │      )                                                        │  │
│     │      # Path: src/auto.txt                                     │  │
│     │      self.load_list()  ← Betöltési kísérlet                  │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  2. BETÖLTÉS (load_list)                                                │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │  def load_list(self):                                         │  │
│     │      if os.path.exists(self.auto_file):                       │  │
│     │          # Fájl létezik → beolvassa                           │  │
│     │          with open(self.auto_file, "r") as f:                 │  │
│     │              self.execution_list = json.loads(f.read())       │  │
│     │      else:                                                    │  │
│     │          # Fájl NEM létezik → üres lista                      │  │
│     │          self.execution_list = []                             │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  3. MENTÉS (save_list) - BÁRMILYEN VÁLTOZÁSKOR                          │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │  Triggerek:                                                   │  │
│     │    - add_item()      → Új elem hozzáadása                     │  │
│     │    - remove_item()   → Elem törlése                           │  │
│     │    - move_item_up()  → Elem felfelé mozgatása                 │  │
│     │    - move_item_down()→ Elem lefelé mozgatása                  │  │
│     │    - load_config_file() → Külső fájl betöltése                │  │
│     │    - select_folder() → Output mappa módosítása                │  │
│     │    - _open_edit_dialog() → Elem szerkesztése                  │  │
│     │                                                               │  │
│     │  def save_list(self):                                         │  │
│     │      with open(self.auto_file, "w") as f:                     │  │
│     │          json.dump(self.execution_list, f, indent=2)          │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Fájl formátum (JSON)

```json
[
  {
    "category": "Statistical Models",
    "model": "ARIMA",
    "params": {
      "p": "1",
      "d": "1",
      "q": "1"
    },
    "output_folder": "C:/Reports/AutoExec",
    "use_gpu": false,
    "horizon": 52,
    "data_mode": "original",
    "panel_mode": false,
    "dual_model": false,
    "auto_stability": false,
    "auto_risk": true
  },
  {
    "category": "Classical Machine Learning",
    "model": "XGBoost",
    "params": {
      "n_estimators": "100",
      "learning_rate": "0.1",
      "max_depth": "6"
    },
    "output_folder": "C:/Reports/AutoExec",
    "use_gpu": true,
    "horizon": 52,
    "data_mode": "forward",
    "panel_mode": true,
    "dual_model": false,
    "auto_stability": true,
    "auto_risk": false
  }
]
```

## Logikai magyarázat

### Miért automatikus mentés?

1. **Session perzisztencia**: Ha bezárod a programot, a lista megmarad
2. **Crash védelem**: Váratlan leállás esetén nincs adatvesztés
3. **Nincs "Mentés" gomb szükség**: A felhasználónak nem kell emlékeznie

### Miért JSON formátum?

1. **Ember által olvasható**: Szerkeszthető szöveges fájl
2. **Könnyű exportálás**: Copy-paste más rendszerekbe
3. **Python natív támogatás**: `json.loads()` / `json.dump()`

### Miért indent=2?

```python
json.dump(self.execution_list, f, indent=2)
```

A `indent=2` formázott (pretty-printed) kimenetet ad, ami:
- Könnyebben olvasható
- Verziókezelőben (git) jobb diff-et ad
- Kézi szerkesztésre alkalmas

---

# 05.09 Lista elemek kezelése

## Lista elem megjelenése

```
┌─────────────────────────────────────────────────────────────────────────┐
│ [-] [▲] [▼] [original/Panel+Stab] ARIMA (Statistical) | P: 1 | D: 1    │
└─────────────────────────────────────────────────────────────────────────┘
  │    │   │         │                    │                  │
  │    │   │         │                    │                  └─ Paraméterek
  │    │   │         │                    └─ Modell és kategória
  │    │   │         └─ Módok: data_mode/panel/dual+riportok
  │    │   └─ Lefelé mozgatás
  │    └─ Felfelé mozgatás
  └─ Eltávolítás
```

## Gombok funkciói

### Eltávolítás gomb (-)

```
┌────┐
│ -  │  ← Piros (#c0392b), 28x28px
└────┘
```

```python
def remove_item(self, index):
    if 0 <= index < len(self.execution_list):
        del self.execution_list[index]
        self.save_list()
        self.refresh_list_ui()
        self.update_start_button_state()
```

### Felfelé mozgatás (▲)

```
┌────┐
│ ▲  │  ← Kék (#3498db), 28x28px
└────┘
```

```python
def move_item_up(self, index):
    if index > 0:
        # Csere az előzővel
        self.execution_list[index], self.execution_list[index - 1] = (
            self.execution_list[index - 1],
            self.execution_list[index],
        )
        self.save_list()
        self.refresh_list_ui()
```

**Letiltva**: Ha az elem már az első helyen van

### Lefelé mozgatás (▼)

```
┌────┐
│ ▼  │  ← Kék (#3498db), 28x28px
└────┘
```

```python
def move_item_down(self, index):
    if index < len(self.execution_list) - 1:
        # Csere a következővel
        self.execution_list[index], self.execution_list[index + 1] = (
            self.execution_list[index + 1],
            self.execution_list[index],
        )
        self.save_list()
        self.refresh_list_ui()
```

**Letiltva**: Ha az elem már az utolsó helyen van

## Dupla kattintás - Szerkesztés

A lista elemre **dupla kattintással** megnyílik egy szerkesztő dialógus:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Edit Task                                 │
├─────────────────────────────────────────────────────────────────┤
│  ARIMA (Statistical Models)                                      │
├─────────────────────────────────────────────────────────────────┤
│  Settings                                                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Data Mode:   [original    ▼]                               │  │
│  │ Horizon:     [52          ]                                │  │
│  │ Use GPU:     [  ] (switch)                                 │  │
│  │ Panel Mode:  [  ] (checkbox)                               │  │
│  │ Dual Model:  [  ] (checkbox)                               │  │
│  │ Stability:   [✓] (checkbox)                                │  │
│  │ Risk:        [  ] (checkbox)                               │  │
│  └───────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Model Parameters                                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ P:            [1          ]                                │  │
│  │ D:            [1          ]                                │  │
│  │ Q:            [1          ]                                │  │
│  └───────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                              [Cancel]  [Save]                    │
└─────────────────────────────────────────────────────────────────┘
```

### Szerkesztés működése

```python
def _open_edit_dialog(self, index):
    item = self.execution_list[index]

    # Popup ablak létrehozás
    popup = ctk.CTkToplevel(self)
    popup.title("Edit Task")
    popup.geometry("600x450")

    # ... Widget-ek létrehozása az aktuális értékekkel ...

    def save_changes():
        # Beállítások mentése
        item["data_mode"] = entry_widgets["_data_mode"].get()
        item["horizon"] = int(entry_widgets["_horizon"].get())
        item["use_gpu"] = entry_widgets["_use_gpu"].get() == 1
        item["panel_mode"] = entry_widgets["_panel_mode"].get()
        item["dual_model"] = entry_widgets["_dual_model"].get()
        item["auto_stability"] = entry_widgets["_auto_stability"].get()
        item["auto_risk"] = entry_widgets["_auto_risk"].get()

        # Paraméterek mentése
        for key, entry in entry_widgets.items():
            if not key.startswith("_"):
                item["params"][key] = entry.get()

        # Azonnali mentés és frissítés
        self.save_list()
        self.refresh_list_ui()
        popup.destroy()
```

## Futás közben

Auto Execution futása alatt **minden szerkesztő funkció letiltott**:

```python
def refresh_list_ui(self):
    is_running = hasattr(self.parent, "is_auto_running") and self.parent.is_auto_running

    # Gombok letiltása futás közben
    if is_running:
        btn_remove.configure(state="disabled", fg_color="#555555")
        btn_up.configure(state="disabled", fg_color="#555555")
        btn_down.configure(state="disabled", fg_color="#555555")

    # Dupla kattintás csak ha nem fut
    if not is_running:
        lbl.bind("<Double-Button-1>", lambda e, idx=i: self._open_edit_dialog(idx))
```

---

# 05.10 Shutdown PC after ALL checkbox

## Megjelenés

```
┌──────────────────────────────┐
│  [ ] Shutdown PC after ALL   │  ← Piros szöveg (#e74c3c)
└──────────────────────────────┘
```

## Funkció

Bejelöléskor a számítógép **automatikusan leáll** miután **MINDEN** modell lefutott a sorban.

## Összehasonlítás az Analysis Tab checkboxszal

| Szempont | Analysis Tab | Auto Exec Window |
|----------|--------------|------------------|
| **Felirat** | "Shutdown" | "Shutdown PC after ALL" |
| **Mikor áll le** | Az aktuális modell után | MINDEN modell után |
| **Queue megszakítás** | IGEN - nem fut a következő | NEM - végigfut minden |
| **Másik checkbox** | Marad aktív | Analysis Tab-ot letiltja |

## Logikai működés

### Auto Exec "after ALL" mód

```python
def run_auto_sequence(self, execution_list, shutdown_after_all=False):
    if shutdown_after_all:
        # Az Analysis Tab checkbox letiltása
        self.var_shutdown_after_run.set(False)  # Kikapcsolás
        self.chk_shutdown.configure(state="disabled", text_color="gray")
        self.log("Shutdown after ALL models - Analysis tab checkbox locked.")

    self.auto_shutdown_after_all = shutdown_after_all
```

### Befejezéskor

```python
def _finish_auto_sequence(self):
    # Újra ellenőrzi a checkbox állapotát!
    auto_exec_shutdown = False
    if after_all_shutdown:
        if self.auto_window.var_shutdown_after_all.get():
            auto_exec_shutdown = True
        else:
            self.log("Shutdown cancelled - checkbox was unchecked.")

    if auto_exec_shutdown:
        self.log("Initiating Shutdown (Auto Exec mode - after all models)...")
        shutdown_handler.trigger_shutdown_sequence()
```

## Folyamat diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│               SHUTDOWN CHECKBOX COMPARISON                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ANALYSIS TAB CHECKBOX bejelölve:                                       │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ Model 1 ──▶ Model 2 ──▶ Model 3 ──▶ ... ──▶ Model N               │ │
│  │    │                                                              │ │
│  │    │ ← Közben bejelöli                                            │ │
│  │    ▼                                                              │ │
│  │  STOP!   ❌ Model 3-N NEM FUT                                     │ │
│  │    ▼                                                              │ │
│  │ SHUTDOWN                                                          │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  AUTO EXEC "after ALL" CHECKBOX bejelölve:                              │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ Model 1 ──▶ Model 2 ──▶ Model 3 ──▶ ... ──▶ Model N               │ │
│  │                                                 │                 │ │
│  │                                                 ▼                 │ │
│  │                                            MINDEN KÉSZ            │ │
│  │                                                 ▼                 │ │
│  │                                            SHUTDOWN               │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Visszavonás lehetősége

Mindkét checkbox **bármikor ki-pipálható** a futás közben, és a shutdown NEM történik meg:

```python
# Végső ellenőrzés shutdown előtt
if self.var_shutdown_after_all.get():  # Még mindig be van jelölve?
    auto_exec_shutdown = True
else:
    self.log("Shutdown cancelled - checkbox was unchecked.")
```

---

# 05.11 Start Auto Execution gomb

## Megjelenés

### Aktív állapot
```
┌─────────────────────────┐
│  Start Auto Execution   │  ← Alapszín, 14pt félkövér
└─────────────────────────┘
```

### Inaktív állapot
```
┌─────────────────────────┐
│  Start Auto Execution   │  ← Szürke, letiltva
└─────────────────────────┘
```

## Funkció

A **Start Auto Execution** gomb elindítja az összes sorban lévő modell egymás utáni automatikus végrehajtását.

## Előfeltételek

| Feltétel | Ellenőrzés |
|----------|------------|
| Lista nem üres | `len(self.execution_list) > 0` |
| Nincs futó Auto Exec | `not self.parent.is_auto_running` |
| Adat be van töltve | `self.parent.processed_data is not None` |

## Állapot frissítés

```python
def update_start_button_state(self):
    if hasattr(self.parent, "is_auto_running") and self.parent.is_auto_running:
        self.btn_start.configure(state="disabled")
        self.lbl_status.configure(text="Auto Execution in progress...")
    else:
        if self.execution_list:
            self.btn_start.configure(state="normal")
        else:
            self.btn_start.configure(state="disabled")
```

## Indítási folyamat

```python
def start_execution(self):
    get_sound_manager().play_button_click()

    # 1. Ellenőrzések
    if not self.execution_list:
        messagebox.showwarning("Empty", "Add models to the list first.")
        return

    if self.parent.processed_data is None:
        messagebox.showwarning("No Data", "Please load a dataset first.")
        return

    # 2. Ablak elrejtése
    self.grab_release()
    self.withdraw()

    # 3. Auto sequence indítás a parent-en
    self.parent.run_auto_sequence(
        self.execution_list,
        shutdown_after_all=self.var_shutdown_after_all.get()
    )
```

## Kapcsolat az Analysis Tab gombjaival

### Stop gomb

Az Analysis Tab **Stop** gombja **működik** Auto Execution közben is:

```python
def stop_analysis(self):
    if hasattr(self, "stop_check"):
        self.stop_check.set()
        # ...

# Auto Execution-ben:
def run_next_auto_model(self):
    # Stop ellenőrzés
    if stop_callback and stop_callback():
        self._finish_auto_sequence()
        return
```

**Hatás**: Megszakítja az aktuális modellt ÉS a teljes queue-t

### Pause gomb

Az Analysis Tab **Pause** gombja **működik** Auto Execution közben is:

```python
def toggle_pause(self):
    if self.pause_event.is_set():
        self.pause_event.clear()  # Szüneteltetés
        self.log("Analysis paused.")
    else:
        self.pause_event.set()    # Folytatás
        self.log("Analysis resumed.")
```

**Hatás**: Szünetelteti az aktuális modellt, folytatáskor onnan folytatja

## Futás közbeni állapot

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AUTO EXECUTION RUNNING STATE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Auto Exec Window:                                                      │
│    - Start gomb: DISABLED                                               │
│    - Lista elemek: DISABLED (nem szerkeszthető)                         │
│    - Ablak: Elrejthető (bezárás = hide)                                 │
│                                                                         │
│  Analysis Tab:                                                          │
│    - Run Analysis: DISABLED                                             │
│    - Stop: ENABLED ← Megállítja a queue-t                               │
│    - Pause: ENABLED ← Szünetelteti az aktuális modellt                  │
│    - Auto gomb: ENABLED (ablak előhozható)                              │
│                                                                         │
│  Működés:                                                               │
│    1. Auto Exec elindítja az első modellt                               │
│    2. Analysis Tab futtatja (stop/pause működik)                        │
│    3. Modell kész → Riport generálás                                    │
│    4. Következő modell automatikusan indul                              │
│    5. Queue vége → Shutdown (ha be volt jelölve)                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Batch feldolgozás és memória kezelés

```python
# auto_execution_mixin.py
AUTO_EXEC_BATCH_SIZE = 5  # Minden 5 modell után nagy GC

def run_auto_reporting_thread(self):
    # ...
    finally:
        # Garbage collection
        gc.collect()

        # Batch határ ellenőrzés
        if self.auto_batch_counter >= AUTO_EXEC_BATCH_SIZE:
            self.log(f"Batch boundary ({AUTO_EXEC_BATCH_SIZE} models) - deep cleanup...")
            gc.collect(0)
            gc.collect(1)
            gc.collect(2)
            self.auto_batch_counter = 0
```

---

*Dokumentum készítése: 2024.12.29*
*Program verzió: v3.60.0 Stable*
*Fejezet: 05 - Auto Execution Window*
