# 07. Comparison Tab - Modellek Összehasonlítása

## 07.00. Comparison Tab Fő Funkciója

A **Comparison Tab** célja, hogy több különböző elemzés eredményeit összehasonlítsa és vizualizálja. Lehetővé teszi a felhasználó számára, hogy:

1. **Konszenzust találjon** - Mely stratégiákat választják a legtöbb modellek
2. **Modelleket hasonlítson össze** - Mely modellek adnak hasonló eredményeket
3. **Stabilitást mérjen** - Melyik modell a legstabilabb időben
4. **Aggregált riportokat generáljon** - HTML és MD formátumban

### Felhasználói Felület Szerkezete

```
+------------------------------------------------------------------+
| Controls Frame                                                    |
| [Select Reports Folder:]                                         |
| [Select Folder] [1 Week] [1 Month] [3 Months] [6 Months] [1 Year]|
| [Main Data] [?] [Main Data AR] [?]                               |
| Selected: /path/to/reports                                       |
+------------------------------------------------------------------+
|                                                                  |
|                    Compare Log Area                              |
|              (Feldolgozási napló megjelenítése)                  |
|                                                                  |
+------------------------------------------------------------------+
```

### Riport Típusok Összefoglalása

| Gomb | Riport Típus | Bemenet | Kimenet |
|------|-------------|---------|---------|
| **1 Week - 1 Year** | Horizon Comparison | Normál .md riportok | `comparison_report_{horizon}.html` |
| **Main Data** | Main Data Comparison | Normál .md riportok (AR_ nélkül) | `main_data_comparison.html/.md` |
| **Main Data AR** | AR Comparison | AR_ prefix .md riportok | `main_data_ar_comparison.html/.md` |

### Fájl Szűrési Logika

```python
# Normál riportok keresése (Horizon és Main Data gombok)
def scan_reports(root_folder, exclude_ar=True):
    excluded_files = {"README.md", "main_data_comparison.md",
                      "main_data_ar_comparison.md"}

    for file in files:
        if file.endswith(".md") and file not in excluded_files:
            if exclude_ar and file.startswith("AR_"):
                continue  # AR_ fájlok kihagyása

            # Tartalomalapú ellenőrzés
            if is_aggregate_report(full_path):
                continue  # Aggregált riportok kihagyása

            report_files.append(full_path)
```

---

## 07.01. Select Folder Gomb

A **Select Folder** gomb a riportokat tartalmazó mappa kiválasztására szolgál.

### Funkciók

1. **Mappa választó dialógus** megnyitása
2. **Kiválasztott útvonal megjelenítése** a label-en
3. **Perzisztencia**: Az utoljára választott mappa mentésre kerül

### Működés

```python
# src/gui/tabs/compare_tab.py:145-155
def select_compare_folder(self):
    """Select folder for comparison reports."""
    get_sound_manager().play_button_click()
    initial_dir = getattr(self, "last_comparison_folder", "") or None
    folder_selected = filedialog.askdirectory(initialdir=initial_dir)

    if folder_selected:
        self.last_comparison_folder = folder_selected
        self.save_window_state()  # Azonnali mentés
        self.selected_compare_path.configure(text=folder_selected, text_color="white")
        self.compare_folder = folder_selected
        self.compare_log.insert("end", f"Selected folder: {folder_selected}\n")
```

### Elvárt Mappastruktúra

A kiválasztott mappának tartalmaznia kell az elemzési riportokat almappákban:

```
Reports/
├── LightGBM_EURUSD_Standard/
│   ├── LightGBM_Analysis_20240115.md
│   ├── LightGBM_Analysis_20240115.html
│   └── AR_LightGBM_Analysis_20240115.md  <- AR riport
├── XGBoost_EURUSD_Standard/
│   ├── XGBoost_Analysis_20240115.md
│   └── AR_XGBoost_Analysis_20240115.md
├── ARIMA_EURUSD_Standard/
│   └── ARIMA_Analysis_20240115.md
└── ...
```

A rendszer rekurzívan végigmegy az összes almappán.

---

## 07.02. Időhorizont Gombok (1 Week - 1 Year)

Az **1 Week**, **1 Month**, **3 Months**, **6 Months**, **1 Year** gombok horizont-specifikus összehasonlító riportokat generálnak.

### Elérhető Horizontok

| Gomb | Horizon Kulcs | Leírás |
|------|---------------|--------|
| **1 Week** | `1 Week` | 1 hetes előrejelzések összehasonlítása |
| **1 Month** | `1 Month` | 1 hónapos előrejelzések összehasonlítása |
| **3 Months** | `3 Months` | 3 hónapos előrejelzések összehasonlítása |
| **6 Months** | `6 Months` | 6 hónapos előrejelzések összehasonlítása |
| **1 Year** | `1 Year` | 1 éves előrejelzések összehasonlítása |

### Feldolgozási Folyamat

```
[Horizont gomb kattintás (pl. "1 Month")]
        │
        ▼
[scan_reports(): Rekurzív .md fájl keresés]
        │
        ▼
[Minden riporthoz: parse_report(file, horizon)]
        │
        ├── Method kinyerése
        ├── Best Strategy kinyerése az adott horizontra
        └── Top 10 lista kinyerése az adott horizontra
        │
        ▼
[generate_html_report(): Aggregált riport generálása]
        │
        ├── Statisztikák számítása
        ├── 12 interaktív grafikon generálása
        └── HTML fájl írása
        │
        ▼
[comparison_report_{horizon}.html]
```

### Riport Parsing Logika

```python
# src/analysis/comparator/horizon.py:18-111
def parse_report(file_path: str, horizon: str = "1 Month") -> Dict[str, Any]:
    """
    Parses a single markdown report to extract Method, Top Strategy, and Top 10.
    """
    data = {
        "file": os.path.basename(file_path),
        "path": file_path,
        "method": "Unknown",
        "top_strategy": None,
        "top_10": []
    }

    # 1. Method kinyerése
    # Keresés: **Method**: {method_name}

    # 2. Best Strategy kinyerése az adott horizontra
    # Keresés a "### Best Strategy Forecast" szekcióban
    # pl. "- **1 Month**: (No. 123) 580.50"

    # 3. Top 10 lista kinyerése
    # Keresés: "## Top 10 Strategies by Horizon" táblázat
    # Az adott horizont oszlopából kinyeri a stratégia ID-kat
```

### Generált Grafikonok (12 db)

| # | Grafikon | Leírás |
|---|----------|--------|
| 1 | **Top Strategy Consensus** | Mely stratégiákat választották legtöbbször győztesnek |
| 2 | **Top 10 Consistency** | Hányszor jelenik meg egy stratégia a Top 10-ben |
| 3 | **Method Performance (Pie)** | Győztes módszerek eloszlása |
| 4 | **Weighted Score Leaderboard** | Súlyozott pontszám (1. hely = 10 pont, 10. hely = 1 pont) |
| 5 | **Rank Volatility (Box)** | Stratégiák helyezéseinek szórása |
| 6 | **Consensus Confidence** | Egyetértés %-ban kifejezve |
| 7 | **Method Similarity Heatmap** | Jaccard index a módszerek Top 10 listái között |
| 8 | **Method → Strategy Sankey** | Sankey diagram a módszer-stratégia kapcsolatokról |
| 9 | **Rank Stability Line** | Helyezések alakulása módszerenként |
| 10 | **Method Agreement Matrix** | Módszerek közötti egyetértés % |
| 11 | **Strategy Coverage (Upset)** | Hány módszer választotta az adott stratégiát |
| 12 | **Top 3 Overlap Analysis** | Top 3 pozíciók átfedése |

### Matematikai Számítások

#### Weighted Score (Súlyozott Pontszám)

```python
# A Top 10 listában elfoglalt pozíció alapján
strategy_scores = {}
for report in parsed_data:
    for rank, strat in enumerate(report["top_10"]):
        if rank >= 10:
            break
        points = 10 - rank  # 1. hely = 10 pont, ..., 10. hely = 1 pont
        strategy_scores[strat] = strategy_scores.get(strat, 0) + points
```

**Példa**:
```
Strategy #123:
  - LightGBM: 1. hely → 10 pont
  - XGBoost: 3. hely → 8 pont
  - ARIMA: 2. hely → 9 pont
  Weighted Score = 10 + 8 + 9 = 27
```

#### Jaccard Similarity Index

A módszerek Top 10 listáinak hasonlóságát méri:

```python
# Jaccard Index = |A ∩ B| / |A ∪ B|
for m1 in unique_methods:
    for m2 in unique_methods:
        set1 = method_sets[m1]  # m1 Top 10 stratégiái
        set2 = method_sets[m2]  # m2 Top 10 stratégiái

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        jaccard = intersection / union if union > 0 else 0
```

**Értelmezés**:
- 1.0 = Tökéletesen egyező Top 10 listák
- 0.0 = Teljesen különböző listák
- 0.5 = 50% átfedés

#### Consensus Confidence

```python
# Hány modell választotta ugyanazt a stratégiát
total_methods = len(unique_methods)
for strat, count in winner_counts.most_common(15):
    confidence = (count / total_methods) * 100
```

**Példa**:
```
10 modell közül 7 választotta a #123 stratégiát
Confidence = (7 / 10) * 100 = 70%
```

### Kimenet

A generált HTML fájl neve: `comparison_report_{horizon}.html`

Példák:
- `comparison_report_1_week.html`
- `comparison_report_1_month.html`
- `comparison_report_3_months.html`
- `comparison_report_6_months.html`
- `comparison_report_1_year.html`

---

## 07.03. Main Data Gomb

A **Main Data** gomb egy átfogó összehasonlító riportot generál az összes normál (nem AR_) riportból.

### Célja

- Minden elemzési riport metaadatainak összegyűjtése
- Stratégia konszenzus vizualizálása
- Futási idők és modell komplexitás elemzése

### Feldolgozási Folyamat

```
[Main Data kattintás]
        │
        ▼
[scan_reports(exclude_ar=True)]
        │  ↳ AR_ prefix fájlok kihagyása
        │
        ▼
[Minden riporthoz: parse_main_data_report()]
        │
        ├── Filename: Fájlnév (extension nélkül)
        ├── Method: **Method**: érték
        ├── Ranking Mode: **Ranking Mode**: érték
        ├── Generating Time: **Generating time**: érték
        ├── Model Settings: ## Model Settings táblázat
        └── Best Strategy: "The top performing strategy..."
        │
        ▼
[generate_main_data_report()]
        │
        ├── 6 interaktív grafikon
        ├── Szűrhető, rendezhető táblázat
        └── MD és HTML kimenet
        │
        ▼
[main_data_comparison.md / .html]
```

### Kinyert Adatok

```python
# src/analysis/comparator/main_data.py:21-90
def parse_main_data_report(file_path: str) -> Dict[str, Any]:
    data = {
        "Filename": os.path.splitext(os.path.basename(file_path))[0],
        "Method": "Unknown",
        "Ranking Mode": "Unknown",
        "Generating Time": "Unknown",
        "Model Settings": "",
        "Best Strategy": "Unknown",
    }

    # Regex alapú kinyerés
    method_match = re.search(r"\*\*Method\*\*: (.+)", content)
    ranking_match = re.search(r"\*\*Ranking Mode\*\*: (.+)", content)
    time_match = re.search(r"\*\*Generating time\*\*: (.+)", content)
    strat_match = re.search(
        r"The top performing strategy for the next month is "
        r"\*\*(Strategy No\. \d+)\*\*", content
    )

    # Model Settings táblázat parsing
    # | Parameter | Value | formátumból
```

### Generált Grafikonok (6 db)

| # | Grafikon | Leírás |
|---|----------|--------|
| 1 | **Strategy Consensus Gauge** | Konszenzus erősség mérő (Top 1 és Top 3) |
| 2 | **Best Strategy Frequency** | Top 20 leggyakrabban választott stratégia |
| 3 | **Generating Time Distribution** | Futási idők box plot módszerenként |
| 4 | **Strategy Selection Heatmap** | Módszer × Stratégia heatmap |
| 5 | **Model Settings Impact** | Beállítások hatása a stratégia választásra |
| 6 | **Execution Time vs Complexity** | Futási idő vs paraméterszám scatter |

### Matematikai Számítások

#### Consensus Gauge (Konszenzus Mérő)

```python
# Top 1 konszenzus
total_reports = len(valid_reports)
most_common_strategy, most_common_count = strategy_counts.most_common(1)[0]
consensus_percentage = (most_common_count / total_reports) * 100

# Top 3 kombinált konszenzus
top3_count = sum([c for _, c in strategy_counts.most_common(3)])
top3_percentage = (top3_count / total_reports) * 100
```

**Értelmezés**:
- **Top 1 > 50%**: Erős konszenzus
- **Top 1 25-50%**: Közepes konszenzus
- **Top 1 < 25%**: Gyenge konszenzus

#### Idő Konverzió

```python
# src/analysis/comparator/base.py:98-116
def parse_time_to_seconds(time_str: str) -> float:
    """Convert time string (HH:MM:SS or MM:SS) to seconds."""
    parts = time_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return 0
```

#### Komplexitás Mérés

```python
# Paraméterek száma a Model Settings alapján
param_count = settings.count(":")  # Kulcs: érték párok száma
```

### HTML Funkciók

- **Kereshető táblázat**: Szűrés fájlnév, módszer, stratégia szerint
- **Rendezhető oszlopok**: Kattintással növekvő/csökkenő rendezés
- **Tooltip-ok**: Hosszú cellák teljes tartalmának megjelenítése

### Kimenet

- `main_data_comparison.md` - Markdown formátum
- `main_data_comparison.html` - Interaktív HTML riport

---

## 07.04. Main Data AR Gomb

A **Main Data AR** gomb az All Results (AR_) prefix riportokat dolgozza fel, amelyek heti bontású előrejelzéseket tartalmaznak.

### Különbség a Main Data-tól

| Jellemző | Main Data | Main Data AR |
|----------|-----------|--------------|
| **Forrás fájlok** | Normál .md riportok | AR_ prefix .md riportok |
| **Tartalom** | 1 best stratégia/riport | Heti bontású stratégia lista |
| **Adatmennyiség** | ~6 mező/riport | 52 hét × stratégia/riport |
| **Elemzés fókusza** | Statikus összehasonlítás | Időbeli stabilitás |

### AR Riport Struktúra

Az AR_ riportok speciális formátumban tartalmazzák a heti előrejelzéseket:

```markdown
# All Results Report - LightGBM Analysis

**Method**: LightGBM
**Forecast Horizon**: 52
**Generating time**: 02:45:30

## Weekly Forecast Breakdown

| Week | Best Strategy | Predicted Profit |
|------|---------------|-----------------|
| Week 1 | No. 123 | 150.50 |
| Week 2 | No. 123 | 145.20 |
| Week 3 | No. 456 | 160.00 |
| ... | ... | ... |
| Week 52 | No. 123 | 120.00 |
```

### Feldolgozási Folyamat

```
[Main Data AR kattintás]
        │
        ▼
[scan_ar_reports(): AR_ prefix fájlok keresése]
        │
        ▼
[Minden AR riporthoz: parse_ar_report()]
        │
        ├── Method
        ├── Forecast Horizon
        ├── Generating Time
        └── Weekly Data (week, strategy_no, profit)
        │
        ▼
[Aggregáció és Metrikák Számítása]
        │
        ├── Weekly Consensus (heti konszenzus)
        ├── Strategy Total Appearances (összes megjelenés)
        ├── Method Stability (stabilitás = stratégia váltások száma)
        └── Method Weekly Data (heti adatok módszerenként)
        │
        ▼
[generate_main_data_ar_report()]
        │
        ├── 8 interaktív grafikon
        ├── 3 részletes táblázat
        └── MD és HTML kimenet
        │
        ▼
[main_data_ar_comparison.md / .html]
```

### Kinyert Adatok

```python
# src/analysis/comparator/main_data_ar.py:22-96
def parse_ar_report(file_path: str) -> Dict[str, Any]:
    data = {
        "file": os.path.basename(file_path),
        "path": file_path,
        "method": "Unknown",
        "forecast_horizon": 52,
        "generating_time": "N/A",
        "weekly_data": [],  # Lista: [{week, strategy_no, profit}, ...]
    }

    # Weekly Forecast Breakdown táblázat parsing
    # | Week | Best Strategy | Predicted Profit |
    # Minden sorból: week szám, stratégia ID, profit érték
```

### Aggregált Metrikák

#### Weekly Consensus (Heti Konszenzus)

```python
# Minden hétre: melyik stratégiát választották a legtöbben
weekly_consensus = {w: Counter() for w in range(1, max_weeks + 1)}

for report in parsed_reports:
    for week_data in report["weekly_data"]:
        week = week_data["week"]
        strat = week_data["strategy_no"]
        weekly_consensus[week][strat] += 1

# Konszenzus erősség számítása
for week in range(1, max_weeks + 1):
    best_strat, best_count = weekly_consensus[week].most_common(1)[0]
    consensus_pct = (best_count / total_reports) * 100
```

#### Method Stability (Modell Stabilitás)

```python
# Hány alkalommal váltott stratégiát a modell
method_stability = {}

for report in parsed_reports:
    method = report["method"]
    prev_strategy = None
    switch_count = 0

    for week_data in report["weekly_data"]:
        strat = week_data["strategy_no"]
        if prev_strategy is not None and strat != prev_strategy:
            switch_count += 1
        prev_strategy = strat

    method_stability[method] = switch_count
```

**Értelmezés**:
- **0 switch**: Tökéletesen stabil (ugyanaz a stratégia minden héten)
- **51 switch**: Maximálisan instabil (minden héten más stratégia)

#### Perfect Switching Comparison

```python
# "Tökéletes váltás" - mindig a legjobb stratégiát választjuk minden héten
perfect_switching_profit = []
for week in range(1, max_weeks + 1):
    best_profit_this_week = max(
        method_data[week]["profit"]
        for method_data in method_weekly_data.values()
        if week in method_data
    )
    perfect_switching_profit.append(best_profit_this_week)

# Kumulatív profit összehasonlítás
perfect_cumulative = cumsum(perfect_switching_profit)
avg_model_cumulative = mean([cumsum(model_profits) for model in models])

advantage = perfect_cumulative[-1] - avg_model_cumulative[-1]
advantage_pct = (advantage / abs(avg_model_cumulative[-1])) * 100
```

### Generált Grafikonok (8 db)

| # | Grafikon | Leírás |
|---|----------|--------|
| 1 | **Weekly Consensus Strength** | Konszenzus % hetente (zöld > 50%, sárga 25-50%, piros < 25%) |
| 2 | **Strategy Selection Heatmap** | Módszer × Hét mátrix (színkód: stratégia index) |
| 3 | **Model Stability Ranking** | Stabilitás rangsor (kevesebb váltás = jobb) |
| 4 | **Strategy Dominance Over Time** | Top 10 stratégia dominanciája időben (stacked area) |
| 5 | **Strategy Distribution Sunburst** | Hierarchikus eloszlás (Módszer → Stratégia) |
| 6 | **Perfect Switching Comparison** | Tökéletes váltás vs átlagos modell teljesítmény |
| 7 | **Strategy Appearance Frequency** | Top 20 leggyakoribb stratégia |
| 8 | **Method Agreement Matrix** | Módszerek közötti egyetértés % hetente |

### Táblázatok

#### 1. Weekly Consensus Details

| Week | Best Strategy | Votes | Consensus % |
|------|---------------|-------|-------------|
| Week 1 | #123 | 8 | 80.0% |
| Week 2 | #123 | 7 | 70.0% |
| ... | ... | ... | ... |

#### 2. Model Stability Ranking

| Method | Strategy Switches |
|--------|-------------------|
| ARIMA | 3 |
| LightGBM | 8 |
| XGBoost | 12 |
| ... | ... |

#### 3. Processed AR Reports

| File | Method | Weeks | Gen. Time |
|------|--------|-------|-----------|
| AR_LightGBM_... | LightGBM | 52 | 02:45:30 |
| AR_XGBoost_... | XGBoost | 52 | 03:12:15 |
| ... | ... | ... | ... |

### Summary Statistics (Összefoglaló)

A riport Executive Summary szekciója tartalmazza:

```
This report aggregates weekly forecast data from 10 AR reports.

The most frequently selected strategy across all weeks and models
is #123 with 420 total appearances.

The most stable model (fewest strategy switches) is ARIMA
with only 3 changes over 52 weeks.
```

### Kimenet

- `main_data_ar_comparison.md` - Markdown formátum
- `main_data_ar_comparison.html` - Interaktív HTML riport

---

## Összefoglaló: Comparison Tab Munkafolyamat

```
                    [Comparison Tab]
                           │
            ┌──────────────┴──────────────┐
            │                             │
      [Select Folder]               [Riport Típus]
            │                             │
            ▼                    ┌────────┴────────┐
    [Mappa kiválasztása]         │                 │
            │             ┌──────┴──────┐   ┌──────┴──────┐
            │             │   Horizon   │   │  Main Data  │
            │             │   Buttons   │   │   Buttons   │
            │             └──────┬──────┘   └──────┬──────┘
            │                    │                  │
            ▼                    ▼                  ▼
    [Rekurzív .md keresés]  [parse_report]   [parse_main_data]
            │                    │           [parse_ar_report]
            ▼                    ▼                  │
    [Parsed adatok]        [generate_html_report]  │
            │                    │                  ▼
            └────────────────────┴──────────────────┘
                                 │
                                 ▼
                    [HTML + MD riport generálás]
                                 │
                                 ▼
                    [Plotly interaktív grafikonok]
```

### Mikor Melyik Gombot Használjuk?

| Használati Eset | Ajánlott Gomb |
|-----------------|---------------|
| Adott horizont konszenzus keresése | **1 Week - 1 Year** |
| Modellek átfogó összehasonlítása | **Main Data** |
| Heti stabilitás elemzése | **Main Data AR** |
| Futási idők elemzése | **Main Data** |
| "Melyik modell a legstabilabb?" | **Main Data AR** |
| "Melyik stratégiát választják leggyakrabban?" | **1 Month** vagy **Main Data** |

### Tippek

1. **Több elemzés futtatása**: A Comparison Tab hatékonyságához több különböző modellel kell elemzést futtatni
2. **AR riportok**: Az Auto Execution során automatikusan generálódnak AR_ riportok
3. **Mappa struktúra**: A riportok különböző almappákban is lehetnek, a rendszer rekurzívan keresi őket
4. **Konszenzus értelmezése**: Magas konszenzus (>50%) erős jelzés, alacsony (<25%) bizonytalan

---

*Dokumentáció generálva: 2024*
*Forrásfájlok: `src/gui/tabs/compare_tab.py`, `src/analysis/comparator/base.py`, `src/analysis/comparator/horizon.py`, `src/analysis/comparator/main_data.py`, `src/analysis/comparator/main_data_ar.py`*
