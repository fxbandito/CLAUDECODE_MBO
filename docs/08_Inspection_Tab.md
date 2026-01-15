# 08. Inspection Tab - Előrejelzések Validálása

## 08.00. Inspection Tab Fő Funkciója

Az **Inspection Tab** célja az előrejelzési modellek pontosságának validálása tényleges benchmark adatokkal. Ez a funkció lehetővé teszi a felhasználó számára, hogy:

1. **Összehasonlítsa** a modellek előrejelzéseit a valós eredményekkel
2. **Mérje** a helyezés-eltérést (Rank Deviation)
3. **Azonosítsa** a legjobb teljesítményű modelleket
4. **Generáljon** részletes validációs riportokat

### Felhasználói Felület Szerkezete

```
+------------------------------------------------------------------+
| Controls Frame                                                    |
| Reports Folder: [_________________] [Browse]  [Run Inspection]   |
| Benchmark Excel: [________________] [Browse]  [Run Inspection AR]|
|                                     [========= Progress =========]|
+------------------------------------------------------------------+
|                                                                  |
|                    Results Table (Treeview)                      |
| Model | Forecast Year | Horizon | Pred Rank | Act Rank | Diff   |
| ...   | ...           | ...     | ...       | ...      | ...    |
|                                                                  |
+------------------------------------------------------------------+
| [Export Report]                                                  |
+------------------------------------------------------------------+
```

### Táblázat Oszlopai

| Oszlop | Leírás |
|--------|--------|
| **Model** | Az előrejelzési modell neve (pl. LightGBM, ARIMA) |
| **Forecast Year** | Az előrejelzés céléve |
| **Horizon** | Időhorizont (1 Week, 1 Month, 3 Months, 6 Months, 1 Year) |
| **Predicted Rank** | Az előrejelzett helyezés |
| **Actual Rank** | A tényleges helyezés a benchmark alapján |
| **Rank Diff** | Helyezés eltérés (|Predicted - Actual|) |
| **Predicted Profit** | Előrejelzett profit |
| **Actual Profit** | Tényleges profit a benchmark-ból |
| **Actual Winner** | Ki lett a tényleges győztes stratégia |
| **Winner Profit** | A győztes tényleges profitja |

### Működési Elv

```
[Elemzési Riportok (.md)]     [Benchmark Excel]
         │                            │
         ▼                            ▼
    [parse_markdown_reports]    [parse_benchmark_excel]
         │                            │
         └──────────┬─────────────────┘
                    │
                    ▼
            [compare_results]
                    │
                    ▼
        [Rank Deviation, Top 1 Accuracy, stb.]
                    │
                    ▼
        [HTML + MD Riport Generálás]
```

---

## 08.01. Reports Folder Gomb

A **Reports Folder** mező és **Browse** gomb az elemzési riportokat tartalmazó mappa kiválasztására szolgál.

### Működés

```python
# src/gui/tabs/inspection_tab.py:149-158
def browse_reports(self):
    """Browse for reports folder."""
    get_sound_manager().play_button_click()
    initial_dir = getattr(self, "last_inspection_folder", "") or None
    path = filedialog.askdirectory(title="Select Reports Folder", initialdir=initial_dir)

    if path:
        self.last_inspection_folder = path
        self.save_window_state()  # Azonnali mentés
        self.entry_reports_path.delete(0, "end")
        self.entry_reports_path.insert(0, path)
```

### Elvárt Mappastruktúra

A kiválasztott mappának Markdown (.md) riportokat kell tartalmaznia:

```
Reports/
├── LightGBM_EURUSD_Standard/
│   ├── LightGBM_Analysis_EURUSD_(2020-2023)_20240115.md
│   └── AR_LightGBM_Analysis_EURUSD_(2020-2023)_20240115.md
├── XGBoost_EURUSD_Standard/
│   ├── XGBoost_Analysis_EURUSD_(2020-2023)_20240115.md
│   └── AR_XGBoost_Analysis_EURUSD_(2020-2023)_20240115.md
└── ...
```

### Fontos Fájlnév Konvenció

A rendszer a fájlnévből nyeri ki a metaadatokat:

```
{Model}_Analysis_{Pair}_{(TrainingYears)}_{Date}.md

Példa: LightGBM_Analysis_EURUSD_(2020-2021-2022-2023)_20240115.md

Kinyert adatok:
- pair: "EURUSD"
- years: "2020-2021-2022-2023"
- forecast_year: 2024 (last_training_year + 1)
```

---

## 08.02. Benchmark Excel Gomb

A **Benchmark Excel** mező és **Browse** gomb a tényleges teljesítményadatokat tartalmazó Excel fájl kiválasztására szolgál.

### Működés

```python
# src/gui/tabs/inspection_tab.py:160-171
def browse_benchmark(self):
    """Browse for benchmark Excel file."""
    get_sound_manager().play_button_click()
    path = filedialog.askopenfilename(
        title="Select Benchmark Excel",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )

    if path:
        self.entry_benchmark_path.delete(0, "end")
        self.entry_benchmark_path.insert(0, path)
```

### Excel Fájl Szerkezete

A benchmark Excel fájlnak speciális struktúrát kell követnie:

```
+----------+----------+----------+----------+---+----------+----------+
| Date_W1  |          |          |          |...|Date_W52  |          |
+----------+----------+----------+----------+---+----------+----------+
| No.      | Profit   | Trades   | PF       |...| No.      | Profit   |
+----------+----------+----------+----------+---+----------+----------+
| 123      | 150.50   | 5        | 1.5      |...| 123      | 120.00   |
| 456      | -50.20   | 3        | 0.8      |...| 456      | 200.30   |
| ...      | ...      | ...      | ...      |...| ...      | ...      |
+----------+----------+----------+----------+---+----------+----------+

- 10 oszlop per hét (No., Profit, Trades, PF, DD%, stb.)
- A fejléc (0. sor) tartalmazza a dátumokat
- Az 1. sor tartalmazza az oszlop neveket (No., Profit, stb.)
- Az adatok a 2. sortól kezdődnek
```

### Beolvasási Logika

```python
# src/analysis/inspection.py:252-375
def parse_benchmark_excel(self, file_path, progress_callback=None):
    """
    Reads the Benchmark Excel file and calculates actual profits/ranks.
    Assumes 10 columns per week.
    """
    df = pd.read_excel(file_path)

    n_cols = df.shape[1]
    n_weeks = n_cols // 10  # 10 oszlop per hét

    weekly_profits = []

    for w in range(n_weeks):
        start_col = w * 10

        # Strategy és Profit oszlopok kinyerése
        week_df = df.iloc[1:, [start_col, start_col + 1]].copy()
        week_df.columns = ["Strategy", "Profit"]

        # Adattisztítás
        week_df["Strategy"] = week_df["Strategy"].astype(str).str.replace("#", "")
        week_df["Profit"] = pd.to_numeric(week_df["Profit"], errors="coerce").fillna(0)

        weekly_profits.append(week_df)
```

### Horizont Aggregáció

A különböző időhorizontokhoz a hetek összegzése:

| Horizont | Hetek | Aggregáció |
|----------|-------|------------|
| **1 Week** | 1. hét | week_1 |
| **1 Month** | 1-4. hét | sum(week_1 : week_4) |
| **3 Months** | 1-13. hét | sum(week_1 : week_13) |
| **6 Months** | 1-26. hét | sum(week_1 : week_26) |
| **1 Year** | 1-52. hét | sum(week_1 : week_52) |

```python
def aggregate_weeks(start_week, end_week):
    """Aggregate profits across weeks for a given horizon."""
    relevant_dfs = weekly_profits[start_week - 1 : end_week]

    big_df = pd.concat(relevant_dfs, ignore_index=True)
    merged = big_df.groupby("Strategy", as_index=False)["Profit"].sum()
    merged.rename(columns={"Profit": "Actual_Profit"}, inplace=True)

    return merged

# Horizont benchmark-ok generálása
benchmarks["1 Week"] = aggregate_weeks(1, 1)
benchmarks["1 Month"] = aggregate_weeks(1, 4)     # 4 hét
benchmarks["3 Months"] = aggregate_weeks(1, 13)   # 13 hét
benchmarks["6 Months"] = aggregate_weeks(1, 26)   # 26 hét
benchmarks["1 Year"] = aggregate_weeks(1, 52)     # 52 hét
```

### Actual Rank Számítása

```python
# Minden horizonthoz rangsor generálása
if not b_1m.empty:
    b_1m["Actual_Rank"] = b_1m["Actual_Profit"].rank(
        ascending=False,  # Magasabb profit = jobb helyezés
        method="min"      # Azonos profit = azonos helyezés
    )
```

**Példa**:
```
Strategy | Actual_Profit | Actual_Rank
---------|---------------|-------------
  123    |    5000.00    |     1.
  456    |    4500.00    |     2.
  789    |    4500.00    |     2.   (azonos profit = azonos rank)
  012    |    4000.00    |     4.
```

---

## 08.03. Run Inspection Funkció

A **Run Inspection** gomb a standard (nem AR_) riportok validálását végzi.

### Teljes Feldolgozási Folyamat

```
[Run Inspection kattintás]
        │
        ▼
[Validáció: mindkét bemenet megadva?]
        │
        ▼
[Háttérszál indítása]
        │
        ├── [1. parse_markdown_reports(mode="standard")]
        │       - AR_ fájlok kizárása
        │       - Top 10 táblázat parsing
        │
        ├── [2. parse_benchmark_excel()]
        │       - Heti adatok aggregálása
        │       - Rank számítás
        │
        ├── [3. compare_results()]
        │       - Merge előrejelzés + benchmark
        │       - Rank Deviation számítás
        │       - Winner azonosítás
        │
        ├── [4. Riport generálás]
        │       - HTML + MD automatikus mentés
        │
        └── [5. GUI frissítés]
                - Treeview táblázat
                - Log üzenetek
```

### Markdown Riport Parsing

```python
# src/analysis/inspection.py:22-250
def parse_markdown_reports(self, folder_path, mode="standard", progress_callback=None):
    """
    Scans a folder for Markdown reports and extracts forecast data.
    mode: 'standard' (exclude AR_) or 'ar' (only AR_)
    """
    md_files = glob.glob(os.path.join(folder_path, "**/*.md"), recursive=True)

    # Szűrés mód alapján
    filtered_files = []
    for f in md_files:
        fname = os.path.basename(f)
        if mode == "standard":
            if not fname.startswith("AR_"):
                filtered_files.append(f)
        elif mode == "ar":
            if fname.startswith("AR_"):
                filtered_files.append(f)
```

### Top 10 Táblázat Parsing

A riportokból a "Top 10 Strategies by Horizon" szekciót olvassa:

```python
# Táblázat keresése
table_section = re.search(
    r"## Top 10 Strategies by Horizon\s+(.*?)\n\n",
    content, re.DOTALL
)

# Cella értelmezése: "#123 (150.50)" formátum
def parse_cell(cell_text):
    m = re.search(r"#(\d+).*?\(.*?([\d,.-]+)\)", cell_text)
    if m:
        return m.group(1), float(m.group(2).replace(",", ""))
    return None, None

# Horizont oszlopok:
# | Rank | 1 Week | 1 Month | 3 Months | 6 Months | 1 Year |
# | 1    | #123 (150) | #456 (580) | ...
```

### Compare Results - Matematikai Logika

```python
# src/analysis/inspection.py:377-432
def compare_results(self, reports_df, benchmarks, progress_callback=None):
    """
    Merges forecast data with benchmark data to calculate accuracy metrics.
    """
    for horizon, bench_df in benchmarks.items():
        horizon_reports = reports_df[reports_df["Horizon"] == horizon].copy()

        # Merge: előrejelzés + benchmark
        merged = pd.merge(horizon_reports, bench_df, on="Strategy", how="left")

        # Győztes azonosítása (Rank 1)
        winners = bench_df[bench_df["Actual_Rank"] == 1]
        merged["Winner_Strategy"] = winners.iloc[0]["Strategy"]
        merged["Winner_Profit"] = winners.iloc[0]["Actual_Profit"]

        # Rank Deviation számítása
        merged["Rank_Deviation"] = abs(
            merged["Predicted_Rank"] - merged["Actual_Rank"]
        )

        # Profit különbség
        merged["Profit_Diff"] = (
            merged["Predicted_Profit"] - merged["Actual_Profit"]
        )
```

### Rank Deviation Értelmezése

| Rank Deviation | Jelentés |
|----------------|----------|
| **0** | Tökéletes előrejelzés - pont eltalálta a helyezést |
| **1-5** | Jó előrejelzés - közel volt |
| **6-20** | Közepes előrejelzés |
| **> 20** | Gyenge előrejelzés |

**Példa**:
```
Predicted Rank: 1 (modell szerint ez lesz a legjobb)
Actual Rank: 5 (valójában az 5. lett)
Rank Deviation: |1 - 5| = 4
```

### Summary Statistics

A riportban megjelenő összefoglaló metrikák:

```python
# Összes előrejelzés száma
total_predictions = len(comparison_df)

# Átlagos helyezés-eltérés
avg_rank_deviation = comparison_df["Rank_Deviation"].mean()

# Top 1 pontosság: Hány esetben lett tényleg 1. a Top 1 előrejelzés
predicted_top1 = comparison_df[comparison_df["Predicted_Rank"] == 1]
top1_hits = predicted_top1[predicted_top1["Actual_Rank"] == 1]
top1_accuracy = len(top1_hits) / len(predicted_top1) * 100

# Top 10 találatok: Hány Top 10 előrejelzés került valóban Top 10-be
top10_hits = len(comparison_df[
    (comparison_df["Predicted_Rank"] <= 10) &
    (comparison_df["Actual_Rank"] <= 10)
])
```

### Generált Grafikonok

| # | Grafikon | Leírás |
|---|----------|--------|
| 1 | **Predicted vs Actual Rank** | Scatter plot, horizontonként színezve |
| 2 | **Actual Profit of Top 1** | Bar chart - a Top 1 előrejelzések tényleges profitja |
| 3 | **Rank Deviation Distribution** | Box plot - eltérés eloszlása modellenkét |

### Kimenet

Automatikusan generált fájlok:
- `Inspect_{Pair}_{(Years)}_{BenchmarkYear}.html`
- `Inspect_{Pair}_{(Years)}_{BenchmarkYear}.md`

---

## 08.04. Run Inspection AR Funkció

A **Run Inspection AR** gomb az All Results (AR_) riportok validálását végzi.

### Különbség a Standard Inspection-től

| Jellemző | Standard | AR |
|----------|----------|-----|
| **Forrás fájlok** | Normál .md (AR_ nélkül) | AR_ prefix .md |
| **Parsing mód** | Top 10 táblázat | Best Strategy Forecast szekció |
| **Kimenet prefix** | `Inspect_` | `AR_Inspect_` |

### AR Riport Parsing

Az AR riportokban különböző szekciókból nyeri ki az adatokat:

```python
# 1. Best Strategy Forecast szekció
# Format: "- **1 Week**: (No. 6127) 185.91"
best_forecast_matches = re.finditer(
    r"\*\*(\d+\s*(?:Week|Month|Months|Year)s?)\*\*:\s*"
    r"\(No\.\s*(\d+)\)\s*([\d,.-]+)",
    content,
)

# Horizont normalizálása
horizon_map = {
    "Week": "1 Week",
    "Month": "1 Month",
    "3 Months": "3 Months",
    "6 Months": "6 Months",
    "Year": "1 Year"
}

# 2. Fallback: Executive Summary
# "The overall best strategy (1 Month horizon) is **Strategy No. 4941**."
summary_matches = re.finditer(
    r"overall best strategy \((.*?) horizon\) is "
    r"\*\*Strategy No\. (\d+)\*\*",
    content,
)

# 3. Weekly Forecast Breakdown (Week 1 only)
# "| Week 1 | No. 123 | 150.50 |"
week1_match = re.search(
    r"\| Week 1 \| No\. (\d+) \|\s*([\d\.-]+)",
    content
)
```

### Folyamat

```
[Run Inspection AR kattintás]
        │
        ▼
[parse_markdown_reports(mode="ar")]
        │  ↳ Csak AR_ prefix fájlok
        │  ↳ Best Strategy Forecast parsing
        │
        ▼
[parse_benchmark_excel()]
        │  ↳ Ugyanaz mint standard
        │
        ▼
[compare_results()]
        │  ↳ Ugyanaz mint standard
        │
        ▼
[Riport generálás AR_ prefixszel]
        │
        ▼
[AR_Inspect_{Pair}_{(Years)}_{BenchmarkYear}.html]
```

### Fájlnév Generálás

```python
# src/analysis/inspection.py:833-851
def generate_auto_filename(self, metadata, excel_year, mode="standard"):
    """
    Generates filename: Inspect_Pair_(Years)_ExcelYear.html
    For AR mode: AR_Inspect_Pair_(Years)_ExcelYear.html
    """
    pair = metadata.get("pair", "UnknownPair")
    years = metadata.get("years", "UnknownYears")

    # AR prefix hozzáadása AR módban
    prefix = "AR_Inspect" if mode == "ar" else "Inspect"

    filename = f"{prefix}_{pair}_{years}_{excel_year}.html"
    return filename
```

---

## 08.05. Export Report Gomb

Az **Export Report** gomb lehetővé teszi a validációs eredmények manuális exportálását.

### Működés

```python
# src/gui/tabs/inspection_tab.py:260-285
def export_inspection_report(self):
    """Export inspection results to HTML and Markdown."""
    get_sound_manager().play_button_click()

    if not hasattr(self, "inspection_results") or self.inspection_results.empty:
        messagebox.showwarning("Warning", "No inspection results to export.")
        return

    path = filedialog.asksaveasfilename(
        defaultextension=".html",
        filetypes=[("HTML files", "*.html")]
    )

    if path:
        engine = InspectionEngine()

        # HTML generálás
        engine.generate_html_report(self.inspection_results, path)

        # MD generálás (automatikusan .md extension)
        md_path = path.rsplit(".", 1)[0] + ".md"
        engine.generate_markdown_report(self.inspection_results, md_path)
```

### HTML Riport Tartalom

A generált HTML riport tartalma:

#### 1. Summary Statistics

```html
<div class="summary-stats">
    <div class="stat-box">
        <h3>150</h3>
        <p>Total Predictions</p>
    </div>
    <div class="stat-box highlight">
        <h3>4.2</h3>
        <p>Avg Rank Deviation</p>
    </div>
    <div class="stat-box success">
        <h3>12.5%</h3>
        <p>Top 1 Accuracy</p>
    </div>
    <div class="stat-box success">
        <h3>85</h3>
        <p>Top 10 Hits</p>
    </div>
</div>
```

#### 2. Interaktív Grafikonok (Plotly)

- **Predicted vs Actual Rank**: Scatter plot facetted by Horizon
- **Actual Profit of Top 1**: Bar chart grouped by Model × Horizon
- **Rank Deviation Distribution**: Box plot by Horizon, colored by Model

#### 3. Részletes Táblázatok Modellenkét

```html
<div class="card">
    <h2>Model: LightGBM</h2>
    <table>
        <thead>
            <tr>
                <th>Forecast Year</th>
                <th>Horizon</th>
                <th>Predicted Rank</th>
                <th>Strategy</th>
                <th>Predicted Profit</th>
                <th>Actual Rank</th>
                <th>Actual Profit</th>
                <th>Rank Deviation</th>
                <th>Winner Strategy</th>
                <th>Winner Profit</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>2024</td>
                <td>1 Month</td>
                <td>1</td>
                <td>123</td>
                <td>$580.50</td>
                <td>3.</td>
                <td>$520.00</td>
                <td>2</td>
                <td>456</td>
                <td>$650.00</td>
            </tr>
            ...
        </tbody>
    </table>
</div>
```

### Markdown Riport Struktúra

```markdown
# Model Inspection Report

**Generated:** 2024-01-15 14:30
**Pair:** EURUSD
**Training Data:** 2020-2021-2022-2023
**Benchmark Year:** 2024

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Predictions | 150 |
| Avg Rank Deviation | 4.2 |
| Top 1 Accuracy | 12.5% |
| Top 10 Hits | 85 |

## Detailed Results by Model

### Model: LightGBM

| Forecast_Year | Horizon | Predicted_Rank | Strategy | ... |
|---------------|---------|----------------|----------|-----|
| 2024 | 1 Month | 1 | 123 | ... |
| 2024 | 3 Months | 1 | 123 | ... |
...

### Model: XGBoost
...
```

---

## Összefoglaló: Inspection Tab Munkafolyamat

```
                    [Inspection Tab]
                           │
        ┌──────────────────┴──────────────────┐
        │                                      │
[Reports Folder]                     [Benchmark Excel]
(.md riportok)                       (Tényleges adatok)
        │                                      │
        ▼                                      ▼
[parse_markdown_reports]           [parse_benchmark_excel]
        │                                      │
        │   ┌────────────────────────────────┐ │
        └──>│          compare_results       │<┘
            │                                │
            │  - Merge on Strategy ID        │
            │  - Rank Deviation = |P - A|    │
            │  - Winner identification       │
            │                                │
            └────────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────┴──────────────┐
              │      Summary Statistics      │
              │                              │
              │  - Total Predictions         │
              │  - Avg Rank Deviation        │
              │  - Top 1 Accuracy            │
              │  - Top 10 Hits               │
              └──────────────┬───────────────┘
                             │
                             ▼
              [HTML + MD Riport Generálás]
```

### Használati Esetek

| Eset | Teendő |
|------|--------|
| **Modell validálás** | Benchmark Excel alapján ellenőrzés |
| **Modell összehasonlítás** | Több modell Rank Deviation összehasonlítása |
| **Best Model kiválasztás** | Legalacsonyabb Avg Rank Deviation keresése |
| **Horizont analízis** | Mely horizonton a legpontosabb az előrejelzés |

### Metrikák Értelmezése

| Metrika | Jó érték | Rossz érték |
|---------|----------|-------------|
| **Avg Rank Deviation** | < 5 | > 20 |
| **Top 1 Accuracy** | > 20% | < 5% |
| **Top 10 Hits** | > 50% | < 20% |

### Tippek

1. **Benchmark fájl**: Mindig az előrejelzés utáni év adatait használja
2. **Training years**: A riport fájlnévben lévő évek a training adatok
3. **Forecast year**: Automatikusan számítódik (last_training_year + 1)
4. **AR vs Standard**: AR riportok részletesebb heti bontást tartalmaznak

---

*Dokumentáció generálva: 2024*
*Forrásfájlok: `src/gui/tabs/inspection_tab.py`, `src/analysis/inspection.py`*
