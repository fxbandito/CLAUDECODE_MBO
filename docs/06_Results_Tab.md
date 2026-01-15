# 06. Results Tab - Eredmények és Riportok

## 06.00. Results Tab Fő Funkciója

A **Results Tab** a program elemzési eredményeinek megjelenítésére, rangsorolására és exportálására szolgál. Ez a tab automatikusan aktiválódik az elemzés befejezése után, és lehetővé teszi a felhasználó számára, hogy:

1. **Megtekintse az eredményeket** - A stratégiák előrejelzéseit táblázatos formában
2. **Rangsorolja a stratégiákat** - Különböző súlyozási módokkal
3. **Riportokat generáljon** - MD és HTML formátumban
4. **Exportálja az adatokat** - CSV fájlba további elemzéshez
5. **Betöltsön korábbi elemzéseket** - .pkl (pickle) fájlokból

### Felhasználói Felület Szerkezete

```
+------------------------------------------------------------------+
| Ranking Controls Frame                                            |
| [Ranking Mode: ▼ Standard] [Description]  [Apply Ranking] [?]    |
+------------------------------------------------------------------+
| Output Controls Frame                                             |
| [Output Folder:] [_________________] [Browse] [Generate Report]  |
| [Export CSV] [All Results]          [Load Analysis] [Show Params]|
+------------------------------------------------------------------+
|                                                                  |
|                    Results Text Area                             |
|              (Táblázatos eredmény megjelenítés)                  |
|                                                                  |
+------------------------------------------------------------------+
```

### Táblázat Oszlopai

A Results Tab a következő oszlopokat jeleníti meg:

| Oszlop | Leírás |
|--------|--------|
| **Rank** | Helyezés (csak Weighted módokban) |
| **No.** | Stratégia azonosító szám |
| **Forecast_1W** | 1 hetes előrejelzés (profit) |
| **Forecast_1M** | 1 hónapos előrejelzés (4 hét) |
| **Forecast_3M** | 3 hónapos előrejelzés (13 hét) |
| **Forecast_6M** | 6 hónapos előrejelzés (26 hét) |
| **Forecast_12M** | 1 éves előrejelzés (52 hét) |
| **Method** | Alkalmazott előrejelzési modell |
| **SS** | Stability Score (0-1) |
| **RS** | Ranking Score (súlyozott mód) |
| **SR** | Sharpe Ratio |
| **AC** | Activity Consistency |
| **PC** | Profit Consistency |
| **DC** | Data Confidence |

---

## 06.01. Ranking Mode és Apply Ranking Funkció

A **Ranking Mode** dropdown lehetővé teszi a stratégiák rangsorolási módjának kiválasztását. Három mód áll rendelkezésre:

| Mód | Belső Kulcs | Leírás |
|-----|-------------|--------|
| **Standard** | `forecast` | Eredeti nézet, 1M előrejelzés szerinti rendezés |
| **Stability Weighted** | `stability` | Stabilitás bónusszal súlyozott rangsor |
| **Risk Adjusted** | `risk_adjusted` | Kockázattal korrigált (Sharpe-alapú) rangsor |

### Apply Ranking Gomb Működése

1. **Kattintás**: A felhasználó kiválasztja a kívánt módot és rákattint
2. **Metrikák Ellenőrzése**: A rendszer ellenőrzi, hogy a stabilitási metrikák léteznek-e
3. **Háttérszál Indítása**: Ha szükséges, háttérszálban számítja a metrikákat
4. **Adatforrás Keresése**:
   - Először a Data Tab-ból próbálja lekérni az adatokat
   - Ha nincs betöltve, az `all_strategies_data`-ból rekonstruálja
5. **Rangsorolás Alkalmazása**: `DataProcessor.apply_ranking()` meghívása
6. **Megjelenítés Frissítése**: `_update_results_display()` frissíti a táblázatot

### Kód Referencia

```python
# src/gui/tabs/results_tab.py:242-276
def _apply_ranking(self):
    """Apply the selected ranking mode to results."""
    # Map combo value to mode key
    mode_map = {
        self.tr("Standard"): "forecast",
        self.tr("Stability Weighted"): "stability",
        self.tr("Risk Adjusted"): "risk_adjusted"
    }
    mode = mode_map.get(self.ranking_combo.get(), "forecast")

    # Check if metrics need calculation
    needs_calc = mode in ["stability", "risk_adjusted"] and not all(
        c in base_df.columns for c in ["Stability_Score"]
    )
```

---

## 06.02. Forecast (Standard) Mód - Matematikai Leírás

A **Standard (Forecast)** mód a legegyszerűbb rangsorolási módszer: kizárólag az előrejelzett profit alapján rendezi a stratégiákat.

### Algoritmus

```
Rendezés: Forecast_1M DESC (csökkenő sorrendben)
Rank = pozíció a listában
```

### Matematikai Definíció

Legyen `F_i^{1M}` az `i` stratégia 1 hónapos előrejelzése.

A rangsor:
```
Rank(i) = |{j : F_j^{1M} > F_i^{1M}}| + 1
```

### Jellemzők

| Tulajdonság | Érték |
|-------------|-------|
| **Súlyozás** | Nincs |
| **Stabilitás figyelembevétele** | Nem |
| **Kockázat figyelembevétele** | Nem |
| **Számítási igény** | Minimális (egyszerű rendezés) |
| **Rank oszlop** | Nem jelenik meg |

### Implementáció

```python
# src/data/processor.py:866-870
if ranking_mode == "forecast":
    # Simple forecast-based ranking
    sort_col = sort_column
    df = df.sort_values(sort_col, ascending=False)
```

### Előnyök és Hátrányok

| Előnyök | Hátrányok |
|---------|-----------|
| Egyszerű, átlátható | Nem veszi figyelembe a kockázatot |
| Gyors számítás | Volatilis stratégiák előnybe kerülhetnek |
| Tiszta előrejelzés-alapú | Rövid múlt esetén megbízhatatlan |

---

## 06.03. Stability Weighted Mód - Matematikai Leírás

A **Stability Weighted** mód bónuszt ad a konzisztens viselkedésű stratégiáknak. A stabilitás mérésére összetett metrikát használ.

### Stabilitási Metrikák Számítása

A `calculate_stability_metrics()` függvény az alábbi metrikákat számítja ki minden stratégiához:

#### 1. Activity Consistency (AC)

Az aktivitás konzisztenciája - mennyire egyenletes a kereskedési aktivitás.

```python
# 13 hetes gördülő aktivitási arány variációs együtthatója alapján
rolling_activity = (Trades > 0).rolling(13).mean()
activity_mean = rolling_activity.mean()
activity_std = rolling_activity.std()

if activity_mean > 0:
    activity_cv = activity_std / activity_mean
    Activity_Consistency = max(0, 1 - activity_cv)
else:
    Activity_Consistency = 0
```

**Interpretáció**:
- 1.0 = Tökéletesen egyenletes kereskedés minden héten
- 0.0 = Rendkívül változékony aktivitás

#### 2. Profit Consistency (PC)

A profitkonzisztencia - a nyerő hetek aránya az aktív hetek között.

```python
positive_weeks = count(Profit > 0)
negative_weeks = count(Profit < 0)

if (positive_weeks + negative_weeks) > 0:
    Profit_Consistency = positive_weeks / (positive_weeks + negative_weeks)
else:
    Profit_Consistency = 0.5
```

**Interpretáció**:
- > 0.5 = Több nyerő hét, mint veszteséges
- = 0.5 = Egyenlő arány
- < 0.5 = Több veszteséges hét

#### 3. Sharpe Ratio (SR)

Kockázattal korrigált hozam mutató.

```python
mean_profit = mean(Profit)
std_profit = std(Profit)

Sharpe_Ratio = mean_profit / std_profit if std_profit > 0 else 0
```

#### 4. Data Confidence (DC)

Az adatok megbízhatósága - mennyi historikus adat áll rendelkezésre.

```python
total_weeks = len(Profit)
Data_Confidence = min(1.0, total_weeks / 104)  # 2 év = 100% megbízhatóság
```

**Interpretáció**:
- 1.0 = Legalább 2 év (104 hét) adat
- 0.5 = 1 év adat
- < 0.25 = Kevesebb mint fél év

### Stability Score (SS) - Összetett Mutató

```python
Stability_Score = (
    0.30 * Activity_Consistency +
    0.35 * Profit_Consistency +
    0.20 * min(1.0, max(0.0, (Sharpe_Ratio + 1) / 3)) +
    0.15 * Data_Confidence
)
```

**Súlyok magyarázata**:
| Komponens | Súly | Indoklás |
|-----------|------|----------|
| Activity Consistency | 30% | Egyenletes kereskedés fontos |
| Profit Consistency | 35% | A nyerési arány a legfontosabb |
| Sharpe normalizált | 20% | Kockázat/hozam arány |
| Data Confidence | 15% | Több adat = megbízhatóbb |

### Ranking Score Számítása (Stability Weighted)

```python
stability_bonus = Stability_Score * 0.3  # Maximum 30% bónusz
Ranking_Score = Forecast_1M * (1 + stability_bonus)
```

**Példa számítás**:
```
Stratégia A:
  Forecast_1M = 1000
  Stability_Score = 0.8
  stability_bonus = 0.8 * 0.3 = 0.24
  Ranking_Score = 1000 * (1 + 0.24) = 1240

Stratégia B:
  Forecast_1M = 1100
  Stability_Score = 0.4
  stability_bonus = 0.4 * 0.3 = 0.12
  Ranking_Score = 1100 * (1 + 0.12) = 1232

Eredmény: A > B (annak ellenére, hogy B előrejelzése magasabb)
```

### Implementáció

```python
# src/data/processor.py:872-896
elif ranking_mode == "stability":
    if "Stability_Score" not in df.columns:
        # Fallback to forecast ranking
        sort_col = sort_column
    else:
        stability_bonus = df["Stability_Score"] * 0.3
        df["Ranking_Score"] = df[sort_column] * (1 + stability_bonus)
        sort_col = "Ranking_Score"
        df = df.sort_values(sort_col, ascending=False)
```

---

## 06.04. Risk Adjusted Mód - Matematikai Leírás

A **Risk Adjusted** mód a Sharpe-ráció alapú kockázatkorrekciót alkalmaz, figyelembe véve az adatok megbízhatóságát is.

### Sharpe Factor Számítása

A Sharpe Ratio-t szorzótényezővé alakítja (0.5 - 1.5 tartomány):

```python
# Sharpe_Ratio klippolva [-1, 2] tartományba
sharpe_clipped = clip(Sharpe_Ratio, -1, 2)

# Normalizálás 0.75 - 1.5 tartományba
sharpe_factor = 1 + (sharpe_clipped / 4)
```

**Leképezés**:
| Sharpe Ratio | sharpe_factor |
|--------------|---------------|
| -1.0 | 0.75 |
| 0.0 | 1.00 |
| 1.0 | 1.25 |
| 2.0 | 1.50 |

### Confidence Factor Számítása

Az adatmennyiség alapján:

```python
confidence_factor = 0.7 + (Data_Confidence * 0.3)
```

**Leképezés**:
| Data_Confidence | confidence_factor |
|-----------------|-------------------|
| 0.0 | 0.70 |
| 0.5 | 0.85 |
| 1.0 | 1.00 |

### Ranking Score Számítása (Risk Adjusted)

```python
Ranking_Score = Forecast_1M * sharpe_factor * confidence_factor
```

**Teljes képlet kibontva**:
```
Ranking_Score = Forecast_1M × (1 + Sharpe_Ratio_clipped/4) × (0.7 + 0.3×Data_Confidence)
```

**Példa számítás**:
```
Stratégia A (alacsony kockázat, sok adat):
  Forecast_1M = 800
  Sharpe_Ratio = 1.5
  Data_Confidence = 1.0

  sharpe_factor = 1 + (1.5/4) = 1.375
  confidence_factor = 0.7 + (1.0 * 0.3) = 1.0
  Ranking_Score = 800 * 1.375 * 1.0 = 1100

Stratégia B (magas kockázat, kevés adat):
  Forecast_1M = 1200
  Sharpe_Ratio = -0.5
  Data_Confidence = 0.3

  sharpe_factor = 1 + (-0.5/4) = 0.875
  confidence_factor = 0.7 + (0.3 * 0.3) = 0.79
  Ranking_Score = 1200 * 0.875 * 0.79 = 829

Eredmény: A > B (A alacsonyabb előrejelzés, de jobb risk-adjusted score)
```

### Implementáció

```python
# src/data/processor.py:904-917
elif ranking_mode == "risk_adjusted":
    if "Sharpe_Ratio" not in df.columns or "Data_Confidence" not in df.columns:
        # Fallback to forecast ranking
        sort_col = sort_column
    else:
        # Normalize Sharpe to a multiplier (0.5 to 1.5 range)
        sharpe_factor = 1 + (df["Sharpe_Ratio"].clip(-1, 2) / 4)
        confidence_factor = 0.7 + (df["Data_Confidence"] * 0.3)
        df["Ranking_Score"] = df[sort_column] * sharpe_factor * confidence_factor
        sort_col = "Ranking_Score"
        df = df.sort_values(sort_col, ascending=False)
```

### A Három Mód Összehasonlítása

| Jellemző | Standard | Stability Weighted | Risk Adjusted |
|----------|----------|-------------------|---------------|
| **Képlet** | F | F × (1 + 0.3×SS) | F × SF × CF |
| **Kockázat súly** | 0% | ~20% (SS-ben) | ~25-50% |
| **Konzisztencia súly** | 0% | ~65% (SS-ben) | 0% |
| **Adatmennyiség súly** | 0% | ~15% (SS-ben) | ~30% |
| **Max bónusz/levonás** | - | +30% bónusz | +50% / -50% |
| **Ideális használat** | Agresszív | Konzervatív | Kockázatkerülő |

---

## 06.05. Output Folder Funkció

Az **Output Folder** mező határozza meg, hová kerülnek a generált riportok.

### Funkciók

1. **Szövegmező**: Aktuális kimeneti mappa útvonala
2. **Browse Gomb**: Mappa választó dialógus megnyitása
3. **Perzisztencia**: Az utoljára választott mappa mentésre kerül (`window_state.pkl`)

### Működés

```python
# src/gui/tabs/results_tab.py:638-647
def browse_output_folder(self):
    """Open folder browser for output directory."""
    get_sound_manager().play_button_click()
    initial_dir = getattr(self, "last_results_output_folder", "") or None
    folder = filedialog.askdirectory(initialdir=initial_dir)
    if folder:
        self.last_results_output_folder = folder
        self.save_window_state()  # Azonnal mentés
        self.entry_output_path.delete(0, "end")
        self.entry_output_path.insert(0, folder)
```

### Mappastruktúra Generáláskor

A riportok az alábbi almappákba kerülnek:

```
{Output Folder}/
├── {Model}_{Currency}_{RankingMode}/     <- Generate Report
│   ├── {Model}_Analysis_{timestamp}.md
│   ├── {Model}_Analysis_{timestamp}.html
│   └── (Auto módban: AR_*.md, AR_*.html)
└── {Model}_{Currency}_AllResults/        <- All Results gomb
    ├── {Model}_Analysis_all.md
    └── {Model}_Analysis_all.html
```

---

## 06.06. Generate Report Gomb - Részletes Működés

A **Generate Report** gomb a kiválasztott elemzésből komplett riportokat generál MD és HTML formátumban.

### Generálási Folyamat

```
[Generate Report kattintás]
        │
        ▼
[Ellenőrzések: results létezik? output folder beállítva?]
        │
        ▼
[Háttérszál indítása: _generate_report_bg()]
        │
        ├── [Ranking mód szerinti mappanév]
        │      {Model}_{Currency}_{Standard|StabilityWeighted|RiskAdjusted}
        │
        ├── [Best stratégia meghatározása aktuális rangsor alapján]
        │      - Ha változott a #1, új stratégia adatait kéri le
        │      - all_strategies_data-ból vagy újrafuttatással
        │
        ├── [ReportExporter példányosítása]
        │
        ├── [MD riport generálása]
        │      create_markdown_report()
        │
        ├── [HTML riport generálása]
        │      create_html_report()
        │      - Plotly interaktív grafikonokkal
        │
        └── [Auto módban: All Results riport is]
```

### Riport Tartalom

#### Markdown Riport Struktúra

```markdown
# Trading Strategy Analysis Report
**Date**: 2024-01-15 14:30
**Generating time**: 45.2s
**Ranking Mode**: Standard
**Method**: LightGBM

## Executive Summary
The analysis evaluated 500 strategies.
The top performing strategy is **Strategy No. 123**.

### Best Strategy Forecast
- **1 Week**: (No. 123) 150.00
- **1 Month**: (No. 123) 580.50
- **3 Months**: (No. 456) 1520.00
...

## Model Settings
| Parameter | Value |
|-----------|-------|
| n_estimators | 100 |
| max_depth | 6 |
...

## Financial Metrics (Strategy 123)
| Metric | Value |
|--------|-------|
| Total Profit | 15420.50 |
| Win Rate | 58.5% |
| Sharpe Ratio | 1.24 |
...

## Top 10 Strategies by Horizon
| Rank | 1 Week | 1 Month | 3 Months | 6 Months | 1 Year |
|------|--------|---------|----------|----------|--------|
| 1 | #123 (150.00) | #123 (580.50) | #456 (1520.00) | ... |
...
```

#### HTML Riport Jellemzők

- **Interaktív Plotly grafikonok**:
  - Profit görbe (historikus)
  - Előrejelzés vizualizáció
  - Drawdown grafikon

- **Reszponzív dizájn**: Mobilon is olvasható
- **Sötét téma**: A program témájával konzisztens
- **Beágyazott adatok**: Nincs külső függőség

### Pénzügyi Metrikák Számítása

A riportban az alábbi metrikák jelennek meg a legjobb stratégiához:

| Metrika | Számítás |
|---------|----------|
| **Total Profit** | Σ(Profit) |
| **Win Rate** | Nyerő hetek / Összes hét × 100% |
| **Profit Factor** | Σ(Pozitív profit) / |Σ(Negatív profit)| |
| **Sharpe Ratio** | mean(Profit) / std(Profit) |
| **Sortino Ratio** | mean(Profit) / std(Negatív profit) |
| **Max Drawdown** | max(Cummax - CumProfit) |
| **Avg Trade** | mean(Profit ahol Trades > 0) |

### Auto Mód Speciális Viselkedés

Auto Execution során a `generate_current_report()` eltérően működik:

```python
def generate_current_report(self, auto_mode=False, base_dir=None, forced_suffix=None):
    """
    Args:
        auto_mode: True esetén nincs popup, szinkron futás
        base_dir: Override a kimenet mappához
        forced_suffix: Pl. "_A01" a szinkronizált elnevezéshez
    """
```

Különbségek Auto módban:
- Nincs UI popup hiba esetén
- Automatikus `_Axx` suffix a mappanévhez
- All Results riport is automatikusan generálódik
- Szinkron végrehajtás (nem háttérszál)

---

## 06.07. Export CSV Funkció

Az **Export CSV** gomb a teljes eredménytáblázatot CSV fájlba exportálja.

### Működés

```python
# src/gui/tabs/results_tab.py:663-683
def export_results_csv(self):
    """Export results to CSV file."""
    get_sound_manager().play_button_click()

    if not hasattr(self, "results_df") or self.results_df is None:
        messagebox.showwarning(self.tr("Warning"), self.tr("No results to export."))
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        title=self.tr("Export Results to CSV")
    )

    if file_path:
        self.results_df.to_csv(file_path, index=False)
        self.log(f"Results exported to: {file_path}")
```

### Exportált Oszlopok

A CSV tartalmazza az összes oszlopot, beleértve:
- Azonosítók: No., Method
- Előrejelzések: Forecast_1W, Forecast_1M, Forecast_3M, Forecast_6M, Forecast_12M
- Stabilitási metrikák (ha számítva voltak): Stability_Score, Activity_Consistency, stb.
- Ranking_Score (ha weighted mód volt aktív)
- Forecasts (52 elemű lista stringként)

### Használati Esetek

1. **Excel további elemzés**: Pivot táblák, egyéni szűrések
2. **Más rendszerbe import**: Kereskedési platform, backtester
3. **Archiválás**: Az eredmények hosszú távú tárolása
4. **Összehasonlítás**: Több elemzés eredményeinek összevetése

---

## 06.08. All Results Gomb - Teljes Logika

Az **All Results** gomb egy speciális riportot generál, amely minden stratégia részletes heti bontású előrejelzését tartalmazza.

### Aktiválás Feltétele

A gomb csak akkor aktív, ha:
- Van betöltött vagy kiszámított eredmény (`last_results` létezik)
- Van beállított output folder

### Generálási Folyamat

```
[All Results kattintás]
        │
        ▼
[Háttérszál: _generate_all_results_bg()]
        │
        ├── [Almappa létrehozása]
        │      {Model}_{Currency}_AllResults/
        │
        ├── [ReportExporter példányosítása]
        │
        └── [create_all_results_report() meghívása]
                │
                ├── MD riport: Minden stratégia táblázatban
                │      - Heti bontású előrejelzések
                │      - Összesített statisztikák
                │
                └── HTML riport: Interaktív táblákat
                       - Szűrhető, rendezhető táblázat
                       - Összehasonlító grafikonok
```

### Riport Struktúra

```markdown
# All Results Report - {Model} Analysis

## Summary Statistics
- Total Strategies Analyzed: 500
- Best 1M Forecast: #123 (580.50)
- Average 1M Forecast: 120.35
- Median 1M Forecast: 95.20

## Weekly Forecast Breakdown

### Strategy #123
| Week | Forecast | Cumulative |
|------|----------|------------|
| 1 | 12.50 | 12.50 |
| 2 | 15.20 | 27.70 |
| ... | ... | ... |
| 52 | 8.30 | 580.50 |

### Strategy #124
...
```

### Kód Referencia

```python
# src/gui/tabs/results_tab.py:984-1057
def generate_all_results_report(self):
    """Generate All Results report with weekly breakdown of forecasts."""

    # Create subfolder: modelname_currencypair_AllResults
    currency_pair = filename_base.split("_")[0]
    folder_name = f"{method}_{currency_pair}_AllResults"

    # Generate report
    md_path, html_path = exporter.create_all_results_report(
        results_df=results_df,
        method_name=method,
        best_strat_data=best_strat_data,
        filename_base=f"{method}_{filename_base}_all",
        params=params,
        execution_time=data.get("execution_time", "N/A"),
        forecast_horizon=forecast_horizon,
    )
```

### All Results vs Generate Report Különbségek

| Jellemző | Generate Report | All Results |
|----------|-----------------|-------------|
| **Fókusz** | Legjobb stratégia | Összes stratégia |
| **Részletesség** | Összefoglaló | Heti bontás |
| **Fájlméret** | Kisebb (~100KB) | Nagyobb (~1-5MB) |
| **Mappanév** | {Model}_{Currency}_{RankingMode} | {Model}_{Currency}_AllResults |
| **Fájlnév prefix** | {Model}_Analysis | {Model}_Analysis_all |

---

## 06.09. Load Analysis State Gomb

A **Load Analysis State** gomb lehetővé teszi korábban mentett elemzések visszatöltését `.pkl` (pickle) fájlokból.

### Mentett Adatok Struktúrája

```python
{
    "version": "1.1",              # Fájlformátum verzió
    "results": DataFrame,          # Eredmény táblázat
    "method": "LightGBM",          # Modell neve
    "params": {...},               # Használt paraméterek
    "filename_base": "...",        # Eredeti fájlnév alap
    "best_strat_id": 123,          # Legjobb stratégia ID
    "best_strat_data": DataFrame,  # Legjobb stratégia historikus adata
    "forecast_values": [...],      # 52 hetes előrejelzés lista
    "ranking_mode": "forecast",    # Mentéskori rangsorolási mód
    "execution_time": "45.2s",     # Futási idő
    "all_strategies_data": {       # Minden stratégia historikus adata
        123: DataFrame,
        124: DataFrame,
        ...
    }
}
```

### Betöltési Folyamat

```
[Load Analysis State kattintás]
        │
        ▼
[Fájl választó dialógus (.pkl)]
        │
        ▼
[Pickle betöltés és validálás]
        │
        ├── [Verzió ellenőrzés]
        │      - 0.9: Legacy, hiányzó mezők pótlása
        │      - 1.0: all_strategies_data hiányzik
        │      - 1.1: Teljes formátum
        │
        ├── [Paraméterek backfill]
        │      - Régi fájlokból hiányzó paraméterek pótlása
        │
        ├── [Állapot visszaállítása]
        │      - results_df
        │      - all_strategies_data
        │      - ranking_mode
        │
        └── [UI frissítése]
               - Táblázat megjelenítése
               - Gombok engedélyezése
               - Ranking label frissítése
```

### Verziókezelés

```python
# src/gui/tabs/results_tab.py:548-559
data_version = loaded_data.get("version", "0.9")

if data_version == "0.9":
    # Legacy file - defaults for missing fields
    loaded_data.setdefault("ranking_mode", "forecast")
    loaded_data.setdefault("execution_time", "N/A")
    loaded_data.setdefault("all_strategies_data", {})
elif data_version == "1.0":
    loaded_data.setdefault("all_strategies_data", {})
```

### Paraméter Backfill

Ha egy régi `.pkl` fájlból hiányoznak paraméterek:

```python
# src/gui/tabs/results_tab.py:561-573
method = loaded_data.get("method", "")
saved_params = loaded_data.get("params", {})

if method and saved_params is not None:
    full_params = self.get_full_model_params(method, saved_params)
    if len(full_params) > len(saved_params):
        loaded_data["params"] = full_params
        backfilled = len(full_params) - len(saved_params)
        self.log(f"Backfilled {backfilled} missing parameters with defaults")
```

### Betöltés Utáni Műveletek

1. **Stabilitási metrikák**: Ha nem léteznek, Apply Ranking újraszámolja
2. **Riport generálás**: Generate Report használhatja a cached adatokat
3. **Re-ranking**: Bármely mód alkalmazható a betöltött adatokra

---

## 06.10. Show Params Gomb

A **Show Params** gomb megjeleníti az aktuális vagy betöltött elemzéshez használt modell paramétereket.

### Működés

```python
# src/gui/tabs/results_tab.py:213-231
def show_loaded_parameters(self):
    """Show parameters from the loaded analysis state."""
    if not hasattr(self, "last_results") or not self.last_results:
        messagebox.showinfo(self.tr("Info"), self.tr("No analysis results loaded."))
        return

    params = self.last_results.get("params", {})
    method = self.last_results.get("method", "Unknown Model")

    if not params:
        messagebox.showinfo(self.tr("Info"), self.tr("No parameters found."))
        return

    # Format for display
    msg = f"Model Parameters: {method}\n\nParameters:\n"
    for k, v in params.items():
        msg += f"- {k}: {v}\n"

    self.show_info_popup("Model Parameters", msg)
```

### Megjelenített Információk

Példa LightGBM modellre:

```
Model Parameters: LightGBM

Parameters:
- n_estimators: 100
- max_depth: 6
- learning_rate: 0.1
- n_jobs: 8
- forecast_horizon: 52
- use_gpu: False
- mode: panel
```

### Felhasználási Esetek

1. **Reprodukálhatóság**: Pontosan milyen beállításokkal készült az elemzés
2. **Összehasonlítás**: Különböző futtatások paramétereinek összevetése
3. **Dokumentáció**: Riportokhoz a beállítások rögzítése
4. **Hibakeresés**: Ha az eredmény nem várt, a paraméterek ellenőrzése

---

## Összefoglaló: Results Tab Munkafolyamat

```
                    [Elemzés befejezése]
                           │
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │                   RESULTS TAB                        │
    │                                                      │
    │  1. RANKING                                          │
    │     ├── Standard: Egyszerű forecast rendezés         │
    │     ├── Stability: Konzisztencia bónusz              │
    │     └── Risk Adjusted: Sharpe-alapú korrekció        │
    │                                                      │
    │  2. EXPORT                                           │
    │     ├── Generate Report: MD + HTML (legjobb strat)   │
    │     ├── All Results: Minden stratégia részletesen    │
    │     └── Export CSV: Nyers adat további feldolgozásra │
    │                                                      │
    │  3. PERZISZTENCIA                                    │
    │     ├── Load Analysis State: Korábbi elemzés betöltése│
    │     └── Show Params: Paraméterek megtekintése        │
    │                                                      │
    └─────────────────────────────────────────────────────┘
```

### Tippek a Hatékony Használathoz

1. **Ranking mód választás**:
   - Agresszív kereskedés → Standard
   - Hosszú távú befektetés → Stability Weighted
   - Alacsony kockázat → Risk Adjusted

2. **Riport generálás**:
   - Áttekintéshez → Generate Report
   - Részletes elemzéshez → All Results
   - Saját feldolgozáshoz → Export CSV

3. **Perzisztencia**:
   - Az elemzések automatikusan mentődnek `.pkl`-be
   - Load Analysis State bármikor visszatölti
   - Régi fájlok is kompatibilisek (verziókezelés)

---

*Dokumentáció generálva: 2024*
*Forrásfájlok: `src/gui/tabs/results_tab.py`, `src/data/processor.py`, `src/reporting/exporter.py`*
