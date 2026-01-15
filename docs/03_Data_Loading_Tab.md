# MBO Trading Strategy Analyzer - Data Loading Tab

## Tartalomjegyzék

- [03.00 Data Loading Tab fő funkciója](#0300-data-loading-tab-fő-funkciója)
- [03.01 Convert Excel to Parquet gomb](#0301-convert-excel-to-parquet-gomb)
- [03.02 Open Parquet gomb](#0302-open-parquet-gomb)
- [03.03 Open Folder gomb](#0303-open-folder-gomb)
- [03.04 Feature Mode selection](#0304-feature-mode-selection)
- [03.05 Original mód](#0305-original-mód)
- [03.06 Forward Calc mód](#0306-forward-calc-mód)
- [03.07 Rolling Window mód](#0307-rolling-window-mód)
- [03.08 Módok összehasonlítása](#0308-módok-összehasonlítása)

---

# 03.00 Data Loading Tab fő funkciója

## Áttekintés

A **Data Loading** (Adatbetöltés) tab az alkalmazás első lépése, ahol a kereskedési stratégia adatokat töltjük be elemzésre. Ez a tab felelős az adatok:

1. **Betöltéséért** - Excel vagy Parquet fájlokból
2. **Konvertálásáért** - Excel → Parquet formátumba
3. **Előfeldolgozásáért** - Tisztítás, típuskonverzió
4. **Feature Engineering-jéért** - Származtatott jellemzők számítása

## Tab felépítése

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌────────────┐ │
│  │Convert Excel to │ │  Open Parquet   │ │   Open Folder   │ │ Path Label │ │
│  │    Parquet      │ │                 │ │                 │ │            │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ └────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│  Feature Mode:  (●) Original  ( ) Forward Calc  ( ) Rolling Window   [?]   │
│                 └─ No additional features - raw data only                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌─────────────────────────────────────────────────┐  │
│  │  Files:          │  │  Preview Table (Treeview)                       │  │
│  │  - file1.xlsx    │  │  ┌────┬────────┬───────┬────┬─────┬──────────┐  │  │
│  │  - file2.xlsx    │  │  │No. │ Profit │Trades │ PF │DD % │   Date   │  │  │
│  │  - file3.parquet │  │  ├────┼────────┼───────┼────┼─────┼──────────┤  │  │
│  │                  │  │  │  1 │  125.5 │   12  │2.1 │ 5.2 │2024-01-05│  │  │
│  │                  │  │  │  2 │  -45.2 │    8  │0.8 │ 8.1 │2024-01-05│  │  │
│  │                  │  │  │... │  ...   │  ...  │... │ ... │   ...    │  │  │
│  └──────────────────┘  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Adatszerkezet

Az Excel fájlok speciális **horizontális időblokk** struktúrával rendelkeznek:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Row 0: │ 2024-01-05 │    │    │    │    │ 2024-01-12 │    │    │    │     │
│         │ (Dátum)    │    │    │    │    │ (Dátum)    │    │    │    │     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Row 1: │ No. │Profit│Trad│ PF │DD %│... │ No. │Profit│Trad│ PF │DD %│... │
│         │     │      │ es │    │    │    │     │      │ es │    │    │    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Row 2: │  1  │ 125  │ 12 │2.1 │5.2 │    │  1  │ 89   │  8 │1.8 │6.1 │    │
│  Row 3: │  2  │ -45  │  8 │0.8 │8.1 │    │  2  │ 112  │ 15 │2.5 │4.2 │    │
│   ...   │ ... │ ...  │... │... │... │    │ ... │ ...  │... │... │... │    │
└─────────────────────────────────────────────────────────────────────────────┘
         │<─── Blokk 1 (1. hét) ───>│    │<─── Blokk 2 (2. hét) ───>│
```

### Oszlopok jelentése

| Oszlop | Típus | Leírás |
|--------|-------|--------|
| **No.** | Integer | Stratégia azonosító (0-6143) |
| **Profit** | Float | Heti profit/veszteség ($) |
| **Trades** | Integer | Kereskedések száma a héten |
| **PF** | Float | Profit Factor (Profit/Veszteség arány) |
| **DD %** | Float | Maximum Drawdown százalékban |
| **Start** | String | Kereskedés kezdete |
| **1st Candle** | String | Első gyertya időzítés |
| **Shift** | Integer | Időeltolás érték |
| **Position** | String | Pozíció típusa |
| **Date** | DateTime | Backtest futtatás dátuma |
| **SourceFile** | String | Forrás fájl neve |

---

# 03.01 Convert Excel to Parquet gomb

## Funkció célja

A **Convert Excel to Parquet** gomb lehetővé teszi több Excel fájl egyetlen, optimalizált Parquet fájlba történő összevonását és konvertálását.

```
┌──────────────────────────┐
│ Convert Excel to Parquet │  <- Lila szín (#9b59b6)
└──────────────────────────┘
```

## Miért Parquet?

| Szempont | Excel (.xlsx) | Parquet (.parquet) |
|----------|--------------|-------------------|
| **Betöltési sebesség** | Lassú (másodpercek) | Gyors (milliszekundum) |
| **Fájlméret** | Nagy | ~5-10x kisebb |
| **Típus megőrzés** | Nem megbízható | Pontos |
| **Oszlopszintű olvasás** | Nem | Igen |
| **Tömörítés** | Nincs | Beépített (snappy) |

## Működési folyamat

### 1. Mappa kiválasztás
```
Felhasználó → [Kiválasztja a mappát Excel fájlokkal]
                     │
                     ▼
              ┌──────────────┐
              │ askdirectory │ (filedialog)
              └──────────────┘
```

### 2. Excel fájlok keresése
```python
files = [f for f in os.listdir(source_folder) if f.endswith(".xlsx")]
```

### 3. Párhuzamos betöltés
```
┌──────────────────────────────────────────────────────────────┐
│  joblib.Parallel(n_jobs=CPU_CORES)                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│  │Worker 1 │ │Worker 2 │ │Worker 3 │ │Worker N │            │
│  │file1.xlsx│file2.xlsx│file3.xlsx│ │fileN.xlsx│            │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘            │
│       │           │           │           │                  │
│       ▼           ▼           ▼           ▼                  │
│  ┌─────────────────────────────────────────────┐             │
│  │         pd.concat(all_data)                 │             │
│  └─────────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────┘
```

### 4. Fájlnév generálás

A kimeneti fájlnév intelligensen generálódik:

**Bemeneti fájlok**:
- `AUDJPY_2022_Weekly_Fix.xlsx`
- `AUDJPY_2023_Weekly_Fix.xlsx`
- `AUDJPY_2024_Weekly_Fix.xlsx`

**Kimeneti fájl**:
- `AUDJPY_2022_2023_2024_Weekly_Fix.parquet`

**Algoritmus**:
```python
# 1. Évszámok kinyerése regex-szel
years = re.findall(r"\d{4}", filename)  # ['2022', '2023', '2024']

# 2. Évek rendezése és összefűzése
sorted_years = sorted(years)  # ['2022', '2023', '2024']
year_str = "_".join(sorted_years)  # "2022_2023_2024"

# 3. Első fájl neve alapján mintázat
# AUDJPY_2022_Weekly_Fix.xlsx → AUDJPY_{years}_Weekly_Fix.parquet
output_name = first_file.replace("2022", year_str)
```

### 5. Adattisztítás konverzió előtt

```python
# PF (Profit Factor) oszlop kezelése
if "PF" in merged_df.columns:
    merged_df["PF"] = pd.to_numeric(merged_df["PF"], errors="coerce").fillna(0)
```

### 6. Mentés Parquet formátumban

```python
merged_df.to_parquet(output_path, index=False)
```

## Matematikai/logikai eredmény

**Bemenet**: N darab Excel fájl, mindegyik M stratégiával és T időponttal

```
Fájl₁: M × T₁ sor
Fájl₂: M × T₂ sor
...
Fájlₙ: M × Tₙ sor
```

**Kimenet**: 1 Parquet fájl

```
Összesített sorok = M × (T₁ + T₂ + ... + Tₙ)
                  = M × Σᵢ Tᵢ
```

**Példa**:
- 3 Excel fájl (2022, 2023, 2024)
- Mindegyik 52 heti adat
- 6144 stratégia

```
Összesen = 6144 × (52 + 52 + 52) = 6144 × 156 = 958,464 sor
```

---

# 03.02 Open Parquet gomb

## Funkció célja

Az **Open Parquet** gomb lehetővé teszi egy vagy több Parquet fájl közvetlen betöltését.

```
┌──────────────────┐
│   Open Parquet   │  <- Kék szín (#3498db)
└──────────────────┘
```

## Működési folyamat

### 1. Fájl kiválasztás dialógus

```python
files = filedialog.askopenfilenames(
    filetypes=[("Parquet Files", "*.parquet")],
    initialdir=last_parquet_folder
)
```

- **Többszörös kiválasztás** támogatott
- Megjegyzi az utolsó használt mappát

### 2. Betöltési folyamat

```
┌─────────────────────────────────────────────────────────────────┐
│                    Parquet Betöltés                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐                                               │
│   │ file1.parquet│ ──→ pd.read_parquet() ──→ DataFrame₁        │
│   └─────────────┘                                               │
│                                                                 │
│   ┌─────────────┐                                               │
│   │ file2.parquet│ ──→ pd.read_parquet() ──→ DataFrame₂        │
│   └─────────────┘                                               │
│                                                                 │
│              ↓                                                  │
│   ┌─────────────────────────────────────┐                       │
│   │  pd.concat([df1, df2], ignore_index=True)                  │
│   └─────────────────────────────────────┘                       │
│              ↓                                                  │
│   ┌─────────────────────────────────────┐                       │
│   │  DataProcessor.clean_data(raw_data)                        │
│   └─────────────────────────────────────┘                       │
│              ↓                                                  │
│   ┌─────────────────────────────────────┐                       │
│   │  _apply_feature_mode(processed)                            │
│   └─────────────────────────────────────┘                       │
│              ↓                                                  │
│   ┌─────────────────────────────────────┐                       │
│   │  self.processed_data = result                              │
│   └─────────────────────────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3. SourceFile oszlop kezelése

```python
# Ha nincs SourceFile oszlop, hozzáadjuk
if "SourceFile" not in df.columns:
    df["SourceFile"] = os.path.basename(filepath)
```

### 4. Szálkezelés

A betöltés **háttérszálon** fut a UI blokkolásának elkerülése érdekében:

```python
threading.Thread(
    target=self.load_files_list,
    args=(files,),
    daemon=True
).start()
```

## Teljesítmény előny

| Művelet | Excel (10 fájl) | Parquet (1 fájl) |
|---------|-----------------|------------------|
| Betöltési idő | ~15-30 mp | ~0.5-2 mp |
| Memória használat | Magasabb | Alacsonyabb |
| I/O műveletek | Sok | Kevés |

---

# 03.03 Open Folder gomb

## Funkció célja

Az **Open Folder** gomb egy teljes mappa tartalmát (Excel és/vagy Parquet fájlokat) tölti be egyszerre, párhuzamos feldolgozással.

```
┌──────────────────┐
│   Open Folder    │  <- Alapértelmezett szín
└──────────────────┘
```

## Működési folyamat

### 1. Mappa kiválasztás

```python
folder = filedialog.askdirectory(initialdir=last_open_folder)
```

### 2. Fájlok felderítése

```python
# Mindkét formátum támogatott
files = [f for f in os.listdir(folder_path)
         if f.endswith(".xlsx") or f.endswith(".parquet")]
```

### 3. Párhuzamos betöltés (joblib)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Parallel Loading Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: folder_path = "C:/Data/Strategies/"                            │
│  Files: [file1.xlsx, file2.xlsx, ..., file10.parquet]                  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  CPU Manager → n_jobs = get_n_jobs()  (75% of cores by default) │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  joblib.Parallel(n_jobs=n_jobs)(                                │   │
│  │      delayed(load_single)(f) for f in files                     │   │
│  │  )                                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│          ┌───────────────────┼───────────────────┐                     │
│          ▼                   ▼                   ▼                     │
│     ┌─────────┐         ┌─────────┐         ┌─────────┐               │
│     │Core 1   │         │Core 2   │         │Core N   │               │
│     │file1    │         │file2    │         │fileN    │               │
│     │ ↓       │         │ ↓       │         │ ↓       │               │
│     │DataFrame│         │DataFrame│         │DataFrame│               │
│     └────┬────┘         └────┬────┘         └────┬────┘               │
│          │                   │                   │                     │
│          └───────────────────┼───────────────────┘                     │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  results = [df1, df2, ..., dfN]  (filter out None)              │   │
│  │  final = pd.concat(results, ignore_index=True)                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4. Egyedi fájl betöltése (load_file)

Az Excel fájlok horizontális blokk struktúrájának feldolgozása:

```python
def load_file(filepath):
    # 1. Teljes fájl beolvasása header nélkül
    df_raw = pd.read_excel(filepath, header=None)

    # 2. Első sor (dátumok) forward fill a merged cellákhoz
    row0_filled = df_raw.iloc[0].ffill()

    # 3. "No." oszlopok keresése (blokk kezdetek)
    row1 = df_raw.iloc[1]
    block_starts = [i for i, val in enumerate(row1)
                    if str(val).strip() == "No."]

    # 4. Blokkok feldolgozása
    for start_col in block_starts:
        date_val = row0_filled[start_col]

        # Header mapping
        headers = df_raw.iloc[1, start_col:start_col+10]
        col_map = {}  # {"No.": 0, "Profit": 1, ...}

        # Adat kinyerés
        block_data = extract_columns(df_raw, col_map)
        block_data["Date"] = date_val

        all_blocks.append(block_data)

    # 5. Blokkok összefűzése
    return pd.concat(all_blocks, ignore_index=True)
```

### 5. Adattisztítás (clean_data)

```python
def clean_data(df):
    # Numerikus oszlopok
    numeric_cols = ["Profit", "Trades", "PF", "DD %"]

    for col in numeric_cols:
        if df[col].dtype == object:
            # Accounting formátum: (123.45) → -123.45
            mask_parens = df[col].str.match(r"^\(.*\)$")
            df.loc[mask_parens, col] = "-" + df.loc[mask_parens, col].str.replace(r"[()]", "")

            # Speciális karakterek eltávolítása: $, %, stb.
            df[col] = df[col].str.replace(r"[^\d.-]", "", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # PF NaN → 0
    df["PF"] = df["PF"].fillna(0)

    # No. integer konverzió
    df["No."] = pd.to_numeric(df["No."], errors="coerce").fillna(0).astype(int)

    return df
```

## Matematikai modell

**Párhuzamos feldolgozás időmegtakarítása**:

```
T_sequential = Σᵢ t_load(fileᵢ)

T_parallel = max(t_load(file₁), ..., t_load(fileₙ)) + t_overhead

Speedup = T_sequential / T_parallel ≈ n_files / n_workers
```

**Példa**:
- 10 Excel fájl, egyenként 3 mp betöltési idő
- 8 worker használata

```
T_sequential = 10 × 3 = 30 mp
T_parallel ≈ ceil(10/8) × 3 + 0.5 = 6.5 mp
Speedup ≈ 4.6x
```

---

# 03.04 Feature Mode selection

## Áttekintés

A **Feature Mode** választó lehetővé teszi, hogy az eredeti adatok mellé származtatott jellemzőket (feature-öket) adjunk, amelyek javíthatják a gépi tanulási modellek teljesítményét.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Feature Mode:  (●) Original  ( ) Forward Calc  ( ) Rolling Window   [?]   │
│                 └─ No additional features - raw data only                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Három mód

| Mód | Leírás | Feature-ök | Sebesség |
|-----|--------|------------|----------|
| **Original** | Nyers adatok | 0 új oszlop | ★★★★★ |
| **Forward Calc** | Teljes történelem alapján | ~8 új oszlop | ★★★★☆ |
| **Rolling Window** | 13 hetes gördülő ablak | ~9 új oszlop | ★★☆☆☆ |

## Automatikus újraszámítás

Ha már betöltöttünk adatot, a mód váltásakor a feature-ök **automatikusan újraszámolódnak**:

```python
def _on_feature_mode_change(self):
    mode = self.feature_mode.get()

    if self.raw_data is not None:
        # Háttérszálon újraszámítás
        threading.Thread(
            target=self._recalculate_features,
            daemon=True
        ).start()
```

---

# 03.05 Original mód

## Leírás

Az **Original** mód a legegyszerűbb: csak az eredeti, nyers adatokat használja, semmilyen származtatott feature nélkül.

```
(●) Original
    └─ No additional features - raw data only
```

## Adatszerkezet

```
┌──────────────────────────────────────────────────────────────┐
│  Original Mode - Oszlopok                                    │
├───────┬─────────┬────────┬──────┬───────┬──────┬────────────┤
│  No.  │ Profit  │ Trades │  PF  │ DD %  │ Date │ SourceFile │
├───────┼─────────┼────────┼──────┼───────┼──────┼────────────┤
│   1   │  125.5  │   12   │ 2.1  │  5.2  │ ...  │  file1.xlsx│
│   2   │  -45.2  │    8   │ 0.8  │  8.1  │ ...  │  file1.xlsx│
│  ...  │   ...   │  ...   │ ...  │  ...  │ ...  │     ...    │
└───────┴─────────┴────────┴──────┴───────┴──────┴────────────┘
```

## Mikor használd?

| Használati eset | Indoklás |
|-----------------|----------|
| **Gyors tesztelés** | Nincs számítási overhead |
| **Baseline modell** | Referencia a feature-ök hatásának méréséhez |
| **Adat vizsgálat** | A nyers értékek megtekintése |
| **Egyszerű modellek** | ARIMA, ETS, amelyek csak az idősorokat használják |
| **Korlátozott memória** | Kevesebb oszlop = kisebb memóriaigény |

## Implementáció

```python
def _apply_feature_mode(self, df):
    mode = self.feature_mode.get()

    if mode == "original":
        self.log("Feature mode: Original (no additional features)")
        return df  # Változatlanul visszaadjuk
```

## Előnyök és hátrányok

| Előnyök | Hátrányok |
|---------|-----------|
| ✅ Leggyorsabb betöltés | ❌ Kevesebb információ a modellnek |
| ✅ Legkisebb memória | ❌ ML modellek gyengébb teljesítménye |
| ✅ Egyszerű értelmezés | ❌ Nincs trend/momentum információ |
| ✅ Nincs adatvesztés | ❌ Nincs volatilitás/konzisztencia adat |

---

# 03.06 Forward Calc mód

## Leírás

A **Forward Calc** mód **kumulatív (bővülő ablakos)** feature-öket számít, amelyek az adott időpontig rendelkezésre álló **teljes történelmet** használják.

```
( ) Forward Calc
    └─ Features from entire history (static per strategy)
```

## Matematikai definíciók

### 1. Weeks Count (Hetek száma)

```
feat_weeks_count(t) = t
```

Egyszerűen a sor sorszáma a stratégián belül.

### 2. Active Ratio (Aktivitási arány)

```
                     Σᵢ₌₁ᵗ I(Tradesᵢ > 0)
feat_active_ratio(t) = ─────────────────────
                              t

ahol I(x) = { 1, ha x igaz
            { 0, egyébként
```

**Jelentés**: A stratégia hány százaléka volt aktív (kereskedett) az eddigi hetekben.

**Tartomány**: [0, 1]

### 3. Profit Consistency (Profit konzisztencia)

```
                          Σᵢ₌₁ᵗ I(Profitᵢ > 0)
feat_profit_consistency(t) = ─────────────────────────────────────────
                             Σᵢ₌₁ᵗ I(Profitᵢ > 0) + Σᵢ₌₁ᵗ I(Profitᵢ < 0)
```

**Jelentés**: A nyereséges hetek aránya az összes nem-nulla héthez képest.

**Tartomány**: [0, 1], ahol 0.5+ = több nyereséges, mint veszteséges hét

### 4. Total Profit (Összesített profit)

```
feat_total_profit(t) = Σᵢ₌₁ᵗ Profitᵢ
```

**Jelentés**: Kumulatív profit az első héttől az aktuális hétig.

### 5. Cumulative Trades (Összesített kereskedések)

```
feat_cumulative_trades(t) = Σᵢ₌₁ᵗ Tradesᵢ
```

### 6. Volatility (Volatilitás)

```
                                    ___________________________
                                   ╱  Σᵢ₌₁ᵗ (Profitᵢ - μₜ)²
feat_volatility(t) = σₜ =        ╱  ──────────────────────────
                               ╲╱           t - 1

ahol μₜ = (Σᵢ₌₁ᵗ Profitᵢ) / t  (átlag t időpontig)
```

**Jelentés**: A profit szórása (standard deviation) az eddigi adatok alapján.

### 7. Sharpe Ratio (Sharpe ráta)

```
                      μₜ
feat_sharpe_ratio(t) = ──── , ha σₜ > 0
                      σₜ

                    = 0   , ha σₜ = 0
```

**Jelentés**: Kockázattal korrigált hozam. Magasabb = jobb teljesítmény egységnyi kockázatra.

### 8. Max Drawdown (Maximum visszaesés)

```
feat_max_drawdown(t) = min(0, CumSumₜ - RunningMaxₜ)

ahol:
  CumSumₜ = Σᵢ₌₁ᵗ Profitᵢ
  RunningMaxₜ = max(CumSum₁, CumSum₂, ..., CumSumₜ)
```

**Jelentés**: A legnagyobb csúcs-völgy visszaesés. Mindig ≤ 0 (negatív vagy nulla).

## Vizuális példa

```
Hét:    1      2      3      4      5      6
Profit: 100   -50    120    -30    80     -20

                                                    Expanding Window
                                                    ================
feat_weeks_count:        1      2      3      4      5      6
feat_total_profit:     100     50    170    140    220    200
feat_volatility:         0   106.1   87.4   78.3   76.8   72.1
feat_sharpe_ratio:       ∞   0.47   0.65   0.45   0.57   0.46
feat_active_ratio:     1.0    1.0    1.0    1.0    1.0    1.0
feat_profit_consistency: 1.0   0.5   0.67   0.5   0.6    0.5
```

## Implementáció (optimalizált, vektorizált)

```python
def add_features_forward(df):
    df = df.sort_values(["No.", "Date"]).copy()

    # Helper oszlopok
    df["_is_active"] = (df["Trades"] > 0).astype(float)
    df["_is_positive"] = (df["Profit"] > 0).astype(float)
    df["_is_negative"] = (df["Profit"] < 0).astype(float)

    grouped = df.groupby("No.", group_keys=False)

    # Weeks count
    df["feat_weeks_count"] = grouped.cumcount() + 1

    # Active ratio: cumulative active / weeks
    df["feat_active_ratio"] = grouped["_is_active"].cumsum() / df["feat_weeks_count"]

    # Profit consistency
    cum_pos = grouped["_is_positive"].cumsum()
    cum_neg = grouped["_is_negative"].cumsum()
    df["feat_profit_consistency"] = np.where(
        (cum_pos + cum_neg) > 0,
        cum_pos / (cum_pos + cum_neg),
        0.5
    )

    # Total profit
    df["feat_total_profit"] = grouped["Profit"].cumsum()

    # Expanding mean/std for Sharpe
    expanding_mean = grouped["Profit"].expanding(min_periods=1).mean()
    expanding_std = grouped["Profit"].expanding(min_periods=2).std()

    df["feat_volatility"] = expanding_std.fillna(0)
    df["feat_sharpe_ratio"] = np.where(
        expanding_std > 0,
        expanding_mean / expanding_std,
        0
    )

    # Max drawdown
    cumsum = grouped["Profit"].cumsum()
    running_max = grouped["Profit"].apply(lambda x: x.cumsum().cummax())
    df["feat_max_drawdown"] = (cumsum - running_max).clip(upper=0)

    return df
```

## Mikor használd?

| Használati eset | Indoklás |
|-----------------|----------|
| **Stratégia profilozás** | Teljes képet ad a stratégia jellemzőiről |
| **Hosszú távú előrejelzés** | Stabil, nem volatilis feature-ök |
| **ML modellek tanítása** | Gazdag feature készlet |
| **Stratégia összehasonlítás** | Azonos mérce minden stratégiához |

---

# 03.07 Rolling Window mód

## Leírás

A **Rolling Window** mód **gördülő 13 hetes ablakot** használ a feature-ök számításához. Ez reálisabb, mert csak a közelmúlt adatait használja, és dinamikusan változik idővel.

```
( ) Rolling Window
    └─ Features from rolling 13-week window (dynamic)
```

## Ablak méret

```
window = 13 hét ≈ 1 negyedév
min_periods = max(4, window // 3) = 4 hét minimum
```

## Matematikai definíciók

### 1. Rolling Active Ratio

```
                             Σᵢ₌ₜ₋₁₂ᵗ I(Tradesᵢ > 0)
feat_rolling_active_ratio(t) = ───────────────────────────
                                        13

ahol t-12 az ablak kezdete (13 héttel ezelőtt)
```

**Jelentés**: Az utolsó 13 hétben aktív hetek aránya.

### 2. Rolling Profit Consistency

```
                                 Σᵢ₌ₜ₋₁₂ᵗ I(Profitᵢ > 0)
feat_rolling_profit_consistency(t) = ─────────────────────────────────────────────────────
                                     Σᵢ₌ₜ₋₁₂ᵗ I(Profitᵢ > 0) + Σᵢ₌ₜ₋₁₂ᵗ I(Profitᵢ < 0)
```

### 3. Rolling Average Profit

```
                              Σᵢ₌ₜ₋₁₂ᵗ Profitᵢ
feat_rolling_avg_profit(t) = ─────────────────
                                   13
```

### 4. Rolling Volatility

```
                                  _____________________________
                                 ╱  Σᵢ₌ₜ₋₁₂ᵗ (Profitᵢ - μ̄)²
feat_rolling_volatility(t) =   ╱  ─────────────────────────────
                             ╲╱            12

ahol μ̄ = feat_rolling_avg_profit(t)
```

### 5. Rolling Sharpe

```
                          feat_rolling_avg_profit(t)
feat_rolling_sharpe(t) = ─────────────────────────────
                          feat_rolling_volatility(t)
```

### 6. Rolling Momentum (4 hét és 13 hét)

```
                              Σᵢ₌ₜ₋₃ᵗ Profitᵢ
feat_rolling_momentum_4w(t) = ────────────────  (4 hetes átlag)
                                   4

                               Σᵢ₌ₜ₋₁₂ᵗ Profitᵢ
feat_rolling_momentum_13w(t) = ─────────────────  (13 hetes átlag)
                                    13
```

**Jelentés**: Rövid- és hosszú távú trend indikátor.

### 7. Rolling Profit Sum

```
feat_rolling_profit_sum(t) = Σᵢ₌ₜ₋₁₂ᵗ Profitᵢ
```

### 8. Rolling Max Drawdown

```
feat_rolling_max_dd(t) = min(0, CumSumₜ - RollingMaxₜ)

ahol RollingMaxₜ = max(CumSumₜ₋₁₂, CumSumₜ₋₁₁, ..., CumSumₜ)
```

## Vizuális példa: Gördülő ablak

```
Hetek:  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
Profit: 50 -20  80 -10  40  60 -30  90  20 -40  70  30  50 -10  80  40

t=13:   [───────────────────── 13 hetes ablak ─────────────────────]
        50 -20  80 -10  40  60 -30  90  20 -40  70  30  50
        rolling_avg = (50-20+80-10+40+60-30+90+20-40+70+30+50)/13 = 30

t=14:       [───────────────────── 13 hetes ablak ─────────────────────]
           -20  80 -10  40  60 -30  90  20 -40  70  30  50 -10
           rolling_avg = (-20+80-10+40+60-30+90+20-40+70+30+50-10)/13 = 25.4

t=15:           [───────────────────── 13 hetes ablak ─────────────────────]
                80 -10  40  60 -30  90  20 -40  70  30  50 -10  80
                rolling_avg = 33.1
```

## Implementáció

```python
def add_features_rolling(df, window=13):
    df = df.sort_values(["No.", "Date"]).copy()
    min_periods = max(4, window // 3)

    grouped = df.groupby("No.", group_keys=False)

    # Rolling active ratio
    df["feat_rolling_active_ratio"] = grouped["_is_active"].transform(
        lambda x: x.rolling(window, min_periods=min_periods).mean()
    )

    # Rolling mean and std
    rolling_mean = grouped["Profit"].transform(
        lambda x: x.rolling(window, min_periods=min_periods).mean()
    )
    rolling_std = grouped["Profit"].transform(
        lambda x: x.rolling(window, min_periods=min_periods).std()
    )

    df["feat_rolling_avg_profit"] = rolling_mean
    df["feat_rolling_volatility"] = rolling_std
    df["feat_rolling_sharpe"] = np.where(rolling_std > 0, rolling_mean / rolling_std, 0)

    # Momentum
    df["feat_rolling_momentum_4w"] = grouped["Profit"].transform(
        lambda x: x.rolling(4, min_periods=2).mean()
    )
    df["feat_rolling_momentum_13w"] = grouped["Profit"].transform(
        lambda x: x.rolling(13, min_periods=4).mean()
    )

    # Fill NaN at the beginning
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    df[feat_cols] = df[feat_cols].fillna(0)

    return df
```

## Mikor használd?

| Használati eset | Indoklás |
|-----------------|----------|
| **Trend követés** | A közelmúlt teljesítménye számít |
| **Adaptív modellek** | Változó piaci környezethez |
| **Idősor előrejelzés** | Dinamikus feature-ök az ML-nek |
| **Rövid távú döntések** | Aktuális állapot fontosabb |

---

# 03.08 Módok összehasonlítása

## Összehasonlító táblázat

| Szempont | Original | Forward Calc | Rolling Window |
|----------|----------|--------------|----------------|
| **Feature-ök száma** | 0 | 8 | 9 |
| **Számítási idő** | Instant | Közepes | Lassú |
| **Memória használat** | Alap | +20-30% | +25-35% |
| **Időérzékenység** | Nincs | Statikus | Dinamikus |
| **Adatvesztés** | Nincs | Nincs | Első 12 hét NaN |
| **ML teljesítmény** | Baseline | Jó | Legjobb* |

*Idősor előrejelzéshez

## Feature oszlopok összehasonlítása

### Original
```
[No., Profit, Trades, PF, DD %, Date, SourceFile]
 └─ 7 oszlop
```

### Forward Calc
```
[No., Profit, Trades, PF, DD %, Date, SourceFile,
 feat_weeks_count, feat_active_ratio, feat_profit_consistency,
 feat_total_profit, feat_cumulative_trades, feat_volatility,
 feat_sharpe_ratio, feat_max_drawdown]
 └─ 15 oszlop (+8 feature)
```

### Rolling Window
```
[No., Profit, Trades, PF, DD %, Date, SourceFile,
 feat_rolling_active_ratio, feat_rolling_profit_consistency,
 feat_rolling_avg_profit, feat_rolling_volatility, feat_rolling_sharpe,
 feat_rolling_momentum_4w, feat_rolling_momentum_13w,
 feat_rolling_profit_sum, feat_rolling_max_dd]
 └─ 16 oszlop (+9 feature)
```

## Számítási komplexitás

| Mód | Idő komplexitás | Tér komplexitás |
|-----|-----------------|-----------------|
| Original | O(N) | O(N) |
| Forward | O(N × M) | O(N) |
| Rolling | O(N × M × W) | O(N) |

Ahol:
- N = sorok száma
- M = stratégiák száma
- W = ablak méret (13)

## Ajánlások modell típusonként

| Modell típus | Ajánlott mód | Indoklás |
|--------------|--------------|----------|
| **ARIMA, SARIMA** | Original | Csak az idősor kell |
| **ETS, Prophet** | Original | Saját feature-öket számítanak |
| **Random Forest, XGBoost** | Forward/Rolling | Gazdag feature-készlet segít |
| **LSTM, GRU** | Rolling | Időérzékeny minták |
| **Transformer** | Rolling | Szekvenciális kontextus |
| **Ensemble** | Forward | Stabil összehasonlítás |

## Vizuális összehasonlítás

```
                    Original           Forward Calc          Rolling Window
                    ────────           ────────────          ──────────────
Feature változás
időben:
                    ┌─────────┐        ┌─────────┐         ┌─────────┐
                    │█████████│        │▓▓▓▓▓████│         │░▓█▓░▓█▓░│
                    │█████████│        │▓▓▓▓█████│         │▓█▓░▓█▓░▓│
                    │█████████│        │▓▓▓██████│         │█▓░▓█▓░▓█│
                    │█████████│        │▓▓███████│         │▓░▓█▓░▓█▓│
                    │█████████│        │▓████████│         │░▓█▓░▓█▓░│
                    │█████████│        │█████████│         │▓█▓░▓█▓░▓│
                    └─────────┘        └─────────┘         └─────────┘

Jelkulcs:           █ = Eredeti adat
                    ▓ = Feature (nő az idővel)
                    ░ = Feature (ablakban változik)
```

## Döntési fa a mód kiválasztásához

```
                        Kezdés
                           │
                           ▼
              ┌─────────────────────────┐
              │ Kell extra feature?     │
              └───────────┬─────────────┘
                    │           │
                   NEM         IGEN
                    │           │
                    ▼           ▼
              ┌─────────┐    ┌─────────────────────────┐
              │Original │    │ A közelmúlt fontosabb?  │
              └─────────┘    └───────────┬─────────────┘
                                   │           │
                                  NEM         IGEN
                                   │           │
                                   ▼           ▼
                            ┌─────────────┐ ┌──────────────┐
                            │Forward Calc │ │Rolling Window│
                            └─────────────┘ └──────────────┘
```

---

*Dokumentum készítése: 2024.12.29*
*Program verzió: v3.60.0 Stable*
*Fejezet: 03 - Data Loading Tab*
