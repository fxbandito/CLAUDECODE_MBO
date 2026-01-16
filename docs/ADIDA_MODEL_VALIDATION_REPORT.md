# ADIDA Model Validation Report
**Date:** 2026-01-16 (Updated)
**Model:** ADIDA (Aggregate-Disaggregate Intermittent Demand Approach)
**Test Data:** AUDJPY_2020_2021_2022_2023_Weekly_Fix.parquet

---

## 1. Executive Summary

Az ADIDA modell sikeresen implementalva es tesztelve lett. Minden teszt sikeresen lefutott az AUDJPY adatokon (6144 strategia, 208 adatpont/strategia).

| Teszt | Eredmeny | Megjegyzes |
|-------|----------|------------|
| Parameter betoltes | PASS | GUI integracio helyes |
| Standard ADIDA | PASS | 0.95-661 ms |
| Croston modszer | PASS | 0.10 ms |
| SBA modszer | PASS | 0.08 ms |
| TSB modszer | PASS | 0.07 ms |
| Batch mod | PASS | 15.82 ms / 5 strategia |
| CPU/GPU kezeles | PASS | CUDA elerheto |
| Feature Mode kompatibilitas | PASS | Figyelmeztetes mukodik |
| Model Capabilities | PASS | Panel/Dual letiltva GUI-n |

### 1.1 Uj Funkciok

- **Feature Mode Ellenorzes**: A modell figyelmezteti a felhasznalot ha Forward Calc vagy Rolling Window modban van
- **GUI Kapcsolok**: GPU, Panel Mode, Dual Mode kapcsolok automatikusan letiltodnak ha a modell nem tamogatja
- **Batch Mode**: Parhuzamos feldolgozas joblib-bal (max 8 szal)
- **Utils Integracios**: Teljes integracios az uj `models/utils/` modulokkal

### 1.2 Utils Modulus Integracios (uj architektura)

Az ADIDA modell az elso, ami használja az uj utils modulokat:

| Modul | Funkcionalitas | Hasznalat ADIDA-ban |
|-------|----------------|---------------------|
| `postprocessing.py` | NaN kezeles, outlier capping | `self.postprocess()` |
| `aggregation.py` | Horizont aggregatumok (h1-h52) | `self.calculate_horizons()` |
| `validation.py` | Input validalas | `self.validate_input()` |
| `monitoring.py` | Teljesitmeny meres | `self.create_timer()` |

**Uj metodus - `forecast_with_pipeline()`:**
```python
result = model.forecast_with_pipeline(
    data=profit_series,
    steps=52,
    params={"method": "croston"},
    strategy_id="STR_001"
)
# Visszaad: {"Forecast_1W", "Forecast_1M", ..., "Success", "Error"}
```

Ez a metodus a regi `strategy_analyzer.py` teljes funkcionalitasat biztositja egyetlen hivassal.

---

## 2. Algoritmikus Elemzés

### 2.1 ADIDA Elméleti Háttér

**Referenciák:**
- [Nikolopoulos et al. (2011)](https://link.springer.com/article/10.1057/jors.2010.32) - Eredeti ADIDA módszer
- [Croston (1972)](https://www.researchgate.net/publication/220636564) - Croston módszer
- [Nixtla - CrostonSBA](https://nixtlaverse.nixtla.io/statsforecast/docs/models/crostonsba.html) - SBA implementáció
- [pyInterDemand](https://github.com/Valdecy/pyInterDemand) - Python könyvtár

### 2.2 Implementált Módszerek

#### Standard ADIDA
```
1. Aggregálás: Y_agg[t] = sum(Y[t*k : (t+1)*k])
2. Előrejelzés: F_agg = BaseModel.forecast(Y_agg)
3. Disaggregálás: F[i] = F_agg * weight[i % k]
```

#### Croston
```
z_hat = alpha * z[t] + (1-alpha) * z_hat  (kereslet méret)
p_hat = beta * p[t] + (1-beta) * p_hat    (időköz)
Forecast = z_hat / p_hat
```

#### SBA (Syntetos-Boylan Approximation)
```
Forecast = (1 - beta/2) * z_hat / p_hat
```
**Megjegyzés:** A torzítás-korrekciós faktor `(1 - beta/2)` helyes az irodalom alapján.

#### TSB (Teunter-Syntetos-Babai)
```
Minden periódusban frissül:
- Van kereslet: p_hat = beta * 1 + (1-beta) * p_hat
- Nincs kereslet: p_hat = beta * 0 + (1-beta) * p_hat
Forecast = p_hat * z_hat
```

---

## 3. Azonosított és Javított Problémák

### 3.1 Régi Kód Hibái

| # | Hiba | Súlyosság | Javítás |
|---|------|-----------|---------|
| 1 | Egy alpha paraméter mindkét simításhoz | Közepes | Külön alpha és beta paraméter |
| 2 | TSB módszer hiányzik | Közepes | Implementálva |
| 3 | Rekurzív hívás auto aggregáció esetén | Alacsony | Külön metódusok |
| 4 | Batch mód nem támogatott | Közepes | Párhuzamos feldolgozás joblib-bal |
| 5 | Edge case kezelés hiányos | Alacsony | Robusztus NaN kezelés |

### 3.2 Matematikai Korrekciók

**SBA Formula:**
- Régi: `(1 - alpha/2) * z_hat / p_hat`
- Új (helyes): `(1 - beta/2) * z_hat / p_hat`

Az irodalom szerint az SBA korrekciós faktor az időköz simítási paramétert (beta) használja, nem a kereslet simítási paramétert (alpha).

---

## 4. Paraméterek és GUI Integráció

### 4.1 Új Paraméter Struktúra

```python
PARAM_DEFAULTS = {
    "method": "standard",
    "aggregation_level": "4",
    "base_model": "SES",
    "alpha": "0.1",      # Kereslet simítás
    "beta": "0.1",       # Időköz simítás
    "use_weighted_disagg": "True",
    "optimize": "False",
    "seasonal_periods": "12",
}

PARAM_OPTIONS = {
    "method": ["standard", "croston", "sba", "tsb"],
    "aggregation_level": ["auto", "2", "3", "4", "5", "6", "8", "12"],
    "base_model": ["SES", "ARIMA", "ETS", "Theta", "Naive"],
    "alpha": ["0.05", "0.1", "0.15", "0.2", "0.3", "0.5"],
    "beta": ["0.05", "0.1", "0.15", "0.2", "0.3", "0.5"],
    "use_weighted_disagg": ["True", "False"],
    "optimize": ["True", "False"],
    "seasonal_periods": ["4", "7", "12", "24", "52"],
}
```

### 4.2 GUI Integráció Ellenőrzése

- [x] Model registry-be regisztrálva
- [x] Kategória: "Statistical Models"
- [x] Paraméterek dinamikusan betöltődnek
- [x] ComboBox-ok helyesen generálódnak
- [x] GPU/Batch támogatás flag-ek helyesek

---

## 5. CPU/GPU Kezelés

### 5.1 Jelenlegi Állapot

```
MODEL_INFO = ModelInfo(
    name="ADIDA",
    category="Statistical Models",
    supports_gpu=False,      # Statisztikai modell
    supports_batch=True,     # Párhuzamos feldolgozás
    gpu_threshold=1000,
)
```

### 5.2 GPU Elemzés

Az ADIDA egy statisztikai módszer, amely nem profitál a GPU gyorsításból:
- Nincs mátrix művelet
- Nincs neurális háló
- Szekvenciális simítási algoritmusok

**Ajánlás:** GPU támogatás nem szükséges, a batch mód elegendő.

### 5.3 Batch Mód Implementáció

```python
def forecast_batch(self, all_data, steps, params):
    from joblib import Parallel, delayed
    n_jobs = min(os.cpu_count() or 4, len(all_data), 8)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_single)(name, data)
        for name, data in all_data.items()
    )
    return dict(results)
```

**Teljesítmény:**
- 5 stratégia: 20.39 ms (4.08 ms/stratégia)
- Párhuzamos feldolgozás: akár 8 szál

---

## 6. Teljesítmény Eredmények

### 6.1 AUDJPY Teszt Eredmények

| Módszer | Idő (ms) | Forecast Minta |
|---------|----------|----------------|
| Standard (SES) | 1.26 | [2076.75, 1987.65, 1477.22, ...] |
| Standard (auto+ARIMA) | 1052.55 | [2181.39, 2397.26, 2411.05, ...] |
| Croston | 0.17 | [3078.03, 3078.03, ...] |
| SBA | 0.14 | [3557.41, 3557.41, ...] |
| TSB | 0.14 | [3032.69, 3032.69, ...] |

### 6.2 Megfigyelések

1. **Standard ADIDA + SES**: Gyors (1.26 ms), változatos forecast
2. **Standard ADIDA + ARIMA**: Lassabb (1052 ms) az auto aggregáció miatt
3. **Croston/SBA/TSB**: Nagyon gyors (0.14-0.17 ms), konstans forecast (elvárt viselkedés)

---

## 7. GPU Optimalizációs Lehetőségek Elemzése

A felhasználó által említett szempontok vizsgálata:

### 7.1 CUDA Graphs
**Nem alkalmazható** - Az ADIDA nem használ GPU-t, nincs statikus input shape.

### 7.2 Non-Blocking Data Transfer
**Nem alkalmazható** - Nincs GPU-ra történő adatátvitel.

### 7.3 GPU Sequential Mode
**Nem releváns** - A modell CPU-n fut.

### 7.4 DataLoader num_workers
**Részben releváns** - A batch mód joblib-ot használ párhuzamos feldolgozásra:
- Jelenlegi: max 8 szál
- Beállítható az `n_jobs` paraméterrel

### 7.5 Párhuzamos Stratégia Feldolgozás
**Implementálva** - A `forecast_batch` metódus párhuzamosan dolgozza fel a stratégiákat.

### 7.6 GPU Kihasználtság
**Nem releváns az ADIDA-nál** - A modell nem használ GPU-t. Más modelleknél (pl. LSTM, Transformer) vizsgálandó.

---

## 8. Javaslatok

### 8.1 Azonnali Javítások (Implementálva)
- [x] Külön alpha és beta paraméter
- [x] TSB módszer hozzáadása
- [x] Batch mód támogatás
- [x] Robusztus NaN kezelés

### 8.2 Jövőbeli Fejlesztések
1. **Optimális paraméter keresés**: Grid search vagy Optuna integráció az alpha/beta optimalizálásra
2. **MAPA (Multiple Aggregation Prediction Algorithm)**: Több aggregációs szint kombinálása
3. **Hibrid módszer**: Standard + Croston kombinálása az adattípus alapján

---

## 9. Teszt Fájlok

- `src/models/statistical/adida.py` - Új implementáció
- `tests/test_adida_model.py` - Validációs tesztek

---

## 10. Konklúzió

Az ADIDA modell sikeresen implementálva lett az új struktúrában:
- Matematikailag helyes algoritmusok
- Teljes paraméter támogatás a GUI-hoz
- Batch mód párhuzamos feldolgozással
- Minden teszt sikeres az AUDJPY adatokon

A modell készen áll a produkcióra.

---

**Készítette:** Claude Opus 4.5
**Verzió:** 2.0 (Utils Integracios)
**Utolsó frissítés:** 2026-01-16

---

## Appendix: Uj Fajlok

### Models Utils (a regi strategy_analyzer.py funkcionalitasa)
- `src/models/utils/__init__.py` - Kozponti export
- `src/models/utils/postprocessing.py` - NaN kezeles, outlier capping
- `src/models/utils/aggregation.py` - Horizont aggregatumok
- `src/models/utils/validation.py` - Input validalas
- `src/models/utils/monitoring.py` - Teljesitmeny monitoring
- `src/models/utils/README.md` - Dokumentacios

### ADIDA Model
- `src/models/statistical/adida.py` - Teljes implementacios
- `tests/test_adida_model.py` - Validaciós tesztek
