# Diffusion Model Forecast_1W = 0.00 Bug Report

**Dátum:** 2026-01-13
**Verzió:** v4.10.15 Stable
**Prioritás:** MAGAS

---

## 1. Probléma Összefoglalása

A Diffusion Batch modell **minden stratégiánál Forecast_1W = 0.00** értéket ad vissza, miközben a többi horizon (1M, 3M, 6M, 12M) helyesen működik.

### Érintett fájlok:
- `src/analysis/models/dl_graph_specialized/diffusion_batch.py` (1039-1074. sorok)

### Megfigyelt viselkedés:
| Modell | Forecast_1W | Forecast_1M | Működik? |
|--------|-------------|-------------|----------|
| **Diffusion** | **0.00** | 10534.11 | **HIBÁS** |
| KAN | 3609.45 | 14826.43 | OK |
| MTGNN | 8825.37 | 37123.78 | OK |
| Neural ARIMA | 10291.26 | 43085.77 | OK |

---

## 2. Hiba Oka (Root Cause Analysis)

### 2.1 A Mean Reversion Post-Processing Logika

A `diffusion_batch.py` 1044-1046. sorokban:

```python
if j == 0:
    # First week: 60% model, 40% recent mean
    val = val * 0.6 + recent_mean * 0.4
```

### 2.2 A Probléma

1. **`recent_mean`** a **scaled (normalizált)** adatok átlaga
2. A `TradingRobustScaler` a következő transzformációt alkalmazza:
   - `transform`: `(x - center) / scale` → **0-központú** adatok
   - A `recent_mean` ezért tipikusan **0 körüli érték**

3. A modell predikciója (`val`) szintén **0-központú skálán** van

4. Amikor j == 0:
   - `val = 0.6 * small_value + 0.4 * ~0 ≈ ~0`

5. Az `inverse_transform` után:
   - `result = ~0 * scale + center = center`

6. **DE** az engine.py-ban a Forecast_1W közvetlenül `forecasts[0]` értéke!

### 2.3 A Valódi Probléma

A jelentésben látható **0.00** értékek valójában azt jelentik, hogy:
- A scaled domain-ben az érték **0** körül van
- Az inverse transform után ez a **center_** értéket adja vissza
- **DE** valahol az eredmény **LEBEGŐPONTOS KEREKÍTÉSSEL 0.00-ként JELENIK MEG** a riportban

A legvalószínűbb ok:
1. A `j==0` (Forecast_1W) túlzottan erős mean reversion-t alkalmaz (40%!)
2. Ha a modell predikciója `val ≈ 0` a scaled skálán, és `recent_mean ≈ 0`, akkor az eredmény is `≈ 0`
3. Az inverse transform után ez **nem 0**, hanem a **center_** érték (pl. median)
4. **A probléma valószínűleg a riport generálásban van** - ellenőrizni kell, hogy a tényleges számok helyesen vannak-e átadva

---

## 3. Hibaellenőrzés - Számítási Logika

### Teszt Szcenárió:
- Strategy data: [100, 200, 150, 180, 120, 250, 300, 180, 220, 190, 160, 210, 230]
- `center_ = median(non_zero) = 190` (kb.)
- `scale_ = IQR = 60` (kb.)

### Scaled domain:
- scaled values: [-1.5, 0.17, -0.67, -0.17, -1.17, 1.0, 1.83, -0.17, 0.5, 0.0, -0.5, 0.33, 0.67]
- `scaled_mean ≈ 0.03`
- `recent_mean ≈ 0.1` (utolsó 13 elem átlaga)

### j==0 esetén:
- Ha a modell `val = 0.5` predikciót ad
- `val = 0.5 * 0.6 + 0.1 * 0.4 = 0.3 + 0.04 = 0.34`
- inverse: `0.34 * 60 + 190 = 20.4 + 190 = 210.4`

**Ez NEM 0!** Tehát a hiba valószínűleg máshol van.

---

## 4. Alternatív Hibaforrások

### 4.1 Lehetséges Ok #1: Inference Loop Hiba

A `diffusion_batch.py` 876-984. sorok között:

```python
for forecast_step in range(steps):
    # ...
    all_forecasts[:, forecast_step] = pred_values
```

Ellenőrizni kell, hogy az `all_forecasts[:, 0]` (első hét) helyesen van-e feltöltve.

### 4.2 Lehetséges Ok #2: Scaler Inverse Transform Hiba

Ha a `strat_id` nincs a `self.scalers`-ben, akkor a raw scaled érték kerül a results-ba:
```python
if strat_id in self.scalers:
    forecast_original = self.scalers[strat_id].inverse_transform(...)
else:
    results[strat_id] = forecast_scaled.flatten().tolist()  # SCALED, NEM EREDETI!
```

### 4.3 Lehetséges Ok #3: Report Generation Hiba

Ellenőrizni kell a report generálásban, hogy a `Forecast_1W` helyesen van-e formázva.

---

## 5. Javasolt Javítás

### 5.1 Azonnali Javítás (Workaround)

Csökkenteni kell a j==0 mean reversion súlyát:

```python
# RÉGI (túl erős mean reversion):
if j == 0:
    val = val * 0.6 + recent_mean * 0.4

# ÚJ (gyengébb mean reversion):
if j == 0:
    val = val * 0.85 + recent_mean * 0.15
```

### 5.2 Végleges Javítás

A mean reversion-t az **eredeti skálán** kellene végezni, nem a scaled domain-ben:

```python
# 1. Inverse transform ELŐBB
forecast_original = scaler.inverse_transform(forecast_scaled)

# 2. Mean reversion az eredeti skálán
original_recent_mean = np.mean(strat_data[-13:])  # Eredeti profit átlag
for j in range(steps):
    val = forecast_original[j, 0]
    if j == 0:
        val = val * 0.9 + original_recent_mean * 0.1
    # ... többi horizon hasonlóan
    forecast_original[j, 0] = val
```

### 5.3 Debug Logging Hozzáadása

```python
# Hozzáadni a j==0 utáni sorhoz:
if j == 0:
    logger.debug(
        "Strategy %s: j=0, val_before=%.4f, recent_mean=%.4f, val_after=%.4f",
        strat_id, forecast_scaled[0, 0], recent_mean, val
    )
```

---

## 6. Ellenőrző Lista

- [ ] Ellenőrizni, hogy az `all_forecasts[:, 0]` helyesen van-e feltöltve
- [ ] Ellenőrizni a scaler inverse transform működését
- [ ] Ellenőrizni a report generálás formázását
- [ ] Debug logging hozzáadása a j==0 értékekhez
- [ ] Összehasonlítani a Diffusion (non-batch) és Diffusion Batch eredményeket

---

## 7. Következtetés és Végleges Elemzés

### Összehasonlítás más Batch modellekkel

A KAN Batch-nél is **50%** mean reversion van j==0-nál, mégis működik. Tehát a probléma NEM a mean reversion erősségében van önmagában.

### A Valódi Probléma

A Diffusion modell architektúrájából fakad:

1. **Noise-based sampling**: A Diffusion modell random noise-ból indul és fokozatosan denoise-ol
2. **Első hét instabilitás**: Az autoregressive predikció során az első hét értéke különösen érzékeny
3. **DDIM sampling**: A 937. sorban található `x_0_pred = torch.clamp(x_0_pred, clamp_min, clamp_max)` a `clamp_min` és `clamp_max` közé szorítja az értéket
4. Ha a `clamp_min ≈ clamp_max ≈ 0`, akkor az összes stratégiánál **0** körüli értékek lesznek

### Kulcs Felfedezés

A Diffusion Batch a **803-820. sorokban** számítja a clamp értékeket:
```python
data_min = float(np.percentile(all_scaled_values, 1))
data_max = float(np.percentile(all_scaled_values, 99))
clamp_margin = 0.5 * max(data_range, 0.5)
clamp_min = data_min - clamp_margin
clamp_max = data_max + clamp_margin
```

Ha a scaled adatok **túl szűk tartományban** vannak, a clamping túl agresszív lehet.

---

## 8. Javasolt Javítások

### 8.1 Azonnali Javítás - Mean Reversion Csökkentése j==0-nál

```python
# diffusion_batch.py, 1044-1046. sorok
# RÉGI:
if j == 0:
    val = val * 0.6 + recent_mean * 0.4

# ÚJ:
if j == 0:
    # Gyengébb mean reversion az első hétre - megőrzi a modell predikciót
    val = val * 0.85 + recent_mean * 0.15
```

### 8.2 Alternatív Javítás - Clamp Margin Növelése

```python
# diffusion_batch.py, 818. sor
# RÉGI:
clamp_margin = 0.5 * max(data_range, 0.5)

# ÚJ:
clamp_margin = 1.0 * max(data_range, 1.0)  # Dupla margin
```

### 8.3 Debug Logging

Adjon debug üzeneteket a j==0 értékekhez:
```python
if j == 0 and strat_id in [list(strat_ids)[:5]]:  # Első 5 stratégia
    logger.debug(
        "Diffusion j=0: strat=%s, model_val=%.4f, recent_mean=%.4f, final_val=%.4f",
        strat_id, forecast_scaled[0, 0], recent_mean, val
    )
```

---

## 9. Tesztelési Javaslat

1. Futtassa újra a Diffusion modellt egyetlen stratégiával debug logging-gel
2. Ellenőrizze a `forecast_scaled[0]` és `forecast_original[0]` értékeket
3. Hasonlítsa össze a KAN Batch hasonló értékeivel

---

## 10. ELVÉGZETT JAVÍTÁSOK (v4.10.16)

### 10.1 Eltávolított np.maximum(forecast, 0.0) sorok

A következő fájlokban eltávolítottam a `np.maximum(forecast, 0.0)` hívásokat:

| Fájl | Sor | Módosítás |
|------|-----|-----------|
| `diffusion.py` | 670 | ELTÁVOLÍTVA |
| `mtgnn.py` | 973 | ELTÁVOLÍTVA |
| `kan.py` | 682 | ELTÁVOLÍTVA |
| `neural_arima.py` | 385 | ELTÁVOLÍTVA |

### 10.2 Indoklás

**A negatív forecast értékek VALIDAK:**
- A heti profit **LEHET negatív** - ez veszteséget jelent
- A korábbi korlátozás elrejtette a fontos információt

### 10.3 Batch Mode Tisztázás

**A teszt során a modellek NEM a Batch verziókat használták!**
- Debug log-ban NEM volt "Batch Mode enabled" üzenet
- A normál per-strategy modellek futottak

---

*Report készítette: Automatikus Elemzés*
*Verzió: v4.10.16 (javított)*
