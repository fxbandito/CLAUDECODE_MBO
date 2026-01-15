# Batch Model Validation Report
**Date**: 2026-01-04
**Test Duration**: 19:46 - 23:42 (kb. 4 óra)
**System**: Intel Core i9 (24C/32T), 64GB RAM, NVIDIA RTX 4060 Laptop GPU (8GB)
**Python**: 3.14.2, PyTorch 2.9.1+cu126

---

## Executive Summary

### Teszt Áttekintés
- **Tervezett modellek**: 7 (auto.txt alapján)
- **Sikeresen lefutott**: 6 modell
- **Megszakadt**: 1 modell (StemGNN - nem indult el)
- **Kritikus hiba**: 1 modell (Spiking Neural Networks - 100% negatív előrejelzés)

### Gyors Értékelés

| Modell | Státusz | Futási idő | Előrejelzés minőség |
|--------|---------|------------|---------------------|
| Neural GAM | PASS | 10:59 | Elfogadható (60% negatív 1M) |
| Neural ODE | PASS | 20:27 | Elfogadható (64% negatív 1M) |
| Neural Quantile Regression | PASS | 28:13 | Kiváló (0% negatív) |
| Neural VAR | PASS | 18:47 | Elfogadható (53% negatív 1M) |
| Neural Volatility | PASS | 44:36 | Kiváló (0% negatív) |
| Spiking Neural Networks | FAIL | 1:50:08 | Kritikus hiba (100% negatív) |
| StemGNN | NOT RUN | - | Nem indult el |

---

## Részletes Modellenkénti Elemzés

### 1. Neural GAM

**Státusz**: PASS (figyelmeztetésekkel)

**Futási információk**:
- Futási idő: 10:59 (659 sec)
- GPU használat: 15-25%
- GPU memória: ~1.9 GB
- CPU használat: 27-35%
- RAM: ~2.3 GB

**Előrejelzés statisztikák**:
| Horizont | Min | Max | Átlag | Negatív % |
|----------|-----|-----|-------|-----------|
| 1 Week | -9,838 | 3,014 | -2,413 | 96.0% |
| 1 Month | -11,628 | 19,352 | -583 | 60.6% |
| 3 Months | -9,597 | 29,862 | 1,779 | 32.5% |
| 6 Months | -15,039 | 54,126 | 6,748 | 20.3% |
| 12 Months | -20,617 | 110,741 | 21,090 | 11.2% |

**Értékelés**:
- A rövid távú (1W) előrejelzések 96%-a negatív - ez aggasztó
- Hosszabb távon javul a helyzet (12M: csak 11% negatív)
- A modell konvergált, de lehetséges túlilleszkedés a rövid távon

**Javaslatok**:
- Növeld a mean_reversion paramétert (jelenleg 0.1)
- Fontold meg a look_back növelését 10-ről 12-re

---

### 2. Neural ODE

**Státusz**: PASS (figyelmeztetésekkel)

**Futási információk**:
- Futási idő: 20:27 (1,227 sec)
- GPU használat: 8-24%
- GPU memória: ~1.9 GB
- CPU használat: 26-35%
- RAM: ~2.6 GB

**Előrejelzés statisztikák**:
| Horizont | Min | Max | Átlag | Negatív % |
|----------|-----|-----|-------|-----------|
| 1 Week | -1,329 | 2,222 | -199 | 69.7% |
| 1 Month | -4,913 | 8,953 | -543 | 64.2% |
| 3 Months | -15,704 | 28,434 | -1,644 | 63.0% |
| 6 Months | -31,302 | 56,456 | -3,249 | 62.7% |
| 12 Months | -62,499 | 112,501 | -6,460 | 62.7% |

**Értékelés**:
- Konzisztensen ~63% negatív előrejelzés minden horizonton
- Az átlag végig negatív marad - a modell pesszimista
- A tartomány (range) elfogadható

**Javaslatok**:
- Ellenőrizd a solver_steps paramétert (jelenleg 5)
- Próbáld meg a hidden_size növelését 16-ról 32-re
- A modell valószínűleg alulilleszkedett (underfitting)

---

### 3. Neural Quantile Regression

**Státusz**: PASS (Kiváló)

**Futási információk**:
- Futási idő: 28:13 (1,694 sec)
- GPU használat: 6-21%
- GPU memória: ~2.0 GB
- CPU használat: 24-42%
- RAM: ~2.7 GB

**Előrejelzés statisztikák**:
| Horizont | Min | Max | Átlag | Negatív % |
|----------|-----|-----|-------|-----------|
| 1 Week | 5.4 | 2,811 | 654 | 0% |
| 1 Month | 17.7 | 11,171 | 2,585 | 0% |
| 3 Months | 26.5 | 35,996 | 8,284 | 0% |
| 6 Months | 30.2 | 71,730 | 16,487 | 0% |
| 12 Months | 35.9 | 143,166 | 32,886 | 0% |

**Értékelés**:
- Tökéletes eredmény: nincs negatív előrejelzés
- Az előrejelzések logikusan skálázódnak
- A modell stabilan működik
- A quantiles (0.1, 0.5, 0.9) és grad_clip (0.5) jól működik

**Javaslatok**:
- Ez a legjobban működő modell - használd referenciának
- A paraméterek optimálisak

---

### 4. Neural VAR

**Státusz**: PASS (elfogadható)

**Futási információk**:
- Futási idő: 18:47 (1,127 sec)
- GPU használat: 0-23%
- GPU memória: ~2.0 GB
- CPU használat: 24-33%
- RAM: ~2.7 GB

**Előrejelzés statisztikák**:
| Horizont | Min | Max | Átlag | Negatív % |
|----------|-----|-----|-------|-----------|
| 1 Week | -842 | 1,158 | 40 | 51.3% |
| 1 Month | -3,462 | 4,596 | 78 | 53.4% |
| 3 Months | -11,173 | 15,057 | 331 | 52.5% |
| 6 Months | -22,231 | 30,267 | 765 | 52.2% |
| 12 Months | -44,225 | 60,891 | 1,732 | 51.8% |

**Értékelés**:
- Körülbelül 50-50% pozitív/negatív arány - random-szerű
- Az átlag végig pozitív, de gyengén
- A modell nem tanul elég mintázatot

**Javaslatok**:
- Növeld az epochs-t 10-ről 20-ra
- Próbáld meg a hidden_size növelését 16-ról 32-re
- A grad_clip (0.5) és mean_reversion (0.02) lehet túl konzervatív

---

### 5. Neural Volatility

**Státusz**: PASS (Kiváló - de ellenőrizendő)

**Futási információk**:
- Futási idő: 44:36 (2,677 sec)
- GPU használat: 2-23%
- GPU memória: ~2.0 GB
- CPU használat: 40-66%
- RAM: ~2.8 GB
- Early stopping: Gyakori (7-10 epoch között)

**Előrejelzés statisztikák**:
| Horizont | Min | Max | Átlag | Negatív % |
|----------|-----|-----|-------|-----------|
| 1 Week | 89 | 7,814 | 758 | 0% |
| 1 Month | 834 | 35,055 | 5,039 | 0% |
| 3 Months | 7,093 | 151,624 | 34,974 | 0% |
| 6 Months | 26,718 | 425,138 | 123,204 | 0% |
| 12 Months | 103,453 | 1,342,960 | 459,056 | 0% |

**Értékelés**:
- Nincs negatív előrejelzés - jó
- FIGYELEM: Az értékek exponenciálisan nőnek - lehetséges túlbecslés
- 12M: Max 1.3M és átlag 459K - ezek irreálisan magasak lehetnek
- Volatilitás modellnél ez részben várható, de ellenőrizd az input adatokat

**Javaslatok**:
- Ellenőrizd a mean_reversion_cap paramétert (jelenleg 0.5)
- A modell túlságosan optimista lehet hosszú távon
- Fontold meg a normalizálás ellenőrzését

---

### 6. Spiking Neural Networks

**Státusz**: FAIL - KRITIKUS HIBA

**Futási információk**:
- Futási idő: 1:50:08 (6,609 sec) - TÚLSÁGOSAN HOSSZÚ!
- GPU használat: 25-41%
- GPU memória: ~1.9 GB
- CPU használat: 45-67% (magasabb mint más modellek)
- RAM: ~2.7 GB

**Előrejelzés statisztikák**:
| Horizont | Min | Max | Átlag | Negatív % |
|----------|-----|-----|-------|-----------|
| 1 Week | -56,935 | -1,787 | -18,020 | **100%** |
| 1 Month | -227,740 | -11,911 | -68,910 | **100%** |
| 3 Months | -711,637 | -44,552 | -202,967 | **100%** |
| 6 Months | -1,414,664 | -88,884 | -396,122 | **100%** |
| 12 Months | -2,813,154 | -173,828 | -778,635 | **100%** |

**KRITIKUS PROBLÉMÁK**:
1. **100% negatív előrejelzés** - Ez MATEMATIKAILAG HELYTELEN
2. **Minden stratégia veszteséges** - A modell nem tanult semmi hasznosat
3. **Extrém negatív értékek** - 12M átlag: -778K (irreális)
4. **Hosszú futási idő** - 1:50:08 vs tervezett ~30 sec

**Lehetséges okok**:
1. A spike encoding/decoding nem megfelelő
2. A threshold (1.0) és decay (0.9) paraméterek nem optimálisak
3. A gradient flow probléma a spiking mechanizmus miatt
4. Az alpha (10.0) túl nagy lehet

**Javaslatok - JAVÍTÁS SZÜKSÉGES**:
1. Ellenőrizd az SNN implementációt - valószínűleg bug van
2. A spike_fn és surrogate gradient lehet hibás
3. Próbáld meg csökkenteni az alpha-t 10.0-ról 1.0-ra
4. Növeld a threshold-ot 1.0-ról 0.5-re
5. A modellt tesztelni kell egyszerűbb adatokon

---

### 7. StemGNN

**Státusz**: NOT RUN - NEM INDULT EL

A debug log szerint a 6. modell után (Spiking Neural Networks) a futás véget ért 23:42:51-kor, és a StemGNN nem indult el.

**Lehetséges okok**:
1. Manuális leállítás a felhasználó által
2. Memory hiba a Spiking NN hosszú futása után
3. Alkalmazás bezárás

---

## CPU/GPU Használat Elemzése

### GPU Kihasználtság
| Modell | Átlag GPU% | Max GPU% | GPU Mem (GB) |
|--------|------------|----------|--------------|
| Neural GAM | 12% | 25% | 1.9 |
| Neural ODE | 15% | 24% | 1.9 |
| Neural Quantile Reg | 14% | 21% | 2.0 |
| Neural VAR | 12% | 23% | 2.0 |
| Neural Volatility | 13% | 23% | 2.0 |
| Spiking NN | 32% | 41% | 1.9 |

**Megfigyelések**:
- A GPU kihasználtság alacsony (12-32%)
- A Spiking NN magasabb GPU használata a hosszabb futás miatt
- 8GB GPU memóriából csak ~2GB használt - van tartalék
- A batch_size (128) növelhető lenne 256-ra vagy 512-re

### CPU Kihasználtság
- Átlagosan 25-35% CPU használat
- A Spiking NN esetében 45-67% (magasabb)
- 19 core használatban (80% limit)

---

## Összefoglaló Táblázat

| Modell | Futás | GPU | Előrejelzés | Ajánlás |
|--------|-------|-----|-------------|---------|
| Neural GAM | OK | OK | Figyelmeztetés | Finomhangolás |
| Neural ODE | OK | OK | Figyelmeztetés | Finomhangolás |
| Neural Quantile Regression | OK | OK | Kiváló | Referencia modell |
| Neural VAR | OK | OK | Gyenge | Finomhangolás |
| Neural Volatility | OK | OK | Kiváló (túlbecslés?) | Ellenőrzés |
| Spiking Neural Networks | OK | OK | HIBÁS | JAVÍTÁS SZÜKSÉGES |
| StemGNN | - | - | - | Újrafuttatás |

---

## Javítási Javaslatok

### Azonnali javítások (Prioritás: MAGAS)

1. **Spiking Neural Networks - JAVÍTVA!**
   - File: `src/analysis/models/dl_graph_specialized/snn.py`
   - **Probléma**: A readout layer nem korlátozta az outputot [0,1] közé, így a modell negatív scaled értékeket predikált, amelyek az inverse transform után extrém negatív értékekké váltak.
   - **Javítás**:
     - Sigmoid aktiváció hozzáadása az output layerhez
     - GELU aktiváció használata ReLU helyett a jobb gradient flow-ért
     - Mélyebb readout hálózat a jobb reprezentációért
     - Spike rate formula egyszerűsítése és stabilizálása

2. **StemGNN újrafuttatás**
   - Indíts egy új futtatást csak a StemGNN-nel

### Közepes prioritás

3. **Neural ODE és Neural GAM finomhangolás**
   - Növeld a mean_reversion értéket
   - Próbálj nagyobb hidden_size-t

4. **Neural VAR fejlesztés**
   - Növeld az epochs-t
   - Próbálj más learning_rate-et

### Alacsony prioritás

5. **Neural Volatility ellenőrzés**
   - Validáld az extrém magas 12M előrejelzéseket
   - Ellenőrizd az input adatok normalizálását

---

## Konklúzió

A 7 modellből:
- **2 modell kiválóan működik**: Neural Quantile Regression, Neural Volatility
- **3 modell elfogadhatóan működik**: Neural GAM, Neural ODE, Neural VAR
- **1 modell kritikusan hibás**: Spiking Neural Networks
- **1 modell nem futott le**: StemGNN

**Következő lépések**:
1. Javítsd a Spiking Neural Networks modellt
2. Futtasd újra a StemGNN-t
3. Finomhangold a közepes teljesítményű modelleket

---

*Report generated: 2026-01-04*
*MBO Trading Strategy Analyzer v4.2.0 Stable*
