# Python Környezet Elemzési Jelentés

**Generálva:** 2026-01-13
**Projekt:** gemini_mbo v4.4.0
**Python verzió:** Python 3.x (global environment)

---

## 1. Összefoglaló

| Kategória | Státusz |
|-----------|---------|
| Telepített csomagok | 176 db |
| requirements.txt-ben definiált | 32 db (fő + opcionális) |
| Hiányzó csomagok | 4 db |
| Elavult csomagok | 32 db |
| Kritikus inkompatibilitások | 2 db |

---

## 2. Kritikus Inkompatibilitások

### 2.1 ONNX verzió konfliktus
```
hummingbird-ml 0.4.12 requires onnx<=1.16.1, but you have onnx 1.20.0
```
**Megoldás:** A requirements.txt-ben a hummingbird-ml ki van kommentezve ("DISABLED: Forces broken ONNX 1.16.1"). Ez szándékos, az onnx 1.20.0 a jobb választás.

### 2.2 Huggingface-hub verzió konfliktus
```
transformers 4.57.3 requires huggingface-hub<1.0,>=0.34.0, but you have huggingface-hub 1.2.3
```
**Javaslat:**
- Downgrade: `pip install huggingface-hub==0.34.0`
- Vagy upgrade transformers ha elérhető újabb verzió

---

## 3. Hiányzó Csomagok

A kódban használt, de **NEM telepített** csomagok:

| Import név | Csomag név | Használat |
|------------|------------|-----------|
| `tensorflow` | tensorflow | 1 fájl (deep learning) |
| `pygame` | pygame | 1 fájl (sound_manager.py - hangok) |
| `nolitsa` | nolitsa | 1 fájl (nemlineáris idősorelemzés) |
| `statsforecast` | statsforecast | 1 fájl (CES model) |

**Megjegyzések:**
- `statsforecast` - requirements.txt-ben kommentezve: "not yet compatible with Python 3.14"
- `tensorflow` - valószínűleg opcionális, torch is használva van
- `pygame` - csak hangokhoz, opcionális funkció
- `nolitsa` - speciális nemlineáris elemzéshez

---

## 4. Hiányzó a requirements.txt-ből

A kódban használt, de a **requirements.txt-ből hiányzó** csomagok:

| Csomag | Telepítve | Leírás |
|--------|-----------|--------|
| `streamlit` | 1.52.2 | Web UI (kommentezve van: "optional") |
| `requests` | 2.32.5 | HTTP kérések |
| `threadpoolctl` | 3.6.0 | Thread pool kontroll |

**Javaslat:** Ezeket érdemes hozzáadni a requirements.txt-hez.

---

## 5. Elavult Csomagok (32 db)

| Csomag | Telepített | Legújabb | Prioritás |
|--------|------------|----------|-----------|
| **scipy** | 1.16.3 | 1.17.0 | Magas |
| **numpy** | 2.3.5 | 2.4.1 | Magas |
| **xgboost** | 3.1.2 | 3.1.3 | Közepes |
| **plotly** | 6.5.0 | 6.5.1 | Közepes |
| **pillow** | 12.0.0 | 12.1.0 | Közepes |
| **psutil** | 7.2.0 | 7.2.1 | Alacsony |
| **ruff** | 0.14.10 | 0.14.11 | Alacsony |
| pysindy | 2.0.0 | 2.1.0 | Közepes |
| onnx | 1.20.0 | 1.20.1 | Alacsony |
| huggingface_hub | 1.2.3 | 1.3.1 | Magas* |
| juliacall | 0.9.26 | 0.9.31 | Közepes |
| certifi | 2025.11.12 | 2026.1.4 | Közepes |
| fsspec | 2025.12.0 | 2026.1.0 | Alacsony |
| holidays | 0.87 | 0.88 | Alacsony |
| jsonschema | 4.25.1 | 4.26.0 | Alacsony |
| alembic | 1.17.2 | 1.18.0 | Alacsony |
| anyio | 4.12.0 | 4.12.1 | Alacsony |
| astroid | 4.0.2 | 4.0.3 | Alacsony |
| Cython | 3.2.3 | 3.2.4 | Alacsony |
| filelock | 3.20.1 | 3.20.3 | Alacsony |
| GitPython | 3.1.45 | 3.1.46 | Alacsony |
| librt | 0.7.5 | 0.7.7 | Alacsony |
| narwhals | 2.14.0 | 2.15.0 | Alacsony |
| pathspec | 0.12.1 | 1.0.3 | Alacsony |
| protobuf | 6.33.2 | 6.33.4 | Alacsony |
| pypdf | 6.5.0 | 6.6.0 | Alacsony |
| tokenizers | 0.22.1 | 0.22.2 | Alacsony |
| tomli | 2.3.0 | 2.4.0 | Alacsony |
| tomlkit | 0.13.3 | 0.14.0 | Alacsony |
| typer-slim | 0.21.0 | 0.21.1 | Alacsony |
| urllib3 | 2.6.2 | 2.6.3 | Alacsony |
| zope.interface | 8.1.1 | 8.2 | Alacsony |

*huggingface_hub - downgrade szükséges a transformers kompatibilitáshoz!

---

## 6. Verzió Ellenőrzés - requirements.txt vs Telepített

| Csomag | requirements.txt | Telepített | Státusz |
|--------|------------------|------------|---------|
| pandas | >=2.0.0,<3.0.0 | 2.3.3 | OK |
| numpy | >=2.0.0 | 2.3.5 | OK |
| scipy | >=1.10.0,<2.0.0 | 1.16.3 | OK |
| openpyxl | >=3.1.0 | 3.1.5 | OK |
| pyarrow | >=14.0.0 | 22.0.0 | OK |
| matplotlib | >=3.7.0 | 3.10.8 | OK |
| seaborn | >=0.12.0 | 0.13.2 | OK |
| plotly | >=5.15.0 | 6.5.0 | OK |
| scikit-learn | >=1.3.0,<2.0.0 | 1.8.0 | OK |
| xgboost | >=2.0.0 | 3.1.2 | OK |
| lightgbm | >=4.0.0 | 4.6.0 | OK |
| gplearn | >=0.4.2 | 0.5.dev0 | OK |
| pysr | >=1.0.0 | 1.5.9 | OK |
| onnx | >=1.14.0 | 1.20.0 | OK |
| torch | >=2.0.0 | 2.9.1+cu130 | OK (CUDA) |
| statsmodels | >=0.14.0 | 0.14.6 | OK |
| pmdarima | >=2.0.0 | 2.1.1 | OK |
| prophet | >=1.1.0 | 1.2.1 | OK |
| arch | >=6.0.0 | 8.0.0 | OK |
| PyWavelets | >=1.4.0 | 1.9.0 | OK |
| stumpy | >=1.12.0 | 1.13.0 | OK |
| customtkinter | >=5.2.0 | 5.2.2 | OK |
| Pillow | >=10.0.0 | 12.0.0 | OK |
| joblib | >=1.3.0 | 1.5.3 | OK |
| psutil | >=5.9.0 | 7.2.0 | OK |
| GPUtil | >=1.4.0 | 1.4.0 | OK |
| optuna | >=4.0.0 | 4.6.0 | OK |
| optuna-dashboard | >=0.15.0 | 0.20.0 | OK |
| kaleido | >=1.0.0 | 1.2.0 | OK |
| ruff | >=0.1.0 | 0.14.10 | OK |
| mypy | >=1.5.0 | 1.19.1 | OK |
| pytest | >=7.4.0 | 9.0.2 | OK |
| MetaTrader5 | >=5.0.0 | 5.0.5488 | OK |

---

## 7. Használati Statisztikák

A leggyakrabban importált csomagok (fájlok száma szerint):

| Csomag | Fájlok száma |
|--------|--------------|
| numpy | 128 |
| torch | 79 |
| pandas | 72 |
| sklearn | 50 |
| customtkinter | 17 |
| statsmodels | 16 |
| scipy | 14 |
| streamlit | 13 |
| psutil | 9 |
| plotly | 7 |

---

## 8. Javaslatok

### Azonnali teendők (kritikus):
1. **transformers/huggingface-hub konfliktus megoldása:**
   ```bash
   pip install huggingface-hub==0.34.0
   ```

### Közepes prioritás:
2. **Hiányzó csomagok telepítése (opcionális):**
   ```bash
   pip install pygame  # hangokhoz
   pip install nolitsa  # nemlineáris elemzéshez
   ```

3. **Fő csomagok frissítése:**
   ```bash
   pip install --upgrade scipy numpy xgboost plotly pillow
   ```

### Alacsony prioritás:
4. **requirements.txt kiegészítése:**
   - Hozzáadni: `streamlit>=1.28.0`, `requests>=2.31.0`, `threadpoolctl>=3.0.0`

5. **Összes frissítés (óvatosan):**
   ```bash
   pip install --upgrade alembic anyio astroid certifi Cython filelock fsspec GitPython holidays jsonschema juliacall librt narwhals onnx pathspec pillow plotly protobuf psutil pypdf pysindy ruff scipy tokenizers tomli tomlkit typer-slim urllib3 xgboost zope.interface
   ```
   **Figyelem:** numpy és huggingface-hub NE legyen frissítve a kompatibilitási problémák miatt!

---

## 9. Környezet Összegzés

- **Fő ML stack:** numpy, pandas, scikit-learn, torch, xgboost, lightgbm - mind rendben
- **Idősorelemzés:** statsmodels, pmdarima, prophet, arch - mind rendben
- **GUI:** customtkinter, streamlit - rendben
- **Optimalizálás:** optuna - rendben
- **GPU támogatás:** CUDA 13.0 (torch 2.9.1+cu130)

A környezet alapvetően jó állapotban van. A két kritikus inkompatibilitás közül az ONNX szándékos (hummingbird-ml letiltva), a huggingface-hub/transformers konfliktust érdemes orvosolni.
