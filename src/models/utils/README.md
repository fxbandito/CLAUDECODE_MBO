# Model Utilities

Közös utility függvények minden forecasting modellhez.

Ez a modul a régi `strategy_analyzer.py` funkcionalitását tartalmazza moduláris, újrafelhasználható formában.

## Modulok

| Modul | Leírás | Eredeti forrás |
|-------|--------|----------------|
| `postprocessing.py` | Forecast utófeldolgozás (NaN, outlier, clip) | `strategy_analyzer.py:464-490` |
| `aggregation.py` | Horizont aggregátumok (h1, h4, h13, h26, h52) | `strategy_analyzer.py:492-497` |
| `validation.py` | Input validálás és előfeldolgozás | `strategy_analyzer.py:149-156` |
| `monitoring.py` | Teljesítmény monitoring és logging | `strategy_analyzer.py:512-517` |

## Gyors használat

### BaseModel wrapper metódusokkal (ajánlott)

```python
from models.base import BaseModel, ModelInfo

class MyModel(BaseModel):
    MODEL_INFO = ModelInfo(name="MyModel", category="Statistical Models")
    PARAM_DEFAULTS = {"param1": "1"}
    PARAM_OPTIONS = {"param1": ["1", "2", "3"]}

    def forecast(self, data, steps, params):
        # 1. Validálás (automatikusan kategória alapján)
        validation = self.validate_input(data)
        if not validation.is_valid:
            return [0.0] * steps

        # 2. Modell logika
        raw_forecasts = self._run_model(validation.processed_data, steps)

        # 3. Utófeldolgozás
        return self.postprocess(raw_forecasts)
```

### Teljes pipeline egyetlen hívással

```python
model = ARIMAModel()
result = model.forecast_with_pipeline(
    data=profit_series,
    steps=52,
    params={"p": 1, "d": 1, "q": 1},
    strategy_id="STR_001"
)

# Eredmény kompatibilis a régi strategy_analyzer.py kimenetével
print(result["Forecast_1M"])  # 1 hónapos előrejelzés
print(result["Success"])       # True/False
```

### Közvetlen import (haladó használat)

```python
from models.utils import (
    postprocess_forecasts,
    calculate_horizons,
    validate_input_data,
    ForecastTimer,
)

# Validálás
result = validate_input_data(data, min_points=24)
if not result.is_valid:
    print(f"Error: {result.error_message}")

# Utófeldolgozás
clean = postprocess_forecasts(raw_forecasts)

# Horizontok
horizons = calculate_horizons(forecasts)
print(f"1M: {horizons.h4}, 3M: {horizons.h13}")

# Időmérés
with ForecastTimer("LSTM", strategy_id="STR_001") as timer:
    result = model.forecast(data, steps)
```

---

## Részletes dokumentáció

### 1. Postprocessing (`postprocessing.py`)

#### Funkciók

| Függvény | Leírás |
|----------|--------|
| `postprocess_forecasts()` | Teljes utófeldolgozás |
| `handle_nan_values()` | NaN/Inf kezelés |
| `cap_outliers()` | 3×IQR outlier capping |
| `clip_extreme_values()` | Min/max clipping |

#### PostprocessingConfig

```python
from models.utils.postprocessing import PostprocessingConfig

config = PostprocessingConfig(
    handle_nan=True,           # NaN → 0.0
    handle_inf=True,           # Inf → 0.0
    cap_outliers=True,         # IQR capping
    iqr_multiplier=3.0,        # Q3 + 3×IQR
    min_data_for_iqr=4,        # Minimum pont IQR-hez
    allow_negative=True,       # Negatív értékek OK
    min_value=None,            # Nincs alsó határ
    max_value=None,            # Nincs felső határ
    nan_replacement=0.0,       # NaN helyettesítő
)
```

#### Speciális postprocessing

```python
from models.utils.postprocessing import (
    postprocess_volatility_forecast,   # Volatilitás (mindig pozitív)
    postprocess_probability_forecast,  # Probability [0, 1]
    postprocess_count_forecast,        # Darabszám (egész, nem-negatív)
)
```

---

### 2. Aggregation (`aggregation.py`)

#### Standard horizontok

```python
STANDARD_HORIZONS = {
    "h1": 1,    # 1 hét
    "h4": 4,    # 1 hónap (4 hét)
    "h13": 13,  # 3 hónap
    "h26": 26,  # 6 hónap
    "h52": 52,  # 12 hónap
}
```

#### HorizonResult

```python
from models.utils.aggregation import calculate_horizons

result = calculate_horizons(forecasts)

# Attribútumok
result.h1    # 1 hetes
result.h4    # 1 hónapos (kumulatív)
result.h13   # 3 hónapos (kumulatív)
result.h26   # 6 hónapos (kumulatív)
result.h52   # 12 hónapos (kumulatív)

# Dictionary export
result.to_dict()  # {"Forecast_1W": h1, "Forecast_1M": h4, ...}

# Egyedi horizont
result.get_horizon(8)  # Első 8 hét összege
```

#### Speciális aggregációk

```python
from models.utils.aggregation import (
    calculate_weighted_horizons,       # Exponenciális decay súlyozás
    calculate_percentile_horizons,     # Monte Carlo percentilis
    calculate_confidence_intervals,    # CI (5%, 50%, 95%)
    calculate_rolling_horizons,        # Gördülő ablak
)
```

---

### 3. Validation (`validation.py`)

#### Minimum adatpontok

```python
RECOMMENDED_MIN_POINTS = {
    "statistical": 24,      # ARIMA, SARIMA
    "smoothing": 12,        # ETS, Holt-Winters
    "ml": 50,               # Random Forest, XGBoost
    "deep_learning": 100,   # LSTM, Transformer
    "spectral": 32,         # FFT, Wavelet
}
```

#### ValidationResult

```python
from models.utils.validation import validate_input_data

result = validate_input_data(data, min_points=24)

result.is_valid          # bool
result.processed_data    # List[float] vagy None
result.original_length   # int
result.processed_length  # int
result.error_message     # str vagy None
result.warnings          # List[str]
result.stats             # {"mean", "std", "min", "max", ...}
```

#### Speciális validációk

```python
from models.utils.validation import (
    validate_multivariate_data,     # DataFrame validálás
    validate_forecast_parameters,   # Paraméter validálás
    check_stationarity,             # ADF teszt
    prepare_forecast_data,          # Egyszerűsített előkészítés
)
```

---

### 4. Monitoring (`monitoring.py`)

#### ForecastTimer

```python
from models.utils.monitoring import ForecastTimer

# Context manager
with ForecastTimer("LSTM", strategy_id="STR_001") as timer:
    result = model.forecast(data, steps)
print(f"Elapsed: {timer.elapsed:.2f}s")

# Decorator
@ForecastTimer.as_decorator("ARIMA")
def run_arima(data, steps):
    return model.forecast(data, steps)
```

#### Batch monitoring

```python
from models.utils.monitoring import BatchPerformanceMonitor

with BatchPerformanceMonitor("LSTM") as monitor:
    for strategy_id, data in all_data.items():
        with monitor.track(strategy_id, len(data)):
            result = model.forecast(data, steps)

# Összesítés
summary = monitor.summary()
print(f"Total: {summary['total_time']:.2f}s")
print(f"Avg: {summary['avg_time']:.2f}s")
print(f"Slow: {summary['slow_count']}")
```

#### Küszöbértékek

```python
SLOW_FORECAST_THRESHOLD = 5.0      # debug log
VERY_SLOW_THRESHOLD = 30.0         # warning log
CRITICAL_SLOW_THRESHOLD = 120.0    # error log
```

---

## Migráció a régi strategy_analyzer.py-ról

### Régi kód

```python
# strategy_analyzer.py - monolitikus
result = analyze_strategy(
    strategy_id,
    strat_data,
    method_name,
    params,
    use_gpu,
    max_horizon
)
```

### Új kód (opció 1: BaseModel wrapper)

```python
# Automatikus validálás, utófeldolgozás, aggregálás
model = ARIMAModel()
result = model.forecast_with_pipeline(
    data=strat_data["Profit"].values,
    steps=max_horizon,
    params=params,
    strategy_id=strategy_id
)
```

### Új kód (opció 2: közvetlen használat)

```python
from models.utils import (
    validate_input_data,
    postprocess_forecasts,
    calculate_horizons,
    ForecastTimer,
)

# 1. Validálás
validation = validate_input_data(data, min_points=24)
if not validation.is_valid:
    return None

# 2. Forecast időméréssel
with ForecastTimer("ARIMA", strategy_id) as timer:
    raw = model.forecast(validation.processed_data, steps, params)

# 3. Utófeldolgozás
forecasts = postprocess_forecasts(raw)

# 4. Horizontok
horizons = calculate_horizons(forecasts)

return {
    "No.": strategy_id,
    "Forecast_1W": horizons.h1,
    "Forecast_1M": horizons.h4,
    ...
}
```

---

## Tesztelés

```bash
# Futtatás a projekt gyökeréből
cd src
python -m pytest models/utils/tests/ -v
```

---

## Changelog

### v1.0.0 (2024-01)

- Eredeti `strategy_analyzer.py` funkcionalitás kiemelése
- Moduláris struktúra létrehozása
- BaseModel wrapper metódusok
- Teljes dokumentáció
