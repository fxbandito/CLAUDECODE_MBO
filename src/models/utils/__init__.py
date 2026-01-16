"""
MBO Trading Strategy Analyzer - Model Utilities
Közös utility függvények minden forecasting modellhez.

Ez a modul a régi strategy_analyzer.py funkcionalitását tartalmazza
moduláris, újrafelhasználható formában.

Modulok:
    - postprocessing: Forecast utófeldolgozás (NaN, outlier, clipping)
    - aggregation: Horizont aggregátumok számítása
    - validation: Input validálás és előfeldolgozás
    - monitoring: Teljesítmény monitoring és logging

Használat:
    from models.utils import postprocess_forecasts, calculate_horizons
    from models.utils import validate_input_data, ForecastTimer
"""

from models.utils.postprocessing import (
    postprocess_forecasts,
    handle_nan_values,
    cap_outliers,
    clip_extreme_values,
    PostprocessingConfig,
)

from models.utils.aggregation import (
    calculate_horizons,
    calculate_cumulative_horizons,
    HorizonResult,
    STANDARD_HORIZONS,
)

from models.utils.validation import (
    validate_input_data,
    prepare_forecast_data,
    ValidationResult,
    MIN_DATA_POINTS,
)

from models.utils.monitoring import (
    ForecastTimer,
    log_slow_forecast,
    PerformanceStats,
)

__all__ = [
    # Postprocessing
    "postprocess_forecasts",
    "handle_nan_values",
    "cap_outliers",
    "clip_extreme_values",
    "PostprocessingConfig",
    # Aggregation
    "calculate_horizons",
    "calculate_cumulative_horizons",
    "HorizonResult",
    "STANDARD_HORIZONS",
    # Validation
    "validate_input_data",
    "prepare_forecast_data",
    "ValidationResult",
    "MIN_DATA_POINTS",
    # Monitoring
    "ForecastTimer",
    "log_slow_forecast",
    "PerformanceStats",
]
