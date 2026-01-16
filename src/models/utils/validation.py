"""
MBO Trading Strategy Analyzer - Input Validation
Input validálás és előfeldolgozás forecasting modellekhez.

A régi strategy_analyzer.py 149-156. soraiból származó logika,
kiterjesztve és általánosítva.

Használat:
    from models.utils.validation import validate_input_data, ValidationResult

    # Egyszerű validálás
    result = validate_input_data(profit_series)
    if not result.is_valid:
        print(f"Validation failed: {result.error_message}")
        return None

    # Előfeldolgozott adat használata
    clean_data = result.processed_data
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# KONSTANSOK
# =============================================================================

# Minimum adatpontok száma (a régi strategy_analyzer.py 151. sorából)
MIN_DATA_POINTS = 24

# Ajánlott minimum különböző modellekhez
RECOMMENDED_MIN_POINTS = {
    "statistical": 24,      # ARIMA, SARIMA, stb.
    "smoothing": 12,        # ETS, Holt-Winters
    "ml": 50,               # Random Forest, XGBoost
    "deep_learning": 100,   # LSTM, Transformer
    "spectral": 32,         # FFT, Wavelet (2^n ajánlott)
}


@dataclass
class ValidationResult:
    """
    Input validálás eredménye.

    Attributes:
        is_valid: A bemenet érvényes-e
        processed_data: Előfeldolgozott adat (None ha invalid)
        original_length: Eredeti adat hossza
        processed_length: Feldolgozott adat hossza
        error_message: Hibaüzenet (None ha valid)
        warnings: Figyelmeztetések listája
        stats: Alapstatisztikák az adatról
    """
    is_valid: bool
    processed_data: Optional[List[float]]
    original_length: int
    processed_length: int
    error_message: Optional[str] = None
    warnings: List[str] = None
    stats: Optional[dict] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def validate_input_data(
    data: Union[List[float], np.ndarray, pd.Series],
    min_points: int = MIN_DATA_POINTS,
    allow_nan: bool = False,
    remove_nan: bool = True,
    check_constant: bool = True,
    check_variance: bool = True,
    model_type: Optional[str] = None
) -> ValidationResult:
    """
    Input adat validálása forecasting modellekhez.

    Args:
        data: Bemeneti idősor
        min_points: Minimum szükséges adatpontok (default: 24)
        allow_nan: NaN értékek engedélyezése (default: False)
        remove_nan: NaN értékek eltávolítása (default: True)
        check_constant: Konstans sorozat ellenőrzése (default: True)
        check_variance: Variancia ellenőrzése (default: True)
        model_type: Modell típusa ajánlott minimum pontokhoz

    Returns:
        ValidationResult az eredménnyel

    Note:
        A régi strategy_analyzer.py 151-155. sorából:
        if len(profit_series) < 24:
            logger.debug("Strategy %s: skipped (only %d data points, need 24+)", ...)
            return None
    """
    warnings = []

    # Modell-specifikus minimum pontok
    if model_type and model_type in RECOMMENDED_MIN_POINTS:
        recommended = RECOMMENDED_MIN_POINTS[model_type]
        if min_points < recommended:
            min_points = recommended

    # Konvertálás numpy array-re
    try:
        if isinstance(data, pd.Series):
            arr = data.values.astype(np.float64)
        elif isinstance(data, list):
            arr = np.array(data, dtype=np.float64)
        else:
            arr = np.array(data, dtype=np.float64)
    except (ValueError, TypeError) as e:
        return ValidationResult(
            is_valid=False,
            processed_data=None,
            original_length=0,
            processed_length=0,
            error_message=f"Cannot convert input to numeric array: {e}"
        )

    original_length = len(arr)

    # Üres adat ellenőrzése
    if original_length == 0:
        return ValidationResult(
            is_valid=False,
            processed_data=None,
            original_length=0,
            processed_length=0,
            error_message="Empty input data"
        )

    # NaN kezelés
    nan_count = np.isnan(arr).sum()
    if nan_count > 0:
        if allow_nan:
            warnings.append(f"Data contains {nan_count} NaN values")
        elif remove_nan:
            arr = arr[~np.isnan(arr)]
            warnings.append(f"Removed {nan_count} NaN values")
        else:
            return ValidationResult(
                is_valid=False,
                processed_data=None,
                original_length=original_length,
                processed_length=0,
                error_message=f"Data contains {nan_count} NaN values (use allow_nan=True or remove_nan=True)"
            )

    # Inf kezelés
    inf_count = np.isinf(arr).sum()
    if inf_count > 0:
        arr = arr[~np.isinf(arr)]
        warnings.append(f"Removed {inf_count} infinite values")

    processed_length = len(arr)

    # Minimum pontok ellenőrzése
    if processed_length < min_points:
        return ValidationResult(
            is_valid=False,
            processed_data=None,
            original_length=original_length,
            processed_length=processed_length,
            error_message=f"Not enough data points: {processed_length} (need {min_points}+)"
        )

    # Konstans sorozat ellenőrzése
    if check_constant:
        if np.all(arr == arr[0]):
            return ValidationResult(
                is_valid=False,
                processed_data=None,
                original_length=original_length,
                processed_length=processed_length,
                error_message="Constant time series (all values are identical)"
            )

    # Variancia ellenőrzése
    if check_variance:
        variance = np.var(arr)
        if variance < 1e-10:
            warnings.append(f"Very low variance: {variance:.2e}")

    # Alapstatisztikák számítása
    stats = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "nan_count": nan_count,
        "inf_count": inf_count,
    }

    return ValidationResult(
        is_valid=True,
        processed_data=arr.tolist(),
        original_length=original_length,
        processed_length=processed_length,
        warnings=warnings,
        stats=stats
    )


def prepare_forecast_data(
    data: Union[List[float], np.ndarray, pd.Series, pd.DataFrame],
    column: Optional[str] = "Profit",
    validate: bool = True,
    min_points: int = MIN_DATA_POINTS
) -> Tuple[Optional[List[float]], Optional[str]]:
    """
    Forecast adat előkészítése (egyszerűsített interfész).

    Args:
        data: Bemeneti adat (lista, array, Series, vagy DataFrame)
        column: Oszlop neve DataFrame esetén (default: "Profit")
        validate: Validálás végrehajtása (default: True)
        min_points: Minimum adatpontok

    Returns:
        Tuple: (előkészített adat vagy None, hibaüzenet vagy None)

    Example:
        >>> data, error = prepare_forecast_data(df)
        >>> if error:
        ...     print(f"Error: {error}")
        ...     return
        >>> forecasts = model.forecast(data, steps=52)
    """
    # DataFrame kezelés
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            return None, f"Column '{column}' not found in DataFrame"
        series = data[column].values
    elif isinstance(data, pd.Series):
        series = data.values
    else:
        series = data

    if not validate:
        return list(np.array(series, dtype=np.float64)), None

    result = validate_input_data(series, min_points=min_points)

    if not result.is_valid:
        return None, result.error_message

    return result.processed_data, None


# =============================================================================
# SPECIÁLIS VALIDÁCIÓS FÜGGVÉNYEK
# =============================================================================

def validate_multivariate_data(
    data: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_points: int = MIN_DATA_POINTS
) -> ValidationResult:
    """
    Többváltozós idősor validálása.

    Args:
        data: DataFrame több oszloppal
        required_columns: Kötelező oszlopok (opcionális)
        min_points: Minimum adatpontok

    Returns:
        ValidationResult
    """
    warnings = []

    if not isinstance(data, pd.DataFrame):
        return ValidationResult(
            is_valid=False,
            processed_data=None,
            original_length=0,
            processed_length=0,
            error_message="Input must be a pandas DataFrame for multivariate models"
        )

    # Oszlopok ellenőrzése
    if required_columns:
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            return ValidationResult(
                is_valid=False,
                processed_data=None,
                original_length=len(data),
                processed_length=0,
                error_message=f"Missing required columns: {missing}"
            )

    original_length = len(data)

    # Sorok ellenőrzése
    if original_length < min_points:
        return ValidationResult(
            is_valid=False,
            processed_data=None,
            original_length=original_length,
            processed_length=0,
            error_message=f"Not enough rows: {original_length} (need {min_points}+)"
        )

    # NaN ellenőrzése oszloponként
    nan_counts = data.isna().sum()
    for col, count in nan_counts.items():
        if count > 0:
            warnings.append(f"Column '{col}' has {count} NaN values")

    # Numerikus oszlopok ellenőrzése
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        return ValidationResult(
            is_valid=False,
            processed_data=None,
            original_length=original_length,
            processed_length=0,
            error_message="No numeric columns found"
        )

    return ValidationResult(
        is_valid=True,
        processed_data=None,  # DataFrame-et nem konvertáljuk listára
        original_length=original_length,
        processed_length=original_length,
        warnings=warnings,
        stats={"numeric_columns": numeric_cols, "total_columns": len(data.columns)}
    )


def validate_forecast_parameters(
    params: dict,
    required: Optional[List[str]] = None,
    valid_values: Optional[dict] = None
) -> Tuple[bool, Optional[str]]:
    """
    Forecast paraméterek validálása.

    Args:
        params: Paraméter dictionary
        required: Kötelező paraméterek listája
        valid_values: Érvényes értékek (kulcs → értéklista)

    Returns:
        Tuple: (érvényes-e, hibaüzenet)

    Example:
        >>> params = {"p": 1, "d": 1, "q": 1}
        >>> valid, error = validate_forecast_parameters(
        ...     params,
        ...     required=["p", "d", "q"],
        ...     valid_values={"d": [0, 1, 2]}
        ... )
    """
    if params is None:
        params = {}

    # Kötelező paraméterek
    if required:
        missing = [p for p in required if p not in params]
        if missing:
            return False, f"Missing required parameters: {missing}"

    # Érvényes értékek
    if valid_values:
        for param, valid in valid_values.items():
            if param in params and params[param] not in valid:
                return False, f"Invalid value for '{param}': {params[param]} (valid: {valid})"

    return True, None


def check_stationarity(
    data: Union[List[float], np.ndarray],
    significance_level: float = 0.05
) -> Tuple[bool, float, str]:
    """
    Idősor stacionaritás ellenőrzése (ADF teszt).

    Args:
        data: Bemeneti idősor
        significance_level: Szignifikancia szint (default: 0.05)

    Returns:
        Tuple: (stacionárius-e, p-érték, üzenet)

    Note:
        Opcionális függőség: statsmodels
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return True, 0.0, "statsmodels not available, skipping stationarity check"

    arr = np.array(data, dtype=np.float64)

    try:
        result = adfuller(arr, autolag='AIC')
        p_value = result[1]
        is_stationary = p_value < significance_level

        if is_stationary:
            message = f"Series is stationary (p={p_value:.4f})"
        else:
            message = f"Series is non-stationary (p={p_value:.4f}), consider differencing"

        return is_stationary, p_value, message

    except Exception as e:
        logger.warning("Stationarity test failed: %s", e)
        return True, 0.0, f"Stationarity test failed: {e}"
