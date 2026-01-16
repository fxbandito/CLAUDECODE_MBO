"""
MBO Trading Strategy Analyzer - Forecast Postprocessing
Forecast utófeldolgozás: NaN kezelés, outlier capping, értékhatárok.

A régi strategy_analyzer.py 464-490. soraiból származó logika,
tisztítva és általánosítva.

Használat:
    from models.utils.postprocessing import postprocess_forecasts, PostprocessingConfig

    # Alapértelmezett beállításokkal
    clean_forecasts = postprocess_forecasts(raw_forecasts)

    # Egyedi beállításokkal
    config = PostprocessingConfig(
        iqr_multiplier=2.0,  # Szigorúbb outlier capping
        allow_negative=False  # Negatív értékek tiltása
    )
    clean_forecasts = postprocess_forecasts(raw_forecasts, config)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PostprocessingConfig:
    """
    Utófeldolgozás konfigurációja.

    Attributes:
        handle_nan: NaN értékek kezelése (True = 0.0-ra cseréli)
        handle_inf: Inf értékek kezelése (True = 0.0-ra cseréli)
        cap_outliers: Outlier capping engedélyezése
        iqr_multiplier: IQR szorzó az outlier határhoz (default: 3.0)
        min_data_for_iqr: Minimum adatpont IQR számításhoz (default: 4)
        allow_negative: Negatív értékek engedélyezése (default: True)
        min_value: Minimum érték (ha allow_negative=False, ez 0.0)
        max_value: Maximum érték (opcionális hard limit)
        nan_replacement: NaN helyettesítő érték (default: 0.0)
    """
    handle_nan: bool = True
    handle_inf: bool = True
    cap_outliers: bool = True
    iqr_multiplier: float = 3.0
    min_data_for_iqr: int = 4
    allow_negative: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    nan_replacement: float = 0.0


# Alapértelmezett konfiguráció (a régi strategy_analyzer.py logikája)
DEFAULT_CONFIG = PostprocessingConfig()


def postprocess_forecasts(
    forecasts: Union[List[float], np.ndarray],
    config: Optional[PostprocessingConfig] = None
) -> List[float]:
    """
    Teljes forecast utófeldolgozás.

    Lépések:
    1. NaN és Inf értékek kezelése
    2. Outlier capping (3×IQR módszerrel)
    3. Értékhatárok alkalmazása (min/max)

    Args:
        forecasts: Nyers forecast értékek
        config: Utófeldolgozás beállításai (opcionális)

    Returns:
        Tisztított forecast lista

    Example:
        >>> raw = [1.0, np.nan, 100.0, 2.0, np.inf]
        >>> postprocess_forecasts(raw)
        [1.0, 0.0, 8.0, 2.0, 0.0]  # NaN/Inf → 0, outlier capped
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Numpy array-re konvertálás
    arr = np.array(forecasts, dtype=np.float64)

    # 1. NaN és Inf kezelés
    if config.handle_nan or config.handle_inf:
        arr = handle_nan_values(
            arr,
            handle_nan=config.handle_nan,
            handle_inf=config.handle_inf,
            replacement=config.nan_replacement
        )

    # 2. Outlier capping
    if config.cap_outliers and len(arr) >= config.min_data_for_iqr:
        arr = cap_outliers(arr, iqr_multiplier=config.iqr_multiplier)

    # 3. Értékhatárok
    arr = clip_extreme_values(
        arr,
        min_value=config.min_value if config.allow_negative else 0.0,
        max_value=config.max_value
    )

    return arr.tolist()


def handle_nan_values(
    arr: np.ndarray,
    handle_nan: bool = True,
    handle_inf: bool = True,
    replacement: float = 0.0
) -> np.ndarray:
    """
    NaN és végtelen értékek kezelése.

    Args:
        arr: Numpy array
        handle_nan: NaN kezelése
        handle_inf: Inf kezelése
        replacement: Helyettesítő érték

    Returns:
        Tisztított array

    Note:
        A régi strategy_analyzer.py 471. sorából:
        forecasts_arr = np.nan_to_num(forecasts_arr, nan=0.0, posinf=0.0, neginf=0.0)
    """
    if not handle_nan and not handle_inf:
        return arr

    nan_val = replacement if handle_nan else np.nan
    posinf_val = replacement if handle_inf else np.inf
    neginf_val = replacement if handle_inf else -np.inf

    result = np.nan_to_num(arr, nan=nan_val, posinf=posinf_val, neginf=neginf_val)

    # Statisztika logolása (debug szinten)
    nan_count = np.isnan(arr).sum() if handle_nan else 0
    inf_count = np.isinf(arr).sum() if handle_inf else 0
    if nan_count > 0 or inf_count > 0:
        logger.debug(
            "NaN/Inf handling: %d NaN, %d Inf values replaced with %.2f",
            nan_count, inf_count, replacement
        )

    return result


def cap_outliers(
    arr: np.ndarray,
    iqr_multiplier: float = 3.0,
    cap_lower: bool = False
) -> np.ndarray:
    """
    Outlier capping IQR módszerrel.

    A 3×IQR módszer permisszívebb mint az 1.5×IQR,
    csak a valóban extrém értékeket vágja le.

    Args:
        arr: Numpy array
        iqr_multiplier: IQR szorzó (default: 3.0)
        cap_lower: Alsó határ alkalmazása is (default: False)

    Returns:
        Capped array

    Note:
        A régi strategy_analyzer.py 480-487. soraiból:
        q1 = np.percentile(forecasts_arr, 25)
        q3 = np.percentile(forecasts_arr, 75)
        iqr = q3 - q1
        upper_bound = q3 + 3.0 * iqr
    """
    if len(arr) < 4:
        logger.debug("Not enough data for IQR calculation (need 4+, got %d)", len(arr))
        return arr

    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1

    # Ha nincs variancia, nem capping-elünk
    if iqr <= 0:
        logger.debug("IQR is zero or negative, skipping outlier capping")
        return arr

    upper_bound = q3 + iqr_multiplier * iqr
    result = arr.copy()

    # Felső határ alkalmazása (csak ha pozitív)
    if upper_bound > 0:
        capped_count = np.sum(result > upper_bound)
        if capped_count > 0:
            logger.debug(
                "Outlier capping: %d values capped at %.2f (IQR=%.2f, multiplier=%.1f)",
                capped_count, upper_bound, iqr, iqr_multiplier
            )
        result = np.minimum(result, upper_bound)

    # Alsó határ (opcionális)
    if cap_lower:
        lower_bound = q1 - iqr_multiplier * iqr
        result = np.maximum(result, lower_bound)

    return result


def clip_extreme_values(
    arr: np.ndarray,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> np.ndarray:
    """
    Értékek vágása megadott határok közé.

    Args:
        arr: Numpy array
        min_value: Minimum érték (None = nincs alsó határ)
        max_value: Maximum érték (None = nincs felső határ)

    Returns:
        Clipped array

    Example:
        >>> clip_extreme_values(np.array([-5, 0, 10, 100]), min_value=0, max_value=50)
        array([ 0,  0, 10, 50])
    """
    if min_value is None and max_value is None:
        return arr

    result = arr.copy()

    if min_value is not None:
        clipped_low = np.sum(result < min_value)
        if clipped_low > 0:
            logger.debug("Clipping %d values below %.2f", clipped_low, min_value)
        result = np.maximum(result, min_value)

    if max_value is not None:
        clipped_high = np.sum(result > max_value)
        if clipped_high > 0:
            logger.debug("Clipping %d values above %.2f", clipped_high, max_value)
        result = np.minimum(result, max_value)

    return result


# =============================================================================
# SPECIÁLIS POSTPROCESSING FÜGGVÉNYEK
# =============================================================================

def postprocess_volatility_forecast(
    forecasts: Union[List[float], np.ndarray],
    ensure_positive: bool = True
) -> List[float]:
    """
    Volatilitás forecast utófeldolgozás.

    A volatilitás mindig pozitív kell legyen.

    Args:
        forecasts: Nyers volatilitás forecast
        ensure_positive: Biztosítja a pozitív értékeket

    Returns:
        Tisztított forecast
    """
    config = PostprocessingConfig(
        allow_negative=not ensure_positive,
        min_value=0.0 if ensure_positive else None,
        iqr_multiplier=3.0
    )
    return postprocess_forecasts(forecasts, config)


def postprocess_probability_forecast(
    forecasts: Union[List[float], np.ndarray]
) -> List[float]:
    """
    Valószínűség forecast utófeldolgozás.

    A valószínűség [0, 1] intervallumban kell legyen.

    Args:
        forecasts: Nyers valószínűség forecast

    Returns:
        [0, 1] közé vágott forecast
    """
    config = PostprocessingConfig(
        allow_negative=False,
        min_value=0.0,
        max_value=1.0,
        cap_outliers=False  # Probability-nél fix határok vannak
    )
    return postprocess_forecasts(forecasts, config)


def postprocess_count_forecast(
    forecasts: Union[List[float], np.ndarray],
    round_values: bool = True
) -> List[float]:
    """
    Darabszám (count) forecast utófeldolgozás.

    A darabszám nem-negatív egész kell legyen.

    Args:
        forecasts: Nyers count forecast
        round_values: Kerekítés egészre

    Returns:
        Tisztított forecast
    """
    config = PostprocessingConfig(
        allow_negative=False,
        min_value=0.0,
        iqr_multiplier=3.0
    )
    result = postprocess_forecasts(forecasts, config)

    if round_values:
        result = [round(x) for x in result]

    return result
