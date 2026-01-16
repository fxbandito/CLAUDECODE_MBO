"""
MBO Trading Strategy Analyzer - Horizon Aggregation
Forecast horizont aggregátumok számítása.

A régi strategy_analyzer.py 492-497. soraiból származó logika,
kiterjesztve és általánosítva.

Standard horizontok:
    - h1:  1 hét (1 periódus)
    - h4:  1 hónap (4 periódus)
    - h13: 3 hónap (13 periódus)
    - h26: 6 hónap (26 periódus)
    - h52: 12 hónap (52 periódus)

Használat:
    from models.utils.aggregation import calculate_horizons, HorizonResult

    forecasts = [1.0, 2.0, 3.0, ...]  # 52 elemű lista
    result = calculate_horizons(forecasts)

    print(result.h1)   # Első periódus
    print(result.h4)   # Első 4 periódus összege
    print(result.h52)  # Mind az 52 periódus összege
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# KONSTANSOK
# =============================================================================

# Standard horizontok (hetek)
STANDARD_HORIZONS = {
    "h1": 1,    # 1 hét
    "h4": 4,    # 1 hónap (4 hét)
    "h13": 13,  # 3 hónap (13 hét)
    "h26": 26,  # 6 hónap (26 hét)
    "h52": 52,  # 12 hónap (52 hét)
}

# Horizon mapping a GUI-hoz
HORIZON_LABELS = {
    "h1": "Forecast_1W",
    "h4": "Forecast_1M",
    "h13": "Forecast_3M",
    "h26": "Forecast_6M",
    "h52": "Forecast_12M",
}


@dataclass
class HorizonResult:
    """
    Horizont aggregátumok eredménye.

    Attributes:
        h1: 1 hetes forecast (első érték)
        h4: 1 hónapos kumulatív forecast (első 4 hét összege)
        h13: 3 hónapos kumulatív forecast (első 13 hét összege)
        h26: 6 hónapos kumulatív forecast (első 26 hét összege)
        h52: 12 hónapos kumulatív forecast (első 52 hét összege)
        raw_forecasts: Eredeti forecast lista
        actual_length: Tényleges forecast hossz
    """
    h1: float
    h4: float
    h13: float
    h26: float
    h52: float
    raw_forecasts: List[float]
    actual_length: int

    def to_dict(self) -> Dict[str, float]:
        """Horizon értékek dictionary-ként."""
        return {
            "Forecast_1W": self.h1,
            "Forecast_1M": self.h4,
            "Forecast_3M": self.h13,
            "Forecast_6M": self.h26,
            "Forecast_12M": self.h52,
        }

    def to_dict_short(self) -> Dict[str, float]:
        """Horizon értékek rövid kulcsokkal."""
        return {
            "h1": self.h1,
            "h4": self.h4,
            "h13": self.h13,
            "h26": self.h26,
            "h52": self.h52,
        }

    def get_horizon(self, periods: int) -> float:
        """
        Tetszőleges horizont lekérése.

        Args:
            periods: Periódusok száma

        Returns:
            Kumulatív érték az adott horizontig
        """
        return calculate_cumulative_horizons(self.raw_forecasts, [periods])[0]


def calculate_horizons(
    forecasts: Union[List[float], np.ndarray],
    allow_negative: bool = True
) -> HorizonResult:
    """
    Standard horizont aggregátumok számítása.

    Args:
        forecasts: Forecast értékek listája
        allow_negative: Negatív összegek engedélyezése (default: True)

    Returns:
        HorizonResult az összes standard horizonttal

    Note:
        A régi strategy_analyzer.py 492-497. soraiból:
        h1 = forecasts[0] if len(forecasts) >= 1 else 0
        h4 = sum(forecasts[:4]) if len(forecasts) >= 4 else sum(forecasts)
        ...

        A 499-501. sor kommentje szerint:
        "Negative forecasts are valid (indicate predicted losses) and should be shown to user"
    """
    arr = np.array(forecasts, dtype=np.float64)
    n = len(arr)

    if n == 0:
        logger.warning("Empty forecast list, returning zeros")
        return HorizonResult(
            h1=0.0, h4=0.0, h13=0.0, h26=0.0, h52=0.0,
            raw_forecasts=[], actual_length=0
        )

    # Kumulatív összegek számítása
    h1 = float(arr[0]) if n >= 1 else 0.0
    h4 = float(np.sum(arr[:4])) if n >= 4 else float(np.sum(arr))
    h13 = float(np.sum(arr[:13])) if n >= 13 else float(np.sum(arr))
    h26 = float(np.sum(arr[:26])) if n >= 26 else float(np.sum(arr))
    h52 = float(np.sum(arr[:52])) if n >= 52 else float(np.sum(arr))

    # Opcionális: negatív értékek nullázása
    # A régi kód ezt eltávolította (499-501. sor), mert a negatív előrejelzések
    # valós veszteségeket jelezhetnek
    if not allow_negative:
        h1 = max(0.0, h1)
        h4 = max(0.0, h4)
        h13 = max(0.0, h13)
        h26 = max(0.0, h26)
        h52 = max(0.0, h52)

    return HorizonResult(
        h1=h1, h4=h4, h13=h13, h26=h26, h52=h52,
        raw_forecasts=arr.tolist(),
        actual_length=n
    )


def calculate_cumulative_horizons(
    forecasts: Union[List[float], np.ndarray],
    horizons: List[int]
) -> List[float]:
    """
    Tetszőleges horizontok kumulatív értékeinek számítása.

    Args:
        forecasts: Forecast értékek
        horizons: Kívánt horizontok listája (pl. [1, 4, 13, 26, 52])

    Returns:
        Kumulatív értékek listája az egyes horizontokra

    Example:
        >>> forecasts = [1, 2, 3, 4, 5]
        >>> calculate_cumulative_horizons(forecasts, [1, 3, 5])
        [1.0, 6.0, 15.0]
    """
    arr = np.array(forecasts, dtype=np.float64)
    n = len(arr)

    results = []
    for h in horizons:
        if h <= 0:
            results.append(0.0)
        elif h > n:
            results.append(float(np.sum(arr)))
        else:
            results.append(float(np.sum(arr[:h])))

    return results


def calculate_rolling_horizons(
    forecasts: Union[List[float], np.ndarray],
    window_size: int,
    step: int = 1
) -> List[float]:
    """
    Gördülő ablakos horizont számítás.

    Args:
        forecasts: Forecast értékek
        window_size: Ablak mérete
        step: Lépésköz (default: 1)

    Returns:
        Gördülő összegek listája

    Example:
        >>> forecasts = [1, 2, 3, 4, 5]
        >>> calculate_rolling_horizons(forecasts, window_size=3, step=1)
        [6.0, 9.0, 12.0]  # [1+2+3, 2+3+4, 3+4+5]
    """
    arr = np.array(forecasts, dtype=np.float64)
    n = len(arr)

    if window_size > n:
        return [float(np.sum(arr))]

    results = []
    for i in range(0, n - window_size + 1, step):
        results.append(float(np.sum(arr[i:i + window_size])))

    return results


# =============================================================================
# SPECIÁLIS AGGREGÁCIÓS FÜGGVÉNYEK
# =============================================================================

def calculate_weighted_horizons(
    forecasts: Union[List[float], np.ndarray],
    weights: Optional[Union[List[float], np.ndarray]] = None,
    decay_factor: float = 0.95
) -> HorizonResult:
    """
    Súlyozott horizont aggregátumok.

    Későbbi periódusok kisebb súlyt kapnak (bizonytalanság növekedése).

    Args:
        forecasts: Forecast értékek
        weights: Egyedi súlyok (opcionális)
        decay_factor: Exponenciális csökkenés faktora (default: 0.95)

    Returns:
        Súlyozott HorizonResult
    """
    arr = np.array(forecasts, dtype=np.float64)
    n = len(arr)

    if weights is None:
        # Exponenciális decay súlyok
        weights = np.array([decay_factor ** i for i in range(n)])
    else:
        weights = np.array(weights, dtype=np.float64)
        if len(weights) < n:
            # Padding nullákkal
            weights = np.pad(weights, (0, n - len(weights)), constant_values=0)
        elif len(weights) > n:
            weights = weights[:n]

    weighted = arr * weights

    h1 = float(weighted[0]) if n >= 1 else 0.0
    h4 = float(np.sum(weighted[:4])) if n >= 4 else float(np.sum(weighted))
    h13 = float(np.sum(weighted[:13])) if n >= 13 else float(np.sum(weighted))
    h26 = float(np.sum(weighted[:26])) if n >= 26 else float(np.sum(weighted))
    h52 = float(np.sum(weighted[:52])) if n >= 52 else float(np.sum(weighted))

    return HorizonResult(
        h1=h1, h4=h4, h13=h13, h26=h26, h52=h52,
        raw_forecasts=weighted.tolist(),
        actual_length=n
    )


def calculate_percentile_horizons(
    forecast_samples: List[List[float]],
    percentile: float = 50.0
) -> HorizonResult:
    """
    Percentilis alapú horizont aggregátumok (ensemble/probabilistic modellek).

    Args:
        forecast_samples: Több forecast minta (pl. Monte Carlo)
        percentile: Kívánt percentilis (default: 50.0 = medián)

    Returns:
        HorizonResult a megadott percentilisre

    Example:
        >>> samples = [[1, 2, 3], [2, 3, 4], [1, 4, 2]]
        >>> result = calculate_percentile_horizons(samples, percentile=50)
        >>> result.h1  # Medián az első pozícióban
        1.0
    """
    if not forecast_samples:
        return HorizonResult(
            h1=0.0, h4=0.0, h13=0.0, h26=0.0, h52=0.0,
            raw_forecasts=[], actual_length=0
        )

    # Konvertálás 2D numpy array-re
    arr = np.array(forecast_samples, dtype=np.float64)

    # Percentilis minden pozícióra
    percentile_forecast = np.percentile(arr, percentile, axis=0)

    return calculate_horizons(percentile_forecast.tolist())


def calculate_confidence_intervals(
    forecast_samples: List[List[float]],
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0
) -> Dict[str, HorizonResult]:
    """
    Konfidencia intervallumok számítása horizontonként.

    Args:
        forecast_samples: Több forecast minta
        lower_percentile: Alsó percentilis (default: 5.0)
        upper_percentile: Felső percentilis (default: 95.0)

    Returns:
        Dict három HorizonResult-tal: 'lower', 'median', 'upper'
    """
    return {
        "lower": calculate_percentile_horizons(forecast_samples, lower_percentile),
        "median": calculate_percentile_horizons(forecast_samples, 50.0),
        "upper": calculate_percentile_horizons(forecast_samples, upper_percentile),
    }
