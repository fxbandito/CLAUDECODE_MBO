"""
ARIMA Model - AutoRegressive Integrated Moving Average

Az ARIMA modell a leggyakrabban használt statisztikai idősor-előrejelzési módszer.
Három komponensből áll:
- AR (AutoRegressive): Az előző értékek lineáris kombinációja
- I (Integrated): Differenciálás a stacionaritás eléréséhez
- MA (Moving Average): Az előző előrejelzési hibák lineáris kombinációja

ARIMA(p, d, q) paraméterek:
- p: AR rend (autoregressive order) - hány korábbi értéket használunk
- d: Differenciálás rendje - hányszor differenciálunk
- q: MA rend (moving average order) - hány korábbi hibát használunk

Trend típusok:
- n: Nincs trend (no trend)
- c: Konstans (constant/intercept)
- t: Lineáris trend (linear trend)
- ct: Konstans + lineáris trend

Referenciák:
- Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). Time Series Analysis
- statsmodels dokumentáció: https://www.statsmodels.org/stable/tsa.html

Használat:
    from models.statistical.arima import ARIMAModel

    model = ARIMAModel()

    # Egyszerű forecast
    forecasts = model.forecast(data, steps=12, params={"p": "1", "d": "1", "q": "1"})

    # Teljes pipeline (validálás + utófeldolgozás + horizontok)
    result = model.forecast_with_pipeline(
        data=profit_series,
        steps=52,
        params={"p": "2", "d": "1", "q": "2", "trend": "c"},
        strategy_id="STR_001"
    )
    print(f"1M forecast: {result['Forecast_1M']}")
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models.base import BaseModel, ModelInfo

logger = logging.getLogger(__name__)


class ARIMAModel(BaseModel):
    """
    ARIMA - AutoRegressive Integrated Moving Average.

    Az ARIMA modell a klasszikus statisztikai idősor-előrejelzés alapja.
    A statsmodels könyvtár ARIMA implementációját használja, amely támogatja:
    - Különböző trend típusokat (konstans, lineáris, mindkettő, egyik sem)
    - Robusztus becslést nem-stacionárius adatokra
    - Konfidencia intervallumokat

    Jellemzők:
    - Gyors futás kis és közepes adathalmazokon
    - Batch mód támogatás nagyobb adathalmazokhoz
    - Robusztus NaN kezelés
    - Automatikus fallback hibakezelés

    Feature Mode Kompatibilitás:
    - Original: TÁMOGATOTT (ajánlott)
    - Forward Calc: TÁMOGATOTT (de újra-illesztés javasolt)
    - Rolling Window: TÁMOGATOTT (rolling horizonthoz optimális)

    Példa:
        model = ARIMAModel()

        # Egyszerű forecast
        forecasts = model.forecast(data, steps=12, params={"p": "1", "d": "1", "q": "1"})

        # Teljes pipeline validálással és utófeldolgozással
        result = model.forecast_with_pipeline(
            data=profit_series,
            steps=52,
            params={"p": "2", "d": "1", "q": "2"},
            strategy_id="STR_001"
        )
    """

    MODEL_INFO = ModelInfo(
        name="ARIMA",
        category="Statistical Models",
        supports_gpu=False,
        supports_batch=True,
        gpu_threshold=1000,
        supports_forward_calc=True,
        supports_rolling_window=True,
        supports_panel_mode=False,
        supports_dual_mode=False,
    )

    PARAM_DEFAULTS = {
        "p": "1",
        "d": "1",
        "q": "1",
        "trend": "c",
    }

    PARAM_OPTIONS = {
        "p": ["0", "1", "2", "3", "4", "5"],
        "d": ["0", "1", "2"],
        "q": ["0", "1", "2", "3", "4", "5"],
        "trend": ["n", "c", "t", "ct"],
    }

    # Minimum adatpontok száma
    MIN_DATA_POINTS = 10

    # =========================================================================
    # FŐ FORECAST METÓDUS
    # =========================================================================

    def forecast(
        self,
        data: List[float],
        steps: int,
        params: Dict[str, Any]
    ) -> List[float]:
        """
        ARIMA előrejelzés készítése.

        Args:
            data: Bemeneti idősor adatok
            steps: Előrejelzési horizont
            params: Paraméterek (p, d, q, trend)

        Returns:
            Előrejelzett értékek listája

        Note:
            Ha teljes pipeline-t szeretnél (validálás + utófeldolgozás + horizontok),
            használd a forecast_with_pipeline() metódust helyette.
        """
        # Paraméterek kinyerése
        full_params = self.get_params_with_defaults(params)

        # Numpy array konverzió
        values = np.array(data, dtype=np.float64)
        n = len(values)

        # Minimum adat ellenőrzés
        p = self._parse_int(full_params.get("p", "1"), 1, 0, 10)
        d = self._parse_int(full_params.get("d", "1"), 1, 0, 2)
        q = self._parse_int(full_params.get("q", "1"), 1, 0, 10)

        # Dinamikus minimum adatpontok: max(p+d+q+1, MIN_DATA_POINTS)
        min_points = max(p + d + q + 1, self.MIN_DATA_POINTS)

        if n < min_points:
            logger.debug(
                "ARIMA: Not enough data points (%d < %d required for p=%d, d=%d, q=%d)",
                n, min_points, p, d, q
            )
            return [0.0] * steps

        # NaN kezelés - interpoláció
        values = self._handle_nan_values(values)
        if values is None or len(values) < min_points:
            return [0.0] * steps

        # ARIMA futtatás
        try:
            raw_forecasts = self._run_arima(values, steps, p, d, q, full_params)

            # Utófeldolgozás (NaN kezelés, outlier capping)
            return self.postprocess(raw_forecasts, allow_negative=True)

        except Exception as e:
            logger.error("ARIMA forecast error: %s", str(e))
            return self._fallback_forecast(values, steps)

    # =========================================================================
    # ARIMA IMPLEMENTÁCIÓ
    # =========================================================================

    def _run_arima(
        self,
        values: np.ndarray,
        steps: int,
        p: int,
        d: int,
        q: int,
        params: Dict[str, Any]
    ) -> List[float]:
        """
        ARIMA modell futtatása statsmodels-szel.

        Args:
            values: Tisztított idősor
            steps: Előrejelzési lépések
            p: AR rend
            d: Differenciálás rendje
            q: MA rend
            params: Összes paraméter

        Returns:
            Előrejelzett értékek
        """
        from statsmodels.tsa.arima.model import ARIMA
        from numpy.linalg import LinAlgError

        # Trend paraméter - statsmodels szabály: trend rend >= d
        # d=0: bármilyen trend (n=0, c=0, t=1, ct=2)
        # d=1: c nem megy (rend 0), de t igen (rend 1), ct igen (rend 2)
        # d=2: c és t sem megy, csak ct (rend 2) vagy n (nincs trend)
        trend = str(params.get("trend", "c"))
        if trend not in ["n", "c", "t", "ct"]:
            trend = "c"

        # Automatikus trend korrekció a differenciálás rendje alapján
        if d >= 2:
            # d=2 esetén csak "ct" vagy "n" működik
            if trend in ["c", "t"]:
                trend = "ct"  # Upgrade to quadratic trend
        elif d == 1:
            # d=1 esetén "c" nem működik, de "t" igen
            if trend == "c":
                trend = "t"  # Upgrade to linear trend

        # Figyelmeztetések elnyomása
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            try:
                # ARIMA modell létrehozása
                model = ARIMA(
                    values,
                    order=(p, d, q),
                    trend=trend,
                    enforce_stationarity=False,  # Robusztusabb nem-stacionárius adatokra
                    enforce_invertibility=False,  # Robusztusabb invertálhatóság nélkül
                )

                # Modell illesztése
                model_fit = model.fit()

                # Előrejelzés
                forecast = model_fit.forecast(steps=steps)

                return forecast.tolist()

            except LinAlgError as e:
                # Debug szint - normál működés, fallback kezeli
                logger.debug("ARIMA convergence issue, trying fallback: %s", str(e))
                return self._try_simpler_arima(values, steps, p, d, q)

            except ValueError as e:
                # Debug szint - trend/parameter issue, fallback kezeli
                logger.debug("ARIMA parameter issue, trying fallback: %s", str(e))
                return self._try_simpler_arima(values, steps, p, d, q)

            except (RuntimeError, FloatingPointError) as e:
                # Debug szint - numerikus probléma, fallback kezeli
                logger.debug("ARIMA numerical issue, trying fallback: %s", str(e))
                return self._try_simpler_arima(values, steps, p, d, q)

    def _try_simpler_arima(
        self,
        values: np.ndarray,
        steps: int,
        orig_p: int,
        orig_d: int,
        orig_q: int
    ) -> List[float]:
        """
        Egyszerűbb ARIMA konfiguráció próbálása hiba esetén.

        Fallback stratégia:
        1. ARIMA(1,1,1) - alap konfiguráció
        2. ARIMA(1,0,0) - csak AR(1)
        3. ARIMA(0,1,1) - csak MA(1) differenciálással
        4. Naive átlag

        Args:
            values: Idősor
            steps: Előrejelzési lépések
            orig_p, orig_d, orig_q: Eredeti paraméterek (logoláshoz)

        Returns:
            Előrejelzett értékek
        """
        from statsmodels.tsa.arima.model import ARIMA
        from numpy.linalg import LinAlgError

        # Fallback konfigurációk: (p, d, q, trend)
        # Szabály: trend rend >= d (d=1: t vagy ct; d=2: ct vagy n)
        fallback_configs = [
            (1, 1, 1, "t"),   # Alap ARIMA lineáris trenddel
            (1, 0, 0, "c"),   # Csak AR(1) konstanssal
            (0, 1, 1, "n"),   # Differenciálás + MA(1) trend nélkül
            (0, 1, 0, "n"),   # Random walk trend nélkül
            (1, 0, 1, "c"),   # ARMA(1,1) konstanssal (d=0)
        ]

        logger.debug(
            "ARIMA(%d,%d,%d) failed, trying fallback configurations",
            orig_p, orig_d, orig_q
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            for p, d, q, trend in fallback_configs:
                try:
                    model = ARIMA(
                        values,
                        order=(p, d, q),
                        trend=trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=steps)

                    logger.debug("ARIMA fallback ARIMA(%d,%d,%d) succeeded", p, d, q)
                    return forecast.tolist()

                except (LinAlgError, ValueError, RuntimeError, FloatingPointError):
                    continue

        # Végső fallback: naive átlag
        logger.warning("All ARIMA configurations failed, using naive forecast")
        return self._fallback_forecast(values, steps)

    def _fallback_forecast(
        self,
        values: np.ndarray,
        steps: int
    ) -> List[float]:
        """
        Végső fallback: utolsó értékek átlaga.

        Args:
            values: Idősor
            steps: Lépések száma

        Returns:
            Konstans előrejelzés
        """
        # Utolsó néhány érték átlaga
        last_vals = values[-min(5, len(values)):]
        mean_val = float(np.nanmean(last_vals)) if not np.all(np.isnan(last_vals)) else 0.0

        return [mean_val] * steps

    # =========================================================================
    # SEGÉD METÓDUSOK
    # =========================================================================

    def _handle_nan_values(self, values: np.ndarray) -> Optional[np.ndarray]:
        """
        NaN értékek kezelése lineáris interpolációval.

        Args:
            values: Idősor NaN-okkal

        Returns:
            Tisztított idősor vagy None ha minden NaN
        """
        if np.all(np.isnan(values)):
            return None

        nan_mask = np.isnan(values)
        if not np.any(nan_mask):
            return values

        # Lineáris interpoláció
        x = np.arange(len(values))
        values[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], values[~nan_mask])

        return values

    @staticmethod
    def _parse_int(value: Any, default: int, min_val: int, max_val: int) -> int:
        """
        Biztonságos int konverzió határokkal.

        Args:
            value: Konvertálandó érték
            default: Alapértelmezett
            min_val: Minimum érték
            max_val: Maximum érték

        Returns:
            Korlátozott int érték
        """
        try:
            parsed = int(value)
            return max(min_val, min(max_val, parsed))
        except (ValueError, TypeError):
            return default

    # =========================================================================
    # FEATURE MODE KOMPATIBILITÁS
    # =========================================================================

    @staticmethod
    def check_feature_mode_compatibility(feature_mode: str) -> Tuple[bool, str]:
        """
        Ellenőrzi, hogy a feature mode kompatibilis-e az ARIMA-val.

        Az ARIMA mindhárom módot támogatja, de a leghatékonyabb az Original
        módban, ahol a nyers idősor adatok állnak rendelkezésre.

        Args:
            feature_mode: "Original", "Forward Calc", vagy "Rolling Window"

        Returns:
            Tuple[bool, str]: (kompatibilis-e, figyelmeztető üzenet)
        """
        if feature_mode == "Original":
            return True, ""

        if feature_mode == "Forward Calc":
            return True, (
                "ARIMA INFO: Forward Calc mode is supported.\n"
                "Note: ARIMA will be re-fitted on each expanded window.\n"
                "This may increase computation time but provides robust forecasts."
            )

        if feature_mode == "Rolling Window":
            return True, (
                "ARIMA INFO: Rolling Window mode is supported.\n"
                "ARIMA adapts well to rolling forecasts.\n"
                "Consider using Auto-ARIMA for automated parameter selection."
            )

        return True, ""
