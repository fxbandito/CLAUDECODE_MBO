"""
ADIDA Model - Aggregate-Disaggregate Intermittent Demand Approach

Az ADIDA módszer az intermittens (szórványos) kereslet előrejelzésére szolgál.
A módszer aggregálja az adatokat időszakokba, majd előrejelzést készít az
aggregált sorozatra, végül visszabontja az eredeti frekvenciára.

Támogatott módszerek:
- Standard ADIDA: Aggregálás -> Előrejelzés -> Disaggregálás
- Croston: Külön kereslet méret és időköz előrejelzés
- SBA (Syntetos-Boylan Approximation): Torzítás-korrigált Croston
- TSB (Teunter-Syntetos-Babai): Valószínűség alapú frissítés minden periódusban

Referenciák:
- Nikolopoulos et al. (2011) "An aggregate-disaggregate intermittent demand approach"
- Croston (1972) "Forecasting and Stock Control for Intermittent Demands"
- Syntetos & Boylan (2005) "On the bias of intermittent demand estimates"
- Teunter, Syntetos & Babai (2011) "Intermittent demand: Linking forecasting to inventory obsolescence"

Használat:
    from models.statistical.adida import ADIDAModel

    model = ADIDAModel()

    # Egyszerű forecast
    forecasts = model.forecast(data, steps=52, params={"method": "croston"})

    # Teljes pipeline (validálás + utófeldolgozás + horizontok)
    result = model.forecast_with_pipeline(
        data=profit_series,
        steps=52,
        params={"method": "sba", "alpha": "0.1"},
        strategy_id="STR_001"
    )
    print(f"1M forecast: {result['Forecast_1M']}")
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models.base import BaseModel, ModelInfo

logger = logging.getLogger(__name__)


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class CrostonResult:
    """Croston/SBA/TSB eredmény tárolása."""
    forecast: float
    z_hat: float  # Simított kereslet méret
    p_hat: float  # Simított időköz (Croston/SBA) vagy valószínűség (TSB)


# =============================================================================
# ADIDA MODEL
# =============================================================================

class ADIDAModel(BaseModel):
    """
    ADIDA - Aggregate-Disaggregate Intermittent Demand Approach.

    Az ADIDA módszer lépései:
    1. Adatok aggregálása 'k' méretű, nem átfedő bucket-ekbe
    2. Az aggregált sorozat előrejelzése alap modellel
    3. Az előrejelzés visszabontása az eredeti frekvenciára

    Jellemzők:
    - Többféle módszer: standard, croston, sba, tsb
    - Automatikus aggregációs szint választás
    - Súlyozott disaggregálás historikus mintázatok alapján
    - Robusztus NaN kezelés (utils/postprocessing)
    - Batch mód támogatás nagyobb adathalmazokhoz

    Feature Mode Kompatibilitás:
    - Original: TÁMOGATOTT (ajánlott)
    - Forward Calc: NEM TÁMOGATOTT (figyelmeztetés)
    - Rolling Window: NEM TÁMOGATOTT (figyelmeztetés)

    Példa:
        model = ADIDAModel()

        # Egyszerű forecast
        forecasts = model.forecast(data, steps=52, params={"method": "croston"})

        # Teljes pipeline validálással és utófeldolgozással
        result = model.forecast_with_pipeline(
            data=profit_series,
            steps=52,
            params={"method": "sba"},
            strategy_id="STR_001"
        )
    """

    MODEL_INFO = ModelInfo(
        name="ADIDA",
        category="Statistical Models",
        supports_gpu=False,
        supports_batch=True,
        gpu_threshold=1000,
        supports_forward_calc=False,
        supports_rolling_window=False,
        supports_panel_mode=False,
        supports_dual_mode=False,
    )

    PARAM_DEFAULTS = {
        "method": "standard",
        "aggregation_level": "4",
        "base_model": "SES",
        "alpha": "0.1",
        "beta": "0.1",
        "use_weighted_disagg": "True",
        "seasonal_periods": "12",
    }

    PARAM_OPTIONS = {
        "method": ["standard", "croston", "sba", "tsb"],
        "aggregation_level": ["auto", "2", "3", "4", "5", "6", "8", "12"],
        "base_model": ["SES", "ARIMA", "ETS", "Theta", "Naive"],
        "alpha": ["0.05", "0.1", "0.15", "0.2", "0.3", "0.5"],
        "beta": ["0.05", "0.1", "0.15", "0.2", "0.3", "0.5"],
        "use_weighted_disagg": ["True", "False"],
        "seasonal_periods": ["4", "7", "12", "24", "52"],
    }

    # ADIDA minimum adatpont - felülírja az alap 24-et
    MIN_DATA_POINTS = 8

    def __init__(self):
        """Inicializálás."""
        super().__init__()

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
        ADIDA előrejelzés készítése.

        Args:
            data: Bemeneti idősor adatok (már validált)
            steps: Előrejelzési horizont
            params: Paraméterek

        Returns:
            Előrejelzett értékek listája (utófeldolgozva)

        Note:
            Ha teljes pipeline-t szeretnél (validálás + utófeldolgozás + horizontok),
            használd a forecast_with_pipeline() metódust helyette.
        """
        # Paraméterek kinyerése
        full_params = self.get_params_with_defaults(params)
        method = str(full_params.get("method", "standard")).lower()

        # Numpy array konverzió
        values = np.array(data, dtype=np.float64)
        n = len(values)

        # Minimum adat ellenőrzés
        if n < self.MIN_DATA_POINTS:
            logger.debug("ADIDA: Not enough data points (%d < %d)", n, self.MIN_DATA_POINTS)
            return [0.0] * steps

        # Paraméterek
        alpha = self._parse_float(full_params.get("alpha", "0.1"), 0.1)
        beta = self._parse_float(full_params.get("beta", "0.1"), 0.1)

        # Módszer szerinti futtatás
        try:
            if method in ["croston", "sba", "tsb"]:
                raw_forecasts = self._run_croston_variant(values, steps, method, alpha, beta)
            else:
                raw_forecasts = self._run_standard_adida(values, steps, full_params)

            # Utófeldolgozás (NaN kezelés, outlier capping)
            return self.postprocess(raw_forecasts, allow_negative=True)

        except Exception as e:
            logger.error("ADIDA forecast error: %s", str(e))
            return [0.0] * steps

    # =========================================================================
    # CROSTON VARIÁNSOK
    # =========================================================================

    def _run_croston_variant(
        self,
        values: np.ndarray,
        steps: int,
        method: str,
        alpha: float,
        beta: float
    ) -> List[float]:
        """
        Croston és variánsainak futtatása.

        Args:
            values: Idősor adatok
            steps: Előrejelzési lépések száma
            method: 'croston', 'sba', vagy 'tsb'
            alpha: Kereslet méret simítási paraméter (0 < alpha < 1)
            beta: Időköz/valószínűség simítási paraméter

        Returns:
            Előrejelzett értékek listája
        """
        # Nem-nulla keresletek és pozícióik
        non_zero_mask = np.abs(values) > 1e-10
        non_zero_indices = np.where(non_zero_mask)[0]

        # Nincs elég nem-nulla érték
        if len(non_zero_indices) < 2:
            mean_val = float(np.nanmean(values)) if not np.all(np.isnan(values)) else 0.0
            return [mean_val] * steps

        # Kereslet méretek és időközök
        z_values = values[non_zero_indices]  # Nem-nulla keresletek
        p_values = np.diff(non_zero_indices).astype(np.float64)  # Időközök

        if len(p_values) == 0:
            return [float(np.nanmean(z_values))] * steps

        # TSB külön kezelés
        if method == "tsb":
            return self._run_tsb(values, steps, alpha, beta)

        # Croston és SBA
        result = self._calculate_croston(z_values, p_values, alpha, beta, method)

        return [result.forecast] * steps

    def _calculate_croston(
        self,
        z_values: np.ndarray,
        p_values: np.ndarray,
        alpha: float,
        beta: float,
        method: str
    ) -> CrostonResult:
        """
        Croston/SBA számítás.

        A Croston módszer két sorozatot simít exponenciálisan:
        - z: kereslet méretek (nem-nulla értékek)
        - p: időközök (periódusok a keresletek között)

        Az előrejelzés: y = z/p (Croston) vagy y = (1-β/2)*z/p (SBA)

        Args:
            z_values: Nem-nulla kereslet értékek
            p_values: Időközök
            alpha: z simítási paraméter
            beta: p simítási paraméter
            method: 'croston' vagy 'sba'

        Returns:
            CrostonResult a számított értékekkel
        """
        # Inicializáció az első értékekkel
        z_hat = float(z_values[0])
        p_hat = float(p_values[0]) if len(p_values) > 0 else 1.0

        # Exponenciális simítás - z (kereslet méret)
        for i in range(1, len(z_values)):
            z_hat = alpha * float(z_values[i]) + (1.0 - alpha) * z_hat

        # Exponenciális simítás - p (időköz)
        for i in range(1, len(p_values)):
            p_hat = beta * float(p_values[i]) + (1.0 - beta) * p_hat

        # Előrejelzés számítása
        if p_hat > 0:
            if method == "sba":
                # SBA: Syntetos-Boylan Approximation - torzítás korrekció
                # A standard Croston felfelé torzít, ezt korrigálja
                forecast_value = (1.0 - beta / 2.0) * z_hat / p_hat
            else:
                # Standard Croston: y = z/p
                forecast_value = z_hat / p_hat
        else:
            forecast_value = z_hat

        return CrostonResult(
            forecast=float(forecast_value),
            z_hat=z_hat,
            p_hat=p_hat
        )

    def _run_tsb(
        self,
        values: np.ndarray,
        steps: int,
        alpha: float,
        beta: float
    ) -> List[float]:
        """
        TSB (Teunter-Syntetos-Babai) módszer.

        A TSB különbözik a Croston-tól: MINDEN periódusban frissíti
        a kereslet valószínűségét (nem csak nem-nulla kereslet esetén).

        Ez jobban kezeli a lassan elavuló termékeket, ahol a kereslet
        valószínűsége folyamatosan csökken.

        Előrejelzés: E[Y] = p * z
        ahol p = kereslet valószínűsége, z = várható kereslet méret

        Args:
            values: Idősor adatok
            steps: Előrejelzési lépések
            alpha: Kereslet simítási paraméter
            beta: Valószínűség simítási paraméter

        Returns:
            Előrejelzett értékek
        """
        n = len(values)

        # Nem-nulla keresletek azonosítása
        non_zero_mask = np.abs(values) > 1e-10
        non_zero_indices = np.where(non_zero_mask)[0]

        if len(non_zero_indices) < 1:
            return [0.0] * steps

        # Inicializálás
        z_hat = float(values[non_zero_indices[0]])  # Első nem-nulla érték
        p_hat = float(np.sum(non_zero_mask)) / n  # Kezdeti valószínűség

        # Iteráció MINDEN időponton
        for t in range(n):
            if non_zero_mask[t]:
                # Van kereslet - mindkettő frissül
                z_hat = alpha * float(values[t]) + (1.0 - alpha) * z_hat
                p_hat = beta * 1.0 + (1.0 - beta) * p_hat
            else:
                # Nincs kereslet - csak valószínűség frissül (csökken)
                p_hat = beta * 0.0 + (1.0 - beta) * p_hat

        # Előrejelzés: E[Y] = p * z
        forecast_value = p_hat * z_hat

        return [float(forecast_value)] * steps

    # =========================================================================
    # STANDARD ADIDA
    # =========================================================================

    def _run_standard_adida(
        self,
        values: np.ndarray,
        steps: int,
        params: Dict[str, Any]
    ) -> List[float]:
        """
        Standard ADIDA módszer futtatása.

        Lépések:
        1. Aggregálás k méretű bucket-ekbe
        2. Előrejelzés az aggregált sorozatra
        3. Disaggregálás (visszabontás) súlyozott módon

        Args:
            values: Idősor adatok
            steps: Előrejelzési horizont
            params: Összes paraméter

        Returns:
            Előrejelzett értékek
        """
        aggregation_level_str = str(params.get("aggregation_level", "4"))
        base_model_name = str(params.get("base_model", "SES"))
        use_weighted_disagg = str(params.get("use_weighted_disagg", "True")).lower() == "true"
        seasonal_periods = int(params.get("seasonal_periods", 12))

        n = len(values)

        # Aggregációs szint meghatározása
        if aggregation_level_str.lower() == "auto":
            aggregation_level = self._find_optimal_aggregation_level(values)
        else:
            aggregation_level = int(aggregation_level_str)

        # Ellenőrzés: van-e elég adat
        if n < aggregation_level * 2:
            logger.debug("ADIDA: Not enough data for aggregation level %d", aggregation_level)
            return [float(np.nanmean(values))] * steps if not np.all(np.isnan(values)) else [0.0] * steps

        # 1. AGGREGÁLÁS
        trimmed_values, within_bucket_weights = self._aggregate_data(
            values, aggregation_level, use_weighted_disagg
        )

        if len(trimmed_values) == 0:
            return [0.0] * steps

        # Aggregált értékek számítása
        aggregated_values = self._calculate_aggregated_values(trimmed_values, aggregation_level)

        if len(aggregated_values) < 4:
            return [0.0] * steps

        # 2. ELŐREJELZÉS az aggregált sorozatra
        agg_steps = (steps + aggregation_level - 1) // aggregation_level

        # Seasonal period skálázása
        new_seasonal_periods = self._adjust_seasonal_periods(
            seasonal_periods, aggregation_level, len(aggregated_values)
        )

        agg_forecast = self._forecast_aggregated(
            aggregated_values,
            agg_steps,
            base_model_name,
            new_seasonal_periods,
            params
        )

        # 3. DISAGGREGÁLÁS
        final_forecast = self._disaggregate_forecast(
            agg_forecast, within_bucket_weights, aggregation_level
        )

        return final_forecast[:steps]

    def _aggregate_data(
        self,
        values: np.ndarray,
        aggregation_level: int,
        use_weighted: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adatok aggregálásra előkészítése és súlyok számítása.

        Args:
            values: Idősor
            aggregation_level: Aggregációs szint (k)
            use_weighted: Súlyozott disaggregálás használata

        Returns:
            Tuple: (trimelt értékek, bucket súlyok)
        """
        n = len(values)

        # Idősor vágása, hogy illeszkedjen az aggregációs szinthez
        remainder = n % aggregation_level
        if remainder != 0:
            trimmed_values = values[remainder:]
        else:
            trimmed_values = values

        # Súlyok számítása
        within_bucket_weights = self._calculate_bucket_weights(
            trimmed_values, aggregation_level, use_weighted
        )

        return trimmed_values, within_bucket_weights

    def _calculate_aggregated_values(
        self,
        trimmed_values: np.ndarray,
        aggregation_level: int
    ) -> np.ndarray:
        """
        Aggregált értékek számítása.

        Args:
            trimmed_values: Trimelt idősor
            aggregation_level: Aggregációs szint

        Returns:
            Aggregált értékek tömbje
        """
        aggregated_values = []

        for i in range(0, len(trimmed_values), aggregation_level):
            chunk = trimmed_values[i:i + aggregation_level]

            if np.all(np.isnan(chunk)):
                aggregated_values.append(np.nan)
            else:
                aggregated_values.append(float(np.nansum(chunk)))

        return np.array(aggregated_values, dtype=np.float64)

    def _calculate_bucket_weights(
        self,
        values: np.ndarray,
        aggregation_level: int,
        use_weighted: bool
    ) -> np.ndarray:
        """
        Bucket-en belüli súlyok számítása a disaggregáláshoz.

        A súlyok megmutatják, hogy történetileg a bucket-en belül
        melyik pozíció mekkora részét képviseli az összértéknek.

        Args:
            values: Trimelt idősor
            aggregation_level: Aggregációs szint
            use_weighted: Súlyozott disaggregálás használata

        Returns:
            Súlyok tömbje (összege = 1)
        """
        if not use_weighted:
            # Egyenletes eloszlás
            return np.ones(aggregation_level) / aggregation_level

        position_sums = np.zeros(aggregation_level)
        position_counts = np.zeros(aggregation_level)

        for i in range(0, len(values), aggregation_level):
            chunk = values[i:i + aggregation_level]
            chunk_sum = np.nansum(np.abs(chunk))

            if chunk_sum > 1e-10:  # Nem nulla összeg
                for j, val in enumerate(chunk):
                    if j < len(chunk) and not np.isnan(val):
                        position_sums[j] += abs(val) / chunk_sum
                        position_counts[j] += 1

        # Súlyok számítása
        weights = np.ones(aggregation_level) / aggregation_level
        for j in range(aggregation_level):
            if position_counts[j] > 0:
                weights[j] = position_sums[j] / position_counts[j]

        # Normalizálás (összeg = 1)
        weight_sum = np.sum(weights)
        if weight_sum > 1e-10:
            weights = weights / weight_sum

        return weights

    def _disaggregate_forecast(
        self,
        agg_forecast: List[float],
        weights: np.ndarray,
        aggregation_level: int
    ) -> List[float]:
        """
        Aggregált előrejelzés visszabontása.

        Args:
            agg_forecast: Aggregált előrejelzés
            weights: Bucket súlyok
            aggregation_level: Aggregációs szint

        Returns:
            Disaggregált előrejelzés
        """
        final_forecast = []

        for val in agg_forecast:
            if np.isnan(val):
                final_forecast.extend([np.nan] * aggregation_level)
            else:
                for weight in weights:
                    final_forecast.append(float(val * weight))

        return final_forecast

    # =========================================================================
    # OPTIMÁLIS AGGREGÁCIÓS SZINT
    # =========================================================================

    def _find_optimal_aggregation_level(
        self,
        values: np.ndarray,
        candidate_levels: Optional[List[int]] = None
    ) -> int:
        """
        Optimális aggregációs szint keresése.

        Cross-validation alapú keresés: teszteli a különböző szinteket
        és a legkisebb MAE-val rendelkezőt választja.

        Args:
            values: Idősor adatok
            candidate_levels: Tesztelendő szintek

        Returns:
            Optimális aggregációs szint
        """
        if candidate_levels is None:
            candidate_levels = [2, 3, 4, 6, 8, 12]

        n = len(values)
        valid_levels = [k for k in candidate_levels if n >= k * 4]

        if not valid_levels:
            return 4

        # Holdout: utolsó 20% (min 4 pont)
        holdout_size = max(4, int(n * 0.2))
        train_values = values[:-holdout_size]
        test_values = values[-holdout_size:]

        best_level = valid_levels[0]
        best_mae = float("inf")

        for k in valid_levels:
            try:
                forecast = self._simple_aggregated_forecast(train_values, k, holdout_size)

                if len(forecast) >= len(test_values):
                    forecast = forecast[:len(test_values)]

                # MAE számítás
                valid_pairs = [
                    (f, a) for f, a in zip(forecast, test_values)
                    if not np.isnan(f) and not np.isnan(a)
                ]

                if valid_pairs:
                    mae = np.mean([abs(f - a) for f, a in valid_pairs])
                    if mae < best_mae:
                        best_mae = mae
                        best_level = k

            except Exception:
                continue

        return best_level

    def _simple_aggregated_forecast(
        self,
        values: np.ndarray,
        aggregation_level: int,
        steps: int
    ) -> List[float]:
        """
        Egyszerű előrejelzés az optimális szint kereséséhez.

        SES alapú előrejelzés a gyorsaság érdekében.
        """
        n = len(values)
        remainder = n % aggregation_level
        trimmed = values[remainder:] if remainder != 0 else values

        # Aggregálás
        agg_values = []
        for i in range(0, len(trimmed), aggregation_level):
            chunk = trimmed[i:i + aggregation_level]
            if not np.all(np.isnan(chunk)):
                agg_values.append(float(np.nansum(chunk)))

        if len(agg_values) < 2:
            return [np.nan] * steps

        # SES előrejelzés
        agg_steps = (steps + aggregation_level - 1) // aggregation_level
        ses_forecast = self._simple_exponential_smoothing(agg_values, agg_steps, alpha=0.1)

        # Disaggregálás (egyenletes)
        result = []
        for val in ses_forecast:
            result.extend([val / aggregation_level] * aggregation_level)

        return result[:steps]

    # =========================================================================
    # ALAP MODELLEK AZ AGGREGÁLT SOROZATHOZ
    # =========================================================================

    def _forecast_aggregated(
        self,
        aggregated_series: np.ndarray,
        steps: int,
        base_model: str,
        seasonal_periods: int,
        params: Dict[str, Any]
    ) -> List[float]:
        """
        Előrejelzés az aggregált sorozatra.

        Args:
            aggregated_series: Aggregált idősor
            steps: Előrejelzési lépések
            base_model: Alap modell neve
            seasonal_periods: Szezonális periódus
            params: Eredeti paraméterek

        Returns:
            Előrejelzett értékek
        """
        alpha = self._parse_float(params.get("alpha", "0.1"), 0.1)

        try:
            if base_model == "SES":
                return self._simple_exponential_smoothing(
                    aggregated_series.tolist(), steps, alpha
                )

            if base_model == "Naive":
                return self._naive_forecast(aggregated_series, steps)

            if base_model == "ARIMA":
                return self._forecast_with_arima(aggregated_series, steps)

            if base_model == "ETS":
                return self._forecast_with_ets(aggregated_series, steps, seasonal_periods)

            if base_model == "Theta":
                return self._forecast_with_theta(aggregated_series, steps, seasonal_periods)

            # Fallback: SES
            return self._simple_exponential_smoothing(
                aggregated_series.tolist(), steps, alpha
            )

        except Exception as e:
            logger.debug("ADIDA base model error: %s, falling back to naive", str(e))
            return self._naive_forecast(aggregated_series, steps)

    def _simple_exponential_smoothing(
        self,
        values: List[float],
        steps: int,
        alpha: float = 0.1
    ) -> List[float]:
        """
        Simple Exponential Smoothing előrejelzés.

        Args:
            values: Idősor értékek
            steps: Előrejelzési lépések
            alpha: Simítási paraméter (0 < alpha < 1)

        Returns:
            Előrejelzett értékek (konstans)
        """
        if not values:
            return [0.0] * steps

        # Simítás
        level = float(values[0])
        for val in values[1:]:
            if not np.isnan(val):
                level = alpha * float(val) + (1.0 - alpha) * level

        return [level] * steps

    def _naive_forecast(
        self,
        series: np.ndarray,
        steps: int
    ) -> List[float]:
        """
        Naive előrejelzés: utolsó néhány érték átlaga.

        Args:
            series: Idősor
            steps: Lépések

        Returns:
            Előrejelzés
        """
        last_vals = series[-min(4, len(series)):]
        last_val = float(np.nanmean(last_vals)) if not np.all(np.isnan(last_vals)) else 0.0
        return [last_val] * steps

    def _forecast_with_arima(
        self,
        series: np.ndarray,
        steps: int
    ) -> List[float]:
        """ARIMA előrejelzés az aggregált sorozatra."""
        try:
            from statsmodels.tsa.arima.model import ARIMA as StatsARIMA

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = StatsARIMA(series, order=(1, 1, 1))
                fitted = model.fit()
                forecast = fitted.forecast(steps=steps)
                return forecast.tolist()

        except Exception:
            return self._simple_exponential_smoothing(series.tolist(), steps)

    def _forecast_with_ets(
        self,
        series: np.ndarray,
        steps: int,
        seasonal_periods: int
    ) -> List[float]:
        """ETS előrejelzés az aggregált sorozatra."""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Szezonalitás ellenőrzése
                use_seasonal = (
                    seasonal_periods >= 2 and
                    len(series) >= seasonal_periods * 2
                )

                if use_seasonal:
                    model = ExponentialSmoothing(
                        series,
                        trend="add",
                        seasonal="add",
                        seasonal_periods=seasonal_periods
                    )
                else:
                    model = ExponentialSmoothing(
                        series,
                        trend="add",
                        seasonal=None
                    )

                fitted = model.fit(optimized=True)
                forecast = fitted.forecast(steps)
                return forecast.tolist()

        except Exception:
            return self._simple_exponential_smoothing(series.tolist(), steps)

    def _forecast_with_theta(
        self,
        series: np.ndarray,
        steps: int,
        seasonal_periods: int
    ) -> List[float]:
        """Theta előrejelzés az aggregált sorozatra."""
        try:
            from statsmodels.tsa.forecasting.theta import ThetaModel

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ThetaModel(series, period=seasonal_periods)
                fitted = model.fit()
                forecast = fitted.forecast(steps)
                return forecast.tolist()

        except Exception:
            return self._simple_exponential_smoothing(series.tolist(), steps)

    # =========================================================================
    # SEGÉD METÓDUSOK
    # =========================================================================

    def _adjust_seasonal_periods(
        self,
        original_periods: int,
        aggregation_level: int,
        data_length: int
    ) -> int:
        """
        Szezonális periódus skálázása az aggregációs szinthez.

        Args:
            original_periods: Eredeti szezonális periódus
            aggregation_level: Aggregációs szint
            data_length: Aggregált adat hossza

        Returns:
            Skálázott szezonális periódus
        """
        new_periods = max(2, original_periods // aggregation_level)
        max_seasonal = data_length // 2

        if new_periods > max_seasonal:
            new_periods = max(2, max_seasonal)

        return new_periods

    @staticmethod
    def _parse_float(value: Any, default: float) -> float:
        """
        Biztonságos float konverzió.

        Args:
            value: Konvertálandó érték
            default: Alapértelmezett érték hiba esetén

        Returns:
            Float érték
        """
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    # =========================================================================
    # FEATURE MODE KOMPATIBILITÁS
    # =========================================================================

    @staticmethod
    def check_feature_mode_compatibility(feature_mode: str) -> Tuple[bool, str]:
        """
        Ellenőrzi, hogy a feature mode kompatibilis-e az ADIDA-val.

        Args:
            feature_mode: "Original", "Forward Calc", vagy "Rolling Window"

        Returns:
            Tuple[bool, str]: (kompatibilis-e, figyelmeztető üzenet)
        """
        if feature_mode == "Original":
            return True, ""

        if feature_mode == "Forward Calc":
            return False, (
                "ADIDA WARNING: Forward Calc mode is not recommended!\n"
                "ADIDA is designed for raw intermittent demand data.\n"
                "Expanded window features may distort the forecast.\n"
                "Please switch to 'Original' mode for best results."
            )

        if feature_mode == "Rolling Window":
            return False, (
                "ADIDA WARNING: Rolling Window mode is not recommended!\n"
                "ADIDA works best with raw time series data.\n"
                "Rolling features may hide the intermittent pattern.\n"
                "Please switch to 'Original' mode for best results."
            )

        return True, ""
