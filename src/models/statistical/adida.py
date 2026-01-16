"""
ADIDA Model - Aggregate-Disaggregate Intermittent Demand Approach

Az ADIDA modszer az intermittens (szorvanyos) kereslet elorejelzesere szolgal.
A modszer aggregalja az adatokat idoszakokba, majd elorejelzest keszit az
aggregalt sorozatra, vegul visszabontja az eredeti frekvenciara.

Tamogatott modszerek:
- Standard ADIDA: Aggregalas -> Elorejelzes -> Disaggrealas
- Croston: Kulon kereslet meret es idokoz elorejelzes
- SBA (Syntetos-Boylan Approximation): Torzitas-korrigalt Croston
- TSB (Teunter-Syntetos-Babai): Valoszinuseg alapu frissites minden periodusban

Referenciak:
- Nikolopoulos et al. (2011) "An aggregate-disaggregate intermittent demand approach"
- Croston (1972) "Forecasting and Stock Control for Intermittent Demands"
- Syntetos & Boylan (2005) "On the bias of intermittent demand estimates"
- Teunter, Syntetos & Babai (2011) "Intermittent demand: Linking forecasting to inventory obsolescence"
"""

import warnings
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Kesobb: Optuna tipusok
from dataclasses import dataclass
import numpy as np
import pandas as pd

from models.base import BaseModel, ModelInfo


@dataclass
class CrostonResult:
    """Croston/SBA/TSB eredmeny tarolasa."""
    forecast: float
    z_hat: float  # Simitott kereslet meret
    p_hat: float  # Simitott idokoz (Croston/SBA) vagy valoszinuseg (TSB)


class ADIDAModel(BaseModel):
    """
    ADIDA - Aggregate-Disaggregate Intermittent Demand Approach.

    Az ADIDA modszer lepesei:
    1. Adatok aggregalasa 'k' meretu, nem atfedo bucket-ekbe
    2. Az aggregalt sorozat elorejelzese alap modellel
    3. Az elorejelzes visszabontasa az eredeti frekvenciara

    Jellemzok:
    - Tobbfele modszer: standard, croston, sba, tsb
    - Automatikus aggregacios szint valasztas
    - Sulyozott disaggrealas historikus mintazatok alapjan
    - Robusztus NaN kezeles
    - Batch mod tamogatas nagyobb adathalmazokhoz
    - Optuna integracios parameter optimalizacio

    FONTOS - Feature Mode Kompatibilitas:
    - Original: TAMOGATOTT (ajanlott)
    - Forward Calc: NEM TAMOGATOTT (figyelmeztes)
    - Rolling Window: NEM TAMOGATOTT (figyelmeztes)

    Az ADIDA intermittens kereslethez keszult, ahol a nyers adatok
    a legmegfeleloebbek. A feature-ok (rolling/forward) torzithatjak
    az eredmenyt.
    """

    MODEL_INFO = ModelInfo(
        name="ADIDA",
        category="Statistical Models",
        supports_gpu=False,  # Statisztikai modell, nincs GPU elony
        supports_batch=True,  # Batch mod tamogatott a parhuzamos feldolgozashoz
        gpu_threshold=1000,
        # Feature mode tamogatas
        supports_forward_calc=False,  # NEM ajanlott - figyelmeztet
        supports_rolling_window=False,  # NEM ajanlott - figyelmeztet
        # Specialis modok
        supports_panel_mode=False,  # Nem tamogatja - egyedi kezeles szukseges
        supports_dual_mode=False,  # Nem tamogatja
    )

    PARAM_DEFAULTS = {
        "method": "standard",
        "aggregation_level": "4",
        "base_model": "SES",  # Simple Exponential Smoothing az alapertelmezett
        "alpha": "0.1",  # Kereslet meret simitasi parameter
        "beta": "0.1",  # Idokoz simitasi parameter (Croston variansokhoz)
        "use_weighted_disagg": "True",
        "optimize": "False",  # Parameter optimalizacio Optuna-val
        "seasonal_periods": "12",
    }

    PARAM_OPTIONS = {
        "method": ["standard", "croston", "sba", "tsb"],
        "aggregation_level": ["auto", "2", "3", "4", "5", "6", "8", "12"],
        "base_model": ["SES", "ARIMA", "ETS", "Theta", "Naive"],
        "alpha": ["0.05", "0.1", "0.15", "0.2", "0.3", "0.5"],
        "beta": ["0.05", "0.1", "0.15", "0.2", "0.3", "0.5"],
        "use_weighted_disagg": ["True", "False"],
        "optimize": ["True", "False"],
        "seasonal_periods": ["4", "7", "12", "24", "52"],
    }

    def __init__(self):
        """Inicializalas."""
        super().__init__()
        self._cache = {}  # Optimalizacio: ismetelt szamitasok cache-elese

    @staticmethod
    def check_feature_mode_compatibility(feature_mode: str) -> Tuple[bool, str]:
        """
        Ellenorzi, hogy a feature mode kompatibilis-e az ADIDA-val.

        Args:
            feature_mode: "Original", "Forward Calc", vagy "Rolling Window"

        Returns:
            Tuple[bool, str]: (kompatibilis-e, figyelmeztezo uzenet)
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

    def forecast(
        self,
        data: List[float],
        steps: int,
        params: Dict[str, Any]
    ) -> List[float]:
        """
        ADIDA elorejelzes keszitese.

        Args:
            data: Bemeneti idosor adatok
            steps: Elorejelzesi horizont
            params: Parameterek

        Returns:
            Elorejelzett ertekek listaja
        """
        # Parameterek kinyerese
        full_params = self.get_params_with_defaults(params)
        method = str(full_params.get("method", "standard")).lower()
        # optimize parameter - kesobb Optuna integracio
        # optimize = str(full_params.get("optimize", "False")).lower() == "true"

        # Adatok konvertalasa numpy tombbe
        values = self._to_numpy(data)
        n = len(values)

        if n < 4:
            return [np.nan] * steps

        alpha = float(full_params.get("alpha", 0.1))
        beta = float(full_params.get("beta", 0.1))

        # Croston variansok kulon kezelese
        if method in ["croston", "sba", "tsb"]:
            return self._run_croston_variant(values, steps, method, alpha, beta)

        # Standard ADIDA modszer
        return self._run_standard_adida(values, steps, full_params)

    def forecast_batch(
        self,
        all_data: Dict[str, List[float]],
        steps: int,
        params: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """
        Batch elorejelzes tobb strategiara.

        Parhuzamos feldolgozas joblib-bal ha elerheto.

        Args:
            all_data: Dict ahol kulcs = strategia nev, ertek = idosor
            steps: Elorejelzesi horizont
            params: Parameterek

        Returns:
            Dict ahol kulcs = strategia nev, ertek = elorejelzes
        """
        try:
            from joblib import Parallel, delayed
            import os

            n_jobs = min(os.cpu_count() or 4, len(all_data), 8)

            def _process_single(name: str, data: List[float]) -> tuple:
                try:
                    result = self.forecast(data, steps, params)
                    return (name, result)
                except Exception:
                    return (name, [np.nan] * steps)

            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_process_single)(name, data)
                for name, data in all_data.items()
            )
            return dict(results)

        except ImportError:
            # Fallback szekvencialis feldolgozas
            results = {}
            for name, data in all_data.items():
                try:
                    results[name] = self.forecast(data, steps, params)
                except Exception:
                    results[name] = [np.nan] * steps
            return results

    # _optimize_parameters - KESOBB: Optuna integraciohoz implementalando
    # Jelenleg az optimize parameter nem csinal semmit

    def _to_numpy(self, data) -> np.ndarray:
        """Adatok konvertalasa numpy tombbe."""
        if isinstance(data, pd.Series):
            return data.values.astype(float)
        if isinstance(data, pd.DataFrame):
            return data.values.flatten().astype(float)
        return np.array(data, dtype=float)

    def _run_croston_variant(
        self,
        values: np.ndarray,
        steps: int,
        method: str,
        alpha: float,
        beta: float
    ) -> List[float]:
        """
        Croston es variansainak futtatasa.

        Args:
            values: Idosor adatok
            steps: Elorejelzesi lepesek szama
            method: 'croston', 'sba', vagy 'tsb'
            alpha: Kereslet meret simitasi parameter
            beta: Idokoz/valoszinuseg simitasi parameter

        Returns:
            Elorejelzett ertekek listaja
        """
        # Nem-nulla keresletek es pozicioik
        non_zero_mask = np.abs(values) > 1e-10
        non_zero_indices = np.where(non_zero_mask)[0]

        # Nincs eleg nem-nulla ertek
        if len(non_zero_indices) < 2:
            mean_val = np.nanmean(values) if not np.all(np.isnan(values)) else 0.0
            return [float(mean_val)] * steps

        # Kereslet meretek es idokozok
        z_values = values[non_zero_indices]  # Nem-nulla keresletek
        p_values = np.diff(non_zero_indices).astype(float)  # Idokozok

        if len(p_values) == 0:
            return [float(np.nanmean(z_values))] * steps

        if method == "tsb":
            # TSB: Teunter-Syntetos-Babai modszer
            return self._run_tsb(values, steps, alpha, beta)

        # Croston es SBA
        # Inicializacio az elso ertekekkel
        z_hat = float(z_values[0])
        p_hat = float(p_values[0]) if len(p_values) > 0 else 1.0

        # Exponencialis simitas mindket sorozatra
        for i in range(1, len(z_values)):
            z_hat = alpha * z_values[i] + (1.0 - alpha) * z_hat

        for i in range(1, len(p_values)):
            p_hat = beta * p_values[i] + (1.0 - beta) * p_hat

        # Elorejelzes szamitasa
        if p_hat > 0:
            if method == "sba":
                # SBA: Syntetos-Boylan Approximation torzitas-korrekcio
                # y_hat = (1 - beta/2) * z_hat / p_hat
                forecast_value = (1.0 - beta / 2.0) * z_hat / p_hat
            else:
                # Standard Croston: y_hat = z_hat / p_hat
                forecast_value = z_hat / p_hat
        else:
            forecast_value = z_hat

        return [float(forecast_value)] * steps

    def _run_tsb(
        self,
        values: np.ndarray,
        steps: int,
        alpha: float,
        beta: float
    ) -> List[float]:
        """
        TSB (Teunter-Syntetos-Babai) modszer.

        A TSB kulonbozik a Croston-tol: minden periodusban frissiti
        a kereslet valoszinuseget (nem csak nem-nulla kereslet eseten).

        Args:
            values: Idosor adatok
            steps: Elorejelzesi lepesek
            alpha: Kereslet simitasi parameter
            beta: Valoszinuseg simitasi parameter

        Returns:
            Elorejelzett ertekek
        """
        n = len(values)

        # Nem-nulla keresletek
        non_zero_mask = np.abs(values) > 1e-10
        non_zero_indices = np.where(non_zero_mask)[0]

        if len(non_zero_indices) < 1:
            return [0.0] * steps

        # Inicializalas
        z_hat = float(values[non_zero_indices[0]])  # Elso nem-nulla ertek
        p_hat = float(np.sum(non_zero_mask) / n)  # Kezdeti valoszinuseg

        # Iteracio minden idoponton
        for t in range(n):
            if non_zero_mask[t]:
                # Van kereslet
                z_hat = alpha * values[t] + (1.0 - alpha) * z_hat
                p_hat = beta * 1.0 + (1.0 - beta) * p_hat
            else:
                # Nincs kereslet - csak valoszinuseg frissul
                p_hat = beta * 0.0 + (1.0 - beta) * p_hat

        # Elorejelzes: E[Y] = p * z
        forecast_value = p_hat * z_hat

        return [float(forecast_value)] * steps

    def _run_standard_adida(
        self,
        values: np.ndarray,
        steps: int,
        params: Dict[str, Any]
    ) -> List[float]:
        """
        Standard ADIDA modszer futtatasa.

        Args:
            values: Idosor adatok
            steps: Elorejelzesi horizont
            params: Osszes parameter

        Returns:
            Elorejelzett ertekek
        """
        aggregation_level_str = str(params.get("aggregation_level", "4"))
        base_model_name = str(params.get("base_model", "SES"))
        use_weighted_disagg = str(params.get("use_weighted_disagg", "True")).lower() == "true"
        seasonal_periods = int(params.get("seasonal_periods", 12))

        n = len(values)

        # Aggregacios szint meghatarozasa
        if aggregation_level_str.lower() == "auto":
            aggregation_level = self._find_optimal_aggregation_level(values)
        else:
            aggregation_level = int(aggregation_level_str)

        # Ellenorzes: van-e eleg adat
        if n < aggregation_level * 2:
            return [np.nan] * steps

        # 1. AGGREGALAS
        # Idosor vagasa, hogy illeszkedjen az aggregacios szinthez
        remainder = n % aggregation_level
        if remainder != 0:
            trimmed_values = values[remainder:]
        else:
            trimmed_values = values

        # Within-bucket sulyok szamitasa (ha szukseges)
        within_bucket_weights = self._calculate_bucket_weights(
            trimmed_values, aggregation_level, use_weighted_disagg
        )

        # Aggregalt ertekek szamitasa
        aggregated_values = []
        for i in range(0, len(trimmed_values), aggregation_level):
            chunk = trimmed_values[i:i + aggregation_level]
            if np.all(np.isnan(chunk)):
                aggregated_values.append(np.nan)
            else:
                aggregated_values.append(float(np.nansum(chunk)))

        aggregated_series = np.array(aggregated_values)

        if len(aggregated_series) < 4:
            return [np.nan] * steps

        # 2. ELOREJELZES az aggregalt sorozatra
        agg_steps = (steps + aggregation_level - 1) // aggregation_level

        # Seasonal period skalazasa
        new_seasonal_periods = max(2, seasonal_periods // aggregation_level)
        max_seasonal = len(aggregated_series) // 2
        if new_seasonal_periods > max_seasonal:
            new_seasonal_periods = max(2, max_seasonal)

        agg_forecast = self._forecast_aggregated(
            aggregated_series,
            agg_steps,
            base_model_name,
            new_seasonal_periods,
            params
        )

        # 3. DISAGGREALAS
        final_forecast = []
        for val in agg_forecast:
            if np.isnan(val):
                final_forecast.extend([np.nan] * aggregation_level)
            else:
                # Sulyozott disaggrealas
                for weight in within_bucket_weights:
                    final_forecast.append(float(val * weight))

        return final_forecast[:steps]

    def _calculate_bucket_weights(
        self,
        values: np.ndarray,
        aggregation_level: int,
        use_weighted: bool
    ) -> np.ndarray:
        """
        Bucket-en beluli sulyok szamitasa a disaggrealashoz.

        Args:
            values: Trimelt idosor
            aggregation_level: Aggregacios szint
            use_weighted: Sulyozott disaggrealas hasznalata

        Returns:
            Sulyok tombje (osszege = 1)
        """
        if not use_weighted:
            return np.ones(aggregation_level) / aggregation_level

        position_sums = np.zeros(aggregation_level)
        position_counts = np.zeros(aggregation_level)

        for i in range(0, len(values), aggregation_level):
            chunk = values[i:i + aggregation_level]
            chunk_sum = np.nansum(np.abs(chunk))

            if chunk_sum > 1e-10:  # Nem nulla osszeg
                for j, val in enumerate(chunk):
                    if j < len(chunk) and not np.isnan(val):
                        position_sums[j] += abs(val) / chunk_sum
                        position_counts[j] += 1

        # Sulyok szamitasa
        weights = np.ones(aggregation_level) / aggregation_level
        for j in range(aggregation_level):
            if position_counts[j] > 0:
                weights[j] = position_sums[j] / position_counts[j]

        # Normalizalas
        weight_sum = np.sum(weights)
        if weight_sum > 1e-10:
            weights = weights / weight_sum
        else:
            weights = np.ones(aggregation_level) / aggregation_level

        return weights

    def _find_optimal_aggregation_level(
        self,
        values: np.ndarray,
        candidate_levels: Optional[List[int]] = None
    ) -> int:
        """
        Optimalis aggregacios szint keresese.

        Cross-validation alapu kereses: teszteli a kulonbozo szinteket
        es a legkisebb MAE-val rendelkezot valasztja.

        Args:
            values: Idosor adatok
            candidate_levels: Tesztelendo szintek

        Returns:
            Optimalis aggregacios szint
        """
        if candidate_levels is None:
            candidate_levels = [2, 3, 4, 6, 8, 12]

        n = len(values)
        valid_levels = [k for k in candidate_levels if n >= k * 4]

        if not valid_levels:
            return 4

        # Holdout: utolso 20% (min 4 pont)
        holdout_size = max(4, int(n * 0.2))
        train_values = values[:-holdout_size]
        test_values = values[-holdout_size:]

        best_level = valid_levels[0]
        best_mae = float("inf")

        for k in valid_levels:
            try:
                # Egyszeru elorejelzes az aggregalt sorozatra
                forecast = self._simple_aggregated_forecast(train_values, k, holdout_size)

                if len(forecast) >= len(test_values):
                    forecast = forecast[:len(test_values)]

                # MAE szamitas
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
        Egyszeru elorejelzes az optimalis szint keresesehez.

        SES alapu elorejelzes a gyorsasag erdekeben.
        """
        n = len(values)
        remainder = n % aggregation_level
        trimmed = values[remainder:] if remainder != 0 else values

        # Aggregalas
        agg_values = []
        for i in range(0, len(trimmed), aggregation_level):
            chunk = trimmed[i:i + aggregation_level]
            if not np.all(np.isnan(chunk)):
                agg_values.append(float(np.nansum(chunk)))

        if len(agg_values) < 2:
            return [np.nan] * steps

        # SES elorejelzes
        agg_steps = (steps + aggregation_level - 1) // aggregation_level
        ses_forecast = self._simple_exponential_smoothing(agg_values, agg_steps, alpha=0.1)

        # Disaggrealas (egyenletes)
        result = []
        for val in ses_forecast:
            result.extend([val / aggregation_level] * aggregation_level)

        return result[:steps]

    def _simple_exponential_smoothing(
        self,
        values: List[float],
        steps: int,
        alpha: float = 0.1
    ) -> List[float]:
        """Simple Exponential Smoothing elorejelzes."""
        if not values:
            return [np.nan] * steps

        # Simitas
        level = values[0]
        for val in values[1:]:
            if not np.isnan(val):
                level = alpha * val + (1 - alpha) * level

        return [level] * steps

    def _forecast_aggregated(
        self,
        aggregated_series: np.ndarray,
        steps: int,
        base_model: str,
        seasonal_periods: int,
        params: Dict[str, Any]
    ) -> List[float]:
        """
        Elorejelzes az aggregalt sorozatra.

        Args:
            aggregated_series: Aggregalt idosor
            steps: Elorejelzesi lepesek
            base_model: Alap modell neve
            seasonal_periods: Szezonalis periodus
            params: Eredeti parameterek

        Returns:
            Elorejelzett ertekek
        """
        alpha = float(params.get("alpha", 0.1))

        try:
            if base_model == "SES":
                return self._simple_exponential_smoothing(
                    aggregated_series.tolist(), steps, alpha
                )

            if base_model == "Naive":
                # Utolso nehany ertek atlaga
                last_vals = aggregated_series[-min(4, len(aggregated_series)):]
                last_val = float(np.nanmean(last_vals)) if not np.all(np.isnan(last_vals)) else 0.0
                return [last_val] * steps

            if base_model == "ARIMA":
                return self._forecast_with_arima(
                    aggregated_series, steps, seasonal_periods
                )

            if base_model == "ETS":
                return self._forecast_with_ets(
                    aggregated_series, steps, seasonal_periods
                )

            if base_model == "Theta":
                return self._forecast_with_theta(
                    aggregated_series, steps, seasonal_periods
                )

            # Fallback: SES
            return self._simple_exponential_smoothing(
                aggregated_series.tolist(), steps, alpha
            )

        except Exception:
            # Barmilyen hiba eseten fallback
            last_vals = aggregated_series[-min(4, len(aggregated_series)):]
            last_val = float(np.nanmean(last_vals)) if not np.all(np.isnan(last_vals)) else 0.0
            return [last_val] * steps

    def _forecast_with_arima(
        self,
        series: np.ndarray,
        steps: int,
        seasonal_periods: int
    ) -> List[float]:
        """ARIMA elorejelzes."""
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
        """ETS elorejelzes."""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Szezonalitas ellenorzese
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
        """Theta elorejelzes."""
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
