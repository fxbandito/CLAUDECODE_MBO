"""
MBO Trading Strategy Analyzer - BaseModel
Absztrakt alap osztály minden forecasting modelhez.

Minden model ÖNÁLLÓ egység - saját konfiggal, paraméterekkel.
Nincs közös config fájl, nincs keveredés!

Elérhető utility függvények a modell implementációkhoz:
=========================================================

Model Utilities (src/models/utils/):
    from models.utils import (
        # Postprocessing - Forecast utófeldolgozás
        postprocess_forecasts,      # Teljes utófeldolgozás (NaN, outlier, clip)
        handle_nan_values,          # NaN/Inf kezelés
        cap_outliers,               # 3×IQR outlier capping
        PostprocessingConfig,       # Egyedi konfiguráció

        # Aggregation - Horizont aggregátumok
        calculate_horizons,         # Standard horizontok (h1, h4, h13, h26, h52)
        HorizonResult,              # Eredmény dataclass
        STANDARD_HORIZONS,          # {h1: 1, h4: 4, ...}

        # Validation - Input validálás
        validate_input_data,        # Adat validálás
        prepare_forecast_data,      # Egyszerűsített előkészítés
        ValidationResult,           # Eredmény dataclass
        MIN_DATA_POINTS,            # 24 (minimum adatpont)

        # Monitoring - Teljesítmény mérés
        ForecastTimer,              # Context manager időméréshez
        log_slow_forecast,          # Lassú forecast logolás
        PerformanceStats,           # Statisztika dataclass
    )

Process Management (src/analysis/process_utils.py):
    from analysis.process_utils import (
        cleanup_cuda_context,        # CUDA memória cleanup
        force_kill_child_processes,  # Worker leállítás
        init_worker_environment,     # Worker thread/env config
        set_process_priority,        # Process prioritás
    )

Data Utilities (src/data/processor.py):
    from data.processor import DataProcessor

    data_mode = DataProcessor.detect_data_mode(df)
    # Visszatérési értékek: "original", "forward", "rolling"

    groups = DataProcessor.group_strategies(df)
    # O(1) stratégia lookup batch feldolgozáshoz

Resource Manager (src/analysis/engine.py):
    from analysis.engine import get_resource_manager

    manager = get_resource_manager()
    n_jobs = manager.get_n_jobs()      # CPU slider alapján
    device = manager.get_device()      # "cuda" vagy "cpu"
"""

import logging

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Model meta információk."""
    name: str
    category: str
    supports_gpu: bool = False
    supports_batch: bool = False
    gpu_threshold: int = 1000  # Minimum adat méret GPU használathoz

    # Feature mode támogatás
    supports_forward_calc: bool = True  # Támogatja-e a Forward Calc módot
    supports_rolling_window: bool = True  # Támogatja-e a Rolling Window módot

    # Speciális módok
    supports_panel_mode: bool = False  # Panel Mode (egy modell minden stratégiára)
    supports_dual_mode: bool = False  # Dual Model (activity + profit külön)


class BaseModel(ABC):
    """
    Absztrakt alap osztály minden forecasting modelhez.

    Minden leszármazott osztálynak definiálnia KELL:
    - MODEL_INFO: ModelInfo instance a model tulajdonságaival
    - PARAM_DEFAULTS: dict az alapértelmezett paraméterekkel
    - PARAM_OPTIONS: dict a választható paraméter értékekkel
    - forecast(): előrejelzés metódus

    Példa implementáció:

        class ARIMAModel(BaseModel):
            MODEL_INFO = ModelInfo(
                name="ARIMA",
                category="Statistical Models",
                supports_gpu=False,
                supports_batch=False,
            )

            PARAM_DEFAULTS = {
                "p": "1",
                "d": "1",
                "q": "1",
            }

            PARAM_OPTIONS = {
                "p": ["0", "1", "2", "3", "4", "5"],
                "d": ["0", "1", "2"],
                "q": ["0", "1", "2", "3", "4", "5"],
            }

            def forecast(self, data, steps, params):
                # Validálás
                validation = self.validate_input(data)
                if not validation.is_valid:
                    return [0.0] * steps

                # Forecast logika...
                raw_forecasts = self._run_arima(validation.processed_data, steps, params)

                # Utófeldolgozás
                return self.postprocess(raw_forecasts)
    """

    # Ezeket KELL definiálni minden leszármazottban
    MODEL_INFO: ModelInfo
    PARAM_DEFAULTS: Dict[str, str]
    PARAM_OPTIONS: Dict[str, List[str]]

    def __init__(self):
        """Inicializálás - ellenőrzi a kötelező attribútumokat."""
        self._validate_class_attributes()

    def _validate_class_attributes(self):
        """Ellenőrzi, hogy a leszármazott definiálta-e a kötelező attribútumokat."""
        if not hasattr(self, 'MODEL_INFO') or self.MODEL_INFO is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define MODEL_INFO"
            )
        if not hasattr(self, 'PARAM_DEFAULTS') or self.PARAM_DEFAULTS is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define PARAM_DEFAULTS"
            )
        if not hasattr(self, 'PARAM_OPTIONS') or self.PARAM_OPTIONS is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define PARAM_OPTIONS"
            )

    @abstractmethod
    def forecast(
        self,
        data: List[float],
        steps: int,
        params: Dict[str, Any]
    ) -> List[float]:
        """
        Előrejelzés készítése.
        
        Subclasses MUST use self.get_n_jobs(params) and self.get_device(params)
        instead of accessing global resource managers!

        Args:
            data: Bemeneti idősor adatok
            steps: Előrejelzési horizont (hány lépés előre)
            params: Paraméterek (a PARAM_DEFAULTS-ból vagy felhasználó által módosítva)

        Returns:
            Előrejelzett értékek listája (hossza = steps)
        """

    def get_n_jobs(self, params: Dict[str, Any]) -> int:
        """
        GUI által vezérelt n_jobs lekérdezése a paraméterekből.
        DECENTRALIZED RESOURCE MANAGEMENT implementation via Injection.
        """
        # 1. Check for injected secure key (highest priority)
        if '_n_jobs' in params:
            return int(params['_n_jobs'])

        # 2. Check for standard param key (if user set it manually in params)
        if 'n_jobs' in params:
            return int(params['n_jobs'])

        # 3. Fallback (should ideally not happen with correct injection)
        return 1

    def get_device(self, params: Dict[str, Any]) -> str:
        """
        GUI által vezérelt device (cpu/cuda) lekérdezése.
        DECENTRALIZED RESOURCE MANAGEMENT implementation via Injection.
        """
        # 1. Check for injected secure key
        if '_device' in params:
            return str(params['_device'])

        # 2. Legacy check (use_gpu param)
        if params.get('use_gpu', False):
            return 'cuda'

        return 'cpu'

    # =========================================================================
    # UTILITY WRAPPER METÓDUSOK - A models/utils modulok egyszerű elérése
    # =========================================================================

    def validate_input(
        self,
        data: Union[List[float], "np.ndarray", "pd.Series"],
        min_points: Optional[int] = None
    ) -> "ValidationResult":
        """
        Input adat validálása.

        Wrapper a models.utils.validation.validate_input_data függvényhez.

        Args:
            data: Bemeneti idősor
            min_points: Minimum adatpontok (None = kategória alapján)

        Returns:
            ValidationResult objektum

        Example:
            def forecast(self, data, steps, params):
                validation = self.validate_input(data)
                if not validation.is_valid:
                    logger.warning("Invalid input: %s", validation.error_message)
                    return [0.0] * steps
                clean_data = validation.processed_data
                # ...
        """
        from models.utils.validation import validate_input_data, RECOMMENDED_MIN_POINTS  # pylint: disable=import-outside-toplevel

        # Kategória alapú minimum pontok
        if min_points is None:
            category = self.MODEL_INFO.category.lower()
            if "deep learning" in category:
                min_points = RECOMMENDED_MIN_POINTS.get("deep_learning", 100)
            elif "machine learning" in category:
                min_points = RECOMMENDED_MIN_POINTS.get("ml", 50)
            elif "spectral" in category or "frequency" in category:
                min_points = RECOMMENDED_MIN_POINTS.get("spectral", 32)
            elif "smoothing" in category:
                min_points = RECOMMENDED_MIN_POINTS.get("smoothing", 12)
            else:
                min_points = RECOMMENDED_MIN_POINTS.get("statistical", 24)

        return validate_input_data(data, min_points=min_points)

    def postprocess(
        self,
        forecasts: Union[List[float], "np.ndarray"],
        allow_negative: bool = True,
        cap_outliers: bool = True,
        iqr_multiplier: float = 3.0
    ) -> List[float]:
        """
        Forecast utófeldolgozás.

        Wrapper a models.utils.postprocessing.postprocess_forecasts függvényhez.
        A régi strategy_analyzer.py 464-490. sorának funkcionalitása.

        Args:
            forecasts: Nyers forecast értékek
            allow_negative: Negatív értékek engedélyezése (default: True)
            cap_outliers: Outlier capping (default: True)
            iqr_multiplier: IQR szorzó (default: 3.0)

        Returns:
            Tisztított forecast lista

        Example:
            def forecast(self, data, steps, params):
                raw_forecasts = self._run_model(data, steps)
                return self.postprocess(raw_forecasts)
        """
        from models.utils.postprocessing import postprocess_forecasts, PostprocessingConfig  # pylint: disable=import-outside-toplevel

        config = PostprocessingConfig(
            handle_nan=True,
            handle_inf=True,
            cap_outliers=cap_outliers,
            iqr_multiplier=iqr_multiplier,
            allow_negative=allow_negative,
        )

        return postprocess_forecasts(forecasts, config)

    def calculate_horizons(
        self,
        forecasts: Union[List[float], "np.ndarray"],
        allow_negative: bool = True
    ) -> "HorizonResult":
        """
        Horizont aggregátumok számítása.

        Wrapper a models.utils.aggregation.calculate_horizons függvényhez.
        A régi strategy_analyzer.py 492-497. sorának funkcionalitása.

        Args:
            forecasts: Forecast értékek
            allow_negative: Negatív összegek engedélyezése

        Returns:
            HorizonResult objektum (h1, h4, h13, h26, h52)

        Example:
            forecasts = self.forecast(data, steps=52, params=params)
            horizons = self.calculate_horizons(forecasts)
            print(f"1 month forecast: {horizons.h4}")
        """
        from models.utils.aggregation import calculate_horizons  # pylint: disable=import-outside-toplevel
        return calculate_horizons(forecasts, allow_negative=allow_negative)

    def create_timer(
        self,
        strategy_id: Optional[str] = None,
        data_points: int = 0
    ) -> "ForecastTimer":
        """
        Teljesítmény timer létrehozása.

        Wrapper a models.utils.monitoring.ForecastTimer osztályhoz.
        A régi strategy_analyzer.py 512-517. sorának funkcionalitása.

        Args:
            strategy_id: Stratégia azonosító
            data_points: Adatpontok száma

        Returns:
            ForecastTimer context manager

        Example:
            def forecast(self, data, steps, params):
                with self.create_timer("STR_001", len(data)) as timer:
                    result = self._run_model(data, steps)
                return result
        """
        from models.utils.monitoring import ForecastTimer  # pylint: disable=import-outside-toplevel
        return ForecastTimer(
            model_name=self.MODEL_INFO.name,
            strategy_id=strategy_id,
            data_points=data_points
        )

    # =========================================================================
    # FORECAST WITH FULL PIPELINE
    # =========================================================================

    def forecast_with_pipeline(
        self,
        data: Union[List[float], "np.ndarray", "pd.Series"],
        steps: int,
        params: Dict[str, Any],
        strategy_id: Optional[str] = None,
        postprocess: bool = True,
        allow_negative: bool = True
    ) -> Dict[str, Any]:
        """
        Teljes forecast pipeline: validálás → forecast → utófeldolgozás → aggregálás.

        Ez a metódus a régi strategy_analyzer.py teljes funkcionalitását biztosítja
        egyetlen hívással.

        Args:
            data: Bemeneti idősor
            steps: Előrejelzési horizont
            params: Modell paraméterek
            strategy_id: Stratégia azonosító (opcionális)
            postprocess: Utófeldolgozás végrehajtása (default: True)
            allow_negative: Negatív értékek engedélyezése (default: True)

        Returns:
            Dictionary a régi strategy_analyzer.py kimenetével kompatibilis formátumban:
            {
                "No.": strategy_id,
                "Forecast_1W": h1,
                "Forecast_1M": h4,
                "Forecast_3M": h13,
                "Forecast_6M": h26,
                "Forecast_12M": h52,
                "Method": model_name,
                "Forecasts": [list of forecasts],
                "Success": True/False,
                "Error": None or error message
            }

        Example:
            model = ARIMAModel()
            result = model.forecast_with_pipeline(
                data=profit_series,
                steps=52,
                params={"p": 1, "d": 1, "q": 1},
                strategy_id="STR_001"
            )
            print(f"1 month forecast: {result['Forecast_1M']}")
        """
        import numpy as np  # pylint: disable=import-outside-toplevel

        # 1. Validálás
        validation = self.validate_input(data)
        if not validation.is_valid:
            logger.warning(
                "Strategy %s: skipped (%s)",
                strategy_id or "unknown",
                validation.error_message
            )
            return {
                "No.": strategy_id,
                "Forecast_1W": 0.0,
                "Forecast_1M": 0.0,
                "Forecast_3M": 0.0,
                "Forecast_6M": 0.0,
                "Forecast_12M": 0.0,
                "Method": self.MODEL_INFO.name,
                "Forecasts": [0.0] * steps,
                "Success": False,
                "Error": validation.error_message,
            }

        # 2. Forecast időméréssel
        try:
            with self.create_timer(strategy_id, validation.processed_length):
                raw_forecasts = self.forecast(
                    validation.processed_data,
                    steps,
                    params
                )
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "Strategy %s: forecast error - %s",
                strategy_id or "unknown",
                str(e)
            )
            return {
                "No.": strategy_id,
                "Forecast_1W": 0.0,
                "Forecast_1M": 0.0,
                "Forecast_3M": 0.0,
                "Forecast_6M": 0.0,
                "Forecast_12M": 0.0,
                "Method": self.MODEL_INFO.name,
                "Forecasts": [np.nan] * steps,
                "Success": False,
                "Error": str(e),
            }

        # 3. Utófeldolgozás
        if postprocess:
            forecasts = self.postprocess(raw_forecasts, allow_negative=allow_negative)
        else:
            forecasts = list(raw_forecasts)

        # 4. Horizont aggregálás
        horizons = self.calculate_horizons(forecasts, allow_negative=allow_negative)

        return {
            "No.": strategy_id,
            "Forecast_1W": horizons.h1,
            "Forecast_1M": horizons.h4,
            "Forecast_3M": horizons.h13,
            "Forecast_6M": horizons.h26,
            "Forecast_12M": horizons.h52,
            "Method": self.MODEL_INFO.name,
            "Forecasts": forecasts,
            "Success": True,
            "Error": None,
        }

    # =========================================================================
    # BATCH FORECAST
    # =========================================================================

    def forecast_batch(
        self,
        all_data: Dict[str, List[float]],
        steps: int,
        params: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """
        Batch előrejelzés több stratégiára egyszerre.

        Alapértelmezés: joblib.Parallel párhuzamos feldolgozás.
        GPU-támogatott modellek felülírhatják hatékonyabb implementációval.

        Args:
            all_data: Dict ahol kulcs = stratégia név, érték = idősor
            steps: Előrejelzési horizont
            params: Paraméterek

        Returns:
            Dict ahol kulcs = stratégia név, érték = előrejelzés

        Példa egyedi batch implementációra:
            def forecast_batch(self, all_data, steps, params):
                # GPU batch ha sok stratégia
                if len(all_data) > 100 and self._use_gpu(params):
                    return self._gpu_batch_forecast(all_data, steps, params)
                # Egyébként default
                return super().forecast_batch(all_data, steps, params)
        """
        if not self.MODEL_INFO.supports_batch:
            raise NotImplementedError(
                f"{self.MODEL_INFO.name} does not support batch mode"
            )

        # Kevés stratégia esetén szekvenciális (overhead elkerülése)
        if len(all_data) <= 3:
            results = {}
            for strategy_name, data in all_data.items():
                results[strategy_name] = self.forecast(data, steps, params)
            return results

        # Párhuzamos feldolgozás joblib-bal
        try:
            from joblib import Parallel, delayed  # pylint: disable=import-outside-toplevel

            # Use injected n_jobs from params
            max_jobs = self.get_n_jobs(params)

            n_jobs = min(max_jobs, len(all_data), 8)

            def _process_single(name: str, data: List[float]):
                return name, self.forecast(data, steps, params)

            results_list = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_process_single)(name, data)
                for name, data in all_data.items()
            )

            return dict(results_list)

        except ImportError:
            # Fallback: szekvenciális ha nincs joblib
            logger.warning("joblib not available, using sequential processing")
            results = {}
            for strategy_name, data in all_data.items():
                results[strategy_name] = self.forecast(data, steps, params)
            return results

    def forecast_batch_with_pipeline(
        self,
        all_data: Dict[str, List[float]],
        steps: int,
        params: Dict[str, Any],
        postprocess: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batch forecast teljes pipeline-nal.

        Args:
            all_data: Dict ahol kulcs = stratégia név, érték = idősor
            steps: Előrejelzési horizont
            params: Paraméterek
            postprocess: Utófeldolgozás

        Returns:
            Dict ahol kulcs = stratégia név, érték = forecast_with_pipeline eredmény
        """
        from models.utils.monitoring import BatchPerformanceMonitor  # pylint: disable=import-outside-toplevel

        results = {}

        with BatchPerformanceMonitor(self.MODEL_INFO.name) as monitor:
            for strategy_id, data in all_data.items():
                with monitor.track(strategy_id, len(data)):
                    results[strategy_id] = self.forecast_with_pipeline(
                        data=data,
                        steps=steps,
                        params=params,
                        strategy_id=strategy_id,
                        postprocess=postprocess
                    )

        return results

    # =========================================================================
    # DEVICE ÉS GPU KEZELÉS
    # =========================================================================



    def get_params_with_defaults(
        self,
        user_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Egyesíti a felhasználói paramétereket az alapértékekkel.

        Args:
            user_params: Felhasználó által megadott paraméterek (opcionális)

        Returns:
            Teljes paraméter dict (defaults + user overrides)
        """
        params = dict(self.PARAM_DEFAULTS)
        if user_params:
            params.update(user_params)
        return params

    @classmethod
    def get_info(cls) -> ModelInfo:
        """Visszaadja a model info-t (GUI-nak)."""
        return cls.MODEL_INFO

    @classmethod
    def get_defaults(cls) -> Dict[str, str]:
        """Visszaadja az alapértelmezett paramétereket (GUI-nak)."""
        return cls.PARAM_DEFAULTS.copy()

    @classmethod
    def get_options(cls) -> Dict[str, List[str]]:
        """Visszaadja a paraméter opciókat (GUI-nak)."""
        return cls.PARAM_OPTIONS.copy()

    def __repr__(self) -> str:
        return f"<{self.MODEL_INFO.name} ({self.MODEL_INFO.category})>"

    # =========================================================================
    # HELPER METÓDUSOK - GPU és Memória kezelés
    # =========================================================================

    def cleanup_after_batch(self) -> None:
        """
        Cleanup hívás batch feldolgozás után.

        Felszabadítja a CUDA memóriát és futtat GC-t.
        Hívd meg hosszú batch műveletek után!

        Példa:
            def forecast_batch(self, all_data, steps, params):
                try:
                    results = self._gpu_batch_process(all_data, steps, params)
                    return results
                finally:
                    self.cleanup_after_batch()
        """
        import gc  # pylint: disable=import-outside-toplevel
        gc.collect()

        try:
            from analysis.process_utils import cleanup_cuda_context  # pylint: disable=import-outside-toplevel
            cleanup_cuda_context()
        except ImportError:
            # process_utils nem elérhető - próbáljuk közvetlenül
            try:
                import torch  # pylint: disable=import-outside-toplevel
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass



    def should_use_gpu(self, data_size: int, params: Dict[str, Any]) -> bool:
        """
        Eldönti, hogy GPU-t használjon-e.

        Args:
            data_size: Adat méret
            params: Paraméterek (use_gpu kulcs)

        Returns:
            bool: True ha GPU ajánlott
        """
        if not self.MODEL_INFO.supports_gpu:
            return False

        use_gpu = params.get("use_gpu", True)
        if isinstance(use_gpu, str):
            use_gpu = use_gpu.lower() == "true"

        if not use_gpu:
            return False

        if data_size < self.MODEL_INFO.gpu_threshold:
            return False

        try:
            import torch  # pylint: disable=import-outside-toplevel
            return torch.cuda.is_available()
        except ImportError:
            return False

    # =========================================================================
    # DUAL MODE TÁMOGATÁS (Opcionális - csak ha supports_dual_mode = True)
    # =========================================================================

    def create_dual_regressor(
        self,
        params: Dict[str, Any],
        n_jobs: int = 1
    ) -> Any:
        """
        Regresszor létrehozása dual mode-hoz.

        Csak azok a modellek implementálják, amelyek támogatják a dual mode-ot.
        A visszaadott objektumnak rendelkeznie kell fit() és predict() metódusokkal.

        Args:
            params: Modell paraméterek
            n_jobs: Párhuzamos szálak száma

        Returns:
            sklearn-kompatibilis regresszor objektum

        Raises:
            NotImplementedError: Ha a modell nem támogatja a dual mode-ot
        """
        raise NotImplementedError(
            f"{self.MODEL_INFO.name} does not support dual mode. "
            f"Set supports_dual_mode=True and implement create_dual_regressor()."
        )


# Type hints a lazy importokhoz
if TYPE_CHECKING:
    from models.utils.validation import ValidationResult
    from models.utils.aggregation import HorizonResult
    from models.utils.monitoring import ForecastTimer
