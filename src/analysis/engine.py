"""
Analysis Engine - MBO Trading Strategy Analyzer
Forecasting motor és központi erőforrás-kezelő.

ResourceManager: Központi GUI vezérlő a CPU/GPU beállításokhoz.
- A modellek SAJÁT MAGUK döntik el, hogyan használják ezeket az értékeket
- Ez csak a GUI globális beállításait tárolja és közvetíti

AnalysisEngine: Forecasting motor az elemzések futtatásához.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from models import get_model_class, get_param_defaults

logger = logging.getLogger(__name__)


# =============================================================================
# RESOURCE MANAGER - Központi GUI erőforrás-kezelő
# =============================================================================


class ResourceManager:
    """
    Singleton osztály a GUI központi erőforrás-kezeléséhez.

    FONTOS: Ez NEM modellenként kezeli az erőforrásokat!
    A modellek saját maguk döntik el, hogyan használják ezeket az értékeket.
    Ez csak a GUI globális beállításait tárolja.

    Használat:
        from analysis.engine import get_resource_manager

        manager = get_resource_manager()
        manager.set_cpu_percentage(75)
        n_jobs = manager.get_n_jobs()
    """

    _instance: Optional["ResourceManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ResourceManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
        self._state_lock = threading.Lock()

        # CPU beállítások
        self._cpu_percentage: int = 85
        self._physical_cores: int = self._detect_physical_cores()
        self._logical_cores: int = self._detect_logical_cores()

        # GPU beállítások
        self._gpu_enabled: bool = False
        self._gpu_available: bool = self._detect_gpu()
        self._gpu_name: str = self._get_gpu_name()

        # Callback rendszer (GUI frissítésekhez)
        self._cpu_callbacks: List[Callable[[int], None]] = []
        self._gpu_callbacks: List[Callable[[bool], None]] = []

        # Környezeti változók alkalmazása
        self._apply_env_vars()

        logger.info(
            "ResourceManager initialized: %d physical cores, %d logical cores, GPU: %s",
            self._physical_cores,
            self._logical_cores,
            self._gpu_name if self._gpu_available else "Not available"
        )

    # =========================================================================
    # CPU DETEKCIÓ
    # =========================================================================

    def _detect_physical_cores(self) -> int:
        """Fizikai CPU magok számának detektálása."""
        try:
            import psutil
            return psutil.cpu_count(logical=False) or 4
        except ImportError:
            # Fallback: logikai magok fele
            logical = os.cpu_count() or 4
            return max(1, logical // 2)

    def _detect_logical_cores(self) -> int:
        """Logikai CPU magok számának detektálása."""
        try:
            import psutil
            return psutil.cpu_count(logical=True) or 8
        except ImportError:
            return os.cpu_count() or 4

    # =========================================================================
    # GPU DETEKCIÓ
    # =========================================================================

    def _detect_gpu(self) -> bool:
        """GPU elérhetőség ellenőrzése."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _get_gpu_name(self) -> str:
        """GPU nevének lekérdezése."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except (ImportError, RuntimeError):
            pass
        return ""

    # =========================================================================
    # CPU VEZÉRLÉS - Publikus API
    # =========================================================================

    @property
    def cpu_percentage(self) -> int:
        """Aktuális CPU százalék (10-100)."""
        return self._cpu_percentage

    @property
    def physical_cores(self) -> int:
        """Fizikai CPU magok száma."""
        return self._physical_cores

    @property
    def logical_cores(self) -> int:
        """Logikai CPU magok száma."""
        return self._logical_cores

    def set_cpu_percentage(self, percentage: int) -> None:
        """
        CPU használat százalékának beállítása.

        Args:
            percentage: CPU százalék (10-100)
        """
        with self._state_lock:
            new_value = max(10, min(100, int(percentage)))

            if new_value == self._cpu_percentage:
                return

            old_value = self._cpu_percentage
            self._cpu_percentage = new_value

            logger.debug("CPU percentage: %d%% -> %d%%", old_value, new_value)

            # Környezeti változók frissítése
            self._apply_env_vars()

            # Callback-ek értesítése
            for callback in self._cpu_callbacks:
                try:
                    callback(new_value)
                except Exception as e:
                    logger.warning("CPU callback error: %s", e)

    def get_n_jobs(self) -> int:
        """
        Ajánlott párhuzamos job-ok száma (fizikai magok alapján).

        Returns:
            int: Job-ok száma (minimum 1)
        """
        allowed = int(self._physical_cores * self._cpu_percentage / 100)
        return max(1, allowed)

    def get_n_jobs_logical(self) -> int:
        """
        Ajánlott job-ok száma logikai magok alapján.

        I/O-bound vagy könnyű CPU feladatokhoz.

        Returns:
            int: Job-ok száma logikai magok alapján
        """
        allowed = int(self._logical_cores * self._cpu_percentage / 100)
        return max(1, allowed)

    def get_reserved_cores(self) -> int:
        """
        GUI/rendszer számára fenntartott magok száma.

        Returns:
            int: Fenntartott magok száma
        """
        used = self.get_n_jobs()
        return max(0, self._physical_cores - used)

    # =========================================================================
    # GPU VEZÉRLÉS - Publikus API
    # =========================================================================

    @property
    def gpu_enabled(self) -> bool:
        """GPU használat engedélyezve van-e."""
        return self._gpu_enabled and self._gpu_available

    @property
    def gpu_available(self) -> bool:
        """Van-e elérhető GPU."""
        return self._gpu_available

    @property
    def gpu_name(self) -> str:
        """GPU neve (üres ha nincs)."""
        return self._gpu_name

    def set_gpu_enabled(self, enabled: bool) -> None:
        """
        GPU használat engedélyezése/tiltása.

        Args:
            enabled: True = GPU engedélyezve
        """
        with self._state_lock:
            if not self._gpu_available and enabled:
                logger.warning("GPU not available, cannot enable")
                return

            if enabled == self._gpu_enabled:
                return

            self._gpu_enabled = enabled
            logger.debug("GPU: %s", "enabled" if enabled else "disabled")

            # Callback-ek értesítése
            for callback in self._gpu_callbacks:
                try:
                    callback(enabled)
                except Exception as e:
                    logger.warning("GPU callback error: %s", e)

    def get_device(self) -> str:
        """
        Aktuális device string (modellek számára).

        Returns:
            "cuda" ha GPU engedélyezve és elérhető, egyébként "cpu"
        """
        if self._gpu_enabled and self._gpu_available:
            return "cuda"
        return "cpu"

    # =========================================================================
    # CALLBACK RENDSZER
    # =========================================================================

    def add_cpu_callback(self, callback: Callable[[int], None]) -> None:
        """CPU változás callback regisztrálása."""
        if callback not in self._cpu_callbacks:
            self._cpu_callbacks.append(callback)

    def remove_cpu_callback(self, callback: Callable[[int], None]) -> None:
        """CPU változás callback eltávolítása."""
        if callback in self._cpu_callbacks:
            self._cpu_callbacks.remove(callback)

    def add_gpu_callback(self, callback: Callable[[bool], None]) -> None:
        """GPU változás callback regisztrálása."""
        if callback not in self._gpu_callbacks:
            self._gpu_callbacks.append(callback)

    def remove_gpu_callback(self, callback: Callable[[bool], None]) -> None:
        """GPU változás callback eltávolítása."""
        if callback in self._gpu_callbacks:
            self._gpu_callbacks.remove(callback)

    # =========================================================================
    # KÖRNYEZETI VÁLTOZÓK
    # =========================================================================

    def _apply_env_vars(self) -> None:
        """Környezeti változók beállítása (BLAS/OpenMP thread control)."""
        n_jobs = str(self.get_n_jobs())

        # Standard thread control variables
        env_vars = {
            "OMP_NUM_THREADS": n_jobs,
            "MKL_NUM_THREADS": n_jobs,
            "OPENBLAS_NUM_THREADS": n_jobs,
            "VECLIB_MAXIMUM_THREADS": n_jobs,
            "NUMEXPR_NUM_THREADS": n_jobs,
        }

        for key, value in env_vars.items():
            os.environ[key] = value

    def get_env_vars(self) -> Dict[str, str]:
        """
        Környezeti változók lekérdezése (gyermek folyamatokhoz).

        Returns:
            Dict a beállítandó környezeti változókkal
        """
        n_jobs = str(self.get_n_jobs())
        return {
            "OMP_NUM_THREADS": n_jobs,
            "MKL_NUM_THREADS": n_jobs,
            "OPENBLAS_NUM_THREADS": n_jobs,
            "VECLIB_MAXIMUM_THREADS": n_jobs,
            "NUMEXPR_NUM_THREADS": n_jobs,
            "MBO_CPU_PERCENTAGE": str(self._cpu_percentage),
            "MBO_CPU_CORES": n_jobs,
            "MBO_GPU_ENABLED": "1" if self._gpu_enabled else "0",
        }

    # =========================================================================
    # STÁTUSZ
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Aktuális erőforrás-státusz lekérdezése.

        Returns:
            Dict az összes releváns információval
        """
        return {
            "cpu_percentage": self._cpu_percentage,
            "n_jobs": self.get_n_jobs(),
            "n_jobs_logical": self.get_n_jobs_logical(),
            "physical_cores": self._physical_cores,
            "logical_cores": self._logical_cores,
            "reserved_cores": self.get_reserved_cores(),
            "gpu_enabled": self._gpu_enabled,
            "gpu_available": self._gpu_available,
            "gpu_name": self._gpu_name,
            "device": self.get_device(),
        }

    def __repr__(self) -> str:
        return (
            f"ResourceManager(cpu={self._cpu_percentage}%, "
            f"n_jobs={self.get_n_jobs()}, "
            f"gpu={'on' if self._gpu_enabled else 'off'})"
        )


# =============================================================================
# SINGLETON HOZZÁFÉRÉS
# =============================================================================

_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """
    Globális ResourceManager singleton lekérése.

    Használat:
        from analysis.engine import get_resource_manager
        manager = get_resource_manager()
    """
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


def init_resource_manager(cpu_percentage: int = 85, gpu_enabled: bool = False) -> ResourceManager:
    """
    ResourceManager inicializálása megadott értékekkel.

    Hívd meg az alkalmazás indításakor.

    Args:
        cpu_percentage: Kezdeti CPU százalék
        gpu_enabled: GPU engedélyezve legyen-e

    Returns:
        ResourceManager instance
    """
    manager = get_resource_manager()
    manager.set_cpu_percentage(cpu_percentage)
    manager.set_gpu_enabled(gpu_enabled)
    return manager


# =============================================================================
# ANALYSIS ENGINE - Forecasting motor
# =============================================================================


@dataclass
class AnalysisResult:
    """Egyetlen stratégia elemzés eredménye."""
    strategy_id: str
    forecasts: List[float]
    elapsed_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class AnalysisContext:
    """Elemzés kontextus - összes paraméter egyben."""
    model_name: str
    params: Dict[str, Any]
    forecast_horizon: int = 12
    use_gpu: bool = False
    use_batch: bool = False


@dataclass
class AnalysisProgress:
    """Elemzés haladási állapot."""
    total_strategies: int = 0
    completed_strategies: int = 0
    current_strategy: str = ""
    elapsed_seconds: float = 0.0
    is_running: bool = False
    is_paused: bool = False
    is_cancelled: bool = False
    results: Dict[str, AnalysisResult] = field(default_factory=dict)


class AnalysisEngine:
    """
    Forecasting motor.

    Használat:
        engine = AnalysisEngine()
        engine.set_progress_callback(my_callback)
        results = engine.run(data, context)
    """

    def __init__(self):
        self._progress = AnalysisProgress()
        self._progress_callback: Optional[Callable[[AnalysisProgress], None]] = None
        self._lock = threading.Lock()
        self._resource_manager = get_resource_manager()

    def set_progress_callback(self, callback: Callable[[AnalysisProgress], None]):
        """Progress callback beállítása."""
        self._progress_callback = callback

    def _notify_progress(self):
        """Callback értesítése a haladásról."""
        if self._progress_callback:
            try:
                self._progress_callback(self._progress)
            except Exception as e:
                logger.warning("Progress callback error: %s", e)

    def extract_strategies(
        self,
        data: pd.DataFrame,
        strategy_col: str = "No.",
        value_col: str = "Profit"
    ) -> Dict[str, List[float]]:
        """
        Stratégiák idősorainak kinyerése a DataFrame-ből.

        Args:
            data: Feldolgozott DataFrame
            strategy_col: Stratégia azonosító oszlop
            value_col: Érték oszlop (tipikusan Profit)

        Returns:
            Dict[strategy_id, values]
        """
        strategies = {}

        if data is None or data.empty:
            return strategies

        # Ha nincs strategy column, használjuk az egész Profit oszlopot
        if strategy_col not in data.columns:
            if value_col in data.columns:
                values = data[value_col].dropna().values.tolist()
                if len(values) >= 10:
                    strategies["all"] = values
            else:
                # Fallback: első numerikus oszlop
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    values = data[numeric_cols[0]].dropna().values.tolist()
                    if len(values) >= 10:
                        strategies["default"] = values
            return strategies

        # Stratégiánként csoportosítunk
        for strategy_id in data[strategy_col].unique():
            strategy_df = data[data[strategy_col] == strategy_id]

            if value_col in strategy_df.columns:
                values = strategy_df[value_col].dropna().values.tolist()
                if len(values) >= 10:  # Minimum adatmennyiség
                    strategies[str(strategy_id)] = values

        return strategies

    def run(
        self,
        data: pd.DataFrame,
        context: AnalysisContext,
        strategy_col: str = "No.",
        value_col: str = "Profit"
    ) -> Dict[str, AnalysisResult]:
        """
        Elemzés futtatása az összes stratégiára.

        Args:
            data: Feldolgozott DataFrame
            context: Elemzés kontextus (modell, paraméterek, stb.)
            strategy_col: Stratégia azonosító oszlop
            value_col: Érték oszlop

        Returns:
            Dict[strategy_id, AnalysisResult]
        """
        # Stratégiák kinyerése
        strategies = self.extract_strategies(data, strategy_col, value_col)

        if not strategies:
            logger.warning("No strategies found in data")
            return {}

        # Progress inicializálás
        with self._lock:
            self._progress = AnalysisProgress(
                total_strategies=len(strategies),
                completed_strategies=0,
                is_running=True,
                is_paused=False,
                is_cancelled=False
            )

        self._notify_progress()

        # Modell betöltése
        model_class = get_model_class(context.model_name)
        if model_class is None:
            logger.error("Model not found: %s", context.model_name)
            return {}

        try:
            model = model_class()
        except Exception as e:
            logger.error("Error creating model %s: %s", context.model_name, e)
            return {}

        # Paraméterek összefűzése
        params = get_param_defaults(context.model_name)
        params.update(context.params)

        start_time = time.time()
        results: Dict[str, AnalysisResult] = {}

        # Batch mode ellenőrzés
        if context.use_batch and model.MODEL_INFO.supports_batch:
            results = self._run_batch(model, strategies, context, params)
        else:
            results = self._run_sequential(model, strategies, context, params)

        # Befejezés
        with self._lock:
            self._progress.is_running = False
            self._progress.elapsed_seconds = time.time() - start_time
            self._progress.results = results

        self._notify_progress()

        return results

    def _run_sequential(
        self,
        model,
        strategies: Dict[str, List[float]],
        context: AnalysisContext,
        params: Dict[str, Any]
    ) -> Dict[str, AnalysisResult]:
        """Szekvenciális feldolgozás."""
        results = {}

        for i, (strategy_id, values) in enumerate(strategies.items()):
            # Cancel/pause flag ellenőrzés
            with self._lock:
                if self._progress.is_cancelled:
                    break

                while self._progress.is_paused and not self._progress.is_cancelled:
                    time.sleep(0.1)  # Várakozás pause alatt

                self._progress.current_strategy = strategy_id

            self._notify_progress()

            # Forecast futtatása
            start = time.perf_counter()
            try:
                forecasts = model.forecast(values, context.forecast_horizon, params)
                elapsed_ms = (time.perf_counter() - start) * 1000

                results[strategy_id] = AnalysisResult(
                    strategy_id=strategy_id,
                    forecasts=forecasts,
                    elapsed_ms=elapsed_ms,
                    success=True
                )
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                results[strategy_id] = AnalysisResult(
                    strategy_id=strategy_id,
                    forecasts=[np.nan] * context.forecast_horizon,
                    elapsed_ms=elapsed_ms,
                    success=False,
                    error=str(e)
                )
                logger.warning("Forecast error for strategy %s: %s", strategy_id, e)

            # Progress frissítés
            with self._lock:
                self._progress.completed_strategies = i + 1
                self._progress.results[strategy_id] = results[strategy_id]

            self._notify_progress()

        return results

    def _run_batch(
        self,
        model,
        strategies: Dict[str, List[float]],
        context: AnalysisContext,
        params: Dict[str, Any]
    ) -> Dict[str, AnalysisResult]:
        """Batch feldolgozás."""
        results = {}

        with self._lock:
            self._progress.current_strategy = "BATCH MODE"

        self._notify_progress()

        start = time.perf_counter()
        try:
            batch_results = model.forecast_batch(strategies, context.forecast_horizon, params)
            elapsed_ms = (time.perf_counter() - start) * 1000

            per_strategy_ms = elapsed_ms / len(strategies) if strategies else 0

            for strategy_id, forecasts in batch_results.items():
                results[strategy_id] = AnalysisResult(
                    strategy_id=strategy_id,
                    forecasts=forecasts,
                    elapsed_ms=per_strategy_ms,
                    success=True
                )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error("Batch forecast error: %s", e)

            # Minden stratégia sikertelen
            for strategy_id in strategies:
                results[strategy_id] = AnalysisResult(
                    strategy_id=strategy_id,
                    forecasts=[np.nan] * context.forecast_horizon,
                    elapsed_ms=elapsed_ms / len(strategies),
                    success=False,
                    error=str(e)
                )

        # Progress frissítés
        with self._lock:
            self._progress.completed_strategies = len(strategies)
            self._progress.results = results

        self._notify_progress()

        return results

    def pause(self):
        """Elemzés szüneteltetése."""
        with self._lock:
            self._progress.is_paused = True

    def resume(self):
        """Elemzés folytatása."""
        with self._lock:
            self._progress.is_paused = False

    def cancel(self):
        """Elemzés megszakítása."""
        with self._lock:
            self._progress.is_cancelled = True
            self._progress.is_paused = False

    @property
    def progress(self) -> AnalysisProgress:
        """Jelenlegi progress."""
        with self._lock:
            return self._progress


# =============================================================================
# SEGÉD FÜGGVÉNYEK
# =============================================================================


def analyze_strategy(
    data: List[float],
    model_name: str,
    steps: int,
    params: Optional[Dict[str, Any]] = None
) -> List[float]:
    """
    Egyetlen stratégia elemzése (segéd függvény).

    Args:
        data: Idősor adatok
        model_name: Modell neve
        steps: Előrejelzési horizont
        params: Opcionális paraméterek

    Returns:
        Forecast lista
    """
    model_class = get_model_class(model_name)
    if model_class is None:
        return [np.nan] * steps

    try:
        model = model_class()
        full_params = get_param_defaults(model_name)
        if params:
            full_params.update(params)

        return model.forecast(data, steps, full_params)
    except Exception as e:
        logger.error("analyze_strategy error: %s", e)
        return [np.nan] * steps
