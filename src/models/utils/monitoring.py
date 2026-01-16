"""
MBO Trading Strategy Analyzer - Performance Monitoring
Teljesítmény monitoring és logging forecasting modellekhez.

A régi strategy_analyzer.py 512-517. soraiból származó logika,
kiterjesztve és általánosítva.

Használat:
    from models.utils.monitoring import ForecastTimer, PerformanceStats

    # Context manager használat
    with ForecastTimer("LSTM", strategy_id="STR_001") as timer:
        forecasts = model.forecast(data, steps=52)

    # Decorator használat
    @ForecastTimer.as_decorator("ARIMA")
    def run_forecast(data, steps):
        return model.forecast(data, steps)
"""

import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# KONSTANSOK
# =============================================================================

# Lassú forecast küszöb másodpercben (a régi strategy_analyzer.py 514. sorából)
SLOW_FORECAST_THRESHOLD = 5.0

# Nagyon lassú forecast küszöb (warning szintű log)
VERY_SLOW_THRESHOLD = 30.0

# Kritikusan lassú (error szintű log)
CRITICAL_SLOW_THRESHOLD = 120.0


@dataclass
class PerformanceStats:
    """
    Teljesítmény statisztikák egy forecast futtatásról.

    Attributes:
        model_name: Modell neve
        strategy_id: Stratégia azonosító
        elapsed_time: Futási idő másodpercben
        data_points: Bemeneti adatpontok száma
        forecast_steps: Előrejelzési lépések száma
        success: Sikeres volt-e a futtatás
        error_message: Hibaüzenet (ha sikertelen)
        memory_mb: Memóriahasználat MB-ban (opcionális)
        gpu_used: GPU volt-e használva
    """
    model_name: str
    strategy_id: Optional[str] = None
    elapsed_time: float = 0.0
    data_points: int = 0
    forecast_steps: int = 0
    success: bool = True
    error_message: Optional[str] = None
    memory_mb: Optional[float] = None
    gpu_used: bool = False
    timestamp: float = field(default_factory=time.time)

    @property
    def is_slow(self) -> bool:
        """Lassú volt-e a futtatás."""
        return self.elapsed_time > SLOW_FORECAST_THRESHOLD

    @property
    def is_very_slow(self) -> bool:
        """Nagyon lassú volt-e a futtatás."""
        return self.elapsed_time > VERY_SLOW_THRESHOLD

    @property
    def is_critical(self) -> bool:
        """Kritikusan lassú volt-e a futtatás."""
        return self.elapsed_time > CRITICAL_SLOW_THRESHOLD

    @property
    def points_per_second(self) -> float:
        """Feldolgozott pontok másodpercenként."""
        if self.elapsed_time > 0:
            return self.data_points / self.elapsed_time
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary reprezentáció."""
        return {
            "model_name": self.model_name,
            "strategy_id": self.strategy_id,
            "elapsed_time": self.elapsed_time,
            "data_points": self.data_points,
            "forecast_steps": self.forecast_steps,
            "success": self.success,
            "error_message": self.error_message,
            "memory_mb": self.memory_mb,
            "gpu_used": self.gpu_used,
            "is_slow": self.is_slow,
            "points_per_second": self.points_per_second,
        }


class ForecastTimer:
    """
    Forecast futási idő mérő context manager és decorator.

    Használat context manager-ként:
        with ForecastTimer("LSTM", strategy_id="STR_001") as timer:
            forecasts = model.forecast(data, steps=52)
        print(f"Elapsed: {timer.elapsed:.2f}s")

    Használat decorator-ként:
        @ForecastTimer.as_decorator("ARIMA")
        def run_arima(data, steps):
            return model.forecast(data, steps)

    Note:
        A régi strategy_analyzer.py 512-517. soraiból:
        _elapsed = _time.time() - _start_time
        if _elapsed > 5.0:
            logger.debug("Strategy %s: completed in %.2fs", strategy_id, _elapsed)
    """

    def __init__(
        self,
        model_name: str,
        strategy_id: Optional[str] = None,
        data_points: int = 0,
        forecast_steps: int = 0,
        log_slow: bool = True,
        slow_threshold: float = SLOW_FORECAST_THRESHOLD
    ):
        """
        Inicializálás.

        Args:
            model_name: Modell neve
            strategy_id: Stratégia azonosító
            data_points: Bemeneti adatpontok száma
            forecast_steps: Előrejelzési lépések
            log_slow: Lassú futtatások logolása
            slow_threshold: Lassú küszöb másodpercben
        """
        self.model_name = model_name
        self.strategy_id = strategy_id
        self.data_points = data_points
        self.forecast_steps = forecast_steps
        self.log_slow = log_slow
        self.slow_threshold = slow_threshold

        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._success = True
        self._error_message: Optional[str] = None

    @property
    def elapsed(self) -> float:
        """Eltelt idő másodpercben."""
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.time()
        return end - self._start_time

    def __enter__(self) -> "ForecastTimer":
        """Context manager belépés - időmérés indítása."""
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager kilépés - időmérés befejezése és logolás."""
        self._end_time = time.time()

        if exc_type is not None:
            self._success = False
            self._error_message = str(exc_val)

        # Lassú futtatás logolása
        if self.log_slow and self.elapsed > self.slow_threshold:
            log_slow_forecast(self.get_stats())

        return False  # Ne nyeljük el a kivételeket

    def get_stats(self) -> PerformanceStats:
        """Teljesítmény statisztikák lekérése."""
        return PerformanceStats(
            model_name=self.model_name,
            strategy_id=self.strategy_id,
            elapsed_time=self.elapsed,
            data_points=self.data_points,
            forecast_steps=self.forecast_steps,
            success=self._success,
            error_message=self._error_message,
        )

    def mark_error(self, error: Exception) -> None:
        """Hiba megjelölése."""
        self._success = False
        self._error_message = str(error)

    @classmethod
    def as_decorator(
        cls,
        model_name: str,
        log_slow: bool = True,
        slow_threshold: float = SLOW_FORECAST_THRESHOLD
    ) -> Callable:
        """
        Decorator factory forecast függvényekhez.

        Args:
            model_name: Modell neve
            log_slow: Lassú futtatások logolása
            slow_threshold: Lassú küszöb

        Returns:
            Decorator függvény

        Example:
            @ForecastTimer.as_decorator("ARIMA")
            def run_arima(data, steps):
                return model.forecast(data, steps)
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with cls(model_name, log_slow=log_slow, slow_threshold=slow_threshold):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


def log_slow_forecast(stats: PerformanceStats) -> None:
    """
    Lassú forecast logolása megfelelő szinten.

    Args:
        stats: Teljesítmény statisztikák

    Note:
        A régi strategy_analyzer.py 514-517. soraiból származó logika,
        kiterjesztve több szinttel.
    """
    elapsed = stats.elapsed_time
    model = stats.model_name
    strategy = stats.strategy_id or "unknown"

    if elapsed > CRITICAL_SLOW_THRESHOLD:
        # Kritikusan lassú - error szint
        logger.error(
            "CRITICAL: %s forecast for %s took %.2fs (threshold: %.0fs)",
            model, strategy, elapsed, CRITICAL_SLOW_THRESHOLD
        )
    elif elapsed > VERY_SLOW_THRESHOLD:
        # Nagyon lassú - warning szint
        logger.warning(
            "SLOW: %s forecast for %s took %.2fs (threshold: %.0fs)",
            model, strategy, elapsed, VERY_SLOW_THRESHOLD
        )
    elif elapsed > SLOW_FORECAST_THRESHOLD:
        # Lassú - debug szint (eredeti viselkedés)
        logger.debug(
            "Strategy %s: %s completed in %.2fs",
            strategy, model, elapsed
        )


# =============================================================================
# BATCH MONITORING
# =============================================================================

class BatchPerformanceMonitor:
    """
    Batch futtatások teljesítmény monitorozása.

    Használat:
        monitor = BatchPerformanceMonitor("LSTM")

        for strategy in strategies:
            with monitor.track(strategy.id):
                result = model.forecast(strategy.data, steps=52)

        print(monitor.summary())
    """

    def __init__(self, model_name: str):
        """
        Inicializálás.

        Args:
            model_name: Modell neve
        """
        self.model_name = model_name
        self._stats: List[PerformanceStats] = []
        self._current_timer: Optional[ForecastTimer] = None

    def track(self, strategy_id: str, data_points: int = 0) -> ForecastTimer:
        """
        Egy stratégia futtatásának követése.

        Args:
            strategy_id: Stratégia azonosító
            data_points: Adatpontok száma

        Returns:
            ForecastTimer context manager
        """
        timer = ForecastTimer(
            model_name=self.model_name,
            strategy_id=strategy_id,
            data_points=data_points,
            log_slow=False  # Batch-ben a végén logolunk összesítve
        )
        self._current_timer = timer
        return timer

    def record(self, stats: PerformanceStats) -> None:
        """Statisztika rögzítése."""
        self._stats.append(stats)

    def __enter__(self) -> "BatchPerformanceMonitor":
        """Batch monitoring indítása."""
        self._batch_start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Batch monitoring befejezése és összesítés logolása."""
        self._batch_end = time.time()
        self._log_summary()
        return False

    def _log_summary(self) -> None:
        """Összesítő log."""
        if not self._stats:
            return

        total_time = sum(s.elapsed_time for s in self._stats)
        avg_time = total_time / len(self._stats)
        slow_count = sum(1 for s in self._stats if s.is_slow)
        failed_count = sum(1 for s in self._stats if not s.success)

        logger.info(
            "Batch %s completed: %d strategies, total %.2fs, avg %.2fs, %d slow, %d failed",
            self.model_name,
            len(self._stats),
            total_time,
            avg_time,
            slow_count,
            failed_count
        )

        # Lassú stratégiák részletezése
        slow_strategies = [s for s in self._stats if s.is_slow]
        if slow_strategies:
            slowest = max(slow_strategies, key=lambda s: s.elapsed_time)
            logger.debug(
                "Slowest: %s (%.2fs)",
                slowest.strategy_id,
                slowest.elapsed_time
            )

    def summary(self) -> Dict[str, Any]:
        """
        Összesítő statisztikák.

        Returns:
            Dictionary az összesített statisztikákkal
        """
        if not self._stats:
            return {"count": 0}

        elapsed_times = [s.elapsed_time for s in self._stats]
        return {
            "model_name": self.model_name,
            "count": len(self._stats),
            "total_time": sum(elapsed_times),
            "avg_time": sum(elapsed_times) / len(elapsed_times),
            "min_time": min(elapsed_times),
            "max_time": max(elapsed_times),
            "slow_count": sum(1 for s in self._stats if s.is_slow),
            "failed_count": sum(1 for s in self._stats if not s.success),
            "success_rate": sum(1 for s in self._stats if s.success) / len(self._stats),
        }

    @property
    def stats(self) -> List[PerformanceStats]:
        """Összes rögzített statisztika."""
        return self._stats.copy()


# =============================================================================
# MEMORY MONITORING (OPCIONÁLIS)
# =============================================================================

def get_memory_usage_mb() -> Optional[float]:
    """
    Aktuális memóriahasználat MB-ban.

    Returns:
        Memóriahasználat MB-ban, vagy None ha nem elérhető
    """
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return None
    except Exception:
        return None


def get_gpu_memory_usage_mb() -> Optional[float]:
    """
    GPU memóriahasználat MB-ban.

    Returns:
        GPU memóriahasználat MB-ban, vagy None ha nem elérhető
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return None
    except ImportError:
        return None
    except Exception:
        return None
