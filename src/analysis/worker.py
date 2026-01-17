"""
Analysis Worker - Multiprocessing worker for GUI responsiveness.

This module runs the analysis in a separate process to prevent GUI freezing.
Communication with the main process happens via multiprocessing.Queue.

Architecture:
    Main Process (GUI)              Worker Process (Analysis)
    ==================              =========================
         |                                    |
         |--- start_process(data, context) -->|
         |                                    |
         |<-- progress_queue.put(progress) ---|  (periodic)
         |                                    |
         |<-- result_queue.put(results) ------|  (on complete)
         |                                    |
"""
# pylint: disable=broad-exception-caught

import gc
import logging
import os
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WorkerProgress:
    """
    Progress data sent from worker to main process.

    Must be pickle-serializable (no complex objects).
    """
    total_strategies: int = 0
    completed_strategies: int = 0
    current_strategy: str = ""
    is_running: bool = False
    is_paused: bool = False
    is_cancelled: bool = False
    error: Optional[str] = None


@dataclass
class WorkerResult:
    """
    Final result sent from worker to main process.

    Must be pickle-serializable.
    """
    success: bool
    results: Dict[str, Dict[str, Any]]  # strategy_id -> {forecasts, elapsed_ms, success, error}
    elapsed_seconds: float
    error: Optional[str] = None


def _setup_worker_env(env_vars: Dict[str, str]):
    """Setup environment variables in worker process."""
    for key, value in env_vars.items():
        os.environ[key] = value


def _convert_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string params to appropriate types."""
    converted = {}
    for key, value in params.items():
        if isinstance(value, str):
            # Try int
            try:
                converted[key] = int(value)
                continue
            except ValueError:
                pass

            # Try float
            try:
                converted[key] = float(value)
                continue
            except ValueError:
                pass

            # Try bool
            if value.lower() in ('true', 'false'):
                converted[key] = value.lower() == 'true'
                continue

            # Keep as string
            converted[key] = value
        else:
            converted[key] = value

    return converted


def analysis_worker(
    data_dict: Dict[str, Any],
    context_dict: Dict[str, Any],
    env_vars: Dict[str, str],
    progress_queue: Queue,
    result_queue: Queue,
    cancel_flag_queue: Queue
):
    """
    Worker function running in separate process.

    Args:
        data_dict: Serialized DataFrame data (to_dict format)
        context_dict: Analysis context as dict
        env_vars: Environment variables to set
        progress_queue: Queue to send progress updates
        result_queue: Queue to send final results
        cancel_flag_queue: Queue to receive cancel signals
    """
    start_time = time.time()

    try:
        # Setup environment
        _setup_worker_env(env_vars)

        # Reconstruct DataFrame
        data = pd.DataFrame(data_dict)

        # Import here to avoid import issues in main process
        from models import get_model_class, get_param_defaults

        # Extract context
        model_name = context_dict['model_name']
        params = _convert_params(context_dict.get('params', {}))
        forecast_horizon = context_dict.get('forecast_horizon', 12)
        use_batch = context_dict.get('use_batch', False)

        # Load model
        model_class = get_model_class(model_name)
        if model_class is None:
            result_queue.put(WorkerResult(
                success=False,
                results={},
                elapsed_seconds=time.time() - start_time,
                error=f"Model not found: {model_name}"
            ))
            return

        model = model_class()

        # Merge params with defaults
        full_params = get_param_defaults(model_name)
        full_params.update(params)

        # Extract strategies
        strategies = _extract_strategies(data)

        if not strategies:
            result_queue.put(WorkerResult(
                success=False,
                results={},
                elapsed_seconds=time.time() - start_time,
                error="No strategies found in data"
            ))
            return

        # Send initial progress
        progress_queue.put(WorkerProgress(
            total_strategies=len(strategies),
            completed_strategies=0,
            is_running=True
        ))

        # Run analysis
        if use_batch and model.MODEL_INFO.supports_batch:
            results = _run_batch_worker(
                model, strategies, forecast_horizon, full_params,
                progress_queue, cancel_flag_queue
            )
        else:
            results = _run_sequential_worker(
                model, strategies, forecast_horizon, full_params,
                progress_queue, cancel_flag_queue
            )

        # Send final result
        elapsed = time.time() - start_time
        result_queue.put(WorkerResult(
            success=True,
            results=results,
            elapsed_seconds=elapsed
        ))

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Worker error: %s\n%s", e, tb)
        result_queue.put(WorkerResult(
            success=False,
            results={},
            elapsed_seconds=time.time() - start_time,
            error=str(e)
        ))
    finally:
        # Cleanup
        gc.collect()


def _extract_strategies(
    data: pd.DataFrame,
    strategy_col: str = "No.",
    value_col: str = "Profit"
) -> Dict[str, List[float]]:
    """Extract strategy time series from DataFrame."""
    strategies = {}

    if data is None or data.empty:
        return strategies

    if strategy_col not in data.columns:
        if value_col in data.columns:
            values = data[value_col].dropna().values.tolist()
            if len(values) >= 10:
                strategies["all"] = values
        return strategies

    for strategy_id in data[strategy_col].unique():
        strategy_df = data[data[strategy_col] == strategy_id]

        if value_col in strategy_df.columns:
            values = strategy_df[value_col].dropna().values.tolist()
            if len(values) >= 10:
                strategies[str(strategy_id)] = values

    return strategies


def _check_cancel(cancel_flag_queue: Queue) -> bool:
    """Check if cancel was requested (non-blocking)."""
    try:
        while not cancel_flag_queue.empty():
            msg = cancel_flag_queue.get_nowait()
            if msg == "CANCEL":
                return True
    except Exception:
        pass
    return False


def _run_sequential_worker(
    model,
    strategies: Dict[str, List[float]],
    forecast_horizon: int,
    params: Dict[str, Any],
    progress_queue: Queue,
    cancel_flag_queue: Queue
) -> Dict[str, Dict[str, Any]]:
    """Sequential processing in worker."""
    results = {}
    total = len(strategies)
    last_progress_time = 0

    for i, (strategy_id, values) in enumerate(strategies.items()):
        # Check cancel
        if _check_cancel(cancel_flag_queue):
            break

        # Forecast
        start = time.perf_counter()
        try:
            forecasts = model.forecast(values, forecast_horizon, params)
            elapsed_ms = (time.perf_counter() - start) * 1000

            results[strategy_id] = {
                'forecasts': forecasts,
                'elapsed_ms': elapsed_ms,
                'success': True,
                'error': None
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            results[strategy_id] = {
                'forecasts': [float('nan')] * forecast_horizon,
                'elapsed_ms': elapsed_ms,
                'success': False,
                'error': str(e)
            }

        # Progress update - throttled (max every 250ms)
        current_time = time.time()
        if current_time - last_progress_time > 0.25:
            progress_queue.put(WorkerProgress(
                total_strategies=total,
                completed_strategies=i + 1,
                current_strategy=strategy_id,
                is_running=True
            ))
            last_progress_time = current_time

    return results


def _run_batch_worker(
    model,
    strategies: Dict[str, List[float]],
    forecast_horizon: int,
    params: Dict[str, Any],
    progress_queue: Queue,
    cancel_flag_queue: Queue
) -> Dict[str, Dict[str, Any]]:
    """Batch processing in worker."""
    results = {}

    # Check cancel before batch
    if _check_cancel(cancel_flag_queue):
        return results

    progress_queue.put(WorkerProgress(
        total_strategies=len(strategies),
        completed_strategies=0,
        current_strategy="BATCH MODE",
        is_running=True
    ))

    start = time.perf_counter()
    try:
        batch_results = model.forecast_batch(strategies, forecast_horizon, params)
        elapsed_ms = (time.perf_counter() - start) * 1000
        per_strategy_ms = elapsed_ms / len(strategies) if strategies else 0

        for strategy_id, forecasts in batch_results.items():
            results[strategy_id] = {
                'forecasts': forecasts,
                'elapsed_ms': per_strategy_ms,
                'success': True,
                'error': None
            }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        for strategy_id in strategies:
            results[strategy_id] = {
                'forecasts': [float('nan')] * forecast_horizon,
                'elapsed_ms': elapsed_ms / len(strategies),
                'success': False,
                'error': str(e)
            }

    return results


class AnalysisWorkerManager:
    """
    Manager for the analysis worker process.

    Handles starting, stopping, and communicating with the worker process.

    Usage:
        manager = AnalysisWorkerManager()
        manager.start(data, context, env_vars)

        # In GUI poll loop:
        progress = manager.get_progress()
        result = manager.get_result()

        # To cancel:
        manager.cancel()
    """

    def __init__(self):
        self._process: Optional[Process] = None
        self._progress_queue: Optional[Queue] = None
        self._result_queue: Optional[Queue] = None
        self._cancel_queue: Optional[Queue] = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Check if worker is running."""
        if self._process is None:
            return False
        return self._process.is_alive()

    def start(
        self,
        data: pd.DataFrame,
        context_dict: Dict[str, Any],
        env_vars: Dict[str, str]
    ):
        """
        Start the worker process.

        Args:
            data: DataFrame to analyze
            context_dict: Analysis context as dict
            env_vars: Environment variables for worker
        """
        if self.is_running:
            logger.warning("Worker already running, stopping first...")
            self.stop()

        # Create queues
        self._progress_queue = Queue()
        self._result_queue = Queue()
        self._cancel_queue = Queue()

        # Convert DataFrame to dict for pickling
        data_dict = data.to_dict()

        # Create and start process
        self._process = Process(
            target=analysis_worker,
            args=(
                data_dict,
                context_dict,
                env_vars,
                self._progress_queue,
                self._result_queue,
                self._cancel_queue
            ),
            daemon=True
        )
        self._process.start()
        self._is_running = True

        logger.info("Worker process started (PID: %d)", self._process.pid)

    def get_progress(self) -> Optional[WorkerProgress]:
        """
        Get latest progress from worker (non-blocking).

        Returns the most recent progress, discarding older ones.
        """
        if self._progress_queue is None:
            return None

        latest = None
        try:
            while not self._progress_queue.empty():
                latest = self._progress_queue.get_nowait()
        except Exception:
            pass

        return latest

    def get_result(self) -> Optional[WorkerResult]:
        """
        Get result from worker (non-blocking).

        Returns None if not ready yet.
        """
        if self._result_queue is None:
            return None

        try:
            if not self._result_queue.empty():
                return self._result_queue.get_nowait()
        except Exception:
            pass

        return None

    def cancel(self):
        """Send cancel signal to worker."""
        if self._cancel_queue is not None:
            try:
                self._cancel_queue.put("CANCEL")
            except Exception:
                pass

    def stop(self):
        """Stop the worker process."""
        self.cancel()

        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)

            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1)

        self._cleanup_queues()
        self._process = None
        self._is_running = False

    def _cleanup_queues(self):
        """Clean up queue resources."""
        for queue in [self._progress_queue, self._result_queue, self._cancel_queue]:
            if queue is not None:
                try:
                    while not queue.empty():
                        queue.get_nowait()
                except Exception:
                    pass

        self._progress_queue = None
        self._result_queue = None
        self._cancel_queue = None
