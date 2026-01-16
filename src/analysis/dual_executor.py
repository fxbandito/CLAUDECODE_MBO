"""
Dual Model Mode Executor - MBO Trading Strategy Analyzer

Handles multi-stage training and prediction for dual-model forecasting:
- Activity Model: Predicts whether strategy will be active (0.0-1.0 probability)
- Profit Model: Predicts profit per active week (regression)

Final Forecast = Activity_Ratio x Expected_Active_Weeks x Profit_Per_Active_Week

IMPORTANT: Memory-safe multiprocessing with rolling data mode support.
"""

import gc
import logging
import multiprocessing
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from analysis.engine import get_resource_manager
from analysis.process_utils import init_worker_environment
from analysis.dual_task import train_dual_model_task

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODE DETECTION
# =============================================================================


def detect_data_mode(df: Optional[pd.DataFrame]) -> str:
    """
    Detect which data mode is being used based on column names.

    This is critical for memory management: rolling mode needs fewer workers.

    Args:
        df: DataFrame with feature columns

    Returns:
        str: "rolling", "forward", or "original"
    """
    if df is None:
        return "original"

    columns = set(df.columns)

    # Rolling mode has specific rolling feature columns
    rolling_indicators = {
        "feat_rolling_active_ratio",
        "feat_rolling_profit_consistency",
        "feat_rolling_sharpe",
        "feat_rolling_volatility",
    }

    if rolling_indicators & columns:
        return "rolling"

    # Forward mode has forward-looking feature columns
    forward_indicators = {
        "feat_forward_return_1w",
        "feat_forward_return_4w",
        "feat_forward_activity_1w",
    }

    if forward_indicators & columns:
        return "forward"

    return "original"


# =============================================================================
# WORKER PROCESS UTILITIES
# =============================================================================


def _init_worker_process() -> None:
    """
    Initialize worker process with safe settings.

    Delegates to process_utils.init_worker_environment() and adds priority settings.
    """
    # Alap környezet inicializálás (GPU tiltás, szál limitek)
    init_worker_environment()

    # Alacsony prioritás beállítása
    if HAS_PSUTIL:
        try:
            proc = psutil.Process()
            if os.name == "nt":
                proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                proc.nice(10)
        except (psutil.Error, OSError):
            pass


def _force_kill_child_processes() -> None:
    """
    Force kill all child processes of the current process.

    Prevents zombie processes that continue consuming CPU after analysis.
    """
    if not HAS_PSUTIL:
        return

    try:
        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)

        if not children:
            return

        logger.debug("Force killing %d child processes...", len(children))

        # First try graceful termination
        for child in children:
            try:
                if child.is_running():
                    child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Wait briefly for graceful exit
        _, alive = psutil.wait_procs(children, timeout=2)

        # Force kill any survivors
        for child in alive:
            try:
                logger.warning("Force killing stubborn process PID %d", child.pid)
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Final wait
        if alive:
            psutil.wait_procs(alive, timeout=1)

    except (OSError, psutil.Error) as e:
        logger.debug("Error in force kill child processes: %s", e)


def _suspend_workers(pool) -> List[int]:
    """Suspend all worker processes in the pool."""
    suspended_pids = []
    if not HAS_PSUTIL:
        return suspended_pids

    try:
        for worker in pool._pool:  # pylint: disable=protected-access
            if worker.is_alive():
                try:
                    proc = psutil.Process(worker.pid)
                    proc.suspend()
                    suspended_pids.append(worker.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        logger.info("Suspended %d worker processes", len(suspended_pids))
    except (psutil.Error, AttributeError) as e:
        logger.debug("Could not suspend workers: %s", e)

    return suspended_pids


def _resume_workers(pids: List[int]) -> None:
    """Resume suspended worker processes."""
    if not HAS_PSUTIL:
        return

    for pid in pids:
        try:
            proc = psutil.Process(pid)
            proc.resume()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    logger.info("Resumed %d worker processes", len(pids))


# =============================================================================
# WORKER COUNT CALCULATION
# =============================================================================


def _get_worker_count(data_mode: str) -> Tuple[int, int]:
    """
    Calculate optimal worker count and threads per worker.

    Args:
        data_mode: "original", "forward", or "rolling"

    Returns:
        (workers, threads_per_worker)
    """
    res_mgr = get_resource_manager()
    n_jobs = res_mgr.get_n_jobs()

    # Rolling mode: Memory-intensive, use fewer workers
    if data_mode == "rolling":
        workers = max(1, min(2, n_jobs // 2))
        threads = max(1, n_jobs // workers)
        logger.info("Rolling mode: Using %d workers with %d threads each", workers, threads)
        return workers, threads

    # Dual mode strategy: Fewer workers, more threads per worker
    workers = max(2, min(4, n_jobs // 2))
    threads = max(1, n_jobs // workers)

    return workers, threads


# =============================================================================
# MAIN EXECUTOR
# =============================================================================


def run_dual_model_mode(
    data: pd.DataFrame,
    method_name: str,
    params: Dict[str, Any],
    max_horizon: int = 52,
    progress_callback: Optional[Callable[[float], None]] = None,
    stop_callback: Optional[Callable[[], bool]] = None,
    pause_event: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Optional[int], Optional[pd.DataFrame], str, Dict]:
    """
    Run analysis in Dual Model mode - parallelized with multiprocessing.

    Args:
        data: Input DataFrame with strategy data
        method_name: Model name (e.g., "XGBoost", "LightGBM")
        params: Model parameters
        max_horizon: Maximum forecast horizon in weeks (default 52)
        progress_callback: Progress update callback (0.0-1.0)
        stop_callback: Stop check callback (returns True to stop)
        pause_event: Threading event for pause/resume

    Returns:
        Tuple of:
        - results_df: DataFrame with forecasts sorted by Forecast_1M
        - best_strat_id: ID of best strategy
        - best_strat_data: Historical data of best strategy
        - filename_base: Suggested filename
        - all_strategies_data: Dict of all strategies' historical data
    """
    logger.info("Dual Model mode: Preparing data for %s...", method_name)

    if progress_callback:
        progress_callback(0.05)

    # Detect data mode
    data_mode = detect_data_mode(data)
    logger.info("Dual Model mode: Detected data mode = '%s'", data_mode)

    # Memory cleanup for rolling mode
    if data_mode == "rolling":
        logger.info("Rolling mode: Performing pre-allocation memory cleanup")
        _clear_cuda_cache()
        gc.collect()
        _clear_cuda_cache()
        time.sleep(0.1)

    # Prepare dual model data
    x_data, y_activity, y_profit = _prepare_dual_model_data(data)

    if x_data is None or len(x_data) == 0:
        logger.error("Dual model mode: Failed to prepare data")
        return pd.DataFrame(), None, None, f"Analysis_{method_name}_FAILED", {}

    if progress_callback:
        progress_callback(0.1)

    # Calculate worker settings
    optimal_workers, threads_per_worker = _get_worker_count(data_mode)

    task_params = params.copy() if params else {}
    task_params["n_jobs"] = threads_per_worker

    # Train/validation split
    train_size = int(len(x_data) * 0.8)
    x_train = x_data.iloc[:train_size]

    # Walk-Forward: Get last max_horizon rows per strategy for prediction
    last_indices = x_data.groupby("No.").tail(max_horizon).index
    x_predict = x_data.loc[last_indices].copy()

    # Create tasks
    tasks = []

    # Activity Tasks (5 horizons)
    y_activity_train = y_activity.iloc[:train_size]
    for col in y_activity_train.columns:
        tasks.append((
            method_name, x_train, y_activity_train[col],
            x_predict, task_params, "activity"
        ))

    # Profit Tasks (5 horizons)
    y_profit_train = y_profit.iloc[:train_size]
    for col in y_profit_train.columns:
        tasks.append((
            method_name, x_train, y_profit_train[col],
            x_predict, task_params, "regression"
        ))

    total_tasks = len(tasks)
    optimal_workers = min(optimal_workers, total_tasks)

    logger.info(
        "Dual Model: %d tasks, %d workers, %d threads/worker",
        total_tasks, optimal_workers, threads_per_worker
    )

    # Execute tasks
    activity_preds = {}
    profit_preds = {}
    pool = None

    try:
        pool = multiprocessing.Pool(
            processes=optimal_workers,
            initializer=_init_worker_process
        )

        time.sleep(0.2)  # Allow workers to initialize

        # Submit all tasks
        async_results = [
            pool.apply_async(train_dual_model_task, args=task)
            for task in tasks
        ]

        # Collect results
        completed = 0
        while completed < total_tasks:
            # Check stop
            if stop_callback and stop_callback():
                logger.info("Analysis stopped by user")
                pool.terminate()
                break

            # Check pause
            if pause_event and not pause_event.is_set():
                logger.info("Analysis paused")
                suspended = _suspend_workers(pool)

                while not pause_event.is_set():
                    time.sleep(0.2)
                    if stop_callback and stop_callback():
                        _resume_workers(suspended)
                        pool.terminate()
                        break

                if stop_callback and stop_callback():
                    break

                _resume_workers(suspended)
                logger.info("Analysis resumed")

            # Process ready results
            for i, ar in enumerate(async_results):
                if ar is None:
                    continue

                if ar.ready():
                    try:
                        result_dict = ar.get(timeout=0.1)

                        if "error" in result_dict:
                            logger.error("Task failed: %s", result_dict.get("error"))
                        else:
                            for k, v in result_dict.items():
                                if v is None:
                                    continue
                                if "activity" in k:
                                    activity_preds[k] = v
                                else:
                                    profit_preds[k] = v

                    except Exception as e:  # pylint: disable=broad-exception-caught
                        error_str = str(e).lower()
                        if "terminated abruptly" in error_str or "broken" in error_str:
                            logger.error(
                                "Process pool crashed (possible OOM). "
                                "Try panel mode or reduce worker count."
                            )
                            for j in range(len(async_results)):
                                async_results[j] = None
                            completed = total_tasks
                            break

                        logger.error("Task error: %s", e)

                    async_results[i] = None
                    completed += 1

                    if progress_callback:
                        progress_callback(0.1 + 0.8 * (completed / total_tasks))

            time.sleep(0.05)

        pool.close()
        pool.join()

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Dual model error: %s", e)
        if pool:
            try:
                pool.terminate()
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    finally:
        if pool:
            try:
                pool.terminate()
                pool.join()
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        _force_kill_child_processes()
        gc.collect()
        logger.info("Dual model pool cleanup completed")

    # Combine predictions
    results = _combine_predictions(x_predict, activity_preds, profit_preds, method_name)

    if progress_callback:
        progress_callback(0.95)

    if not results:
        return pd.DataFrame(), None, None, f"Analysis_{method_name}_EMPTY", {}

    results_df = pd.DataFrame(results).sort_values("Forecast_1M", ascending=False)
    best_strat_id = results_df.iloc[0]["No."]
    best_strat_data = data[data["No."] == best_strat_id].copy()

    # Collect all strategies' data
    all_strategies_data = {
        strategy_id: data[data["No."] == strategy_id].copy()
        for strategy_id in results_df["No."].values
    }

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"Analysis_{method_name}_Dual_{ts_str}"

    if stop_callback and stop_callback():
        filename_base += "_STOPPED"

    if progress_callback:
        progress_callback(1.0)

    logger.info("Dual model mode complete: %d strategies", len(results))

    return results_df, best_strat_id, best_strat_data, filename_base, all_strategies_data


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _clear_cuda_cache() -> None:
    """Clear CUDA cache if available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _prepare_dual_model_data(
    df: pd.DataFrame
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Prepare data for dual model training.

    Creates activity and profit targets for multiple horizons.

    Args:
        df: Input DataFrame with strategy data

    Returns:
        Tuple of (X_features, Y_activity, Y_profit)
    """
    if df is None or df.empty:
        return None, None, None

    required_cols = ["No.", "Profit"]
    if not all(col in df.columns for col in required_cols):
        logger.error("Missing required columns: %s", required_cols)
        return None, None, None

    df = df.sort_values(["No.", "Date"]).copy() if "Date" in df.columns else df.copy()

    # Define horizons
    horizons = [1, 4, 13, 26, 52]

    # Create targets
    y_activity = pd.DataFrame(index=df.index)
    y_profit = pd.DataFrame(index=df.index)

    has_trades = "Trades" in df.columns

    for h in horizons:
        # Activity: Was there trading activity in next h weeks?
        if has_trades:
            activity = df.groupby("No.")["Trades"].transform(
                lambda x: (x.shift(-h).rolling(h, min_periods=1).sum() > 0).astype(float)
            )
        else:
            activity = df.groupby("No.")["Profit"].transform(
                lambda x: (x.shift(-h).rolling(h, min_periods=1).apply(lambda s: (s != 0).any())).astype(float)
            )
        y_activity[f"target_activity_{h}w"] = activity.fillna(0.5)

        # Profit: Average profit over next h weeks
        profit = df.groupby("No.")["Profit"].transform(
            lambda x: x.shift(-h).rolling(h, min_periods=1).mean()
        )
        y_profit[f"target_profit_{h}w"] = profit.fillna(0)

    # Features: Use all columns except targets
    exclude_cols = {"Date", "SourceFile"} | set(y_activity.columns) | set(y_profit.columns)
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    x_data = df[feature_cols].copy()

    # Fill NaN in features
    for col in x_data.columns:
        if x_data[col].dtype in [np.float64, np.float32]:
            x_data[col] = x_data[col].fillna(0)

    logger.info(
        "Prepared dual model data: %d samples, %d features, %d activity targets, %d profit targets",
        len(x_data), len(feature_cols), len(y_activity.columns), len(y_profit.columns)
    )

    return x_data, y_activity, y_profit


def _combine_predictions(
    x_predict: pd.DataFrame,
    activity_preds: Dict[str, np.ndarray],
    profit_preds: Dict[str, np.ndarray],
    method_name: str
) -> List[Dict[str, Any]]:
    """
    Combine activity and profit predictions into final forecasts.

    Formula: Final = Activity_Ratio x Weeks x Profit_Per_Active_Week

    Args:
        x_predict: Features used for prediction (contains strategy IDs)
        activity_preds: Dict of activity predictions by target name
        profit_preds: Dict of profit predictions by target name
        method_name: Model name for labeling

    Returns:
        List of result dictionaries
    """
    results = []

    x_predict = x_predict.reset_index(drop=True)
    grouped = x_predict.groupby("No.")

    # Horizon mapping: (activity_col, profit_col, output_name, weeks)
    horizon_mapping = {
        1: ("target_activity_1w", "target_profit_1w", "Forecast_1W", 1),
        4: ("target_activity_4w", "target_profit_4w", "Forecast_1M", 4),
        13: ("target_activity_13w", "target_profit_13w", "Forecast_3M", 13),
        26: ("target_activity_26w", "target_profit_26w", "Forecast_6M", 26),
        52: ("target_activity_52w", "target_profit_52w", "Forecast_12M", 52),
    }

    for strat_id, group in grouped:
        indices = group.index.values

        result = {
            "No.": strat_id,
            "Method": f"{method_name} (Dual)",
        }

        forecasts_sequence = []

        for horizon, (act_col, prof_col, forecast_name, weeks) in horizon_mapping.items():
            # Get predictions for this strategy
            act_slice = activity_preds.get(act_col, np.full(len(indices), 0.5))[indices] \
                if act_col in activity_preds else np.full(len(indices), 0.5)

            prof_slice = profit_preds.get(prof_col, np.zeros(len(indices)))[indices] \
                if prof_col in profit_preds else np.zeros(len(indices))

            # Handle array indexing
            if isinstance(act_slice, np.ndarray) and len(act_slice) > len(indices):
                act_slice = act_slice[indices] if max(indices) < len(act_slice) else np.full(len(indices), 0.5)
            if isinstance(prof_slice, np.ndarray) and len(prof_slice) > len(indices):
                prof_slice = prof_slice[indices] if max(indices) < len(prof_slice) else np.zeros(len(indices))

            # Calculate combined forecast
            expected_active = act_slice * weeks
            combined = expected_active * prof_slice

            # Use last value for scalar report
            if len(combined) > 0:
                result[forecast_name] = float(combined[-1])
                result[f"Activity_{horizon}w"] = float(act_slice[-1])
                result[f"Profit_per_active_{horizon}w"] = float(prof_slice[-1])
                result[f"Expected_active_weeks_{horizon}w"] = float(expected_active[-1])
            else:
                result[forecast_name] = 0.0

            # Store 1W forecast sequence for trajectory
            if horizon == 1:
                forecasts_sequence = combined.tolist() if hasattr(combined, 'tolist') else list(combined)

        # Ensure we have a forecast sequence
        if not forecasts_sequence:
            forecasts_sequence = [0.0] * 52

        result["Forecasts"] = forecasts_sequence
        results.append(result)

    return results


# =============================================================================
# SUPPORTED MODELS - 100% Registry-alapú, NINCS hardcoded lista
# =============================================================================


def get_dual_mode_models() -> list:
    """
    Visszaadja a Dual Mode-ot támogató modellek listáját.

    A lista KIZÁRÓLAG a ModelInfo.supports_dual_mode flag alapján generálódik.
    NINCS fallback - ha egy modell nem implementálta, nem támogatja.
    """
    from models import get_all_models, get_model_info
    return [
        name for name in get_all_models()
        if get_model_info(name) and get_model_info(name).supports_dual_mode
    ]


# Legacy alias - DINAMIKUS, nem hardcoded
DUAL_MODE_SUPPORTED_MODELS = get_dual_mode_models()
