"""
Panel Mode Executor - 100% Modell-fuggetlen architektura.

A Panel Mode egy model tanit az OSSZES strategiara egyszerre (panel adatstruktura).
Sokkal gyorsabb ML-alapu modelleknel, mint az egyenkenti strategia-feldolgozas.

Ez a modul KIZAROLAG az altalanos panel mode infrastrukturat biztositja:
- Panel adat eloallitas (lag feature-okkel)
- Time-aware train/test split (data leakage elkerules)
- Worker process kezeles (ProcessPoolExecutor)

A KONKRET MODELL IMPLEMENTACIOK a models/ mappaban talalhatoak.
Minden modell a sajat create_dual_regressor() metodusat implementalja.

FONTOS: Ez a fajl NEM TARTALMAZ semmilyen modell-specifikus kodot!
"""

import gc
import logging
import multiprocessing
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.engine import get_resource_manager
from analysis.process_utils import (
    cleanup_cuda_context,
    force_kill_child_processes,
    init_worker_environment,
)
from analysis.dual_task import train_dual_model_task

logger = logging.getLogger(__name__)


# =============================================================================
# PANEL DATA PREPARATION
# =============================================================================


def prepare_panel_data(
    df: pd.DataFrame,
    target_col: str = "Profit"
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[str]]:
    """
    Panel adat eloallitas lag feature-okkel.

    Panel mode-ban minden strategia adatait egyetlen dataset-be vonjuk ossze,
    es lag feature-oket keszitunk az idosoros jelleg megorzesere.

    Args:
        df: Input DataFrame (No., Date, Profit oszlopokkal)
        target_col: Celvaltozo oszlop neve

    Returns:
        Tuple of (X_features, Y_targets, feature_column_names)
    """
    if df is None or df.empty:
        return None, None, []

    required_cols = ["No.", target_col]
    if not all(col in df.columns for col in required_cols):
        logger.error("Missing required columns: %s", required_cols)
        return None, None, []

    # Rendezés stratégia és dátum szerint
    df = df.sort_values(["No.", "Date"]).copy() if "Date" in df.columns else df.copy()

    # Lag feature-ok létrehozása stratégiánként
    lag_periods = [1, 2, 4, 8, 13, 26]

    for lag in lag_periods:
        df[f"lag_profit_{lag}"] = df.groupby("No.")[target_col].shift(lag)

    # Rolling statisztikák
    for window in [4, 13]:
        df[f"lag_rolling_mean_{window}"] = df.groupby("No.")[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    df["lag_rolling_std_13"] = df.groupby("No.")[target_col].transform(
        lambda x: x.shift(1).rolling(13, min_periods=1).std()
    )

    # Target oszlopok (különböző horizontok)
    horizons = {"1w": 1, "4w": 4, "13w": 13, "26w": 26, "52w": 52}
    y_targets = pd.DataFrame(index=df.index)

    for name, h in horizons.items():
        y_targets[f"target_{name}"] = df.groupby("No.")[target_col].transform(
            lambda x: x.shift(-h).rolling(h, min_periods=1).mean()
        )

    # Feature oszlopok (exclude: targets, meta oszlopok)
    exclude_cols = {"Date", "SourceFile"} | set(y_targets.columns)
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    x_data = df[feature_cols].copy()

    # NaN kezelés
    for col in x_data.columns:
        if x_data[col].dtype in [np.float64, np.float32]:
            x_data[col] = x_data[col].fillna(0)

    y_targets = y_targets.fillna(0)

    logger.info(
        "Panel data prepared: %d samples, %d features, %d targets",
        len(x_data), len(feature_cols), len(y_targets.columns)
    )

    return x_data, y_targets, feature_cols


# =============================================================================
# TIME-AWARE SPLIT
# =============================================================================


def time_aware_split(
    x_data: pd.DataFrame,
    y_data: pd.DataFrame,
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-aware train/test split - data leakage elkerulese.

    Minden strategia adatait kulon bontja, igy biztositva hogy
    csak a multat hasznaljuk a jovo elorejelzesere.

    Args:
        x_data: Feature DataFrame (No. oszloppal)
        y_data: Target DataFrame
        train_ratio: Training adatok aranya (0.0-1.0)

    Returns:
        (x_train, y_train, x_test, y_test)
    """
    train_indices = []
    test_indices = []

    for strat_id in x_data["No."].unique():
        strat_mask = x_data["No."] == strat_id
        strat_indices = x_data[strat_mask].index.tolist()

        split_point = int(len(strat_indices) * train_ratio)
        train_indices.extend(strat_indices[:split_point])
        test_indices.extend(strat_indices[split_point:])

    x_train = x_data.loc[train_indices]
    y_train = y_data.loc[train_indices]
    x_test = x_data.loc[test_indices]
    y_test = y_data.loc[test_indices]

    logger.info(
        "Time-aware split: %d train, %d test from %d strategies",
        len(x_train), len(x_test), len(x_data["No."].unique())
    )

    return x_train, y_train, x_test, y_test


# =============================================================================
# WORKER COUNT CALCULATION
# =============================================================================


def _get_panel_worker_count() -> Tuple[int, int]:
    """
    Optimalis worker szam szamitasa panel mode-hoz.

    Panel mode CPU-intenziv, ezert kevesebb worker, tobb thread/worker.

    Returns:
        (workers, threads_per_worker)
    """
    res_mgr = get_resource_manager()
    n_jobs = res_mgr.get_n_jobs()

    # Panel mode: 1-2 worker, sok thread
    # Ez optimalis a nagy osszekapcsolt dataset-hez
    workers = max(1, min(2, n_jobs // 2))
    threads = max(1, n_jobs // workers)

    logger.debug("Panel mode workers: %d, threads/worker: %d", workers, threads)
    return workers, threads


# =============================================================================
# MAIN EXECUTOR
# =============================================================================


def run_panel_mode(
    data: pd.DataFrame,
    method_name: str,
    params: Dict[str, Any],
    max_horizon: int = 52,
    progress_callback: Optional[Callable[[float], None]] = None,
    stop_callback: Optional[Callable[[], bool]] = None,
    pause_event: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Optional[int], Optional[pd.DataFrame], str, Dict]:
    """
    Analzis futtatasa Panel Mode-ban - egy modell az osszes strategiara.

    Args:
        data: Input DataFrame a strategia adatokkal
        method_name: Modell neve (pl. "XGBoost", "LightGBM")
        params: Modell parameterek
        max_horizon: Maximum elorejelzesi horizont hetekben (default 52)
        progress_callback: Progress update callback (0.0-1.0)
        stop_callback: Stop ellenorzes callback (True = leallitas)
        pause_event: Threading event a szunetelteteshez

    Returns:
        Tuple:
        - results_df: Elorejelzesek DataFrame (Forecast_1M szerint rendezve)
        - best_strat_id: Legjobb strategia ID
        - best_strat_data: Legjobb strategia torteneti adatai
        - filename_base: Javasolt fajlnev
        - all_strategies_data: Minden strategia torteneti adatai
    """
    logger.info("Panel mode: Preparing data for %s...", method_name)

    if progress_callback:
        progress_callback(0.05)

    # Ellenorzes: a modell tamogatja-e a panel mode-ot?
    from models import get_model_info
    model_info = get_model_info(method_name)
    if model_info and not model_info.supports_panel_mode:
        logger.warning(
            "Model '%s' does not officially support panel mode. "
            "Attempting anyway with dual mode regressor.", method_name
        )

    # Memory cleanup
    cleanup_cuda_context()
    gc.collect()

    # Panel adat elokeszites
    x_data, y_data, _feature_cols = prepare_panel_data(data)

    if x_data is None or len(x_data) == 0:
        logger.error("Panel mode: Failed to prepare data")
        return pd.DataFrame(), None, None, f"Analysis_{method_name}_FAILED", {}

    if progress_callback:
        progress_callback(0.1)

    # Stop ellenorzes
    if stop_callback and stop_callback():
        return pd.DataFrame(), None, None, f"Analysis_{method_name}_STOPPED", {}

    # Pause kezeles
    if pause_event and not pause_event.is_set():
        logger.info("Analysis paused by user (panel mode)...")
        while not pause_event.is_set():
            time.sleep(0.2)
            if stop_callback and stop_callback():
                return pd.DataFrame(), None, None, f"Analysis_{method_name}_STOPPED", {}
        logger.info("Analysis resumed (panel mode).")

    # Time-aware split
    x_train, y_train, _x_test, _y_test = time_aware_split(x_data, y_data)

    # Prediction set: utolso sor minden strategiabol
    last_indices = x_data.groupby("No.").tail(1).index
    x_predict = x_data.loc[last_indices].copy()

    # Worker settings
    optimal_workers, threads_per_worker = _get_panel_worker_count()

    task_params = params.copy() if params else {}
    task_params["n_jobs"] = threads_per_worker

    # Target oszlopok
    target_cols = y_train.columns.tolist()

    logger.info(
        "Panel mode: %d targets, %d workers, %d threads/worker",
        len(target_cols), optimal_workers, threads_per_worker
    )

    # Task-ok letrehozasa
    tasks = []
    for col in target_cols:
        task_type = "regression"
        task_p = task_params.copy()

        # 1-hetes target: rekurziv elorejelzes
        if col == "target_1w":
            task_p["recursive_horizon"] = max_horizon

        tasks.append((
            method_name, x_train, y_train[col],
            x_predict, task_p, task_type
        ))

    # Task vegrehajas
    predictions = {}
    pool = None

    try:
        pool = multiprocessing.Pool(
            processes=optimal_workers,
            initializer=init_worker_environment,
            initargs=(threads_per_worker,)
        )

        time.sleep(0.1)  # Worker init

        async_results = [
            pool.apply_async(train_dual_model_task, args=task)
            for task in tasks
        ]

        completed = 0
        total_tasks = len(tasks)

        while completed < total_tasks:
            # Stop ellenorzes
            if stop_callback and stop_callback():
                logger.info("Panel mode stopped by user")
                pool.terminate()
                break

            # Pause kezeles
            if pause_event and not pause_event.is_set():
                logger.info("Panel mode paused")
                while not pause_event.is_set():
                    time.sleep(0.2)
                    if stop_callback and stop_callback():
                        pool.terminate()
                        break
                if stop_callback and stop_callback():
                    break
                logger.info("Panel mode resumed")

            # Kesz eredmenyek feldolgozasa
            for i, ar in enumerate(async_results):
                if ar is None:
                    continue

                if ar.ready():
                    try:
                        result_dict = ar.get(timeout=0.1)

                        if "error" in result_dict:
                            logger.error("Task failed: %s", result_dict.get("error"))
                        else:
                            predictions.update(result_dict)

                    except Exception as e:  # pylint: disable=broad-exception-caught
                        logger.error("Task error: %s", e)

                    async_results[i] = None
                    completed += 1

                    if progress_callback:
                        progress_callback(0.1 + 0.7 * (completed / total_tasks))

            time.sleep(0.05)

        pool.close()
        pool.join()

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Panel mode error: %s", e)
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

        force_kill_child_processes()
        gc.collect()
        logger.info("Panel mode pool cleanup completed")

    # Eredmenyek konvertalasa
    results = _panel_predictions_to_results(x_predict, predictions, method_name)

    if progress_callback:
        progress_callback(0.95)

    if not results:
        return pd.DataFrame(), None, None, f"Analysis_{method_name}_EMPTY", {}

    results_df = pd.DataFrame(results).sort_values("Forecast_1M", ascending=False)
    best_strat_id = results_df.iloc[0]["No."]
    best_strat_data = data[data["No."] == best_strat_id].copy()

    # Minden strategia adatainak osszegyujtese
    all_strategies_data = {
        strategy_id: data[data["No."] == strategy_id].copy()
        for strategy_id in results_df["No."].values
    }

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"Analysis_{method_name}_Panel_{ts_str}"

    if stop_callback and stop_callback():
        filename_base += "_STOPPED"

    if progress_callback:
        progress_callback(1.0)

    logger.info("Panel mode complete: %d strategies analyzed", len(results))

    return results_df, best_strat_id, best_strat_data, filename_base, all_strategies_data


# =============================================================================
# RESULTS CONVERSION
# =============================================================================


# Target -> Forecast mapping (konstans, nem kell fuggveny)
TARGET_TO_FORECAST = {
    "target_1w": "Forecast_1W",
    "target_4w": "Forecast_1M",
    "target_13w": "Forecast_3M",
    "target_26w": "Forecast_6M",
    "target_52w": "Forecast_12M",
}


def _panel_predictions_to_results(
    x_predict: pd.DataFrame,
    predictions: Dict[str, Any],
    method_name: str
) -> List[Dict[str, Any]]:
    """
    Panel model predikciok konvertalasa strategiankenti eredmeny formatum.

    Args:
        x_predict: Prediction feature-ok (No. oszloppal)
        predictions: Predikciok dict-je target nevenként
        method_name: Modell neve

    Returns:
        List of result dicts (minden strategiahoz egy)
    """
    results = []
    strategies = x_predict["No."].unique()

    for i, strat_id in enumerate(strategies):
        result = {
            "No.": strat_id,
            "Method": f"{method_name} (Panel)",
        }

        # Standard forecast ertekek
        for target_col, forecast_name in TARGET_TO_FORECAST.items():
            if target_col in predictions and predictions[target_col] is not None:
                pred_array = predictions[target_col]
                if i < len(pred_array):
                    result[forecast_name] = float(pred_array[i])
                else:
                    result[forecast_name] = 0.0
            else:
                result[forecast_name] = 0.0

        # Rekurziv predikciok (ha elerheto)
        if "recursive_predictions" in predictions and predictions["recursive_predictions"]:
            try:
                rec_preds = predictions["recursive_predictions"]
                if i < len(rec_preds):
                    result["Forecasts"] = rec_preds[i]
                else:
                    result["Forecasts"] = [result.get("Forecast_1W", 0.0)] * 52
            except (IndexError, TypeError):
                result["Forecasts"] = [result.get("Forecast_1W", 0.0)] * 52
        else:
            # Fallback: 1 hetes elorejelzes ismetlese
            result["Forecasts"] = [result.get("Forecast_1W", 0.0)] * 52

        results.append(result)

    return results


# =============================================================================
# SUPPORTED MODELS - 100% Registry-alapu, NINCS hardcoded lista
# =============================================================================


def get_panel_mode_models() -> List[str]:
    """
    Visszaadja a Panel Mode-ot tamogato modellek listajat.

    A lista KIZAROLAG a ModelInfo.supports_panel_mode flag alapjan generalodik.
    NINCS fallback - ha egy modell nem implementalta, nem tamogatja.
    """
    from models import get_all_models, get_model_info
    return [
        name for name in get_all_models()
        if get_model_info(name) and get_model_info(name).supports_panel_mode
    ]


# Legacy alias - DINAMIKUS, nem hardcoded
PANEL_MODE_SUPPORTED_MODELS = get_panel_mode_models()
