"""
Dual Mode Task - 100% Modell-független worker infrastruktúra.

Ez a modul KIZÁRÓLAG az általános dual mode infrastruktúrát biztosítja:
- Worker process környezet inicializálás
- Rekurzív előrejelzés algoritmus
- Task wrapper a multiprocessing-hez

A KONKRÉT MODELL IMPLEMENTÁCIÓK a models/ mappában találhatók.
Minden modell a saját create_dual_regressor() metódusát implementálja.

FONTOS: Ez a fájl NEM TARTALMAZ semmilyen modell-specifikus kódot!
"""

import os
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd


# =============================================================================
# WORKER PROCESS INICIALIZÁLÁS
# =============================================================================


def init_worker_environment() -> None:
    """
    Worker process környezet inicializálása.

    Általános beállítások minden modellhez:
    - GPU letiltása (CUDA konfliktusok elkerülése multiprocessing-ben)
    - Szál limitek beállítása
    - Numerikus könyvtárak konfigurálása
    """
    # GPU letiltása worker-ekben - megelőzi a CUDA context konfliktusokat
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["FAISS_OPT_LEVEL"] = "generic"

    # Single-thread mód numerikus könyvtárakhoz
    # A modellek saját maguk döntik el, hány szálat használnak
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Worker flag - modellek ellenőrizhetik
    os.environ["MBO_MP_WORKER"] = "1"

    # PyTorch konfiguráció (ha elérhető)
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass

    # Sklearn warning elnyomása
    warnings.filterwarnings(
        "ignore",
        message="Loky-backed parallel loops cannot be called"
    )


# =============================================================================
# REKURZÍV ELŐREJELZÉS - Általános algoritmus
# =============================================================================


# Lag oszlopok és history buffer indexek közötti megfeleltetés
# Ez az adatstruktúra standard a rendszerben
LAG_COLUMN_MAPPING = {
    "lag_profit_1": 0,
    "lag_profit_2": 1,
    "lag_profit_4": 3,
    "lag_profit_8": 7,
    "lag_profit_13": 12,
    "lag_profit_26": 25,
}

# History buffer mérete (elég nagy a lag_26-hoz + tartalék)
HISTORY_BUFFER_SIZE = 30


def apply_recursive_forecasting(
    model: Any,
    x_predict: pd.DataFrame,
    initial_predictions: np.ndarray,
    horizon: int,
    task_type: str = "regression"
) -> np.ndarray:
    """
    Rekurzív előrejelzés végrehajtása.

    Ez az algoritmus MODELL-FÜGGETLEN - bármely sklearn-kompatibilis
    regresszorral működik (fit() és predict() metódusok szükségesek).

    Az 1-lépéses modellt használva többlépéses előrejelzést készít,
    minden lépésnél frissítve a lag feature-öket.

    Args:
        model: Betanított modell (fit() már meghívva, predict() szükséges)
        x_predict: Predikciós feature-ök
        initial_predictions: Első lépés predikciói
        horizon: Előrejelzési horizont (hány lépés)
        task_type: "regression" vagy "activity"

    Returns:
        Shape: (n_strategies, horizon) - minden stratégiára minden időpontra
    """
    n_strategies = len(x_predict)
    current_x = x_predict.copy()
    week_predictions = [initial_predictions]

    # History buffer inicializálása lag értékekkel
    history = _init_history_buffer(x_predict, n_strategies)
    current_pred = initial_predictions

    # Rekurzív lépések (2. héttől a horizontig)
    for _ in range(1, horizon):
        # History frissítése: új predikció beszúrása az elejére
        history = np.roll(history, 1, axis=1)
        history[:, 0] = current_pred

        # Lag oszlopok frissítése
        _update_lag_columns(current_x, history)

        # Rolling statisztikák frissítése
        _update_rolling_stats(current_x, history)

        # Következő lépés predikciója
        next_pred = model.predict(current_x)

        if task_type == "activity":
            next_pred = np.clip(next_pred, 0, 1)

        week_predictions.append(next_pred)
        current_pred = next_pred

    # Transzponálás: (horizon, n_strategies) -> (n_strategies, horizon)
    return np.array(week_predictions).T


def _init_history_buffer(
    x_predict: pd.DataFrame,
    n_strategies: int
) -> np.ndarray:
    """
    History buffer inicializálása a lag oszlopokból.

    Args:
        x_predict: Feature DataFrame
        n_strategies: Stratégiák száma

    Returns:
        Shape: (n_strategies, HISTORY_BUFFER_SIZE)
    """
    history = np.zeros((n_strategies, HISTORY_BUFFER_SIZE))

    for col, idx in LAG_COLUMN_MAPPING.items():
        if col in x_predict.columns and idx < HISTORY_BUFFER_SIZE:
            history[:, idx] = x_predict[col].values

    return history


def _update_lag_columns(x_data: pd.DataFrame, history: np.ndarray) -> None:
    """
    Lag oszlopok frissítése a history buffer alapján.

    Args:
        x_data: Feature DataFrame (helyben módosul)
        history: History buffer
    """
    for col, idx in LAG_COLUMN_MAPPING.items():
        if col in x_data.columns and idx < HISTORY_BUFFER_SIZE:
            x_data[col] = history[:, idx]


def _update_rolling_stats(x_data: pd.DataFrame, history: np.ndarray) -> None:
    """
    Rolling statisztikák frissítése a history buffer alapján.

    Args:
        x_data: Feature DataFrame (helyben módosul)
        history: History buffer
    """
    if "lag_rolling_mean_4" in x_data.columns:
        x_data["lag_rolling_mean_4"] = np.mean(history[:, :4], axis=1)

    if "lag_rolling_mean_13" in x_data.columns:
        x_data["lag_rolling_mean_13"] = np.mean(history[:, :13], axis=1)

    if "lag_rolling_std_13" in x_data.columns:
        # Kis epsilon a nulla std elkerülésére
        x_data["lag_rolling_std_13"] = np.std(history[:, :13], axis=1) + 1e-6


# =============================================================================
# FŐ TASK FÜGGVÉNY (Multiprocessing worker)
# =============================================================================


def train_dual_model_task(
    model_name: str,
    x_train: pd.DataFrame,
    y: pd.Series,
    x_predict: pd.DataFrame,
    params: Dict[str, Any],
    task_type: str = "regression"
) -> Dict[str, Any]:
    """
    Dual mode modell tanítása worker process-ben.

    Ez a függvény fut a multiprocessing.Pool-ban.
    A modellt a models registry-ből tölti be és a create_dual_regressor()
    metódust hívja - NINCS FALLBACK, a modellnek implementálnia KELL.

    Args:
        model_name: Modell neve (pl. "LightGBM", "XGBoost")
        x_train: Tanító feature-ök
        y: Tanító célváltozó
        x_predict: Predikciós feature-ök
        params: Modell paraméterek (a modell értelmezi)
        task_type: "activity" vagy "regression"

    Returns:
        {target_name: predictions} vagy {target_name: None, "error": str}
    """
    # Worker környezet inicializálása
    init_worker_environment()

    target_name = y.name

    try:
        # Modell betöltése a registry-ből
        from models import get_model_class

        model_class = get_model_class(model_name)
        if model_class is None:
            return {
                target_name: None,
                "error": f"Model not found in registry: {model_name}"
            }

        # Modell instance létrehozása
        model_instance = model_class()

        # Ellenőrzés: támogatja-e a dual mode-ot?
        if not model_instance.MODEL_INFO.supports_dual_mode:
            return {
                target_name: None,
                "error": f"Model '{model_name}' does not support dual mode. "
                         f"Set supports_dual_mode=True in MODEL_INFO."
            }

        # Regresszor létrehozása - a MODELL dönti el a paramétereket!
        n_jobs = int(params.get("n_jobs", 1))
        regressor = model_instance.create_dual_regressor(params, n_jobs)

        # Tanítás
        regressor.fit(x_train, y)

        # Predikció
        preds = regressor.predict(x_predict)

        # Activity értékek klippelése [0, 1] közé
        if task_type == "activity":
            preds = np.clip(preds, 0, 1)

        # Rekurzív előrejelzés (ha kért)
        recursive_horizon = int(params.get("recursive_horizon", 0))
        if recursive_horizon > 0:
            recursive_preds = apply_recursive_forecasting(
                model=regressor,
                x_predict=x_predict,
                initial_predictions=preds,
                horizon=recursive_horizon,
                task_type=task_type
            )
            return {
                target_name: preds,
                "recursive_predictions": recursive_preds.tolist(),
            }

        return {target_name: preds}

    except NotImplementedError as e:
        return {
            target_name: None,
            "error": f"Model '{model_name}' must implement create_dual_regressor(): {e}"
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        return {target_name: None, "error": str(e)}
