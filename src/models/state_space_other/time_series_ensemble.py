"""
Time Series Ensemble Model
Ensemble of multiple time series models
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class TimeSeriesEnsembleModel(BaseModel):
    """Time Series Ensemble - Ensemble of multiple models."""

    MODEL_INFO = ModelInfo(
        name="Time Series Ensemble",
        category="State Space & Other",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "models": "arima,ets,theta",
        "combination": "mean",
        "weights": "equal",
    }

    PARAM_OPTIONS = {
        "models": ["arima,ets", "arima,ets,theta", "arima,ets,theta,prophet", "all"],
        "combination": ["mean", "median", "weighted", "best"],
        "weights": ["equal", "inverse_variance", "optimal"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Time Series Ensemble előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Time Series Ensemble forecast not yet implemented")
