"""
XGBoost Model
Extreme Gradient Boosting for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class XGBoostModel(BaseModel):
    """XGBoost - Extreme Gradient Boosting."""

    MODEL_INFO = ModelInfo(
        name="XGBoost",
        category="Classical Machine Learning",
        supports_gpu=True,
        supports_batch=True,
        gpu_threshold=500,
    )

    PARAM_DEFAULTS = {
        "n_estimators": "100",
        "learning_rate": "0.1",
        "max_depth": "6",
        "min_child_weight": "1",
        "subsample": "1.0",
        "colsample_bytree": "1.0",
        "gamma": "0",
        "lags": "12",
    }

    PARAM_OPTIONS = {
        "n_estimators": ["50", "100", "200", "300", "500"],
        "learning_rate": ["0.01", "0.05", "0.1", "0.2", "0.3"],
        "max_depth": ["3", "4", "5", "6", "8", "10"],
        "min_child_weight": ["1", "3", "5", "7"],
        "subsample": ["0.6", "0.7", "0.8", "0.9", "1.0"],
        "colsample_bytree": ["0.6", "0.7", "0.8", "0.9", "1.0"],
        "gamma": ["0", "0.1", "0.2", "0.5"],
        "lags": ["6", "12", "24", "36", "52"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """XGBoost előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("XGBoost forecast not yet implemented")
