"""
LightGBM Model
Light Gradient Boosting Machine for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class LightGBMModel(BaseModel):
    """LightGBM - Light Gradient Boosting Machine."""

    MODEL_INFO = ModelInfo(
        name="LightGBM",
        category="Classical Machine Learning",
        supports_gpu=True,
        supports_batch=True,
        gpu_threshold=500,
    )

    PARAM_DEFAULTS = {
        "n_estimators": "100",
        "learning_rate": "0.1",
        "max_depth": "-1",
        "num_leaves": "31",
        "min_child_samples": "20",
        "subsample": "1.0",
        "colsample_bytree": "1.0",
        "lags": "12",
    }

    PARAM_OPTIONS = {
        "n_estimators": ["50", "100", "200", "300", "500"],
        "learning_rate": ["0.01", "0.05", "0.1", "0.2"],
        "max_depth": ["-1", "3", "5", "7", "10"],
        "num_leaves": ["15", "31", "63", "127"],
        "min_child_samples": ["10", "20", "30", "50"],
        "subsample": ["0.7", "0.8", "0.9", "1.0"],
        "colsample_bytree": ["0.7", "0.8", "0.9", "1.0"],
        "lags": ["6", "12", "24", "36", "52"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """LightGBM előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("LightGBM forecast not yet implemented")
