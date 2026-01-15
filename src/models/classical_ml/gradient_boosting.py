"""
Gradient Boosting Model
Gradient Boosting Regressor for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class GradientBoostingModel(BaseModel):
    """Gradient Boosting - Gradient Boosting Regressor."""

    MODEL_INFO = ModelInfo(
        name="Gradient Boosting",
        category="Classical Machine Learning",
        supports_gpu=False,
        supports_batch=True,
    )

    PARAM_DEFAULTS = {
        "n_estimators": "100",
        "learning_rate": "0.1",
        "max_depth": "3",
        "min_samples_split": "2",
        "min_samples_leaf": "1",
        "subsample": "1.0",
        "lags": "12",
    }

    PARAM_OPTIONS = {
        "n_estimators": ["50", "100", "200", "300", "500"],
        "learning_rate": ["0.01", "0.05", "0.1", "0.2"],
        "max_depth": ["2", "3", "4", "5", "6", "8"],
        "min_samples_split": ["2", "5", "10"],
        "min_samples_leaf": ["1", "2", "4"],
        "subsample": ["0.7", "0.8", "0.9", "1.0"],
        "lags": ["6", "12", "24", "36", "52"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Gradient Boosting előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Gradient Boosting forecast not yet implemented")
