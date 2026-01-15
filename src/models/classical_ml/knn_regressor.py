"""
KNN Regressor Model
K-Nearest Neighbors Regressor for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class KNNRegressorModel(BaseModel):
    """KNN Regressor - K-Nearest Neighbors Regressor."""

    MODEL_INFO = ModelInfo(
        name="KNN Regressor",
        category="Classical Machine Learning",
        supports_gpu=False,
        supports_batch=True,
    )

    PARAM_DEFAULTS = {
        "n_neighbors": "5",
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": "30",
        "lags": "12",
    }

    PARAM_OPTIONS = {
        "n_neighbors": ["3", "5", "7", "10", "15", "20"],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": ["20", "30", "40", "50"],
        "lags": ["6", "12", "24", "36", "52"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """KNN Regressor előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("KNN Regressor forecast not yet implemented")
