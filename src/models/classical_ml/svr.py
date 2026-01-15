"""
SVR Model
Support Vector Regression for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class SVRModel(BaseModel):
    """SVR - Support Vector Regression."""

    MODEL_INFO = ModelInfo(
        name="SVR",
        category="Classical Machine Learning",
        supports_gpu=False,
        supports_batch=True,
    )

    PARAM_DEFAULTS = {
        "kernel": "rbf",
        "C": "1.0",
        "epsilon": "0.1",
        "gamma": "scale",
        "lags": "12",
    }

    PARAM_OPTIONS = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "C": ["0.1", "1.0", "10.0", "100.0"],
        "epsilon": ["0.01", "0.1", "0.2", "0.5"],
        "gamma": ["scale", "auto", "0.001", "0.01", "0.1"],
        "lags": ["6", "12", "24", "36", "52"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """SVR előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("SVR forecast not yet implemented")
