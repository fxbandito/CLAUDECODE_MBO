"""
Random Forest Model
Random Forest Regressor for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class RandomForestModel(BaseModel):
    """Random Forest - Random Forest Regressor."""

    MODEL_INFO = ModelInfo(
        name="Random Forest",
        category="Classical Machine Learning",
        supports_gpu=False,
        supports_batch=True,
    )

    PARAM_DEFAULTS = {
        "n_estimators": "100",
        "max_depth": "None",
        "min_samples_split": "2",
        "min_samples_leaf": "1",
        "max_features": "sqrt",
        "bootstrap": "True",
        "lags": "12",
    }

    PARAM_OPTIONS = {
        "n_estimators": ["50", "100", "200", "300", "500"],
        "max_depth": ["None", "5", "10", "15", "20"],
        "min_samples_split": ["2", "5", "10"],
        "min_samples_leaf": ["1", "2", "4"],
        "max_features": ["sqrt", "log2", "0.5", "1.0"],
        "bootstrap": ["True", "False"],
        "lags": ["6", "12", "24", "36", "52"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Random Forest előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Random Forest forecast not yet implemented")
