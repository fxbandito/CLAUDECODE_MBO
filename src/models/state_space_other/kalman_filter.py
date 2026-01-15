"""
Kalman Filter Model
Kalman Filter for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class KalmanFilterModel(BaseModel):
    """Kalman Filter - Kalman Filter for time series."""

    MODEL_INFO = ModelInfo(
        name="Kalman Filter",
        category="State Space & Other",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "level": "True",
        "trend": "False",
        "seasonal": "None",
        "seasonal_periods": "12",
        "damped_trend": "False",
    }

    PARAM_OPTIONS = {
        "level": ["True", "False"],
        "trend": ["True", "False"],
        "seasonal": ["None", "12", "52"],
        "seasonal_periods": ["4", "7", "12", "52"],
        "damped_trend": ["True", "False"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Kalman Filter előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Kalman Filter forecast not yet implemented")
