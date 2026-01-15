"""
Exponential Smoothing Model
Holt-Winters Exponential Smoothing
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class ExponentialSmoothingModel(BaseModel):
    """Exponential Smoothing - Holt-Winters method."""

    MODEL_INFO = ModelInfo(
        name="Exponential Smoothing",
        category="Smoothing & Decomposition",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "trend": "add",
        "seasonal": "add",
        "seasonal_periods": "12",
        "damped_trend": "False",
        "use_boxcox": "False",
    }

    PARAM_OPTIONS = {
        "trend": ["add", "mul", "None"],
        "seasonal": ["add", "mul", "None"],
        "seasonal_periods": ["4", "7", "12", "24", "52"],
        "damped_trend": ["True", "False"],
        "use_boxcox": ["True", "False"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Exponential Smoothing előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Exponential Smoothing forecast not yet implemented")
