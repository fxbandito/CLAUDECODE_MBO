"""
ETS Model
Error-Trend-Seasonality (Exponential Smoothing State Space Model)
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class ETSModel(BaseModel):
    """ETS - Error-Trend-Seasonality."""

    MODEL_INFO = ModelInfo(
        name="ETS",
        category="Smoothing & Decomposition",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "error": "add",
        "trend": "add",
        "seasonal": "add",
        "seasonal_periods": "12",
        "damped_trend": "False",
    }

    PARAM_OPTIONS = {
        "error": ["add", "mul"],
        "trend": ["add", "mul", "None"],
        "seasonal": ["add", "mul", "None"],
        "seasonal_periods": ["4", "7", "12", "24", "52"],
        "damped_trend": ["True", "False"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """ETS előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("ETS forecast not yet implemented")
