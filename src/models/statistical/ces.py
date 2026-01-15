"""
CES Model
Complex Exponential Smoothing
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class CESModel(BaseModel):
    """CES - Complex Exponential Smoothing."""

    MODEL_INFO = ModelInfo(
        name="CES",
        category="Statistical Models",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "seasonality": "none",
        "seasonal_periods": "12",
    }

    PARAM_OPTIONS = {
        "seasonality": ["none", "simple", "partial", "full"],
        "seasonal_periods": ["4", "7", "12", "24", "52"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """CES előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("CES forecast not yet implemented")
