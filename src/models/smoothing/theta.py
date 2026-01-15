"""
Theta Model
Theta method for time series forecasting
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class ThetaModel(BaseModel):
    """Theta - Theta method for time series forecasting."""

    MODEL_INFO = ModelInfo(
        name="Theta",
        category="Smoothing & Decomposition",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "theta": "2.0",
        "deseasonalize": "True",
        "use_test": "False",
    }

    PARAM_OPTIONS = {
        "theta": ["0.5", "1.0", "2.0", "3.0"],
        "deseasonalize": ["True", "False"],
        "use_test": ["True", "False"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Theta előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Theta forecast not yet implemented")
