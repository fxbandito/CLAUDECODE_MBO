"""
State Space Model
State Space Model for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class StateSpaceModelModel(BaseModel):
    """State Space Model - State Space Model for time series."""

    MODEL_INFO = ModelInfo(
        name="State Space Model",
        category="State Space & Other",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "level": "True",
        "trend": "False",
        "seasonal": "None",
        "ar_order": "0",
        "ma_order": "0",
    }

    PARAM_OPTIONS = {
        "level": ["True", "False"],
        "trend": ["True", "False"],
        "seasonal": ["None", "12", "52"],
        "ar_order": ["0", "1", "2", "3"],
        "ma_order": ["0", "1", "2", "3"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """State Space Model előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("State Space Model forecast not yet implemented")
