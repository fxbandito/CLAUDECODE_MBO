"""
ARIMA Model
AutoRegressive Integrated Moving Average
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class ARIMAModel(BaseModel):
    """ARIMA - AutoRegressive Integrated Moving Average."""

    MODEL_INFO = ModelInfo(
        name="ARIMA",
        category="Statistical Models",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "p": "1",
        "d": "1",
        "q": "1",
        "trend": "c",
    }

    PARAM_OPTIONS = {
        "p": ["0", "1", "2", "3", "4", "5"],
        "d": ["0", "1", "2"],
        "q": ["0", "1", "2", "3", "4", "5"],
        "trend": ["n", "c", "t", "ct"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """ARIMA előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("ARIMA forecast not yet implemented")
