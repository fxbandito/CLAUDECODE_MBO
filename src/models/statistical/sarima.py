"""
SARIMA Model
Seasonal ARIMA
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class SARIMAModel(BaseModel):
    """SARIMA - Seasonal ARIMA."""

    MODEL_INFO = ModelInfo(
        name="SARIMA",
        category="Statistical Models",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "p": "1",
        "d": "1",
        "q": "1",
        "P": "1",
        "D": "1",
        "Q": "1",
        "s": "12",
        "trend": "c",
    }

    PARAM_OPTIONS = {
        "p": ["0", "1", "2", "3", "4", "5"],
        "d": ["0", "1", "2"],
        "q": ["0", "1", "2", "3", "4", "5"],
        "P": ["0", "1", "2"],
        "D": ["0", "1"],
        "Q": ["0", "1", "2"],
        "s": ["4", "7", "12", "24", "52"],
        "trend": ["n", "c", "t", "ct"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """SARIMA előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("SARIMA forecast not yet implemented")
