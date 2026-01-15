"""
Auto-ARIMA Model
Automatic ARIMA model selection
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class AutoARIMAModel(BaseModel):
    """Auto-ARIMA - Automatic ARIMA model selection."""

    MODEL_INFO = ModelInfo(
        name="Auto-ARIMA",
        category="Statistical Models",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "max_p": "5",
        "max_d": "2",
        "max_q": "5",
        "seasonal": "False",
        "ic": "aic",
        "stepwise": "True",
    }

    PARAM_OPTIONS = {
        "max_p": ["3", "4", "5", "6", "7"],
        "max_d": ["1", "2", "3"],
        "max_q": ["3", "4", "5", "6", "7"],
        "seasonal": ["True", "False"],
        "ic": ["aic", "bic", "hqic"],
        "stepwise": ["True", "False"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Auto-ARIMA előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Auto-ARIMA forecast not yet implemented")
