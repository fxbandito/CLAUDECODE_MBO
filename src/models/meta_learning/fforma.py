"""
FFORMA Model
Feature-based Forecast Model Averaging
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class FFORMAModel(BaseModel):
    """FFORMA - Feature-based Forecast Model Averaging."""

    MODEL_INFO = ModelInfo(
        name="FFORMA",
        category="Meta-Learning & AutoML",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "n_estimators": "100",
        "learning_rate": "0.1",
        "max_depth": "6",
        "base_models": "arima,ets,theta",
    }

    PARAM_OPTIONS = {
        "n_estimators": ["50", "100", "200", "300"],
        "learning_rate": ["0.01", "0.05", "0.1", "0.2"],
        "max_depth": ["3", "4", "5", "6", "8"],
        "base_models": ["arima,ets", "arima,ets,theta", "arima,ets,theta,naive"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """FFORMA előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("FFORMA forecast not yet implemented")
