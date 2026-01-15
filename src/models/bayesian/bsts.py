"""
BSTS Model
Bayesian Structural Time Series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class BSTSModel(BaseModel):
    """BSTS - Bayesian Structural Time Series."""

    MODEL_INFO = ModelInfo(
        name="BSTS",
        category="Bayesian & Probabilistic Methods",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "niter": "1000",
        "burn": "200",
        "seasonal_periods": "12",
        "trend": "True",
        "seasonal": "True",
    }

    PARAM_OPTIONS = {
        "niter": ["500", "1000", "2000", "5000"],
        "burn": ["100", "200", "500"],
        "seasonal_periods": ["4", "7", "12", "52"],
        "trend": ["True", "False"],
        "seasonal": ["True", "False"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """BSTS előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("BSTS forecast not yet implemented")
