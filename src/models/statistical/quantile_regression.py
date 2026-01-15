"""
Quantile Regression Model
Regression for quantiles of the response variable
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class QuantileRegressionModel(BaseModel):
    """Quantile Regression - Regression for quantiles of the response variable."""

    MODEL_INFO = ModelInfo(
        name="Quantile Regression",
        category="Statistical Models",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "quantile": "0.5",
        "max_iter": "1000",
    }

    PARAM_OPTIONS = {
        "quantile": ["0.1", "0.25", "0.5", "0.75", "0.9", "0.95"],
        "max_iter": ["500", "1000", "2000", "5000"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Quantile Regression előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Quantile Regression forecast not yet implemented")
