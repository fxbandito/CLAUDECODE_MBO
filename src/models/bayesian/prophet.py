"""
Prophet Model
Facebook Prophet for time series forecasting
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class ProphetModel(BaseModel):
    """Prophet - Facebook Prophet."""

    MODEL_INFO = ModelInfo(
        name="Prophet",
        category="Bayesian & Probabilistic Methods",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "seasonality_mode": "multiplicative",
        "yearly_seasonality": "auto",
        "weekly_seasonality": "auto",
        "daily_seasonality": "False",
        "changepoint_prior_scale": "0.05",
        "seasonality_prior_scale": "10",
        "interval_width": "0.8",
    }

    PARAM_OPTIONS = {
        "seasonality_mode": ["additive", "multiplicative"],
        "yearly_seasonality": ["auto", "True", "False"],
        "weekly_seasonality": ["auto", "True", "False"],
        "daily_seasonality": ["auto", "True", "False"],
        "changepoint_prior_scale": ["0.001", "0.01", "0.05", "0.1", "0.5"],
        "seasonality_prior_scale": ["0.01", "0.1", "1", "10"],
        "interval_width": ["0.8", "0.9", "0.95"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Prophet előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Prophet forecast not yet implemented")
