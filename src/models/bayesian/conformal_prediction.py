"""
Conformal Prediction Model
Conformal Prediction for time series uncertainty
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class ConformalPredictionModel(BaseModel):
    """Conformal Prediction - Conformal Prediction for uncertainty."""

    MODEL_INFO = ModelInfo(
        name="Conformal Prediction",
        category="Bayesian & Probabilistic Methods",
        supports_gpu=True,
        supports_batch=True,
        gpu_threshold=200,
    )

    PARAM_DEFAULTS = {
        "base_model": "ridge",
        "significance": "0.1",
        "calibration_size": "0.2",
        "method": "aci",
    }

    PARAM_OPTIONS = {
        "base_model": ["ridge", "lasso", "rf", "xgb"],
        "significance": ["0.05", "0.1", "0.2"],
        "calibration_size": ["0.1", "0.2", "0.3"],
        "method": ["aci", "cqr", "naive"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Conformal Prediction előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Conformal Prediction forecast not yet implemented")
