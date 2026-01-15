"""
Gaussian Process Model
Gaussian Process Regression for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class GaussianProcessModel(BaseModel):
    """Gaussian Process - Gaussian Process Regression."""

    MODEL_INFO = ModelInfo(
        name="Gaussian Process",
        category="Bayesian & Probabilistic Methods",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "kernel": "rbf",
        "alpha": "1e-10",
        "n_restarts": "3",
        "normalize_y": "True",
    }

    PARAM_OPTIONS = {
        "kernel": ["rbf", "matern", "rational_quadratic", "exp_sine_squared"],
        "alpha": ["1e-12", "1e-10", "1e-8", "1e-5"],
        "n_restarts": ["0", "1", "3", "5"],
        "normalize_y": ["True", "False"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Gaussian Process előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Gaussian Process forecast not yet implemented")
