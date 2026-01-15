"""
Monte Carlo Model
Monte Carlo simulation for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class MonteCarloModel(BaseModel):
    """Monte Carlo - Monte Carlo simulation."""

    MODEL_INFO = ModelInfo(
        name="Monte Carlo",
        category="Bayesian & Probabilistic Methods",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "n_simulations": "1000",
        "method": "bootstrap",
        "block_size": "10",
        "seed": "42",
    }

    PARAM_OPTIONS = {
        "n_simulations": ["100", "500", "1000", "5000", "10000"],
        "method": ["bootstrap", "parametric", "block_bootstrap"],
        "block_size": ["5", "10", "20", "30"],
        "seed": ["42", "123", "None"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Monte Carlo előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Monte Carlo forecast not yet implemented")
