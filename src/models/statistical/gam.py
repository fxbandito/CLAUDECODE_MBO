"""
GAM Model
Generalized Additive Model
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class GAMModel(BaseModel):
    """GAM - Generalized Additive Model."""

    MODEL_INFO = ModelInfo(
        name="GAM",
        category="Statistical Models",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "n_splines": "20",
        "spline_order": "3",
        "lam": "0.6",
    }

    PARAM_OPTIONS = {
        "n_splines": ["10", "15", "20", "25", "30", "50"],
        "spline_order": ["2", "3", "4"],
        "lam": ["0.1", "0.3", "0.6", "1.0", "10.0"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """GAM előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("GAM forecast not yet implemented")
