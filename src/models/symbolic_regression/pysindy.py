"""
PySindy Model
PySINDy - Sparse Identification of Nonlinear Dynamics
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class PySindyModel(BaseModel):
    """PySindy - Sparse Identification of Nonlinear Dynamics."""

    MODEL_INFO = ModelInfo(
        name="PySindy",
        category="Symbolic Regression",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "optimizer": "STLSQ",
        "threshold": "0.1",
        "alpha": "0.05",
        "max_iter": "20",
        "degree": "2",
        "look_back": "12",
    }

    PARAM_OPTIONS = {
        "optimizer": ["STLSQ", "SR3", "SSR", "FROLS"],
        "threshold": ["0.01", "0.05", "0.1", "0.2"],
        "alpha": ["0.01", "0.05", "0.1"],
        "max_iter": ["10", "20", "30", "50"],
        "degree": ["1", "2", "3", "4"],
        "look_back": ["6", "12", "24", "36"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """PySindy előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("PySindy forecast not yet implemented")
