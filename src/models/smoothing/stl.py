"""
STL Model
Seasonal-Trend decomposition using LOESS
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class STLModel(BaseModel):
    """STL - Seasonal-Trend decomposition using LOESS."""

    MODEL_INFO = ModelInfo(
        name="STL",
        category="Smoothing & Decomposition",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "period": "12",
        "seasonal": "7",
        "trend": "None",
        "robust": "False",
    }

    PARAM_OPTIONS = {
        "period": ["4", "7", "12", "24", "52"],
        "seasonal": ["7", "11", "13", "15", "21"],
        "trend": ["None", "13", "23", "51"],
        "robust": ["True", "False"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """STL előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("STL forecast not yet implemented")
