"""
MSTL Model
Multiple Seasonal-Trend decomposition using LOESS
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class MSTLModel(BaseModel):
    """MSTL - Multiple Seasonal-Trend decomposition using LOESS."""

    MODEL_INFO = ModelInfo(
        name="MSTL",
        category="Smoothing & Decomposition",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "periods": "12,52",
        "stl_kwargs": "{}",
    }

    PARAM_OPTIONS = {
        "periods": ["7,365", "12,52", "24,168", "4,52"],
        "stl_kwargs": ["{}"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """MSTL előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("MSTL forecast not yet implemented")
