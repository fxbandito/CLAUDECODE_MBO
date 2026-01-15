"""
VAR Model
Vector AutoRegression
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class VARModel(BaseModel):
    """VAR - Vector AutoRegression."""

    MODEL_INFO = ModelInfo(
        name="VAR",
        category="Statistical Models",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "maxlags": "5",
        "ic": "aic",
        "trend": "c",
    }

    PARAM_OPTIONS = {
        "maxlags": ["1", "2", "3", "5", "10", "15"],
        "ic": ["aic", "bic", "hqic", "fpe"],
        "trend": ["n", "c", "ct", "ctt"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """VAR előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("VAR forecast not yet implemented")
