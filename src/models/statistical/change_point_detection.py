"""
Change Point Detection Model
Detects structural changes in time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class ChangePointDetectionModel(BaseModel):
    """Change Point Detection - Detects structural changes in time series."""

    MODEL_INFO = ModelInfo(
        name="Change Point Detection",
        category="Statistical Models",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "method": "pelt",
        "penalty": "bic",
        "min_size": "2",
    }

    PARAM_OPTIONS = {
        "method": ["pelt", "binseg", "bottomup", "window"],
        "penalty": ["bic", "aic", "mbic", "hannan"],
        "min_size": ["1", "2", "3", "5", "10"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Change Point Detection előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Change Point Detection forecast not yet implemented")
