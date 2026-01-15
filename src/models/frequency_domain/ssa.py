"""
SSA Model
Singular Spectrum Analysis for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class SSAModel(BaseModel):
    """SSA - Singular Spectrum Analysis."""

    MODEL_INFO = ModelInfo(
        name="SSA",
        category="Frequency Domain & Signal Processing",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "window_size": "None",
        "n_components": "None",
        "groups": "None",
    }

    PARAM_OPTIONS = {
        "window_size": ["None", "12", "24", "52", "104"],
        "n_components": ["None", "3", "5", "10", "15"],
        "groups": ["None", "auto"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """SSA előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("SSA forecast not yet implemented")
