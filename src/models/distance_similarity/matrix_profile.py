"""
Matrix Profile Model
Matrix Profile for time series motif discovery
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class MatrixProfileModel(BaseModel):
    """Matrix Profile - Matrix Profile for motif discovery."""

    MODEL_INFO = ModelInfo(
        name="Matrix Profile",
        category="Distance & Similarity-based",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "window_size": "24",
        "top_k": "3",
        "normalize": "True",
    }

    PARAM_OPTIONS = {
        "window_size": ["12", "24", "36", "52", "104"],
        "top_k": ["1", "3", "5", "10"],
        "normalize": ["True", "False"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Matrix Profile előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Matrix Profile forecast not yet implemented")
