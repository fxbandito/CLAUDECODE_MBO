"""
DTW Model
Dynamic Time Warping for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class DTWModel(BaseModel):
    """DTW - Dynamic Time Warping."""

    MODEL_INFO = ModelInfo(
        name="DTW",
        category="Distance & Similarity-based",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "look_back": "24",
        "top_k": "5",
        "window": "None",
        "metric": "euclidean",
    }

    PARAM_OPTIONS = {
        "look_back": ["12", "24", "36", "52"],
        "top_k": ["1", "3", "5", "10"],
        "window": ["None", "5", "10", "20"],
        "metric": ["euclidean", "squared", "dtw"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """DTW előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("DTW forecast not yet implemented")
