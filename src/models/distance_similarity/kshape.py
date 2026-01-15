"""
k-Shape Model
k-Shape clustering for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class KShapeModel(BaseModel):
    """k-Shape - k-Shape clustering for time series."""

    MODEL_INFO = ModelInfo(
        name="k-Shape",
        category="Distance & Similarity-based",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "n_clusters": "3",
        "max_iter": "100",
        "n_init": "10",
        "look_back": "24",
    }

    PARAM_OPTIONS = {
        "n_clusters": ["2", "3", "4", "5", "8", "10"],
        "max_iter": ["50", "100", "200"],
        "n_init": ["5", "10", "20"],
        "look_back": ["12", "24", "36", "52"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """k-Shape előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("k-Shape forecast not yet implemented")
