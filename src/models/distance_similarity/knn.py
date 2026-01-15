"""
k-NN Model
k-Nearest Neighbors for time series (distance-based)
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class KNNModel(BaseModel):
    """k-NN - k-Nearest Neighbors (distance-based)."""

    MODEL_INFO = ModelInfo(
        name="k-NN",
        category="Distance & Similarity-based",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "look_back": "24",
        "n_neighbors": "5",
        "weights": "distance",
        "metric": "dtw",
    }

    PARAM_OPTIONS = {
        "look_back": ["12", "24", "36", "52"],
        "n_neighbors": ["1", "3", "5", "7", "10"],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "dtw", "manhattan"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """k-NN előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("k-NN forecast not yet implemented")
