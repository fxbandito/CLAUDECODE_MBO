"""
TDA Model
Topological Data Analysis for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class TDAModel(BaseModel):
    """TDA - Topological Data Analysis."""

    MODEL_INFO = ModelInfo(
        name="TDA",
        category="State Space & Other",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "embedding_dim": "3",
        "time_delay": "1",
        "window_size": "24",
        "max_homology_dim": "1",
    }

    PARAM_OPTIONS = {
        "embedding_dim": ["2", "3", "4", "5"],
        "time_delay": ["1", "2", "3", "5"],
        "window_size": ["12", "24", "36", "52"],
        "max_homology_dim": ["0", "1", "2"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """TDA előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("TDA forecast not yet implemented")
