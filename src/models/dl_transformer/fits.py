"""
FiTS Model
Fitting Time Series with interpolation
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class FiTSModel(BaseModel):
    """FiTS - Fitting Time Series."""

    MODEL_INFO = ModelInfo(
        name="FiTS",
        category="Deep Learning - Transformer-based",
        supports_gpu=True,
        supports_batch=True,
        gpu_threshold=100,
    )

    PARAM_DEFAULTS = {
        "epochs": "25",
        "batch_size": "64",
        "look_back": "24",
        "learning_rate": "0.001",
        "cut_freq": "0",
        "individual": "False",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128", "256"],
        "look_back": ["12", "24", "36", "48", "96"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "cut_freq": ["0", "1", "2", "4", "8"],
        "individual": ["True", "False"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """FiTS előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("FiTS forecast not yet implemented")
