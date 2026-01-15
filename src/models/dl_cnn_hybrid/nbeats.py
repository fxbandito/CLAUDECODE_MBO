"""
N-BEATS Model
Neural Basis Expansion Analysis for Time Series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class NBEATSModel(BaseModel):
    """N-BEATS - Neural Basis Expansion Analysis."""

    MODEL_INFO = ModelInfo(
        name="N-BEATS",
        category="Deep Learning - CNN & Hybrid Architectures",
        supports_gpu=True,
        supports_batch=True,
        gpu_threshold=100,
    )

    PARAM_DEFAULTS = {
        "epochs": "25",
        "batch_size": "64",
        "look_back": "24",
        "learning_rate": "0.001",
        "num_stacks": "2",
        "num_blocks": "3",
        "hidden_size": "256",
        "theta_dims": "4",
        "share_weights": "False",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128", "256"],
        "look_back": ["12", "24", "36", "48", "96"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "num_stacks": ["1", "2", "3"],
        "num_blocks": ["1", "2", "3", "4"],
        "hidden_size": ["128", "256", "512"],
        "theta_dims": ["2", "4", "8"],
        "share_weights": ["True", "False"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """N-BEATS előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("N-BEATS forecast not yet implemented")
