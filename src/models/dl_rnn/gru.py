"""
GRU Model
Gated Recurrent Unit for time series forecasting
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class GRUModel(BaseModel):
    """GRU - Gated Recurrent Unit."""

    MODEL_INFO = ModelInfo(
        name="GRU",
        category="Deep Learning - RNN-based",
        supports_gpu=True,
        supports_batch=True,
        gpu_threshold=250,
    )

    PARAM_DEFAULTS = {
        "epochs": "25",
        "batch_size": "64",
        "hidden_size": "64",
        "num_layers": "2",
        "dropout": "0.1",
        "look_back": "12",
        "learning_rate": "0.001",
        "bidirectional": "False",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128", "256"],
        "hidden_size": ["32", "64", "128", "256"],
        "num_layers": ["1", "2", "3", "4"],
        "dropout": ["0.0", "0.1", "0.2", "0.3"],
        "look_back": ["6", "12", "24", "36", "52"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "bidirectional": ["True", "False"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """GRU előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("GRU forecast not yet implemented")
