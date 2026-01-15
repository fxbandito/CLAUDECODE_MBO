"""
ES-RNN Model
Exponential Smoothing with Recurrent Neural Networks (Hybrid)
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class ESRNNModel(BaseModel):
    """ES-RNN - Exponential Smoothing with RNN (Hybrid)."""

    MODEL_INFO = ModelInfo(
        name="ES-RNN",
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
        "seasonality": "12",
        "dilations": "1,2,4,8",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128", "256"],
        "hidden_size": ["32", "64", "128", "256"],
        "num_layers": ["1", "2", "3"],
        "dropout": ["0.0", "0.1", "0.2", "0.3"],
        "look_back": ["6", "12", "24", "36", "52"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "seasonality": ["4", "7", "12", "24", "52"],
        "dilations": ["1,2", "1,2,4", "1,2,4,8", "1,2,4,8,16"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """ES-RNN előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("ES-RNN forecast not yet implemented")
