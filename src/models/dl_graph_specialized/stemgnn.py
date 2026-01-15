"""
StemGNN Model
Spectral Temporal Graph Neural Network
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class StemGNNModel(BaseModel):
    """StemGNN - Spectral Temporal Graph Neural Network."""

    MODEL_INFO = ModelInfo(
        name="StemGNN",
        category="Deep Learning - Graph & Specialized Neural Networks",
        supports_gpu=True,
        supports_batch=True,
        gpu_threshold=100,
    )

    PARAM_DEFAULTS = {
        "epochs": "25",
        "batch_size": "64",
        "look_back": "24",
        "learning_rate": "0.001",
        "hidden_size": "64",
        "num_layers": "2",
        "stack_cnt": "2",
        "multi_layer": "5",
        "dropout": "0.1",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128"],
        "look_back": ["12", "24", "36", "48"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "hidden_size": ["32", "64", "128"],
        "num_layers": ["1", "2", "3"],
        "stack_cnt": ["1", "2", "3"],
        "multi_layer": ["3", "5", "7"],
        "dropout": ["0.0", "0.1", "0.2", "0.3"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """StemGNN előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("StemGNN forecast not yet implemented")
