"""
TFT Model
Temporal Fusion Transformer for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class TFTModel(BaseModel):
    """TFT - Temporal Fusion Transformer."""

    MODEL_INFO = ModelInfo(
        name="TFT",
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
        "hidden_size": "64",
        "lstm_layers": "2",
        "attention_head_size": "4",
        "dropout": "0.1",
        "hidden_continuous_size": "8",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128", "256"],
        "look_back": ["12", "24", "36", "48", "96"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "hidden_size": ["32", "64", "128", "256"],
        "lstm_layers": ["1", "2", "3"],
        "attention_head_size": ["1", "2", "4"],
        "dropout": ["0.0", "0.1", "0.2", "0.3"],
        "hidden_continuous_size": ["4", "8", "16"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """TFT előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("TFT forecast not yet implemented")
