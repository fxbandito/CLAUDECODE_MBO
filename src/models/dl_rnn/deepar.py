"""
DeepAR Model
Deep AutoRegressive probabilistic forecasting
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class DeepARModel(BaseModel):
    """DeepAR - Deep AutoRegressive probabilistic forecasting."""

    MODEL_INFO = ModelInfo(
        name="DeepAR",
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
        "cell_type": "LSTM",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128", "256"],
        "hidden_size": ["32", "64", "128", "256"],
        "num_layers": ["1", "2", "3", "4"],
        "dropout": ["0.0", "0.1", "0.2", "0.3"],
        "look_back": ["6", "12", "24", "36", "52"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "cell_type": ["LSTM", "GRU"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """DeepAR előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("DeepAR forecast not yet implemented")
