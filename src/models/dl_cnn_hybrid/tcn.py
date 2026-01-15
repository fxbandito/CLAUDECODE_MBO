"""
TCN Model
Temporal Convolutional Network for time series forecasting
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class TCNModel(BaseModel):
    """TCN - Temporal Convolutional Network."""

    MODEL_INFO = ModelInfo(
        name="TCN",
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
        "num_channels": "64,64,64",
        "kernel_size": "3",
        "dropout": "0.1",
        "dilation_base": "2",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128", "256"],
        "look_back": ["12", "24", "36", "48", "96"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "num_channels": ["32,32", "64,64", "64,64,64", "128,128,128"],
        "kernel_size": ["2", "3", "4", "5", "7"],
        "dropout": ["0.0", "0.1", "0.2", "0.3"],
        "dilation_base": ["2", "3", "4"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """TCN előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("TCN forecast not yet implemented")
