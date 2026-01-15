"""
DLinear Model
Decomposition Linear for time series forecasting
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class DLinearModel(BaseModel):
    """DLinear - Decomposition Linear."""

    MODEL_INFO = ModelInfo(
        name="DLinear",
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
        "individual": "False",
        "moving_avg": "25",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128", "256"],
        "look_back": ["12", "24", "36", "48", "96"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "individual": ["True", "False"],
        "moving_avg": ["13", "25", "49"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """DLinear előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("DLinear forecast not yet implemented")
