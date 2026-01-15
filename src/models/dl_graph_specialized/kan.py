"""
KAN Model
Kolmogorov-Arnold Networks for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class KANModel(BaseModel):
    """KAN - Kolmogorov-Arnold Networks."""

    MODEL_INFO = ModelInfo(
        name="KAN",
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
        "hidden_layers": "64,64",
        "grid_size": "5",
        "spline_order": "3",
        "dropout": "0.1",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128", "256"],
        "look_back": ["12", "24", "36", "48"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "hidden_layers": ["32,32", "64,64", "64,64,64", "128,128"],
        "grid_size": ["3", "5", "7", "10"],
        "spline_order": ["2", "3", "4"],
        "dropout": ["0.0", "0.1", "0.2"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """KAN előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("KAN forecast not yet implemented")
