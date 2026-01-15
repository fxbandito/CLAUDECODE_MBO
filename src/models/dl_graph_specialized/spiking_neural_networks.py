"""
Spiking Neural Networks Model
Spiking Neural Networks for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class SpikingNeuralNetworksModel(BaseModel):
    """Spiking Neural Networks - SNN for time series."""

    MODEL_INFO = ModelInfo(
        name="Spiking Neural Networks",
        category="Deep Learning - Graph & Specialized Neural Networks",
        supports_gpu=True,
        supports_batch=True,
        gpu_threshold=200,
    )

    PARAM_DEFAULTS = {
        "epochs": "25",
        "batch_size": "64",
        "look_back": "24",
        "learning_rate": "0.001",
        "hidden_size": "128",
        "num_layers": "2",
        "threshold": "1.0",
        "tau": "2.0",
        "dropout": "0.1",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128", "256"],
        "look_back": ["12", "24", "36", "48"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "hidden_size": ["64", "128", "256"],
        "num_layers": ["1", "2", "3"],
        "threshold": ["0.5", "1.0", "1.5"],
        "tau": ["1.0", "2.0", "4.0"],
        "dropout": ["0.0", "0.1", "0.2"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Spiking Neural Networks előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Spiking Neural Networks forecast not yet implemented")
