"""
Neural GAM Model
Neural Generalized Additive Model
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class NeuralGAMModel(BaseModel):
    """Neural GAM - Neural Generalized Additive Model."""

    MODEL_INFO = ModelInfo(
        name="Neural GAM",
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
        "hidden_size": "64",
        "num_layers": "2",
        "n_splines": "20",
        "dropout": "0.1",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128", "256"],
        "look_back": ["12", "24", "36", "48"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "hidden_size": ["32", "64", "128"],
        "num_layers": ["1", "2", "3"],
        "n_splines": ["10", "15", "20", "30"],
        "dropout": ["0.0", "0.1", "0.2"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Neural GAM előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Neural GAM forecast not yet implemented")
