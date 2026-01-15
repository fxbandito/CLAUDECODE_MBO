"""
MoE Model
Mixture of Experts for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class MoEModel(BaseModel):
    """MoE - Mixture of Experts."""

    MODEL_INFO = ModelInfo(
        name="MoE",
        category="Meta-Learning & AutoML",
        supports_gpu=True,
        supports_batch=True,
        gpu_threshold=200,
    )

    PARAM_DEFAULTS = {
        "epochs": "25",
        "batch_size": "64",
        "look_back": "24",
        "learning_rate": "0.001",
        "num_experts": "4",
        "hidden_size": "64",
        "top_k": "2",
        "dropout": "0.1",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128", "256"],
        "look_back": ["12", "24", "36", "48"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "num_experts": ["2", "4", "6", "8"],
        "hidden_size": ["32", "64", "128"],
        "top_k": ["1", "2", "3"],
        "dropout": ["0.0", "0.1", "0.2"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """MoE előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("MoE forecast not yet implemented")
