"""
TimesNet Model
Temporal 2D-Variation Modeling for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class TimesNetModel(BaseModel):
    """TimesNet - Temporal 2D-Variation Modeling."""

    MODEL_INFO = ModelInfo(
        name="TimesNet",
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
        "d_model": "64",
        "d_ff": "64",
        "num_kernels": "6",
        "top_k": "3",
        "e_layers": "2",
        "dropout": "0.1",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128", "256"],
        "look_back": ["12", "24", "36", "48", "96"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "d_model": ["32", "64", "128"],
        "d_ff": ["32", "64", "128", "256"],
        "num_kernels": ["4", "6", "8"],
        "top_k": ["1", "2", "3", "5"],
        "e_layers": ["1", "2", "3"],
        "dropout": ["0.0", "0.1", "0.2", "0.3"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """TimesNet előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("TimesNet forecast not yet implemented")
