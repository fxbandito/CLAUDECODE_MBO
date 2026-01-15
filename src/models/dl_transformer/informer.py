"""
Informer Model
Beyond Efficient Transformer for Long Sequence Time Series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class InformerModel(BaseModel):
    """Informer - Beyond Efficient Transformer."""

    MODEL_INFO = ModelInfo(
        name="Informer",
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
        "d_model": "64",
        "n_heads": "4",
        "e_layers": "2",
        "d_layers": "1",
        "d_ff": "128",
        "factor": "5",
        "attn": "prob",
        "distil": "True",
        "dropout": "0.1",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128", "256"],
        "look_back": ["12", "24", "36", "48", "96"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "d_model": ["32", "64", "128", "256"],
        "n_heads": ["2", "4", "8"],
        "e_layers": ["1", "2", "3"],
        "d_layers": ["1", "2"],
        "d_ff": ["64", "128", "256", "512"],
        "factor": ["3", "5", "7"],
        "attn": ["prob", "full"],
        "distil": ["True", "False"],
        "dropout": ["0.0", "0.1", "0.2", "0.3"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Informer előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Informer forecast not yet implemented")
