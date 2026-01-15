"""
TiDE Model
Time-series Dense Encoder for time series forecasting
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class TiDEModel(BaseModel):
    """TiDE - Time-series Dense Encoder."""

    MODEL_INFO = ModelInfo(
        name="TiDE",
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
        "hidden_size": "256",
        "num_encoder_layers": "2",
        "num_decoder_layers": "2",
        "decoder_output_dim": "8",
        "dropout": "0.1",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128", "256"],
        "look_back": ["12", "24", "36", "48", "96"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "hidden_size": ["128", "256", "512"],
        "num_encoder_layers": ["1", "2", "3"],
        "num_decoder_layers": ["1", "2", "3"],
        "decoder_output_dim": ["4", "8", "16"],
        "dropout": ["0.0", "0.1", "0.2", "0.3"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """TiDE előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("TiDE forecast not yet implemented")
