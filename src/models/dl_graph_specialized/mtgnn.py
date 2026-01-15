"""
MTGNN Model
Multivariate Time Series Graph Neural Network
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class MTGNNModel(BaseModel):
    """MTGNN - Multivariate Time Series Graph Neural Network."""

    MODEL_INFO = ModelInfo(
        name="MTGNN",
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
        "gcn_depth": "2",
        "num_nodes": "10",
        "hidden_dim": "32",
        "tanhalpha": "3",
        "dilation_exponential": "2",
        "layers": "3",
        "dropout": "0.3",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128"],
        "look_back": ["12", "24", "36", "48"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "gcn_depth": ["1", "2", "3"],
        "num_nodes": ["5", "10", "20"],
        "hidden_dim": ["16", "32", "64"],
        "tanhalpha": ["1", "3", "5"],
        "dilation_exponential": ["1", "2"],
        "layers": ["2", "3", "4"],
        "dropout": ["0.1", "0.2", "0.3"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """MTGNN előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("MTGNN forecast not yet implemented")
