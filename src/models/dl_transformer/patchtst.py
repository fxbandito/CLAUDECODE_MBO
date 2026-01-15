"""
PatchTST Model
Patch Time Series Transformer
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class PatchTSTModel(BaseModel):
    """PatchTST - Patch Time Series Transformer."""

    MODEL_INFO = ModelInfo(
        name="PatchTST",
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
        "d_ff": "128",
        "patch_len": "16",
        "stride": "8",
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
        "d_ff": ["64", "128", "256", "512"],
        "patch_len": ["8", "16", "24", "32"],
        "stride": ["4", "8", "16"],
        "dropout": ["0.0", "0.1", "0.2", "0.3"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """PatchTST előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("PatchTST forecast not yet implemented")
