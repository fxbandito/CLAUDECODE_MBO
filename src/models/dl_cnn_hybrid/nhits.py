"""
N-HiTS Model
Neural Hierarchical Interpolation for Time Series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class NHiTSModel(BaseModel):
    """N-HiTS - Neural Hierarchical Interpolation."""

    MODEL_INFO = ModelInfo(
        name="N-HiTS",
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
        "num_stacks": "3",
        "num_blocks": "1",
        "hidden_size": "256",
        "pooling_kernel_sizes": "2,2,2",
        "downsample_frequencies": "4,2,1",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128", "256"],
        "look_back": ["12", "24", "36", "48", "96"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "num_stacks": ["2", "3", "4"],
        "num_blocks": ["1", "2", "3"],
        "hidden_size": ["128", "256", "512"],
        "pooling_kernel_sizes": ["2,2,2", "4,4,4", "2,4,8"],
        "downsample_frequencies": ["4,2,1", "8,4,1", "2,2,1"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """N-HiTS előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("N-HiTS forecast not yet implemented")
