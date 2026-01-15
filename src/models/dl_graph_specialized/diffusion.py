"""
Diffusion Model
Diffusion-based time series forecasting
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class DiffusionModel(BaseModel):
    """Diffusion - Diffusion-based forecasting."""

    MODEL_INFO = ModelInfo(
        name="Diffusion",
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
        "hidden_size": "64",
        "num_layers": "4",
        "diffusion_steps": "100",
        "beta_schedule": "linear",
        "dropout": "0.1",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128", "256"],
        "look_back": ["12", "24", "36", "48"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "hidden_size": ["32", "64", "128", "256"],
        "num_layers": ["2", "3", "4", "6"],
        "diffusion_steps": ["50", "100", "200", "500"],
        "beta_schedule": ["linear", "cosine", "quadratic"],
        "dropout": ["0.0", "0.1", "0.2"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Diffusion előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Diffusion forecast not yet implemented")
