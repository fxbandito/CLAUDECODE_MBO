"""
Multi-task Learning Model
Multi-task learning for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class MultiTaskLearningModel(BaseModel):
    """Multi-task Learning - Multi-task learning for time series."""

    MODEL_INFO = ModelInfo(
        name="Multi-task Learning",
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
        "shared_layers": "2",
        "task_layers": "1",
        "hidden_size": "64",
        "dropout": "0.1",
    }

    PARAM_OPTIONS = {
        "epochs": ["10", "15", "20", "25", "30", "50"],
        "batch_size": ["32", "64", "128", "256"],
        "look_back": ["12", "24", "36", "48"],
        "learning_rate": ["0.0001", "0.0005", "0.001", "0.005"],
        "shared_layers": ["1", "2", "3"],
        "task_layers": ["1", "2"],
        "hidden_size": ["32", "64", "128"],
        "dropout": ["0.0", "0.1", "0.2"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Multi-task Learning előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Multi-task Learning forecast not yet implemented")
