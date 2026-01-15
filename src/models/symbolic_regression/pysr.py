"""
PySR Model
PySR - High-Performance Symbolic Regression
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class PySRModel(BaseModel):
    """PySR - High-Performance Symbolic Regression."""

    MODEL_INFO = ModelInfo(
        name="PySR",
        category="Symbolic Regression",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "niterations": "40",
        "populations": "15",
        "population_size": "33",
        "maxsize": "20",
        "binary_operators": "+,-,*,/",
        "unary_operators": "sin,cos,exp,log",
        "look_back": "12",
    }

    PARAM_OPTIONS = {
        "niterations": ["20", "40", "60", "100"],
        "populations": ["10", "15", "20", "30"],
        "population_size": ["20", "33", "50"],
        "maxsize": ["10", "15", "20", "30"],
        "binary_operators": ["+,-,*,/", "+,-,*,/,^"],
        "unary_operators": ["", "sin,cos", "sin,cos,exp,log", "sin,cos,exp,log,sqrt"],
        "look_back": ["6", "12", "24", "36"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """PySR előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("PySR forecast not yet implemented")
