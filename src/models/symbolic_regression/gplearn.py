"""
GPLearn Model
Genetic Programming for symbolic regression
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class GPLearnModel(BaseModel):
    """GPLearn - Genetic Programming for symbolic regression."""

    MODEL_INFO = ModelInfo(
        name="GPLearn",
        category="Symbolic Regression",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "generations": "20",
        "population_size": "1000",
        "tournament_size": "20",
        "parsimony_coefficient": "0.001",
        "function_set": "add,sub,mul,div",
        "look_back": "12",
    }

    PARAM_OPTIONS = {
        "generations": ["10", "20", "30", "50", "100"],
        "population_size": ["500", "1000", "2000", "5000"],
        "tournament_size": ["10", "20", "30"],
        "parsimony_coefficient": ["0.0001", "0.001", "0.01"],
        "function_set": ["add,sub,mul,div", "add,sub,mul,div,sqrt,log", "add,sub,mul,div,sin,cos"],
        "look_back": ["6", "12", "24", "36"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """GPLearn előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("GPLearn forecast not yet implemented")
