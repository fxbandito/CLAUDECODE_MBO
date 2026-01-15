"""
OGARCH Model
Orthogonal GARCH
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class OGARCHModel(BaseModel):
    """OGARCH - Orthogonal GARCH."""

    MODEL_INFO = ModelInfo(
        name="OGARCH",
        category="Statistical Models",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "p": "1",
        "q": "1",
        "n_components": "3",
    }

    PARAM_OPTIONS = {
        "p": ["1", "2", "3"],
        "q": ["1", "2", "3"],
        "n_components": ["1", "2", "3", "4", "5"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """OGARCH előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("OGARCH forecast not yet implemented")
