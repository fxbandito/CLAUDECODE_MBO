"""
GARCH Family Model
Generalized AutoRegressive Conditional Heteroskedasticity
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class GARCHFamilyModel(BaseModel):
    """GARCH Family - Generalized AutoRegressive Conditional Heteroskedasticity."""

    MODEL_INFO = ModelInfo(
        name="GARCH Family",
        category="Statistical Models",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "p": "1",
        "q": "1",
        "model_type": "GARCH",
        "dist": "normal",
    }

    PARAM_OPTIONS = {
        "p": ["1", "2", "3"],
        "q": ["1", "2", "3"],
        "model_type": ["GARCH", "EGARCH", "GJR-GARCH", "TGARCH", "FIGARCH"],
        "dist": ["normal", "t", "skewt", "ged"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """GARCH Family előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("GARCH Family forecast not yet implemented")
