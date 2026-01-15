"""
VECM Model
Vector Error Correction Model
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class VECMModel(BaseModel):
    """VECM - Vector Error Correction Model."""

    MODEL_INFO = ModelInfo(
        name="VECM",
        category="Statistical Models",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "k_ar_diff": "1",
        "coint_rank": "1",
        "deterministic": "ci",
    }

    PARAM_OPTIONS = {
        "k_ar_diff": ["1", "2", "3", "4", "5"],
        "coint_rank": ["1", "2", "3"],
        "deterministic": ["nc", "co", "ci", "lo", "li"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """VECM előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("VECM forecast not yet implemented")
