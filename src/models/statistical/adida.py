"""
ADIDA Model
Aggregate-Disaggregate Intermittent Demand Approach
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class ADIDAModel(BaseModel):
    """ADIDA - Aggregate-Disaggregate Intermittent Demand Approach."""

    MODEL_INFO = ModelInfo(
        name="ADIDA",
        category="Statistical Models",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "aggregation_level": "4",
        "decomposition_method": "multiplicative",
    }

    PARAM_OPTIONS = {
        "aggregation_level": ["2", "3", "4", "5", "6", "8", "12"],
        "decomposition_method": ["additive", "multiplicative"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """ADIDA előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("ADIDA forecast not yet implemented")
