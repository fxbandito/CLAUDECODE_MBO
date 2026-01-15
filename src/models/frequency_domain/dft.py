"""
DFT Model
Discrete Fourier Transform for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class DFTModel(BaseModel):
    """DFT - Discrete Fourier Transform."""

    MODEL_INFO = ModelInfo(
        name="DFT",
        category="Frequency Domain & Signal Processing",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "n_harmonics": "10",
        "detrend": "True",
    }

    PARAM_OPTIONS = {
        "n_harmonics": ["3", "5", "10", "15", "20", "30"],
        "detrend": ["True", "False"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """DFT előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("DFT forecast not yet implemented")
