"""
Periodogram Model
Periodogram for spectral analysis
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class PeriodogramModel(BaseModel):
    """Periodogram - Periodogram for spectral analysis."""

    MODEL_INFO = ModelInfo(
        name="Periodogram",
        category="Frequency Domain & Signal Processing",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "window": "hann",
        "detrend": "constant",
        "scaling": "density",
    }

    PARAM_OPTIONS = {
        "window": ["boxcar", "hann", "hamming", "blackman", "bartlett"],
        "detrend": ["constant", "linear", "False"],
        "scaling": ["density", "spectrum"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Periodogram előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Periodogram forecast not yet implemented")
