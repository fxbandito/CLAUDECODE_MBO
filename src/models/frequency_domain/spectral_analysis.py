"""
Spectral Analysis Model
Spectral analysis for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class SpectralAnalysisModel(BaseModel):
    """Spectral Analysis - Spectral analysis for time series."""

    MODEL_INFO = ModelInfo(
        name="Spectral Analysis",
        category="Frequency Domain & Signal Processing",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "method": "welch",
        "window": "hann",
        "nperseg": "256",
        "noverlap": "None",
    }

    PARAM_OPTIONS = {
        "method": ["welch", "periodogram", "multitaper"],
        "window": ["hann", "hamming", "blackman", "bartlett"],
        "nperseg": ["64", "128", "256", "512"],
        "noverlap": ["None", "64", "128"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Spectral Analysis előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Spectral Analysis forecast not yet implemented")
