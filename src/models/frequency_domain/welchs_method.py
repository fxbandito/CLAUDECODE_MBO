"""
Welchs Method Model
Welch's method for spectral density estimation
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class WelchsMethodModel(BaseModel):
    """Welchs Method - Welch's method for spectral density."""

    MODEL_INFO = ModelInfo(
        name="Welchs Method",
        category="Frequency Domain & Signal Processing",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "window": "hann",
        "nperseg": "256",
        "noverlap": "None",
        "detrend": "constant",
        "scaling": "density",
    }

    PARAM_OPTIONS = {
        "window": ["hann", "hamming", "blackman", "bartlett", "boxcar"],
        "nperseg": ["64", "128", "256", "512", "1024"],
        "noverlap": ["None", "32", "64", "128"],
        "detrend": ["constant", "linear", "False"],
        "scaling": ["density", "spectrum"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """Welchs Method előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("Welchs Method forecast not yet implemented")
