"""
FFT Model
Fast Fourier Transform for time series
"""

from typing import List, Dict, Any
from models.base import BaseModel, ModelInfo


class FFTModel(BaseModel):
    """FFT - Fast Fourier Transform."""

    MODEL_INFO = ModelInfo(
        name="FFT",
        category="Frequency Domain & Signal Processing",
        supports_gpu=False,
        supports_batch=False,
    )

    PARAM_DEFAULTS = {
        "n_harmonics": "10",
        "detrend": "True",
        "window": "None",
    }

    PARAM_OPTIONS = {
        "n_harmonics": ["3", "5", "10", "15", "20", "30"],
        "detrend": ["True", "False"],
        "window": ["None", "hann", "hamming", "blackman"],
    }

    def forecast(self, data: List[float], steps: int, params: Dict[str, Any]) -> List[float]:
        """FFT előrejelzés."""
        # TODO: Implementáció
        raise NotImplementedError("FFT forecast not yet implemented")
