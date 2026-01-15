"""
MBO Trading Strategy Analyzer - BaseModel
Absztrakt alap osztály minden forecasting modelhez.

Minden model ÖNÁLLÓ egység - saját konfiggal, paraméterekkel.
Nincs közös config fájl, nincs keveredés!
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Model meta információk."""
    name: str
    category: str
    supports_gpu: bool = False
    supports_batch: bool = False
    gpu_threshold: int = 1000  # Minimum adat méret GPU használathoz


class BaseModel(ABC):
    """
    Absztrakt alap osztály minden forecasting modelhez.

    Minden leszármazott osztálynak definiálnia KELL:
    - MODEL_INFO: ModelInfo instance a model tulajdonságaival
    - PARAM_DEFAULTS: dict az alapértelmezett paraméterekkel
    - PARAM_OPTIONS: dict a választható paraméter értékekkel
    - forecast(): előrejelzés metódus

    Példa implementáció:

        class ARIMAModel(BaseModel):
            MODEL_INFO = ModelInfo(
                name="ARIMA",
                category="Statistical Models",
                supports_gpu=False,
                supports_batch=False,
            )

            PARAM_DEFAULTS = {
                "p": "1",
                "d": "1",
                "q": "1",
            }

            PARAM_OPTIONS = {
                "p": ["0", "1", "2", "3", "4", "5"],
                "d": ["0", "1", "2"],
                "q": ["0", "1", "2", "3", "4", "5"],
            }

            def forecast(self, data, steps, params):
                # implementáció
                pass
    """

    # Ezeket KELL definiálni minden leszármazottban
    MODEL_INFO: ModelInfo
    PARAM_DEFAULTS: Dict[str, str]
    PARAM_OPTIONS: Dict[str, List[str]]

    def __init__(self):
        """Inicializálás - ellenőrzi a kötelező attribútumokat."""
        self._validate_class_attributes()

    def _validate_class_attributes(self):
        """Ellenőrzi, hogy a leszármazott definiálta-e a kötelező attribútumokat."""
        if not hasattr(self, 'MODEL_INFO') or self.MODEL_INFO is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define MODEL_INFO"
            )
        if not hasattr(self, 'PARAM_DEFAULTS') or self.PARAM_DEFAULTS is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define PARAM_DEFAULTS"
            )
        if not hasattr(self, 'PARAM_OPTIONS') or self.PARAM_OPTIONS is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define PARAM_OPTIONS"
            )

    @abstractmethod
    def forecast(
        self,
        data: List[float],
        steps: int,
        params: Dict[str, Any]
    ) -> List[float]:
        """
        Előrejelzés készítése.

        Args:
            data: Bemeneti idősor adatok
            steps: Előrejelzési horizont (hány lépés előre)
            params: Paraméterek (a PARAM_DEFAULTS-ból vagy felhasználó által módosítva)

        Returns:
            Előrejelzett értékek listája (hossza = steps)
        """

    def forecast_batch(
        self,
        all_data: Dict[str, List[float]],
        steps: int,
        params: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """
        Batch előrejelzés több stratégiára egyszerre.

        Alapértelmezés: egyenként hívja a forecast()-ot.
        GPU-támogatott modellek felülírhatják hatékonyabb implementációval.

        Args:
            all_data: Dict ahol kulcs = stratégia név, érték = idősor
            steps: Előrejelzési horizont
            params: Paraméterek

        Returns:
            Dict ahol kulcs = stratégia név, érték = előrejelzés
        """
        if not self.MODEL_INFO.supports_batch:
            raise NotImplementedError(
                f"{self.MODEL_INFO.name} does not support batch mode"
            )

        # Alapértelmezett: egyenként feldolgozás
        results = {}
        for strategy_name, data in all_data.items():
            results[strategy_name] = self.forecast(data, steps, params)
        return results

    def get_device(self, data_size: int, use_gpu: bool = True):
        """
        Visszaadja a megfelelő device-t (CPU/GPU).

        A döntés a MODEL SAJÁT gpu_threshold értéke alapján történik.

        Args:
            data_size: Adat méret
            use_gpu: Felhasználó kéri-e a GPU használatot

        Returns:
            torch.device vagy "cpu" string
        """
        if not self.MODEL_INFO.supports_gpu:
            return "cpu"

        if not use_gpu:
            return "cpu"

        # Késleltetett import - csak ha tényleg kell
        try:
            import torch
            if torch.cuda.is_available() and data_size >= self.MODEL_INFO.gpu_threshold:
                return torch.device("cuda")
        except ImportError:
            pass

        return "cpu"

    def get_params_with_defaults(self, user_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Egyesíti a felhasználói paramétereket az alapértékekkel.

        Args:
            user_params: Felhasználó által megadott paraméterek (opcionális)

        Returns:
            Teljes paraméter dict (defaults + user overrides)
        """
        params = dict(self.PARAM_DEFAULTS)
        if user_params:
            params.update(user_params)
        return params

    @classmethod
    def get_info(cls) -> ModelInfo:
        """Visszaadja a model info-t (GUI-nak)."""
        return cls.MODEL_INFO

    @classmethod
    def get_defaults(cls) -> Dict[str, str]:
        """Visszaadja az alapértelmezett paramétereket (GUI-nak)."""
        return cls.PARAM_DEFAULTS.copy()

    @classmethod
    def get_options(cls) -> Dict[str, List[str]]:
        """Visszaadja a paraméter opciókat (GUI-nak)."""
        return cls.PARAM_OPTIONS.copy()

    def __repr__(self) -> str:
        return f"<{self.MODEL_INFO.name} ({self.MODEL_INFO.category})>"
