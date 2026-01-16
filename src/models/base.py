"""
MBO Trading Strategy Analyzer - BaseModel
Absztrakt alap osztály minden forecasting modelhez.

Minden model ÖNÁLLÓ egység - saját konfiggal, paraméterekkel.
Nincs közös config fájl, nincs keveredés!

Elérhető utility függvények a modell implementációkhoz:
=========================================================

Process Management (src/analysis/process_utils.py):
    from analysis.process_utils import (
        cleanup_cuda_context,        # CUDA memória cleanup
        force_kill_child_processes,  # Worker leállítás
        init_worker_environment,     # Worker thread/env config
        set_process_priority,        # Process prioritás
    )

Data Utilities (src/data/processor.py):
    from data.processor import DataProcessor

    data_mode = DataProcessor.detect_data_mode(df)
    # Visszatérési értékek: "original", "forward", "rolling"

    groups = DataProcessor.group_strategies(df)
    # O(1) stratégia lookup batch feldolgozáshoz

Resource Manager (src/analysis/engine.py):
    from analysis.engine import get_resource_manager

    manager = get_resource_manager()
    n_jobs = manager.get_n_jobs()      # CPU slider alapján
    device = manager.get_device()      # "cuda" vagy "cpu"
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Model meta információk."""
    name: str
    category: str
    supports_gpu: bool = False
    supports_batch: bool = False
    gpu_threshold: int = 1000  # Minimum adat méret GPU használathoz

    # Feature mode támogatás
    supports_forward_calc: bool = True  # Támogatja-e a Forward Calc módot
    supports_rolling_window: bool = True  # Támogatja-e a Rolling Window módot

    # Speciális módok
    supports_panel_mode: bool = False  # Panel Mode (egy modell minden stratégiára)
    supports_dual_mode: bool = False  # Dual Model (activity + profit külön)


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

        Alapértelmezés: joblib.Parallel párhuzamos feldolgozás.
        GPU-támogatott modellek felülírhatják hatékonyabb implementációval.

        Args:
            all_data: Dict ahol kulcs = stratégia név, érték = idősor
            steps: Előrejelzési horizont
            params: Paraméterek

        Returns:
            Dict ahol kulcs = stratégia név, érték = előrejelzés

        Példa egyedi batch implementációra:
            def forecast_batch(self, all_data, steps, params):
                # GPU batch ha sok stratégia
                if len(all_data) > 100 and self._use_gpu(params):
                    return self._gpu_batch_forecast(all_data, steps, params)
                # Egyébként default
                return super().forecast_batch(all_data, steps, params)
        """
        if not self.MODEL_INFO.supports_batch:
            raise NotImplementedError(
                f"{self.MODEL_INFO.name} does not support batch mode"
            )

        # Kevés stratégia esetén szekvenciális (overhead elkerülése)
        if len(all_data) <= 3:
            results = {}
            for strategy_name, data in all_data.items():
                results[strategy_name] = self.forecast(data, steps, params)
            return results

        # Párhuzamos feldolgozás joblib-bal
        try:
            from joblib import Parallel, delayed

            # ResourceManager-től kérjük a CPU beállítást
            try:
                from analysis.engine import get_resource_manager
                manager = get_resource_manager()
                max_jobs = manager.get_n_jobs()
            except ImportError:
                max_jobs = os.cpu_count() or 4

            n_jobs = min(max_jobs, len(all_data), 8)

            def _process_single(name: str, data: List[float]):
                return name, self.forecast(data, steps, params)

            results_list = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_process_single)(name, data)
                for name, data in all_data.items()
            )

            return dict(results_list)

        except ImportError:
            # Fallback: szekvenciális ha nincs joblib
            logger.warning("joblib not available, using sequential processing")
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

    # =========================================================================
    # HELPER METÓDUSOK - GPU és Memória kezelés
    # =========================================================================

    def cleanup_after_batch(self) -> None:
        """
        Cleanup hívás batch feldolgozás után.

        Felszabadítja a CUDA memóriát és futtat GC-t.
        Hívd meg hosszú batch műveletek után!

        Példa:
            def forecast_batch(self, all_data, steps, params):
                try:
                    results = self._gpu_batch_process(all_data, steps, params)
                    return results
                finally:
                    self.cleanup_after_batch()
        """
        import gc
        gc.collect()

        try:
            from analysis.process_utils import cleanup_cuda_context
            cleanup_cuda_context()
        except ImportError:
            # process_utils nem elérhető - próbáljuk közvetlenül
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def get_n_jobs(self, max_limit: int = 8) -> int:
        """
        Ajánlott párhuzamos job szám lekérése.

        A ResourceManager CPU slider beállítása alapján.

        Args:
            max_limit: Maximum job szám (default: 8)

        Returns:
            int: Ajánlott job szám
        """
        try:
            from analysis.engine import get_resource_manager
            manager = get_resource_manager()
            return min(manager.get_n_jobs(), max_limit)
        except ImportError:
            return min(os.cpu_count() or 4, max_limit)

    def should_use_gpu(self, data_size: int, params: Dict[str, Any]) -> bool:
        """
        Eldönti, hogy GPU-t használjon-e.

        Args:
            data_size: Adat méret
            params: Paraméterek (use_gpu kulcs)

        Returns:
            bool: True ha GPU ajánlott
        """
        if not self.MODEL_INFO.supports_gpu:
            return False

        use_gpu = params.get("use_gpu", True)
        if isinstance(use_gpu, str):
            use_gpu = use_gpu.lower() == "true"

        if not use_gpu:
            return False

        if data_size < self.MODEL_INFO.gpu_threshold:
            return False

        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # =========================================================================
    # DUAL MODE TÁMOGATÁS (Opcionális - csak ha supports_dual_mode = True)
    # =========================================================================

    def create_dual_regressor(
        self,
        params: Dict[str, Any],
        n_jobs: int = 1
    ) -> Any:
        """
        Regresszor létrehozása dual mode-hoz.

        Csak azok a modellek implementálják, amelyek támogatják a dual mode-ot.
        A visszaadott objektumnak rendelkeznie kell fit() és predict() metódusokkal.

        Args:
            params: Modell paraméterek
            n_jobs: Párhuzamos szálak száma

        Returns:
            sklearn-kompatibilis regresszor objektum

        Raises:
            NotImplementedError: Ha a modell nem támogatja a dual mode-ot
        """
        raise NotImplementedError(
            f"{self.MODEL_INFO.name} does not support dual mode. "
            f"Set supports_dual_mode=True and implement create_dual_regressor()."
        )
