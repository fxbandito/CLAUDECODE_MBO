"""
Process Utility Functions - Worker és Process Management.

Párhuzamos végrehajtáshoz szükséges segédfüggvények:
- Worker process cleanup
- CUDA context reset
- Child process management
"""

import gc
import logging
import os
from typing import Optional, Set

logger = logging.getLogger(__name__)

# psutil opcionális - graceful fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.debug("psutil not available - process management limited")


def cleanup_cuda_context() -> None:
    """
    CUDA context cleanup - memória felszabadítás és szinkronizáció.

    Használat:
        - Model futtatás után
        - Shutdown előtt
        - Pool restart között
    """
    try:
        import torch  # pylint: disable=import-outside-toplevel
        if not torch.cuda.is_available():
            return

        # Szinkronizálás - várakozás a futó műveletekre
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass  # Context már korrupt lehet

        # Cache ürítés
        torch.cuda.empty_cache()

        # GC a Python objektumokhoz
        gc.collect()

        # Második cache ürítés a GC által felszabadított memóriához
        torch.cuda.empty_cache()

        logger.debug("CUDA context cleanup completed")

    except ImportError:
        pass  # torch nem elérhető
    except (RuntimeError, AttributeError) as e:
        logger.debug("CUDA cleanup warning: %s", e)


def force_kill_child_processes(protected_pids: Optional[Set[int]] = None) -> int:
    """
    Child processek erőltetett leállítása.

    Használat:
        - AnalysisEngine shutdown
        - Pool terminate után cleanup
        - Zombie processek eltávolítása

    Args:
        protected_pids: Védett PID-ek halmaza (nem lesznek kilőve)

    Returns:
        int: Leállított processek száma
    """
    if not PSUTIL_AVAILABLE:
        logger.debug("psutil not available, skipping process cleanup")
        return 0

    protected = protected_pids or set()
    killed_count = 0

    try:
        current = psutil.Process(os.getpid())
        children = current.children(recursive=True)

        if not children:
            return 0

        # Védett processek kiszűrése
        to_kill = [c for c in children if c.pid not in protected]

        if not to_kill:
            logger.debug("All %d children are protected", len(children))
            return 0

        skipped = len(children) - len(to_kill)
        logger.debug(
            "Terminating %d child processes (skipping %d protected)",
            len(to_kill), skipped
        )

        # 1. Graceful terminate
        for child in to_kill:
            try:
                if child.is_running():
                    child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # 2. Várakozás graceful exit-re
        gone, alive = psutil.wait_procs(to_kill, timeout=2)
        killed_count = len(gone)

        # 3. Force kill a túlélőkre
        for child in alive:
            try:
                logger.warning("Force killing stubborn process PID %d", child.pid)
                child.kill()
                killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # 4. Végső várakozás
        if alive:
            psutil.wait_procs(alive, timeout=1)

        # 5. Ellenőrzés
        remaining = [
            p for p in current.children(recursive=True)
            if p.pid not in protected
        ]

        if remaining:
            logger.warning("Still %d children after cleanup", len(remaining))
        else:
            logger.debug("All child processes terminated")

    except (OSError, psutil.Error) as e:
        logger.error("Process cleanup error: %s", e)

    return killed_count


def init_worker_environment(n_threads: int = 1) -> None:
    """
    Worker process környezet inicializálása.

    KÖZPONTI függvény minden multiprocessing worker-hez.
    Korlátozza a thread számot, letiltja a GPU-t és konfigurálja
    a numerikus könyvtárakat a biztonságos párhuzamos futtatáshoz.

    Args:
        n_threads: Megengedett thread szám (default: 1)

    Használat:
        - multiprocessing.Pool initializer-ként
        - Bármely worker process indulásakor

    Miért fontos:
        - Megakadályozza a "halálspirált" (túl sok thread)
        - Elkerüli a CUDA context konfliktusokat
        - Megelőzi a FAISS AVX2/CUDA ütközéseket
    """
    import warnings  # pylint: disable=import-outside-toplevel

    threads_str = str(n_threads)

    # =========================================================================
    # 1. GPU LETILTÁS - KRITIKUS a multiprocessing stabilitáshoz
    # =========================================================================
    # Worker-ek NEM használhatnak GPU-t - elkerüli:
    # - CUDA context konfliktusokat
    # - Access violation hibákat
    # - Memória fragmentációt több CUDA context-ből
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # FAISS: generic CPU implementáció - elkerüli AVX2/CUDA konfliktusokat
    os.environ["FAISS_OPT_LEVEL"] = "generic"

    # =========================================================================
    # 2. THREAD LIMITEK - Megelőzi a túlterhelést
    # =========================================================================
    # Single-thread mód numerikus könyvtárakhoz
    # 12 worker × 12 thread = 144 thread → rendszer lefagyás
    # 12 worker × 1 thread = 12 thread → stabil futás
    os.environ["OMP_NUM_THREADS"] = threads_str
    os.environ["MKL_NUM_THREADS"] = threads_str
    os.environ["OPENBLAS_NUM_THREADS"] = threads_str
    os.environ["VECLIB_MAXIMUM_THREADS"] = threads_str
    os.environ["NUMEXPR_NUM_THREADS"] = threads_str

    # =========================================================================
    # 3. WORKER JELÖLÉS - Más modulok ellenőrizhetik
    # =========================================================================
    os.environ["MBO_MP_WORKER"] = "1"

    # =========================================================================
    # 4. PYTORCH KONFIGURÁCIÓ
    # =========================================================================
    try:
        import torch  # pylint: disable=import-outside-toplevel
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(n_threads)
    except ImportError:
        pass
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.debug("torch thread config error: %s", e)

    # =========================================================================
    # 5. WARNING ELNYOMÁS - Tisztább logok
    # =========================================================================
    # Sklearn loky warning elnyomása (nested parallelism)
    warnings.filterwarnings(
        "ignore",
        message="Loky-backed parallel loops cannot be called"
    )

    logger.debug("Worker initialized: %d threads, GPU disabled, FAISS generic", n_threads)


def set_process_priority(priority: str = "below_normal") -> bool:
    """
    Process prioritás beállítása.

    Args:
        priority: "idle", "below_normal", "normal", "above_normal"

    Returns:
        bool: Sikeres volt-e
    """
    if not PSUTIL_AVAILABLE:
        return False

    try:
        import sys  # pylint: disable=import-outside-toplevel
        p = psutil.Process()

        if sys.platform == "win32":
            priority_map = {
                "idle": psutil.IDLE_PRIORITY_CLASS,
                "below_normal": psutil.BELOW_NORMAL_PRIORITY_CLASS,
                "normal": psutil.NORMAL_PRIORITY_CLASS,
                "above_normal": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
            }
            p.nice(priority_map.get(priority, psutil.BELOW_NORMAL_PRIORITY_CLASS))
        else:
            # Unix: nice values (higher = lower priority)
            nice_map = {
                "idle": 19,
                "below_normal": 10,
                "normal": 0,
                "above_normal": -5,
            }
            p.nice(nice_map.get(priority, 10))

        logger.debug("Process priority set to %s", priority)
        return True

    except (psutil.AccessDenied, OSError) as e:
        logger.debug("Could not set priority: %s", e)
        return False
