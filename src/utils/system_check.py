"""
System Check Module - Rendszer állapot ellenőrzés és diagnosztika.

Ez a modul ellenőrzi:
- CPU, RAM, GPU erőforrásokat
- CUDA és PyTorch elérhetőséget
- Python verzió és fontos csomagok állapotát
- Zombie/beragadt processeket
- Lemez helyet
- Környezeti változókat

Használat:
    from utils.system_check import SystemChecker
    checker = SystemChecker()
    report = checker.run_full_check()
"""

import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime

# Optional imports with graceful fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class SystemStatus:  # pylint: disable=too-many-instance-attributes
    """Rendszer állapot adatstruktúra."""

    # Általános
    timestamp: str = ""
    platform: str = ""
    python_version: str = ""

    # CPU
    cpu_physical_cores: int = 0
    cpu_logical_cores: int = 0
    cpu_usage_percent: float = 0.0
    cpu_freq_mhz: float = 0.0

    # RAM
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    ram_used_percent: float = 0.0

    # GPU/CUDA
    cuda_available: bool = False
    cuda_version: str = ""
    gpu_count: int = 0
    gpu_devices: list = field(default_factory=list)
    gpu_memory_total_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0

    # PyTorch
    torch_version: str = ""
    torch_cuda_version: str = ""

    # Disk
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_used_percent: float = 0.0

    # Processes
    zombie_processes: list = field(default_factory=list)
    python_processes: list = field(default_factory=list)
    high_memory_processes: list = field(default_factory=list)

    # Packages
    key_packages: dict = field(default_factory=dict)

    # Warnings
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    # Summary
    status_ok: bool = True
    summary: str = ""


class SystemChecker:
    """
    Rendszer állapot ellenőrző osztály.

    Összegyűjti és elemzi a rendszer erőforrásait,
    ellenőrzi a szükséges szoftvereket és figyelmeztet
    a potenciális problémákra.
    """

    # Minimum követelmények
    MIN_RAM_GB = 4.0
    MIN_DISK_FREE_GB = 5.0
    MAX_CPU_USAGE_PERCENT = 90.0
    MAX_RAM_USAGE_PERCENT = 90.0

    # Fontos csomagok listája
    KEY_PACKAGES = [
        "pandas",
        "numpy",
        "torch",
        "sklearn",
        "xgboost",
        "lightgbm",
        "statsmodels",
        "customtkinter",
    ]

    def __init__(self):
        """Inicializálás."""
        self.status = SystemStatus()

    def run_full_check(self) -> SystemStatus:
        """
        Teljes rendszer ellenőrzés futtatása.

        Returns:
            SystemStatus: Az összegyűjtött rendszer információk.
        """
        self.status = SystemStatus()
        self.status.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Alap információk
        self._check_platform()
        self._check_python()

        # Erőforrások
        if PSUTIL_AVAILABLE:
            self._check_cpu()
            self._check_ram()
            self._check_disk()
            self._check_processes()
        else:
            self.status.warnings.append("psutil not available - limited diagnostics")

        # GPU/CUDA
        self._check_cuda()

        # Csomagok
        self._check_packages()

        # Összefoglaló generálás
        self._generate_summary()

        return self.status

    def _check_platform(self) -> None:
        """Platform információk gyűjtése."""
        self.status.platform = f"{platform.system()} {platform.release()}"

    def _check_python(self) -> None:
        """Python verzió ellenőrzése."""
        ver = sys.version_info
        self.status.python_version = f"{ver.major}.{ver.minor}.{ver.micro}"

        # Python verzió figyelmeztetés
        if sys.version_info < (3, 9):
            self.status.warnings.append(
                f"Python {self.status.python_version} - recommend 3.9+"
            )

    def _check_cpu(self) -> None:
        """CPU információk gyűjtése."""
        if not PSUTIL_AVAILABLE:
            return

        self.status.cpu_physical_cores = psutil.cpu_count(logical=False) or 0
        self.status.cpu_logical_cores = psutil.cpu_count(logical=True) or 0
        self.status.cpu_usage_percent = psutil.cpu_percent(interval=0.1)

        # CPU frekvencia (ha elérhető)
        try:
            freq = psutil.cpu_freq()
            if freq:
                self.status.cpu_freq_mhz = freq.current
        except (AttributeError, OSError, RuntimeError):
            pass  # Nem minden rendszeren elérhető

        # Figyelmeztetés magas CPU használatnál
        if self.status.cpu_usage_percent > self.MAX_CPU_USAGE_PERCENT:
            self.status.warnings.append(
                f"High CPU usage: {self.status.cpu_usage_percent:.1f}%"
            )

    def _check_ram(self) -> None:
        """RAM információk gyűjtése."""
        if not PSUTIL_AVAILABLE:
            return

        mem = psutil.virtual_memory()
        self.status.ram_total_gb = mem.total / (1024**3)
        self.status.ram_available_gb = mem.available / (1024**3)
        self.status.ram_used_percent = mem.percent

        # Figyelmeztetések
        if self.status.ram_available_gb < self.MIN_RAM_GB:
            self.status.warnings.append(
                f"Low RAM: {self.status.ram_available_gb:.1f} GB available"
            )

        if self.status.ram_used_percent > self.MAX_RAM_USAGE_PERCENT:
            self.status.warnings.append(
                f"High RAM usage: {self.status.ram_used_percent:.1f}%"
            )

    def _check_disk(self) -> None:
        """Lemez információk gyűjtése."""
        if not PSUTIL_AVAILABLE:
            return

        try:
            # Aktuális munkakönyvtár lemeze
            disk = psutil.disk_usage(os.getcwd())
            self.status.disk_total_gb = disk.total / (1024**3)
            self.status.disk_free_gb = disk.free / (1024**3)
            self.status.disk_used_percent = disk.percent

            if self.status.disk_free_gb < self.MIN_DISK_FREE_GB:
                self.status.warnings.append(
                    f"Low disk space: {self.status.disk_free_gb:.1f} GB free"
                )
        except (OSError, PermissionError):
            pass  # Lemez elérési hiba

    def _check_cuda(self) -> None:
        """CUDA és GPU információk gyűjtése."""
        if not TORCH_AVAILABLE:
            self.status.warnings.append("PyTorch not available - GPU check skipped")
            return

        self.status.torch_version = torch.__version__

        # CUDA elérhetőség
        self.status.cuda_available = torch.cuda.is_available()

        if self.status.cuda_available:
            self.status.torch_cuda_version = torch.version.cuda or "N/A"
            self.status.gpu_count = torch.cuda.device_count()

            # GPU eszközök információi
            for i in range(self.status.gpu_count):
                try:
                    props = torch.cuda.get_device_properties(i)
                    gpu_info = {
                        "index": i,
                        "name": props.name,
                        "total_memory_gb": props.total_memory / (1024**3),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                    self.status.gpu_devices.append(gpu_info)

                    # Memória használat
                    mem_total = torch.cuda.get_device_properties(i).total_memory
                    mem_allocated = torch.cuda.memory_allocated(i)
                    self.status.gpu_memory_total_gb += mem_total / (1024**3)
                    self.status.gpu_memory_used_gb += mem_allocated / (1024**3)
                except (RuntimeError, AssertionError) as e:
                    self.status.errors.append(f"GPU {i} info error: {e}")
        else:
            # CUDA nem elérhető - ok info
            cuda_build = getattr(torch.version, 'cuda', None)
            if cuda_build:
                self.status.warnings.append(
                    f"CUDA {cuda_build} built but not available - check drivers"
                )

    def _check_processes(self) -> None:
        """Process ellenőrzés - zombie és problémás processek keresése."""
        if not PSUTIL_AVAILABLE:
            return

        current_pid = os.getpid()
        python_procs = []
        zombies = []
        high_mem = []

        try:
            for proc in psutil.process_iter(['pid', 'name', 'status', 'memory_percent', 'cmdline']):
                try:
                    info = proc.info
                    pid = info['pid']

                    # Saját process kihagyása
                    if pid == current_pid:
                        continue

                    # Zombie processek
                    if info['status'] == psutil.STATUS_ZOMBIE:
                        zombies.append({
                            "pid": pid,
                            "name": info['name'],
                        })

                    # Python processek (MBO-val kapcsolatosak lehetnek)
                    name = (info['name'] or '').lower()
                    cmdline = ' '.join(info.get('cmdline') or []).lower()

                    if 'python' in name or 'python' in cmdline:
                        if 'mbo' in cmdline or 'main' in cmdline:
                            python_procs.append({
                                "pid": pid,
                                "name": info['name'],
                                "memory_percent": info.get('memory_percent', 0),
                            })

                    # Magas memória használatú processek (>10%)
                    mem_pct = info.get('memory_percent', 0)
                    if mem_pct and mem_pct > 10:
                        high_mem.append({
                            "pid": pid,
                            "name": info['name'],
                            "memory_percent": mem_pct,
                        })

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except (psutil.Error, OSError):
            pass  # Process iteration error

        self.status.zombie_processes = zombies
        self.status.python_processes = python_procs
        self.status.high_memory_processes = sorted(
            high_mem, key=lambda x: x['memory_percent'], reverse=True
        )[:5]  # Top 5

        # Figyelmeztetések
        if zombies:
            self.status.warnings.append(
                f"Found {len(zombies)} zombie process(es)"
            )

        if len(python_procs) > 1:
            self.status.warnings.append(
                f"Found {len(python_procs)} other MBO-related Python process(es)"
            )

    def _check_packages(self) -> None:
        """Fontos csomagok verzióinak ellenőrzése."""
        for pkg_name in self.KEY_PACKAGES:
            try:
                if pkg_name == "sklearn":
                    # pylint: disable=import-outside-toplevel
                    import sklearn
                    self.status.key_packages[pkg_name] = sklearn.__version__
                elif pkg_name == "torch":
                    if TORCH_AVAILABLE:
                        self.status.key_packages[pkg_name] = torch.__version__
                else:
                    pkg = __import__(pkg_name)
                    version = getattr(pkg, '__version__', 'unknown')
                    self.status.key_packages[pkg_name] = version
            except ImportError:
                self.status.key_packages[pkg_name] = "NOT INSTALLED"
                if pkg_name in ("pandas", "numpy", "customtkinter"):
                    self.status.errors.append(f"Critical package missing: {pkg_name}")

    def _generate_summary(self) -> None:
        """Összefoglaló generálása."""
        # Státusz meghatározása
        self.status.status_ok = len(self.status.errors) == 0

        # Rövid összefoglaló
        parts = []

        # CPU/RAM
        if PSUTIL_AVAILABLE:
            parts.append(
                f"CPU: {self.status.cpu_physical_cores}c/{self.status.cpu_logical_cores}t "
                f"({self.status.cpu_usage_percent:.0f}%)"
            )
            parts.append(
                f"RAM: {self.status.ram_available_gb:.1f}/{self.status.ram_total_gb:.1f} GB "
                f"({self.status.ram_used_percent:.0f}%)"
            )

        # GPU
        if self.status.cuda_available:
            gpu_names = [g['name'] for g in self.status.gpu_devices]
            parts.append(f"GPU: {', '.join(gpu_names)} (CUDA {self.status.torch_cuda_version})")
        else:
            parts.append("GPU: Not available (CPU mode)")

        # Figyelmeztetések száma
        if self.status.warnings:
            parts.append(f"Warnings: {len(self.status.warnings)}")

        if self.status.errors:
            parts.append(f"ERRORS: {len(self.status.errors)}")

        self.status.summary = " | ".join(parts)

    def get_detailed_report(self) -> str:
        """
        Részletes szöveges jelentés generálása (log fájlhoz).

        Returns:
            str: Formázott jelentés szöveg.
        """
        lines = [
            "=" * 70,
            "SYSTEM DIAGNOSTIC REPORT",
            f"Generated: {self.status.timestamp}",
            "=" * 70,
            "",
            "--- SYSTEM INFO ---",
            f"Platform: {self.status.platform}",
            f"Python: {self.status.python_version}",
            "",
            "--- CPU ---",
            f"Physical cores: {self.status.cpu_physical_cores}",
            f"Logical cores: {self.status.cpu_logical_cores}",
            f"Current usage: {self.status.cpu_usage_percent:.1f}%",
        ]

        if self.status.cpu_freq_mhz:
            lines.append(f"Frequency: {self.status.cpu_freq_mhz:.0f} MHz")

        lines.extend([
            "",
            "--- MEMORY ---",
            f"Total RAM: {self.status.ram_total_gb:.2f} GB",
            f"Available: {self.status.ram_available_gb:.2f} GB",
            f"Used: {self.status.ram_used_percent:.1f}%",
            "",
            "--- DISK ---",
            f"Total: {self.status.disk_total_gb:.1f} GB",
            f"Free: {self.status.disk_free_gb:.1f} GB",
            f"Used: {self.status.disk_used_percent:.1f}%",
            "",
            "--- GPU/CUDA ---",
            f"CUDA available: {self.status.cuda_available}",
            f"PyTorch version: {self.status.torch_version}",
        ])

        if self.status.cuda_available:
            lines.append(f"CUDA version: {self.status.torch_cuda_version}")
            lines.append(f"GPU count: {self.status.gpu_count}")
            for gpu in self.status.gpu_devices:
                lines.append(
                    f"  [{gpu['index']}] {gpu['name']} - "
                    f"{gpu['total_memory_gb']:.1f} GB "
                    f"(Compute {gpu['compute_capability']})"
                )

        lines.extend([
            "",
            "--- KEY PACKAGES ---",
        ])
        for pkg, version in self.status.key_packages.items():
            status = "OK" if version != "NOT INSTALLED" else "MISSING"
            lines.append(f"  {pkg}: {version} [{status}]")

        # Processes
        if self.status.zombie_processes:
            lines.extend([
                "",
                "--- ZOMBIE PROCESSES ---",
            ])
            for z in self.status.zombie_processes:
                lines.append(f"  PID {z['pid']}: {z['name']}")

        if self.status.python_processes:
            lines.extend([
                "",
                "--- OTHER MBO PYTHON PROCESSES ---",
            ])
            for p in self.status.python_processes:
                lines.append(
                    f"  PID {p['pid']}: {p['name']} "
                    f"(mem: {p['memory_percent']:.1f}%)"
                )

        if self.status.high_memory_processes:
            lines.extend([
                "",
                "--- HIGH MEMORY PROCESSES (TOP 5) ---",
            ])
            for p in self.status.high_memory_processes:
                lines.append(
                    f"  PID {p['pid']}: {p['name']} "
                    f"(mem: {p['memory_percent']:.1f}%)"
                )

        # Warnings and Errors
        if self.status.warnings:
            lines.extend([
                "",
                "--- WARNINGS ---",
            ])
            for w in self.status.warnings:
                lines.append(f"  [!] {w}")

        if self.status.errors:
            lines.extend([
                "",
                "--- ERRORS ---",
            ])
            for e in self.status.errors:
                lines.append(f"  [X] {e}")

        lines.extend([
            "",
            "=" * 70,
            f"STATUS: {'OK' if self.status.status_ok else 'ISSUES DETECTED'}",
            f"SUMMARY: {self.status.summary}",
            "=" * 70,
        ])

        return "\n".join(lines)

    def get_short_report(self) -> str:
        """
        Rövid jelentés a GUI-hoz.

        Returns:
            str: Tömör összefoglaló szöveg.
        """
        lines = ["System Check:"]

        # OS és Python verzió
        lines.append(
            f"  OS: {self.status.platform} | Python: {self.status.python_version}"
        )

        # CPU/RAM egy sorban
        if PSUTIL_AVAILABLE:
            lines.append(
                f"  CPU: {self.status.cpu_physical_cores} cores "
                f"| RAM: {self.status.ram_available_gb:.1f} GB free"
            )

        # GPU
        if self.status.cuda_available:
            gpu_name = self.status.gpu_devices[0]['name'] if self.status.gpu_devices else "Unknown"
            lines.append(f"  GPU: {gpu_name} (CUDA {self.status.torch_cuda_version})")
        else:
            lines.append("  GPU: Not available - using CPU")

        # Problémák
        if self.status.warnings or self.status.errors:
            issues = len(self.status.warnings) + len(self.status.errors)
            lines.append(f"  Issues: {issues} (check logs for details)")

        return "\n".join(lines)


def run_system_check(verbose: bool = False) -> SystemStatus:
    """
    Convenience function rendszer ellenőrzéshez.

    Args:
        verbose: Ha True, kiírja a részletes jelentést.

    Returns:
        SystemStatus: A rendszer állapot.
    """
    checker = SystemChecker()
    status = checker.run_full_check()

    if verbose:
        print(checker.get_detailed_report())

    return status


# Modul szintű gyors elérés
def get_system_summary() -> str:
    """Gyors összefoglaló lekérése."""
    checker = SystemChecker()
    checker.run_full_check()
    return checker.status.summary


if __name__ == "__main__":
    # Standalone teszt
    run_system_check(verbose=True)
