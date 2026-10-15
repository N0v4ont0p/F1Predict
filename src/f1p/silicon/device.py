"""What machine are we on? CPU clusters, memory, GPU, Neural Engine, BLAS and ML backends."""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CpuCluster:
    name: str  # e.g. "Super", "Performance", "Efficiency" (as macOS reports them)
    cores: int


@dataclass(frozen=True, slots=True)
class Device:
    chip: str
    os: str
    arch: str
    apple_silicon: bool
    clusters: tuple[CpuCluster, ...]
    logical_cpus: int
    memory_gb: float
    gpu_cores: int | None
    neural_engine: bool
    blas: str
    backends: dict[str, bool] = field(default_factory=dict)

    @property
    def strong_cores(self) -> int:
        """Cores suited to compute lanes (everything but efficiency cores)."""
        strong = sum(c.cores for c in self.clusters if not _is_efficiency(c.name))
        return strong or max(1, self.logical_cpus // 2)

    @property
    def efficiency_cores(self) -> int:
        return sum(c.cores for c in self.clusters if _is_efficiency(c.name))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["strong_cores"] = self.strong_cores
        d["efficiency_cores"] = self.efficiency_cores
        return d


def _is_efficiency(name: str) -> bool:
    return name.lower().startswith("eff")


def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _sysctl(key: str) -> str | None:
    return _run(["sysctl", "-n", key])


def _mac_clusters() -> tuple[CpuCluster, ...]:
    levels = int(_sysctl("hw.nperflevels") or 0)
    out = []
    for i in range(levels):
        name = _sysctl(f"hw.perflevel{i}.name") or f"level{i}"
        cores = int(_sysctl(f"hw.perflevel{i}.physicalcpu") or 0)
        if cores:
            out.append(CpuCluster(name, cores))
    return tuple(out)


def _mac_gpu_cores() -> int | None:
    raw = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=15.0)
    if not raw:
        return None
    try:
        for gpu in json.loads(raw).get("SPDisplaysDataType", []):
            cores = gpu.get("sppci_cores")
            if cores:
                return int(cores)
    except (ValueError, TypeError):
        return None
    return None


def _linux_memory_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return 0.0


def _linux_chip() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def blas_vendor() -> str:
    """Which BLAS NumPy links (Accelerate on a correctly built macOS NumPy)."""
    try:
        import numpy as np

        cfg = np.show_config(mode="dicts")  # type: ignore[call-arg]
        info = cfg.get("Build Dependencies", {}).get("blas", {})  # type: ignore[union-attr]
        return str(info.get("name", "unknown"))
    except Exception:  # pragma: no cover - very old numpy
        return "unknown"


def available_backends() -> dict[str, bool]:
    names = {"mlx": "mlx.core", "jax": "jax", "coremltools": "coremltools", "torch": "torch"}
    return {k: importlib.util.find_spec(v.split(".")[0]) is not None for k, v in names.items()}


@functools.cache
def detect() -> Device:
    """Discover the machine (cached; cheap after the first call)."""
    arch = platform.machine()
    logical = os.cpu_count() or 1
    if sys.platform == "darwin":
        chip = _sysctl("machdep.cpu.brand_string") or "Apple"
        memory = int(_sysctl("hw.memsize") or 0) / 1024**3
        apple = arch == "arm64"
        clusters = _mac_clusters() if apple else (CpuCluster("Performance", logical),)
        gpu = _mac_gpu_cores() if apple else None
        ane = apple
    else:
        chip = _linux_chip()
        memory = _linux_memory_gb()
        apple = False
        clusters = (CpuCluster("Performance", logical),)
        gpu, ane = None, False
    return Device(
        chip=chip,
        os=f"{platform.system()} {platform.release()}",
        arch=arch,
        apple_silicon=apple,
        clusters=clusters,
        logical_cpus=logical,
        memory_gb=round(memory, 1),
        gpu_cores=gpu,
        neural_engine=ane,
        blas=blas_vendor(),
        backends=available_backends(),
    )
