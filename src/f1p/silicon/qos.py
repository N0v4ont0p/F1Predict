"""Quality-of-service lanes (plan §11a.3-4).

macOS has no hard core pinning; threads are steered to super/performance or efficiency
cores by their QoS class. A lane is a thread pool whose threads set their QoS on start, so
the libraries they call (Polars, DuckDB, XLA, …) inherit the right placement. Elsewhere the
QoS call is a no-op and lanes are plain pools.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, IntEnum

from f1p.silicon.device import detect


class QoS(IntEnum):
    """``qos_class_t`` values from ``<sys/qos.h>``."""

    USER_INTERACTIVE = 0x21
    USER_INITIATED = 0x19
    DEFAULT = 0x15
    UTILITY = 0x11
    BACKGROUND = 0x09


class Lane(Enum):
    INTERACTIVE = "interactive"  # super cores: CLI/API requests, live updates
    COMPUTE = "compute"  # performance + super cores: inference, fitting, simulation prep
    BACKGROUND = "background"  # efficiency cores: fetching, compaction, upkeep

    @property
    def qos(self) -> QoS:
        return {
            Lane.INTERACTIVE: QoS.USER_INTERACTIVE,
            Lane.COMPUTE: QoS.USER_INITIATED,
            Lane.BACKGROUND: QoS.UTILITY,
        }[self]


def _libsystem() -> ctypes.CDLL | None:
    if sys.platform != "darwin":
        return None
    path = ctypes.util.find_library("System")
    return ctypes.CDLL(path) if path else None


def set_thread_qos(qos: QoS) -> bool:
    """Give the calling thread a QoS class. Returns ``False`` where unsupported."""
    lib = _libsystem()
    if lib is None:
        return False
    fn = lib.pthread_set_qos_class_self_np
    fn.argtypes = [ctypes.c_uint, ctypes.c_int]
    fn.restype = ctypes.c_int
    return fn(int(qos), 0) == 0


def lane_size(lane: Lane) -> int:
    d = detect()
    if lane is Lane.BACKGROUND:
        return max(2, d.efficiency_cores or d.logical_cpus // 2)
    if lane is Lane.INTERACTIVE:
        return 2
    return max(1, d.strong_cores)


def lane_pool(lane: Lane, max_workers: int | None = None) -> ThreadPoolExecutor:
    """A thread pool whose workers carry the lane's QoS."""
    return ThreadPoolExecutor(
        max_workers=max_workers or lane_size(lane),
        thread_name_prefix=f"f1p-{lane.value}",
        initializer=set_thread_qos,
        initargs=(lane.qos,),
    )
