"""Compute accounting for long runs (plan M6 §0): wall and CPU time, peak memory and the
machine's thermal state, recorded next to every result so speed-ups and regressions are
visible.

Energy is read from ``powermetrics`` only when the run already has root (it never prompts);
otherwise it is left out and the record says so.
"""

from __future__ import annotations

import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any


def thermal_state() -> str | None:
    """macOS thermal pressure (``pmset -g therm``): ``nominal`` or the first warning line,
    ``None`` elsewhere."""
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    warnings = [
        ln.strip()
        for ln in out.splitlines()
        if "warning level" in ln.lower() and "no " not in ln.lower()
    ]
    return warnings[0] if warnings else "nominal"


def _peak_rss_mb(who: int) -> float:
    rss = resource.getrusage(who).ru_maxrss
    # bytes on macOS, kilobytes on Linux
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def _cpu_s() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + kids.ru_utime + kids.ru_stime


@dataclass
class ComputeRecord:
    label: str
    workers: int = 1
    wall_s: float = 0.0
    cpu_s: float = 0.0
    peak_rss_mb: float = 0.0  # this process
    # the largest worker; on Linux a spawned child's high-water mark starts at its parent's
    # resident size when it was forked, so this is an upper bound there
    peak_child_rss_mb: float = 0.0
    thermal_start: str | None = None
    thermal_end: str | None = None
    energy_j: float | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class compute_log:  # noqa: N801 - used as a context manager, reads like a function
    """``with compute_log("eval run", workers=8) as rec: ...`` fills ``rec`` on exit."""

    def __init__(self, label: str, *, workers: int = 1, **notes: Any) -> None:
        self.record = ComputeRecord(label, workers=workers, notes=dict(notes))
        self._t0 = 0.0
        self._cpu0 = 0.0

    def __enter__(self) -> ComputeRecord:
        self.record.thermal_start = thermal_state()
        self._t0 = time.perf_counter()
        self._cpu0 = _cpu_s()
        return self.record

    def __exit__(self, *exc: object) -> None:
        rec = self.record
        rec.wall_s = round(time.perf_counter() - self._t0, 3)
        rec.cpu_s = round(_cpu_s() - self._cpu0, 3)
        rec.peak_rss_mb = round(_peak_rss_mb(resource.RUSAGE_SELF), 1)
        rec.peak_child_rss_mb = round(_peak_rss_mb(resource.RUSAGE_CHILDREN), 1)
        rec.thermal_end = thermal_state()
        if os.geteuid() != 0:
            rec.notes.setdefault("energy", "not measured (powermetrics needs root)")
