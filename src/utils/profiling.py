"""Lightweight memory & time profiling utilities (no heavy deps).

Uses ``resource.getrusage`` for peak RSS, which is available on macOS/Linux. On macOS
``ru_maxrss`` is reported in **bytes**; on Linux it is **kilobytes**. We normalise to MB.
"""
from __future__ import annotations

import gc
import platform
import resource
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


def _maxrss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    if platform.system() == "Darwin":
        return raw / (1024 * 1024)
    return raw / 1024


@dataclass
class ProfileResult:
    label: str
    seconds: float
    peak_rss_mb: float

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "seconds": round(self.seconds, 4),
            "peak_rss_mb": round(self.peak_rss_mb, 2),
        }


@contextmanager
def profile(label: str = "block") -> Iterator[ProfileResult]:
    """Context manager yielding a :class:`ProfileResult` populated on exit."""
    gc.collect()
    result = ProfileResult(label=label, seconds=0.0, peak_rss_mb=0.0)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.seconds = time.perf_counter() - start
        result.peak_rss_mb = _maxrss_mb()
