"""The unified-memory budget (plan §11a.5): tasks reserve memory before they run.

The sum of active reservations never exceeds the working budget (16 GB of 24 GB by
default). A task that does not fit waits until enough is released — or fails at once if it
could never fit — so concurrent work on the CPU, GPU and Neural Engine cannot push macOS
into swap.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from f1p.core.config import MemoryBudget


class BudgetError(MemoryError):
    """A reservation larger than the whole working budget."""


class MemoryLedger:
    def __init__(self, budget: MemoryBudget | None = None):
        self.budget = budget or MemoryBudget()
        self._lock = threading.Condition()
        self._active: dict[int, tuple[str, float]] = {}
        self._next = 0

    @property
    def reserved_gb(self) -> float:
        with self._lock:
            return sum(gb for _, gb in self._active.values())

    @property
    def available_gb(self) -> float:
        return self.budget.working_gb - self.reserved_gb

    @contextmanager
    def reserve(
        self, gb: float, label: str = "task", *, timeout: float | None = None
    ) -> Iterator[None]:
        """Hold ``gb`` of the working budget for the duration of the block."""
        if gb > self.budget.working_gb:
            raise BudgetError(
                f"{label} wants {gb:g} GB; the working budget is {self.budget.working_gb:g} GB"
            )
        with self._lock:
            ok = self._lock.wait_for(
                lambda: sum(g for _, g in self._active.values()) + gb <= self.budget.working_gb,
                timeout=timeout,
            )
            if not ok:
                raise TimeoutError(f"{label}: {gb:g} GB not available within {timeout}s")
            token = self._next
            self._next += 1
            self._active[token] = (label, gb)
        try:
            yield
        finally:
            with self._lock:
                del self._active[token]
                self._lock.notify_all()

    def snapshot(self) -> list[tuple[str, float]]:
        with self._lock:
            return list(self._active.values())


def batch_size(bytes_per_item: int, gb: float, *, minimum: int = 1) -> int:
    """How many items fit in ``gb`` (for Monte Carlo batches, posterior draws, …)."""
    return max(minimum, int(gb * 1024**3 // max(bytes_per_item, 1)))
