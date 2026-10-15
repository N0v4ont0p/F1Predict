"""Knowability time: the single gate that keeps the future out of every feature (plan §6.2).

Every curated row carries ``observed_at`` — the UTC instant the fact became public (a race
result at the chequered flag, a stewards' penalty at its decision time, a forecast at its
issue time). Anything that builds features or fits models reads data only through
:func:`known_at`, so a prediction "as of" time *t* cannot see what happened after *t*. This
replaces the scattered ``shift(1)`` conventions of the legacy code (defect D4).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl

OBSERVED_AT = "observed_at"


@dataclass(frozen=True, slots=True, order=True)
class AsOf:
    """A point in time from which the world is viewed. Always timezone-aware UTC."""

    at: datetime

    def __post_init__(self) -> None:
        if self.at.tzinfo is None or self.at.utcoffset() is None:
            raise ValueError("AsOf requires a timezone-aware datetime")
        object.__setattr__(self, "at", self.at.astimezone(UTC))

    @classmethod
    def now(cls) -> AsOf:
        return cls(datetime.now(UTC))

    @classmethod
    def parse(cls, text: str) -> AsOf:
        """Parse ISO-8601; a trailing ``Z`` is accepted. Naive input is rejected."""
        return cls(datetime.fromisoformat(text.replace("Z", "+00:00")))

    def __str__(self) -> str:
        return self.at.isoformat().replace("+00:00", "Z")


def known_at[F: (pl.DataFrame, pl.LazyFrame)](
    frame: F, as_of: AsOf, column: str = OBSERVED_AT
) -> F:
    """Rows of ``frame`` that were public at ``as_of`` (``observed_at <= as_of``).

    Rows with a null ``observed_at`` are treated as *unknown* and excluded: a fact with no
    known publication time must not leak into a backtest.
    """
    return frame.filter(pl.col(column).is_not_null() & (pl.col(column) <= as_of.at))
