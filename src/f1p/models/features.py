"""Per-car signals read from an :class:`~f1p.eval.context.EventContext` (all causal)."""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray

from f1p.eval.context import EventContext

FloatArray = NDArray[np.float64]

MAX_PACE_GAP = 5.0  # percent; slower (or missing) laps are capped here


def quali_pace(ctx: EventContext) -> FloatArray | None:
    """Minus the percentage gap of each car's best qualifying lap to pole.

    0 for the fastest car, ``-MAX_PACE_GAP`` for anyone more than that far off (or with no
    time). Magnitude matters: a 0.5 s pole is a different race from a 0.01 s one.
    """
    q = ctx.known("qualifying").filter(
        (pl.col("event_id") == ctx.event_id) & (pl.col("session") == "quali")
    )
    if q.is_empty():
        return None
    best = q.select(
        "driver_id",
        pl.min_horizontal("best_ms", "q1_ms", "q2_ms", "q3_ms").alias("t"),
    ).filter(pl.col("t").is_not_null() & (pl.col("t") > 0))
    if best.is_empty():
        return None
    pole = float(best["t"].min())  # type: ignore[arg-type]
    gap = {d: 100.0 * (float(t) / pole - 1.0) for d, t in best.iter_rows()}
    return np.array([-min(gap.get(d, MAX_PACE_GAP), MAX_PACE_GAP) for d in ctx.drivers])
