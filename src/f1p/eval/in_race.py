"""The in-race checkpoint ``C5`` (plan M6 §D.2): the race as it stands after lap ``k``.

At ``C5`` a model sees everything public at the start (the tables are cut at the C4 time,
as before) **plus the race itself up to the end of lap k** — the live-timing feed is public
as it happens. :func:`race_so_far` is the single place that cut is made, the in-race
counterpart of :func:`f1p.core.clock.known_at`.

``k`` is :data:`IN_RACE_FRACTION` of the race distance (one third: most cars have not yet
made their first stop, so the order on the road is not yet the order of the race). The
distance is the laps the race actually ran — F1DB rarely has the scheduled laps; a
shortened race (red flag, time limit) therefore puts ``k`` a little earlier than the
schedule would.

Two approximations, both small at one third distance:

* every car's laps ``1..k`` are visible, and a lapped car completes its own lap ``k``
  up to one lap after the leader;
* whether a car is still running comes from the classification's laps completed (a car
  that completed fewer than ``k`` laps had stopped by then — a fact public at lap ``k``),
  with the laps table's own records as the fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from f1p.eval.context import IN_RACE, IN_RACE_FRACTION, EventContext, cut_lap, is_in_race

__all__ = ["IN_RACE", "IN_RACE_FRACTION", "RaceState", "cut_lap", "race_so_far", "race_state"]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def race_so_far(ctx: EventContext) -> pl.DataFrame:
    """This race's laps ``1..k`` (all the in-race data a ``C5`` model may use)."""
    if not is_in_race(ctx.checkpoint):
        raise ValueError(f"{ctx.checkpoint} is not an in-race checkpoint")
    key = "__race_so_far"
    if key not in ctx._cache:
        k = int(ctx.meta["lap"])
        ctx._cache[key] = ctx.tables["laps"].filter(
            (pl.col("event_id") == ctx.event_id) & (pl.col("lap") <= k)
        )
    return ctx._cache[key]


@dataclass(frozen=True, slots=True)
class RaceState:
    """The race after lap ``lap``, one entry per driver of the context (``ctx.drivers``)."""

    lap: int
    race_laps: int
    running: BoolArray  # still in the race
    position: FloatArray  # running position at lap k (retired cars: behind, by laps run)
    gap_s: FloatArray  # seconds behind the leader (NaN: retired or no timing)
    stops: IntArray  # stops made
    compound: list[str | None]  # tyres fitted at lap k
    tyre_age: FloatArray  # laps on them (NaN unknown)
    neutral: bool  # lap k ran under SC / VSC

    @property
    def laps_left(self) -> int:
        return self.race_laps - self.lap


def _laps_completed(ctx: EventContext) -> dict[str, int]:
    r = ctx.tables["results"].filter(
        (pl.col("event_id") == ctx.event_id) & (pl.col("session") == "race")
    )
    return {str(d): int(n) for d, n in r.select("driver_id", "laps").iter_rows() if n is not None}


def race_state(ctx: EventContext) -> RaceState:
    """Positions, gaps, stops and tyres after lap ``k`` (see the module notes)."""
    k = int(ctx.meta["lap"])
    laps = race_so_far(ctx)
    n = ctx.n
    idx = ctx.index()
    done = _laps_completed(ctx)
    last = laps.group_by("driver_id").agg(pl.col("lap").max().alias("last"))
    seen = dict(last.iter_rows())
    running = np.array(
        [done[d] >= k if d in done else seen.get(d, 0) >= k - 1 for d in ctx.drivers]
    )

    # the latest record of each car (lap k, or its last lap before a timing gap)
    latest = laps.sort("lap").group_by("driver_id", maintain_order=True).last()
    stops_by = dict(
        laps.group_by("driver_id").agg(pl.col("in_lap").cast(pl.Int64).sum()).iter_rows()
    )
    position = np.full(n, np.nan)
    compound: list[str | None] = [None] * n
    tyre_age = np.full(n, np.nan)
    stops = np.zeros(n, dtype=np.int64)
    for row in latest.iter_rows(named=True):
        i = idx.get(row["driver_id"])
        if i is None:
            continue
        position[i] = float(row["position"]) if row["position"] is not None else np.nan
        compound[i] = row["compound"]
        tyre_age[i] = float(row["tyre_age"]) if row["tyre_age"] is not None else np.nan
        stops[i] = int(stops_by.get(row["driver_id"], 0))
    gap = _gaps(laps, k, ctx.drivers)

    # running cars without a position (timing gaps) go behind the timed ones; retired cars
    # behind every running car, those that ran longer first
    order_key = np.where(np.isnan(position), 1e3, position)
    laps_run = np.array([done.get(d, seen.get(d, 0)) for d in ctx.drivers], dtype=np.float64)
    key = np.where(running, order_key, 1e6 - laps_run)
    rank = np.empty(n)
    rank[np.argsort(key, kind="stable")] = np.arange(1.0, n + 1.0)
    gap = np.where(running, gap, np.nan)
    lap_k = laps.filter(pl.col("lap") == k)
    neutral = bool(lap_k["neutral"].mean() >= 0.5) if lap_k.height else False  # type: ignore[operator]
    return RaceState(
        k,
        int(ctx.meta["race_laps"]),
        running,
        rank,
        gap,
        stops,
        compound,
        tyre_age,
        neutral,
    )


def _gaps(laps: pl.DataFrame, k: int, drivers: tuple[str, ...]) -> FloatArray:
    """Seconds behind the leader at the end of lap ``k``: the chain of intervals to the car
    ahead; where an interval is missing (lapped cars, sources without intervals), the
    difference in summed lap times to the leader, never less than the car ahead's gap."""
    at_k = laps.filter(pl.col("lap") == k).sort("position")
    total = dict(laps.group_by("driver_id").agg(pl.col("lap_ms").sum() / 1000.0).iter_rows())
    out = dict.fromkeys(drivers, np.nan)
    acc, lead = 0.0, np.nan
    for j, row in enumerate(at_k.iter_rows(named=True)):
        d = row["driver_id"]
        mine = float(total.get(d, np.nan))
        if j == 0:
            acc, lead = 0.0, mine
        elif row["gap_ahead_s"] is not None and np.isfinite(row["gap_ahead_s"]):
            acc += max(float(row["gap_ahead_s"]), 0.0)
        elif np.isfinite(mine - lead):
            acc = max(acc, mine - lead)
        # else: no timing at all; level with the car ahead
        if d in out:
            out[d] = acc
    return np.array([out[d] for d in drivers], dtype=np.float64)
