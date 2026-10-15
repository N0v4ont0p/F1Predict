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


def circuit_affinity(
    ctx: EventContext,
    *,
    by: str = "team",
    visits: int = 3,
    lineage: dict[tuple[int, str], str] | None = None,
) -> FloatArray:
    """How much better than its season form each car has finished at this circuit.

    For each of the last ``visits`` races at ``ctx.circuit_id`` (results public at
    ``ctx.as_of``), a car's gain is its season-average classified position minus its position
    there, divided by the field size. The feature is the mean gain of the car's team lineage
    (``by="team"``) or driver (``by="driver"``), shrunk towards 0 with few visits. 0 for
    newcomers and new circuits.
    """
    results = ctx.known("results").filter(pl.col("session") == "race")
    events = ctx.tables["events"].select("event_id", "circuit_id")
    r = results.join(events, on="event_id").filter(pl.col("classified"))
    if r.is_empty():
        return np.zeros(ctx.n)
    key = "driver_id"
    if by == "team":
        cs = ctx.tables["constructor_seasons"].select("season", "constructor_id", "lineage_id")
        r = r.join(cs, on=["season", "constructor_id"], how="left").with_columns(
            pl.coalesce("lineage_id", "constructor_id").alias("team")
        )
        key = "team"
    season_avg = r.group_by("season", key).agg(pl.col("position").mean().alias("avg"))
    field = r.group_by("event_id").agg(pl.len().alias("field"))
    here = (
        r.filter(pl.col("circuit_id") == ctx.circuit_id)
        .join(season_avg, on=["season", key])
        .join(field, on="event_id")
        .with_columns(((pl.col("avg") - pl.col("position")) / pl.col("field")).alias("gain"))
    )
    last = here.select("event_id", "season", "round").unique().sort("season", "round").tail(visits)
    here = here.join(last.select("event_id"), on="event_id")
    agg = here.group_by(key).agg(pl.col("gain").sum().alias("s"), pl.len().alias("k"))
    score = {k: s / (n + 1.0) for k, s, n in agg.iter_rows()}  # one pseudo-visit of 0 gain
    if by == "team":
        lin = lineage or {}
        ids = [lin.get((ctx.season, c), c) for c in ctx.constructors]
    else:
        ids = list(ctx.drivers)
    return np.array([score.get(i, 0.0) for i in ids])


def race_pace_signal(ctx: EventContext, *, decay: float = 0.8, window: int = 12) -> FloatArray:
    """Expected pace advantage of each car from recent clean-lap race pace (2018+).

    Team part: exponentially weighted mean (``decay`` per race back, last ``window`` races
    public at ``ctx.as_of``) of the lineage's average deficit. Driver part: the weighted mean
    of the driver's deficit relative to their own team in those races. The signal is minus
    their sum, centred on the field; cars without history get the field mean. Zero when no
    pace data is known (the combiner then learns nothing from it for that race).
    """
    if "race_pace" not in ctx.tables:
        return np.zeros(ctx.n)
    pace = ctx.known("race_pace")
    if pace.is_empty():
        return np.zeros(ctx.n)
    res = (
        ctx.known("results")
        .filter(pl.col("session") == "race")
        .select("event_id", "season", "round", "driver_id", "constructor_id")
    )
    cs = ctx.tables["constructor_seasons"].select("season", "constructor_id", "lineage_id")
    df = (
        pace.join(res, on=["event_id", "driver_id"])
        .join(cs, on=["season", "constructor_id"], how="left")
        .with_columns(pl.coalesce("lineage_id", "constructor_id").alias("team"))
    )
    recent = df.select("event_id", "season", "round").unique().sort("season", "round").tail(window)
    if recent.is_empty():
        return np.zeros(ctx.n)
    ages = {e: len(recent) - 1 - i for i, e in enumerate(recent["event_id"].to_list())}
    df = df.filter(pl.col("event_id").is_in(list(ages))).with_columns(
        pl.col("event_id").replace_strict(ages, return_dtype=pl.Int64).alias("age")
    )
    df = df.with_columns(
        (decay ** pl.col("age")).alias("w"),
        pl.col("pace_deficit_pct").mean().over("event_id", "team").alias("team_deficit"),
    ).with_columns((pl.col("pace_deficit_pct") - pl.col("team_deficit")).alias("vs_team"))
    team = {
        t: float(v)
        for t, v in df.unique(["event_id", "team"])
        .group_by("team")
        .agg(((pl.col("team_deficit") * pl.col("w")).sum() / pl.col("w").sum()).alias("v"))
        .iter_rows()
    }
    driver = {
        d: float(v)
        for d, v in df.group_by("driver_id")
        .agg(((pl.col("vs_team") * pl.col("w")).sum() / pl.col("w").sum()).alias("v"))
        .iter_rows()
    }
    lineages = cs.filter(pl.col("season") == ctx.season)
    lin = {c: (lineage or c) for _, c, lineage in lineages.iter_rows()}
    raw = []
    for d, c in zip(ctx.drivers, ctx.constructors, strict=True):
        t = team.get(lin.get(c, c))
        raw.append(None if t is None else t + driver.get(d, 0.0))
    known = [v for v in raw if v is not None]
    if not known:
        return np.zeros(ctx.n)
    fill = float(np.mean(known))
    arr = np.array([fill if v is None else v for v in raw])
    return -(arr - arr.mean())
