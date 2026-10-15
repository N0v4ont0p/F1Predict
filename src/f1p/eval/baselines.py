"""Baseline forecasters every real model must beat (plan §10.3).

Each rank baseline turns one ordering known at the checkpoint (grid, qualifying, last race,
championship standings) into a Plackett-Luce model ``log s_i = -β · rank_i``. The single
parameter β ("how much does this ordering predict the race?") is fitted by maximum likelihood
on earlier events only — the same rolling, causal protocol as every other model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from numpy.typing import NDArray

from f1p.eval.context import EventContext, Forecaster
from f1p.eval.forecast import Forecast, pl_log_likelihood
from f1p.eval.metrics import Outcome

FloatArray = NDArray[np.float64]
History = Sequence[tuple[EventContext, object, Outcome]]

TRAINING_SEASONS = 8
BETA_BOUNDS = (0.0, 3.0)


def golden_section_max(
    f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-4
) -> float:
    """Maximise a unimodal function on [lo, hi]."""
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def finisher_indices(ctx: EventContext, outcome: Outcome) -> NDArray[np.intp]:
    idx = ctx.index()
    return np.array([idx[d] for d in outcome.classified if d in idx], dtype=np.intp)


class Uniform:
    """Every driver equally likely — the floor any model must clear."""

    name = "uniform"
    checkpoints = frozenset({"C1", "C4"})

    def prepare(self, ctx: EventContext) -> None:
        return None

    def forecast(self, ctx: EventContext, signal: object, history: History) -> Forecast:
        return Forecast.from_log_strength(
            ctx.event_id, ctx.checkpoint, self.name, ctx.drivers, np.zeros(ctx.n)
        )


def _ranks_from(order: dict[str, float], drivers: tuple[str, ...]) -> FloatArray:
    """Rank vector aligned to ``drivers``; unknown drivers share the slot behind the last known."""
    back = (max(order.values()) + 1.0) if order else 1.0
    return np.array([order.get(d, back) for d in drivers], dtype=np.float64)


def grid_ranks(ctx: EventContext) -> FloatArray | None:
    g = ctx.known("grid").filter(
        (pl.col("event_id") == ctx.event_id) & (pl.col("session") == "race")
    )
    if g.is_empty():
        return None
    order = {
        r["driver_id"]: float(r["grid_position"] or r["display_order"])
        for r in g.iter_rows(named=True)
    }
    return _ranks_from(order, ctx.drivers)


def quali_ranks(ctx: EventContext) -> FloatArray | None:
    q = ctx.known("qualifying").filter(
        (pl.col("event_id") == ctx.event_id) & (pl.col("session") == "quali")
    )
    if q.is_empty():
        return None
    order = {
        r["driver_id"]: float(r["position"] or r["display_order"]) for r in q.iter_rows(named=True)
    }
    return _ranks_from(order, ctx.drivers)


def last_race_ranks(ctx: EventContext) -> FloatArray | None:
    """Finishing order of the most recent race known at ``as_of`` (retirements by laps run)."""
    r = ctx.known("results").filter(pl.col("session") == "race")
    if r.is_empty():
        return None
    last = r.sort("season", "round").tail(1).row(0, named=True)["event_id"]
    rows = r.filter(pl.col("event_id") == last).sort(
        pl.col("position").is_null(),
        "position",
        pl.col("laps").fill_null(0),
        descending=[False, False, True],
    )
    order = {d: float(i + 1) for i, d in enumerate(rows["driver_id"].to_list())}
    return _ranks_from(order, ctx.drivers)


def standings_ranks(ctx: EventContext) -> FloatArray | None:
    """Driver championship order after the latest known round (this season, else last)."""
    st = ctx.known("driver_standings")
    if st.is_empty():
        return None
    this = st.filter(pl.col("season") == ctx.season)
    src = this if this.height else st.filter(pl.col("season") == st["season"].max())
    latest = src.filter(pl.col("round") == src["round"].max()).sort("display_order")
    order = {d: float(i + 1) for i, d in enumerate(latest["driver_id"].to_list())}
    return _ranks_from(order, ctx.drivers)


@dataclass
class RankPL:
    """Plackett-Luce on one ordering: ``log s_i = -β · rank_i`` with β fitted causally."""

    name: str
    feature: Callable[[EventContext], FloatArray | None]
    checkpoints: frozenset[str]
    training_seasons: int = TRAINING_SEASONS
    _beta: dict[tuple[int, str], float] = field(default_factory=dict, repr=False)

    def prepare(self, ctx: EventContext) -> FloatArray | None:
        return self.feature(ctx)

    def _fit(self, ctx: EventContext, history: History) -> float:
        key = (ctx.season, ctx.checkpoint)
        if key in self._beta:
            return self._beta[key]
        train = [
            (sig, finisher_indices(c, o))
            for c, sig, o in history
            if sig is not None and ctx.season - self.training_seasons <= c.season < ctx.season
        ]
        train = [(np.asarray(s), f) for s, f in train if len(f) >= 2]
        if not train:
            beta = 0.15  # weak, sensible prior until history exists
        else:

            def loglik(beta: float) -> float:
                return sum(pl_log_likelihood(-beta * s, f) for s, f in train)

            beta = golden_section_max(loglik, *BETA_BOUNDS)
        self._beta[key] = beta
        return beta

    def forecast(self, ctx: EventContext, signal: object, history: History) -> Forecast | None:
        if signal is None:
            return None
        beta = self._fit(ctx, history)
        ranks = np.asarray(signal, dtype=np.float64)
        return Forecast.from_log_strength(
            ctx.event_id,
            ctx.checkpoint,
            self.name,
            ctx.drivers,
            -beta * ranks,
            meta={"beta": beta},
        )


def default_baselines() -> list[Forecaster]:
    pre = frozenset({"C1", "C4"})
    post = frozenset({"C4"})
    return [
        Uniform(),
        RankPL("standings", standings_ranks, pre),
        RankPL("last_race", last_race_ranks, pre),
        RankPL("quali", quali_ranks, post),
        RankPL("grid", grid_ranks, post),
    ]
