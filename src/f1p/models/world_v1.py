"""World model v1 as a forecaster: rating filter + checkpoint signals → Plackett-Luce.

At every checkpoint the model reads the rating filter's beliefs *as of* ``ctx.as_of`` (only
races whose results were public by then are assimilated) and turns them into
uncertainty-aware log-strengths. At C4 it adds what qualifying revealed — grid slot and
qualifying order — and a linear Plackett-Luce combiner, refitted each season on the previous
seasons only, weighs the signals.

Retirements are forecast too: the reliability filter's cause-specific rates give each car a
raw P(classified), recalibrated by a two-parameter logistic fitted on the same past seasons.
The forecast is then fully generative — who finishes, and in what order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from f1p.eval.baselines import History, finisher_indices, grid_ranks, quali_ranks
from f1p.eval.context import EventContext, Tables
from f1p.eval.forecast import Forecast
from f1p.eval.metrics import Outcome
from f1p.models import linear_pl
from f1p.models.features import quali_pace
from f1p.models.ratings import (
    FITTED_V1,
    RaceObs,
    RatingFilter,
    RatingParams,
    lineage_map,
    predictive_log_strength,
    race_observations,
    run_filter,
)
from f1p.models.reliability import ReliabilityFilter, run_reliability

FloatArray = NDArray[np.float64]

FEATURES = {
    "C1": ("rating",),
    "C4": ("rating", "grid", "quali"),
}
# Weak priors for the combiner: trust the rating at face value, grid and quali a little.
PRIORS = {"rating": 1.0, "grid": 0.5, "quali": 0.5, "pace": 0.5}


@dataclass
class _FilterState:
    filter: RatingFilter
    reliability: ReliabilityFilter
    races: list[RaceObs]
    known: FloatArray  # running max of result publication time (epoch s), per race
    lineage: dict[tuple[int, str], str]

    def step_at(self, as_of_ts: float) -> int:
        """Number of leading races whose results were public at ``as_of``."""
        return int(np.searchsorted(self.known, as_of_ts, side="right"))


def _log_rank(ranks: FloatArray | None, n: int) -> FloatArray:
    if ranks is None:
        return np.zeros(n)
    return -np.log(ranks)


def _logit(p: FloatArray) -> FloatArray:
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


@dataclass(frozen=True, slots=True)
class Signal:
    x: FloatArray  # n x p features for the Plackett-Luce combiner
    finish_logit: FloatArray  # raw logit P(classified) from the reliability filter


def fit_finish_calibration(
    samples: list[tuple[FloatArray, FloatArray]], *, l2: float = 1.0
) -> FloatArray:
    """MAP ``(a, b)`` of ``logit p = a + b · raw_logit`` (prior: identity, i.e. ``(0, 1)``)."""
    prior = np.array([0.0, 1.0])
    if not samples:
        return prior
    z = np.concatenate([s for s, _ in samples])
    y = np.concatenate([f for _, f in samples])
    from scipy.optimize import minimize

    def objective(w: FloatArray) -> tuple[float, FloatArray]:
        eta = w[0] + w[1] * z
        p = 1.0 / (1.0 + np.exp(-eta))
        ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
        r = y - p
        d = w - prior
        return -ll + 0.5 * l2 * float(d @ d), -np.array([r.sum(), (r * z).sum()]) + l2 * d

    return np.asarray(minimize(objective, prior, jac=True, method="L-BFGS-B").x)


@dataclass
class WorldV1:
    """Rating filter + linear Plackett-Luce combiner, refitted per season (causal)."""

    params: RatingParams = FITTED_V1
    name: str = "world_v1"
    checkpoints: frozenset[str] = frozenset({"C1", "C4"})
    training_seasons: int = 8
    l2: float = 1.0
    retirements: bool = True
    features: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(FEATURES))
    _states: dict[int, _FilterState] = field(default_factory=dict, repr=False)
    _weights: dict[tuple[int, str], FloatArray] = field(default_factory=dict, repr=False)
    _finish: dict[tuple[int, str], FloatArray] = field(default_factory=dict, repr=False)

    # -- the filter, run once per table set --------------------------------------------------
    def _state(self, tables: Tables) -> _FilterState:
        key = id(tables["results"])
        st = self._states.get(key)
        if st is None:
            races = race_observations(tables)
            filt, _ = run_filter(races, self.params, record=True)
            times = np.array(
                [r.known_at.timestamp() if r.known_at is not None else np.inf for r in races]
            )
            known = np.maximum.accumulate(times) if times.size else times
            st = _FilterState(filt, run_reliability(races), races, known, lineage_map(tables))
            self._states = {key: st}  # one table set at a time
        return st

    def _teams_step(self, ctx: EventContext) -> tuple[_FilterState, list[str], int]:
        st = self._state(ctx.tables)
        teams = [st.lineage.get((ctx.season, c), c) for c in ctx.constructors]
        return st, teams, st.step_at(ctx.as_of.timestamp())

    def beliefs(self, ctx: EventContext) -> tuple[FloatArray, FloatArray]:
        """(μ, σ²) of every car in the field, from races public at ``ctx.as_of``."""
        st, teams, step = self._teams_step(ctx)
        return st.filter.peek(ctx.drivers, teams, ctx.season, before_step=step)

    def raw_p_finish(self, ctx: EventContext) -> FloatArray:
        st, teams, step = self._teams_step(ctx)
        return st.reliability.p_finish(ctx.drivers, teams, step)

    # -- Forecaster protocol -------------------------------------------------------------
    def prepare(self, ctx: EventContext) -> Signal:
        mu, var = self.beliefs(ctx)
        cols = {"rating": predictive_log_strength(mu, var, self.params.beta)}
        if ctx.checkpoint == "C4":
            cols["grid"] = _log_rank(grid_ranks(ctx), ctx.n)
            cols["quali"] = _log_rank(quali_ranks(ctx), ctx.n)
            pace = quali_pace(ctx)
            cols["pace"] = pace if pace is not None else np.zeros(ctx.n)
        x = np.column_stack([cols[f] for f in self.features[ctx.checkpoint]])
        return Signal(x, _logit(self.raw_p_finish(ctx)))

    def _train(
        self, ctx: EventContext, history: History
    ) -> list[tuple[EventContext, Signal, Outcome]]:
        return [
            (c, sig, o)
            for c, sig, o in history
            if isinstance(sig, Signal)
            and ctx.season - self.training_seasons <= c.season < ctx.season
        ]

    def _fit(self, ctx: EventContext, history: History) -> FloatArray:
        key = (ctx.season, ctx.checkpoint)
        if key not in self._weights:
            names = self.features[ctx.checkpoint]
            samples = [(sig.x, finisher_indices(c, o)) for c, sig, o in self._train(ctx, history)]
            prior = np.array([PRIORS[n] for n in names])
            self._weights[key] = linear_pl.fit(
                samples, prior=prior, l2=self.l2, bounds=[(0.0, None)] * len(names)
            )
        return self._weights[key]

    def _fit_finish(self, ctx: EventContext, history: History) -> FloatArray:
        key = (ctx.season, ctx.checkpoint)
        if key not in self._finish:
            samples = []
            for c, sig, o in self._train(ctx, history):
                done = set(o.classified)
                samples.append(
                    (sig.finish_logit, np.array([d in done for d in c.drivers], dtype=float))
                )
            self._finish[key] = fit_finish_calibration(samples)
        return self._finish[key]

    def forecast(self, ctx: EventContext, signal: Any, history: History) -> Forecast:
        sig: Signal = signal
        w = self._fit(ctx, history)
        names = self.features[ctx.checkpoint]
        meta = {f"w_{n}": float(v) for n, v in zip(names, w, strict=True)}
        p_finish = None
        if self.retirements:
            a, b = self._fit_finish(ctx, history)
            p_finish = 1.0 / (1.0 + np.exp(-(a + b * sig.finish_logit)))
            meta |= {"finish_a": float(a), "finish_b": float(b)}
        return Forecast.from_log_strength(
            ctx.event_id,
            ctx.checkpoint,
            self.name,
            ctx.drivers,
            sig.x @ w,
            meta=meta,
            p_finish=p_finish,
        )


def ratings_table(model: WorldV1, tables: Tables, *, top: int = 20) -> pl.DataFrame:
    """Current driver and team beliefs, strongest first (for inspection and the CLI)."""
    st = model._state(tables)
    rows = [
        {"kind": "driver", "id": k, "mu": b.mu, "sigma": b.var**0.5, "season": b.season}
        for k, b in st.filter.drivers.items()
    ] + [
        {"kind": "team", "id": k, "mu": b.mu, "sigma": b.var**0.5, "season": b.season}
        for k, b in st.filter.teams.items()
    ]
    df = pl.DataFrame(rows)
    latest = int(df["season"].max())  # type: ignore[arg-type]
    return (
        df.filter(pl.col("season") == latest)
        .sort("mu", descending=True)
        .group_by("kind", maintain_order=True)
        .head(top)
    )
