"""World model v1 as a forecaster: rating filter + checkpoint signals → Plackett-Luce.

At every checkpoint the model reads the rating filter's beliefs *as of* ``ctx.as_of`` (only
races whose results were public by then are assimilated) and turns them into
uncertainty-aware log-strengths. At C4 it adds what qualifying revealed — grid slot and
qualifying order — and a linear Plackett-Luce combiner, refitted each season on the previous
seasons only, weighs the signals. The order is drawn place by place with a temperature that
falls smoothly down the field (the front is far more predictable than the midfield), the
curve itself fitted alongside the combiner.

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
from f1p.models.features import circuit_affinity, quali_pace
from f1p.models.ratings import (
    FITTED_V1,
    RaceObs,
    RatingFilter,
    RatingParams,
    lineage_map,
    observations,
    predictive_log_strength,
    run_filter,
)
from f1p.models.reliability import ReliabilityFilter, run_reliability

FloatArray = NDArray[np.float64]

FEATURES = {
    "C1": ("rating",),
    "C4": ("rating", "grid", "quali"),
}
# Weak priors for the combiner: trust the rating at face value, grid and quali a little.
PRIORS = {
    "rating": 1.0,
    "rating_q": 0.0,
    "grid": 0.5,
    "quali": 0.5,
    "pace": 0.5,
    "team_affinity": 0.0,
    "driver_affinity": 0.0,
}


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
    rain: float = 0.0  # P(wet race) as knowable at the checkpoint


def finish_design(sig_logit: FloatArray, x: FloatArray, rain: float | None = None) -> FloatArray:
    """Design matrix for P(classified): intercept, raw reliability logit, every pace signal
    (faster cars run at the front, away from first-lap incidents) and, if given, rain."""
    cols = [np.ones(sig_logit.size), sig_logit, x]
    if rain is not None:
        cols.append(np.full(sig_logit.size, rain))
    return np.column_stack(cols)


def rain_signal(ctx: EventContext) -> float:
    """P(wet race) as knowable at the checkpoint: the latest race-window forecast if one is
    public, else the circuit's climatology for the date (0 without weather data)."""
    if "weather_forecast" not in ctx.tables:
        return 0.0
    w = ctx.known("weather_forecast").filter(pl.col("event_id") == ctx.event_id)
    for kind in ("forecast", "climatology"):
        row = w.filter(pl.col("kind") == kind)
        if row.height:
            return float(row["wet_prob"][0])
    return 0.0


def fit_finish_calibration(
    samples: list[tuple[FloatArray, FloatArray]], *, n_features: int, l2: float = 1.0
) -> FloatArray:
    """MAP logistic weights for :func:`finish_design` (prior: the raw rate as-is)."""
    prior = np.zeros(n_features)
    prior[1] = 1.0
    if not samples:
        return prior
    z = np.concatenate([s for s, _ in samples])
    y = np.concatenate([f for _, f in samples])
    from scipy.optimize import minimize

    def objective(w: FloatArray) -> tuple[float, FloatArray]:
        eta = z @ w
        p = 1.0 / (1.0 + np.exp(-eta))
        ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
        d = w - prior
        return -ll + 0.5 * l2 * float(d @ d), -(z.T @ (y - p)) + l2 * d

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
    quali_obs: bool = False  # also assimilate qualifying sessions into the rating filter
    features: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(FEATURES))
    _states: dict[tuple[int, bool], _FilterState] = field(default_factory=dict, repr=False)
    n_stages: int = 0  # free per-place temperatures for the first places (if not smooth)
    smooth_stages: bool = True  # temperatures follow a 2-parameter curve over all places
    n_samples: int = 10_000  # Monte Carlo orders per forecast (P(win) is exact regardless)
    weather: bool = True  # rain changes the retirement rate (and, optionally, the temperature)
    rain_temperature: bool = False  # also let rain flatten the order (unstable: off by default)
    _weights: dict[tuple[int, str], linear_pl.PLFit] = field(default_factory=dict, repr=False)
    _finish: dict[tuple[int, str], FloatArray] = field(default_factory=dict, repr=False)

    # -- the filter, run once per table set --------------------------------------------------
    def _state(self, tables: Tables, quali: bool | None = None) -> _FilterState:
        quali = self.quali_obs if quali is None else quali
        key = (id(tables["results"]), quali)
        st = self._states.get(key)
        if st is None:
            races = observations(tables, quali=quali)
            filt, _ = run_filter(races, self.params, record=True)
            times = np.array(
                [r.known_at.timestamp() if r.known_at is not None else np.inf for r in races]
            )
            known = np.maximum.accumulate(times) if times.size else times
            st = _FilterState(filt, run_reliability(races), races, known, lineage_map(tables))
            # one table set at a time
            self._states = {k: v for k, v in self._states.items() if k[0] == key[0]}
            self._states[key] = st
        return st

    def _teams_step(
        self, ctx: EventContext, quali: bool | None = None
    ) -> tuple[_FilterState, list[str], int]:
        st = self._state(ctx.tables, quali)
        teams = [st.lineage.get((ctx.season, c), c) for c in ctx.constructors]
        return st, teams, st.step_at(ctx.as_of.timestamp())

    def beliefs(
        self, ctx: EventContext, quali: bool | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """(μ, σ²) of every car in the field, from sessions public at ``ctx.as_of`` (races
        only, or with ``quali`` also qualifying and sprint sessions)."""
        st, teams, step = self._teams_step(ctx, quali)
        return st.filter.peek(ctx.drivers, teams, ctx.season, before_step=step)

    def raw_p_finish(self, ctx: EventContext) -> FloatArray:
        st, teams, step = self._teams_step(ctx)
        return st.reliability.p_finish(ctx.drivers, teams, step)

    # -- Forecaster protocol -------------------------------------------------------------
    def prepare(self, ctx: EventContext) -> Signal:
        mu, var = self.beliefs(ctx)
        cols = {"rating": predictive_log_strength(mu, var, self.params.beta)}
        if "rating_q" in self.features[ctx.checkpoint]:
            mu_q, var_q = self.beliefs(ctx, quali=True)
            cols["rating_q"] = predictive_log_strength(mu_q, var_q, self.params.beta)
        wanted = self.features[ctx.checkpoint]
        if "team_affinity" in wanted:
            st = self._state(ctx.tables)
            cols["team_affinity"] = circuit_affinity(ctx, by="team", lineage=st.lineage)
        if "driver_affinity" in wanted:
            cols["driver_affinity"] = circuit_affinity(ctx, by="driver")
        if ctx.checkpoint == "C4":
            cols["grid"] = _log_rank(grid_ranks(ctx), ctx.n)
            cols["quali"] = _log_rank(quali_ranks(ctx), ctx.n)
            pace = quali_pace(ctx)
            cols["pace"] = pace if pace is not None else np.zeros(ctx.n)
        x = np.column_stack([cols[f] for f in self.features[ctx.checkpoint]])
        rain = rain_signal(ctx) if self.weather else 0.0
        return Signal(x, _logit(self.raw_p_finish(ctx)), rain)

    def _train(
        self, ctx: EventContext, history: History
    ) -> list[tuple[EventContext, Signal, Outcome]]:
        return [
            (c, sig, o)
            for c, sig, o in history
            if isinstance(sig, Signal)
            and ctx.season - self.training_seasons <= c.season < ctx.season
        ]

    def _fit(self, ctx: EventContext, history: History) -> linear_pl.PLFit:
        """Combiner weights, stage log-temperatures and the rain effect, from earlier
        seasons only."""
        key = (ctx.season, ctx.checkpoint)
        if key not in self._weights:
            names = self.features[ctx.checkpoint]
            train = self._train(ctx, history)
            samples = [(sig.x, finisher_indices(c, o)) for c, sig, o in train]
            prior = np.array([PRIORS[n] for n in names])
            n_stages = linear_pl.MAX_STAGES if self.smooth_stages else self.n_stages
            self._weights[key] = linear_pl.fit_model(
                samples,
                prior=prior,
                n_stages=n_stages,
                l2=self.l2,
                bounds=[(0.0, None)] * len(names),
                basis=linear_pl.smooth_stage_basis(n_stages) if self.smooth_stages else None,
                race_z=[sig.rain for _, sig, _ in train] if self._rain_temp else None,
            )
        return self._weights[key]

    def _fit_finish(self, ctx: EventContext, history: History) -> FloatArray:
        key = (ctx.season, ctx.checkpoint)
        if key not in self._finish:
            samples = []
            for c, sig, o in self._train(ctx, history):
                done = set(o.classified)
                samples.append(
                    (
                        finish_design(sig.finish_logit, sig.x, self._rain(sig)),
                        np.array([d in done for d in c.drivers], dtype=float),
                    )
                )
            n_features = 2 + len(self.features[ctx.checkpoint]) + int(self.weather)
            self._finish[key] = fit_finish_calibration(samples, n_features=n_features)
        return self._finish[key]

    @property
    def _rain_temp(self) -> bool:
        return self.weather and self.rain_temperature

    def _rain(self, sig: Signal) -> float | None:
        return sig.rain if self.weather else None

    def forecast(self, ctx: EventContext, signal: Any, history: History) -> Forecast:
        sig: Signal = signal
        fit = self._fit(ctx, history)
        theta = fit.theta
        names = self.features[ctx.checkpoint]
        meta = {f"w_{n}": float(v) for n, v in zip(names, fit.w, strict=True)}
        meta |= {f"stage_scale_{k + 1}": float(np.exp(t)) for k, t in enumerate(theta)}
        meta |= {"rain": sig.rain, "rain_gamma": fit.gamma}
        p_finish = None
        if self.retirements:
            wf = self._fit_finish(ctx, history)
            design = finish_design(sig.finish_logit, sig.x, self._rain(sig))
            p_finish = 1.0 / (1.0 + np.exp(-(design @ wf)))
            meta |= {"finish_a": float(wf[0]), "finish_b": float(wf[1])}
            meta |= {f"finish_{n}": float(v) for n, v in zip(names, wf[2:], strict=False)}
        return Forecast.from_log_strength(
            ctx.event_id,
            ctx.checkpoint,
            self.name,
            ctx.drivers,
            fit.log_strength(sig.x, sig.rain if self._rain_temp else 0.0),
            meta=meta,
            p_finish=p_finish,
            stage_scales=np.exp(theta) if theta.size else None,
            n_samples=self.n_samples,
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
