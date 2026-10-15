"""Lap-by-lap race simulator (plan item 7, Phase 6).

A generative race, vectorised over simulated races x cars and stepped lap by lap:

1. **Start.** Cars leave in grid order (0.2 s per slot) and lap 1 adds start chaos.
2. **Every lap.** Each car runs the base lap of the circuit scaled by its pace from the lap
   model, tyre degradation since its last stop, fuel burn and lap-to-lap noise.
   * **Pit stops** follow a strategy drawn from the circuit's history (how many stops, and
     roughly when); a stop costs the circuit's measured pit loss.
   * **Overtaking.** A car that would end the lap within ``min_gap`` of the car ahead on
     the road gets past with a logistic probability in its pace advantage that lap.
     Otherwise it is held behind, which is how track position and dirty air keep slower
     cars ahead.
   * **Safety cars** arrive with the circuit's historical hazard. They neutralise the
     field for a few laps, make stops cheaper (cars due to stop soon pit), and bunch the
     field up when they end.
   * **Retirements** come from each car's P(finish), spread over the race.
3. **Finish.** Classified cars by race time, retirements behind them by laps completed.

It returns the same :class:`~f1p.eval.forecast.Forecast` as every model, so it plugs into
the harness and the stacking. Its implied Plackett-Luce strengths come from a projection: the
PL maximum-likelihood fit to its own sampled orders. Those strengths let it be scored on the
same order likelihood as the others, and pooled.

Inputs come from what is public at the checkpoint:

* pace from :mod:`f1p.models.lap_model`;
* the grid — the real grid at C4, and at C1 grids sampled from ``quali_v1``'s C1
  qualifying forecast;
* reliability from the world model's filter;
* circuit profiles (laps, base lap, pit loss, SC rate, stop counts) from the ``laps`` table.

The few free parameters are calibrated per season on earlier seasons only, by maximising
how much probability the simulator gave to the positions cars actually finished in.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from f1p.eval.baselines import History, grid_ranks
from f1p.eval.context import EventContext
from f1p.eval.forecast import Forecast, position_matrix, sample_orders, stable_seed
from f1p.models.lap_model import LapModelCache, car_pace

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]

GLOBAL_PROFILE = {"laps": 57, "base_lap_s": 90.0, "pit_loss_s": 22.0, "sc_per_race": 0.6}
GLOBAL_STOPS = (0.05, 0.6, 0.3, 0.05)  # P(0, 1, 2, 3 stops)
PROFILE_VISITS = 3


@dataclass(frozen=True, slots=True)
class CircuitProfile:
    laps: int
    base_lap_s: float
    pit_loss_s: float
    sc_per_race: float
    stops: tuple[float, ...]  # P(0, 1, 2, 3 stops)
    deg_per_lap: float = 0.0008  # fraction of a lap per lap of tyre age
    fuel_per_lap: float = 0.0006  # fraction of a lap per lap of fuel on board


@dataclass
class Trace:
    """Lap-by-lap state of the first ``k`` simulated races: race time after each lap (NaN
    once retired) and whether that lap ran behind the safety car."""

    k: int
    cum: list[FloatArray] = field(default_factory=list)  # one (k, n) array per lap
    sc: list[NDArray[np.bool_]] = field(default_factory=list)  # one (k,) array per lap


@dataclass(frozen=True, slots=True)
class SimParams:
    pace_scale: float = 1.0  # multiplies the lap model's pace gaps
    lap_noise: float = 0.004  # sd of a lap time, fraction of a lap
    start_sd: float = 0.8  # s of lap-1 chaos
    pass_a: float = -0.5  # overtaking logistic: intercept ...
    pass_b: float = 2.5  # ... and slope per second of pace advantage that lap
    min_gap: float = 0.4  # s: closer than this behind a car means attempting to pass
    sc_laps: int = 4
    sc_slow: float = 1.4  # lap-time factor behind the safety car
    sc_gap: float = 0.8  # s between cars when the field restarts
    sc_pit_factor: float = 0.55  # share of the pit loss paid under a safety car
    sc_pit_window: int = 6  # laps: cars due to stop this soon pit under a safety car


# -- circuit profile -------------------------------------------------------------------------
def _neutral_periods(laps: pl.DataFrame) -> pl.DataFrame:
    """Per event: the number of distinct neutralised periods (SC / VSC / red flag)."""
    by_lap = (
        laps.group_by("event_id", "lap")
        .agg(pl.col("neutral").mean().alias("share"))
        .with_columns((pl.col("share") >= 0.5).alias("n"))
        .sort("event_id", "lap")
    )
    return by_lap.group_by("event_id", maintain_order=True).agg(
        ((pl.col("n").cast(pl.Int8).diff().fill_null(pl.col("n").cast(pl.Int8)) == 1).sum()).alias(
            "periods"
        )
    )


def circuit_profile(
    known_laps: pl.DataFrame, events: pl.DataFrame, circuit_id: str, *, shrink: float = 2.0
) -> CircuitProfile:
    """The circuit's last few visits in ``known_laps`` (already cut at the checkpoint),
    shrunk towards the all-circuit averages."""
    if known_laps.is_empty():
        g = GLOBAL_PROFILE
        return CircuitProfile(
            int(g["laps"]), g["base_lap_s"], g["pit_loss_s"], g["sc_per_race"], GLOBAL_STOPS
        )
    from f1p.models.lap_model import pit_loss

    ev = events.select("event_id", "circuit_id", "season", "round")
    laps = known_laps.join(ev, on="event_id")
    per_event = laps.group_by("event_id", "circuit_id", "season", "round").agg(
        pl.col("lap").max().alias("laps"),
        (pl.col("lap_ms").filter(pl.col("clean")).median() / 1000.0).alias("base"),
    )
    periods = _neutral_periods(laps)
    loss = pit_loss(laps)
    stints = (
        laps.group_by("event_id", "driver_id")
        .agg(pl.col("stint").max().alias("stints"), pl.col("lap").max().alias("done"))
        .join(per_event.select("event_id", "laps"), on="event_id")
        .filter(pl.col("done") >= pl.col("laps") - 1)  # cars that ran to the end
        .with_columns((pl.col("stints") - 1).clip(0, 3).alias("stops"))
    )
    per_event = per_event.join(periods, on="event_id", how="left").join(
        loss, on="event_id", how="left"
    )
    g_sc = float(per_event["periods"].mean() or GLOBAL_PROFILE["sc_per_race"])  # type: ignore[arg-type]
    g_loss = float(per_event["pit_loss_s"].median() or GLOBAL_PROFILE["pit_loss_s"])  # type: ignore[arg-type]
    g_stops = np.bincount(stints["stops"].to_numpy(), minlength=4)[:4].astype(float)
    g_stops = g_stops / g_stops.sum() if g_stops.sum() else np.array(GLOBAL_STOPS)

    here = per_event.filter(pl.col("circuit_id") == circuit_id).sort("season", "round")
    here = here.tail(PROFILE_VISITS)
    k = float(here.height)
    if k == 0:
        return CircuitProfile(
            int(GLOBAL_PROFILE["laps"]),
            GLOBAL_PROFILE["base_lap_s"],
            g_loss,
            g_sc,
            tuple(float(x) for x in g_stops),
        )
    last = here.row(-1, named=True)
    sc = (float(here["periods"].fill_null(0).sum()) + shrink * g_sc) / (k + shrink)
    loss_here = here["pit_loss_s"].drop_nulls()
    pl_loss = float(loss_here.median()) if loss_here.len() else g_loss  # type: ignore[arg-type]
    st = stints.filter(pl.col("event_id").is_in(here["event_id"].to_list()))
    counts = np.bincount(st["stops"].to_numpy(), minlength=4)[:4].astype(float)
    stops = (counts + shrink * 10 * g_stops) / (counts.sum() + shrink * 10)
    return CircuitProfile(
        int(last["laps"]),
        float(last["base"] or GLOBAL_PROFILE["base_lap_s"]),
        float(np.clip(pl_loss, 12.0, 40.0)),
        sc,
        tuple(float(x) for x in stops),
    )


# -- the simulation --------------------------------------------------------------------------
def _stop_plan(
    profile: CircuitProfile, s: int, n: int, rng: np.random.Generator
) -> NDArray[np.bool_]:
    """``(s, n, laps + 1)``: True on the laps each car is planned to pit."""
    laps = profile.laps
    plan = np.zeros((s, n, laps + 1), dtype=bool)
    k = rng.choice(len(profile.stops), size=(s, n), p=np.asarray(profile.stops))
    for stops in range(1, len(profile.stops)):
        sel = np.argwhere(k == stops)
        if sel.size == 0:
            continue
        for j in range(1, stops + 1):
            centre = laps * j / (stops + 1)
            lap = np.rint(centre + rng.normal(0.0, 0.08 * laps, len(sel))).astype(int)
            lap = np.clip(lap, 2, laps - 2)
            plan[sel[:, 0], sel[:, 1], lap] = True
    return plan


def simulate(
    pace: FloatArray,
    grid: IntArray,
    p_finish: FloatArray,
    profile: CircuitProfile,
    params: SimParams,
    rng: np.random.Generator,
    record: Trace | None = None,
) -> tuple[IntArray, NDArray[np.bool_]]:
    """Simulate ``pace.shape[0]`` races.

    ``pace`` is ``(S, n)``: each car's lap-time effect as a fraction of a lap (lower is
    faster). ``grid`` is ``(S, n)``: each car's starting slot (0 = pole). Returns
    ``(orders, finished)`` like :func:`f1p.eval.forecast.sample_orders`. With ``record``,
    the first ``record.k`` races are also traced lap by lap (for the replay page).
    """
    s, n = pace.shape
    laps = profile.laps
    base = profile.base_lap_s
    rows = np.arange(s)
    pace = params.pace_scale * (pace - pace.mean(axis=1, keepdims=True))
    cum = 0.2 * grid.astype(np.float64) + rng.normal(0.0, params.start_sd, (s, n))
    age = np.zeros((s, n))
    plan = _stop_plan(profile, s, n, rng)
    retire_lap = np.where(
        rng.random((s, n)) < p_finish[None, :], laps + 1, rng.integers(1, laps + 1, (s, n))
    )
    alive = np.ones((s, n), dtype=bool)
    done_laps = np.zeros((s, n), dtype=np.int64)
    sc_left = np.zeros(s, dtype=np.int64)
    sc_hazard = min(profile.sc_per_race / max(laps - 3, 1), 0.5)
    for lap in range(1, laps + 1):
        # safety car: a new period may start (not on the last laps)
        start = (sc_left == 0) & (rng.random(s) < sc_hazard) & (lap > 1) & (lap < laps - 2)
        sc_left[start] = params.sc_laps
        under_sc = sc_left > 0
        if start.any():
            # cars due to stop within the window pit now instead
            window = plan[start, :, lap : lap + params.sc_pit_window + 1]
            due = window.any(axis=2)
            first = window.argmax(axis=2)
            sub = plan[start]
            idx_s, idx_c = np.nonzero(due)
            sub[idx_s, idx_c, lap + first[idx_s, idx_c]] = False
            sub[idx_s, idx_c, lap] = True
            plan[start] = sub
        pit = plan[:, :, lap] & alive
        lap_time = base * (
            1.0
            + pace
            + profile.deg_per_lap * age
            + profile.fuel_per_lap * (laps - lap)
            + rng.normal(0.0, params.lap_noise, (s, n))
        )
        lap_time = np.where(under_sc[:, None], base * params.sc_slow, lap_time)
        loss = np.where(under_sc[:, None], params.sc_pit_factor, 1.0) * profile.pit_loss_s
        lap_time = lap_time + pit * loss
        age = np.where(pit, 0.0, age + 1.0)

        # track position: resolve each car against the one ahead on the road
        road = np.argsort(np.where(alive, cum, np.inf), axis=1)
        new = np.where(alive, cum + lap_time, np.inf)
        prev_new = new[rows, road[:, 0]]
        prev_lt = lap_time[rows, road[:, 0]]
        for p in range(1, n):
            car = road[:, p]
            mine = new[rows, car]
            lt = lap_time[rows, car]
            close = np.isfinite(mine) & np.isfinite(prev_new) & (mine < prev_new + params.min_gap)
            advantage = prev_lt - lt
            p_pass = 1.0 / (1.0 + np.exp(-(params.pass_a + params.pass_b * advantage)))
            passed = close & ~under_sc & (rng.random(s) < p_pass)
            held = close & ~passed
            mine = np.where(held, prev_new + params.min_gap, mine)
            # a successful pass puts the car ahead; the car it passed is now behind it
            new[rows, car] = mine
            better = passed & (mine < prev_new)
            prev_new = np.where(better, prev_new, mine)
            prev_lt = np.where(better, prev_lt, lt)
        cum = new

        # the safety car ends: the field closes up behind the leader
        sc_left = np.maximum(sc_left - 1, 0)
        restart = under_sc & (sc_left == 0)
        if restart.any():
            order = np.argsort(cum[restart], axis=1)
            lead = np.take_along_axis(cum[restart], order[:, :1], axis=1)
            packed = lead + params.sc_gap * np.arange(n)[None, :]
            sub = cum[restart]
            np.put_along_axis(sub, order, packed, axis=1)
            sub[~alive[restart]] = np.inf
            cum[restart] = sub

        # retirements
        out = alive & (retire_lap == lap)
        alive &= ~out
        cum[out] = np.inf
        done_laps[alive] = lap
        if record is not None:
            k = record.k
            record.cum.append(np.where(alive[:k], cum[:k], np.nan))
            record.sc.append(under_sc[:k].copy())
    key = np.where(alive, cum, 1e9 - done_laps)
    return np.argsort(key, axis=1), alive


def implied_log_strength(orders: IntArray, n: int, iters: int = 200) -> FloatArray:
    """Plackett-Luce maximum likelihood fit to sampled full orders (Hunter's MM algorithm):
    the PL model closest to the simulator, used to score and pool it."""
    s = orders.shape[0]
    w = np.ones(n)
    wins = np.full(n, float(s))
    last = np.bincount(orders[:, -1], minlength=n)
    wins -= last  # every car "wins" a stage at each position except the last
    wins = np.maximum(wins, 0.5)
    pos = np.empty_like(orders)
    np.put_along_axis(pos, orders, np.arange(n)[None, :].repeat(s, 0), axis=1)
    for _ in range(iters):
        ws = w[orders]
        denom = np.cumsum(ws[:, ::-1], axis=1)[:, ::-1]  # suffix sums: stage t's remaining set
        inv = 1.0 / denom[:, :-1]  # the last stage has a single car: no information
        cuminv = np.concatenate([np.cumsum(inv, axis=1), np.cumsum(inv, axis=1)[:, -1:]], axis=1)
        exposure = np.take_along_axis(cuminv, pos, axis=1).sum(axis=0)
        w_new = wins / exposure
        w_new /= np.exp(np.mean(np.log(w_new)))
        if np.max(np.abs(np.log(w_new) - np.log(w))) < 1e-6:
            w = w_new
            break
        w = w_new
    return np.log(w)


# -- the forecaster --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SimSignal:
    pace_mean: FloatArray
    pace_sd: FloatArray
    grid: FloatArray | None  # C4: grid slots (rank - 1)
    p_finish: FloatArray
    profile: CircuitProfile
    has_pace: bool


@dataclass
class RaceSim:
    """The lap-by-lap simulator as a :class:`~f1p.eval.context.Forecaster`."""

    name: str = "race_sim_v1"
    checkpoints: frozenset[str] = frozenset({"C1", "C4"})
    targets: frozenset[str] = frozenset({"race"})
    n_sims: int = 4_000
    calibration_sims: int = 400
    calibration_events: int = 30
    training_seasons: int = 3
    grid_samples: int = 4_000
    keep_traces: int = 0  # also trace this many races of each forecast (replay page)
    last_traces: dict[str, Any] | None = field(default=None, repr=False)
    _laps: LapModelCache = field(default_factory=LapModelCache, repr=False)
    _params: dict[tuple[int, str], SimParams] = field(default_factory=dict, repr=False)
    _world: Any = field(default=None, repr=False)
    _quali: dict[int, Any] = field(default_factory=dict, repr=False)

    def _world_model(self) -> Any:
        if self._world is None:
            from f1p.models.world_v1 import WorldV1

            self._world = WorldV1()
        return self._world

    def prepare(self, ctx: EventContext) -> SimSignal | None:
        if "laps" not in ctx.tables:
            return None
        fitted = self._laps.fit_for(ctx)
        if fitted is None:
            return None
        mean, var = car_pace(fitted, ctx, self._laps.priors)
        has = ~np.isnan(mean)
        if has.sum() < max(10, ctx.n // 2):
            return None
        fill = float(np.nanmean(mean))
        worst = float(np.nanmax(mean))
        # cars without lap history: the slower half's average (newcomers are rarely quick)
        mean = np.where(has, mean, 0.5 * (fill + worst))
        var = np.where(has, var, np.nanmax(var) if np.isfinite(np.nanmax(var)) else 1e-4)
        grid = None
        if ctx.checkpoint == "C4":
            ranks = grid_ranks(ctx)
            if ranks is not None:
                grid = np.argsort(np.argsort(ranks, kind="stable"), kind="stable").astype(float)
        p_finish = self._world_model().raw_p_finish(ctx)
        known = ctx.known("laps")
        profile = circuit_profile(known, ctx.tables["events"], ctx.circuit_id)
        deg = fitted.degradation("MEDIUM")
        fuel = fitted.value("fuel")
        profile = replace(
            profile,
            deg_per_lap=float(np.clip(deg, 0.0, 0.003)) or profile.deg_per_lap,
            fuel_per_lap=float(np.clip(fuel, 0.0, 0.002)) or profile.fuel_per_lap,
        )
        return SimSignal(mean, np.sqrt(var), grid, p_finish, profile, True)

    # grids for C1: sampled from the C1 qualifying forecast
    def _grids(self, ctx: EventContext, s: int, rng: np.random.Generator) -> IntArray:
        from f1p.eval.harness import EventForecaster
        from f1p.models.quali_v1 import QualiV1

        key = id(ctx.tables["results"])
        if key not in self._quali:
            self._quali = {key: EventForecaster(ctx.tables, QualiV1(), "C1", target="quali")}
        fc, qctx = None, None
        try:
            fc, qctx = self._quali[key].forecast(ctx.event_id, as_of=ctx.as_of)
        except (KeyError, ValueError):
            fc = None
        if fc is None or fc.log_strength is None or qctx is None:
            orders = np.argsort(rng.random((s, ctx.n)), axis=1)
        else:
            index = {d: i for i, d in enumerate(ctx.drivers)}
            ls = np.full(ctx.n, float(np.min(fc.log_strength)) - 1.0)
            for d, v in zip(qctx.drivers, fc.log_strength, strict=True):
                if d in index:
                    ls[index[d]] = v
            orders, _ = sample_orders(ls, s, rng, stage_scales=fc.stage_scales)
        grid = np.empty_like(orders)
        np.put_along_axis(grid, orders, np.arange(ctx.n)[None, :].repeat(s, 0), axis=1)
        return grid

    def _inputs(
        self, ctx: EventContext, sig: SimSignal, s: int, rng: np.random.Generator
    ) -> tuple[FloatArray, IntArray]:
        """Per simulated race: each car's pace (drawn from the lap model's uncertainty) and
        its grid slot."""
        pace = sig.pace_mean[None, :] + rng.normal(0.0, 1.0, (s, ctx.n)) * sig.pace_sd[None, :]
        if sig.grid is not None:
            grid = np.broadcast_to(sig.grid.astype(np.intp), (s, ctx.n))
        else:
            grid = self._grids(ctx, s, rng)
        return pace, grid

    def _calibrate(self, ctx: EventContext, history: History) -> SimParams:
        key = (ctx.season, ctx.checkpoint)
        if key in self._params:
            return self._params[key]
        train = [
            (c, sig, o)
            for c, sig, o in history
            if isinstance(sig, SimSignal)
            and ctx.season - self.training_seasons <= c.season < ctx.season
        ][-self.calibration_events :]
        best, best_score = SimParams(), -np.inf
        inputs = []
        for c, sig, o in train:
            rng = np.random.default_rng(stable_seed(c.event_id, c.checkpoint, "cal"))
            inputs.append((c, sig, o, *self._inputs(c, sig, self.calibration_sims, rng)))
        if inputs:
            for scale in (0.7, 1.0, 1.4):
                for pass_a in (-2.0, -0.5, 1.0):
                    params = SimParams(pace_scale=scale, pass_a=pass_a)
                    score = 0.0
                    for c, sig, o, pace, grid in inputs:
                        rng = np.random.default_rng(stable_seed(c.event_id, "sim"))
                        orders, _ = simulate(pace, grid, sig.p_finish, sig.profile, params, rng)
                        pos = position_matrix(orders, c.n)
                        idx = c.index()
                        for k, d in enumerate(o.classified):
                            if d in idx:
                                score += float(np.log(pos[idx[d], k] + 1e-3))
                    if score > best_score:
                        best, best_score = params, score
        self._params[key] = best
        return best

    def forecast(self, ctx: EventContext, signal: Any, history: History) -> Forecast | None:
        if not isinstance(signal, SimSignal):
            return None
        params = self._calibrate(ctx, history)
        rng = np.random.default_rng(stable_seed(ctx.event_id, ctx.checkpoint, self.name))
        pace, grid = self._inputs(ctx, signal, self.n_sims, rng)
        record = Trace(self.keep_traces) if self.keep_traces else None
        orders, finished = simulate(
            pace, grid, signal.p_finish, signal.profile, params, rng, record
        )
        if record is not None:
            self.last_traces = {
                "event_id": ctx.event_id,
                "checkpoint": ctx.checkpoint,
                "drivers": list(ctx.drivers),
                "laps": signal.profile.laps,
                "base_lap_s": signal.profile.base_lap_s,
                "grid": grid[: record.k].tolist(),
                "cum": np.stack(record.cum, axis=1),  # (k, laps, n)
                "sc": np.stack(record.sc, axis=1),  # (k, laps)
                "orders": orders[: record.k].tolist(),
            }
        n = ctx.n
        pos = position_matrix(orders, n)
        wins = np.bincount(orders[:, 0], minlength=n).astype(float)
        p_win = (wins + 0.5) / (wins.sum() + 0.5 * n)  # no car is ever quite impossible
        return Forecast(
            ctx.event_id,
            ctx.checkpoint,
            self.name,
            ctx.drivers,
            pos,
            implied_log_strength(orders, n),
            {
                "pace_scale": params.pace_scale,
                "pass_a": params.pass_a,
                "laps": float(signal.profile.laps),
                "pit_loss_s": signal.profile.pit_loss_s,
                "sc_per_race": signal.profile.sc_per_race,
            },
            p_win,
            finished.mean(axis=0),
        )
