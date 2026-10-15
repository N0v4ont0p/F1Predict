"""Lap-level pace model (plan item 6, Phase 4b).

Every clean race lap in a trailing window of events is explained as

    log lap_s = base[event]
              + car[team, event] + driver[d]
              + compound[c, event] + deg[c, event] · tyre_age
              + fuel[event] · laps_remaining
              + traffic · 1{gap to the car ahead < 1 s}
              + ε,     ε ~ Student-t(nu)

which, on the log scale, is the plan's multiplicative model (effects are fractions of a lap:
0.01 = 1%). Priors make it a hierarchical model:

* car pace follows a **random walk** over each team's events (upgrades), with a larger step
  across a season boundary (new cars);
* drivers, per-event compound offsets, degradation and fuel slopes are shrunk towards zero
  or towards global effects shared by all events;
* the noise is Student-t, fitted by iteratively reweighted least squares, so traffic
  moments, lock-ups and slow laps that survived the 107% filter do not drag the estimates.

The posterior is the exact Gaussian (Laplace) approximation around the MAP — a few hundred
parameters, solved densely in well under a second — which makes it cheap to refit causally
before every event of a backtest, the same way the rating filter is read. (A NumPyro version
with full MCMC is a possible later check; at this size the Laplace posterior is accurate.)

What it gives the rest of the system:

* :func:`lap_pace_signal` — each car's expected pace (car + driver) at the next event as a %
  advantage, with its uncertainty, for the race combiner (the ``lap_pace`` feature);
* the fitted compound offsets, degradation per compound and fuel effect, plus
  :func:`pit_loss` — the inputs the lap-by-lap race simulator needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import sparse
from threadpoolctl import threadpool_limits

from f1p.eval.context import EventContext, Tables

FloatArray = NDArray[np.float64]

COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN")
TRAFFIC_GAP = 1.0  # s: following this closely costs (dirty air) or gains (DRS) time


@dataclass(frozen=True, slots=True)
class LapPriors:
    """Prior scales on the log-lap-time scale (0.01 = 1% of a lap)."""

    car0: float = 0.02  # first appearance of a team in the window
    car_step: float = 0.0004  # random-walk step between a team's races (tuned on 2024)
    car_reset: float = 0.012  # step across a season boundary (new cars)
    driver: float = 0.004
    compound_global: float = 0.05
    compound_event: float = 0.01
    deg_global: float = 0.002  # per lap of tyre age
    deg_event: float = 0.0008
    fuel_global: float = 0.002  # per lap of fuel left
    fuel_event: float = 0.0004
    traffic: float = 0.01
    nu: float = 4.0  # Student-t degrees of freedom


@dataclass
class LapFit:
    """A fitted window: parameter estimates, their posterior covariance and the index."""

    events: list[str]
    names: list[str]
    index: dict[str, int]
    mean: FloatArray
    cov: FloatArray
    sigma: float
    team_last: dict[str, tuple[int, str]]  # team → (param index of its latest event, season)
    team_season: dict[str, int]
    n_laps: int

    def value(self, name: str) -> float:
        return float(self.mean[self.index[name]]) if name in self.index else 0.0

    def sd(self, name: str) -> float:
        i = self.index.get(name)
        return float(np.sqrt(self.cov[i, i])) if i is not None else 0.0

    def compound_offset(self, compound: str) -> float:
        return self.value(f"compound:{compound}")

    def degradation(self, compound: str) -> float:
        return self.value(f"deg:{compound}")


class _Design:
    """Builds the sparse-ish design (dense is fine at this size) and the prior."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self.index: dict[str, int] = {}
        self.prior_prec: list[float] = []
        self.links: list[tuple[str, str, float]] = []  # (a, b, sd): a - b ~ N(0, sd²)

    def param(self, name: str, sd: float) -> int:
        if name not in self.index:
            self.index[name] = len(self.names)
            self.names.append(name)
            self.prior_prec.append(0.0 if sd <= 0 else 1.0 / sd**2)
        return self.index[name]

    def link(self, a: str, b: str, sd: float) -> None:
        self.links.append((a, b, sd))

    def precision(self) -> FloatArray:
        p = len(self.names)
        prec = np.diag(np.asarray(self.prior_prec, dtype=np.float64))
        for a, b, sd in self.links:
            i, j, w = self.index[a], self.index[b], 1.0 / sd**2
            prec[i, i] += w
            prec[j, j] += w
            prec[i, j] -= w
            prec[j, i] -= w
        return prec + np.eye(p) * 1e-10


def _prepare(laps: pl.DataFrame, tables: Tables) -> pl.DataFrame:
    """Clean laps with team lineage, season order and the regressors."""
    ev = tables["events"].select("event_id", "season", "round")
    res = (
        tables["results"]
        .filter(pl.col("session") == "race")
        .select("event_id", "driver_id", "constructor_id")
    )
    cs = tables["constructor_seasons"].select("season", "constructor_id", "lineage_id")
    return (
        laps.with_columns(
            (pl.col("lap").max().over("event_id") - pl.col("lap")).cast(pl.Float64).alias("fuel")
        )
        .filter(pl.col("clean"))
        .join(ev, on="event_id")
        .join(res, on=["event_id", "driver_id"])
        .join(cs, on=["season", "constructor_id"], how="left")
        .with_columns(
            pl.coalesce("lineage_id", "constructor_id").alias("team"),
            pl.col("compound").fill_null("UNKNOWN"),
            pl.col("tyre_age").fill_null(pl.col("lap")).cast(pl.Float64),
            (pl.col("gap_ahead_s").fill_null(99.0) < TRAFFIC_GAP).cast(pl.Float64).alias("traffic"),
            (pl.col("lap_ms").cast(pl.Float64) / 1000.0).log().alias("y"),
        )
    )


def fit(
    laps: pl.DataFrame,
    tables: Tables,
    *,
    priors: LapPriors | None = None,
    iterations: int = 6,
) -> LapFit | None:
    """Fit the window of clean laps in ``laps`` (already restricted by the caller)."""
    pr = priors or LapPriors()
    df = _prepare(laps, tables)
    if df.height < 50:
        return None
    order = df.select("event_id", "season", "round").unique().sort("season", "round")
    events = order["event_id"].to_list()
    d = _Design()
    for e in events:
        d.param(f"base:{e}", 0.0)  # flat
        d.param(f"fuel:{e}", pr.fuel_event)
    for c in COMPOUNDS:
        if c != "MEDIUM":
            d.param(f"compound:{c}", pr.compound_global)
        d.param(f"deg:{c}", pr.deg_global)
    d.param("fuel", pr.fuel_global)
    d.param("traffic", pr.traffic)
    # per-event deviations around the global compound, degradation and fuel effects
    for e, c in df.select("event_id", "compound").unique().iter_rows():
        if c != "MEDIUM":
            d.param(f"compound:{c}:{e}", pr.compound_event)
        d.param(f"deg:{c}:{e}", pr.deg_event)
    # car pace: random walk over each team's events
    team_events = (
        df.select("team", "event_id").unique().join(order, on="event_id").sort("season", "round")
    )
    team_last: dict[str, tuple[int, str]] = {}
    team_season: dict[str, int] = {}
    for team, grp in team_events.group_by("team", maintain_order=True):
        t = str(team[0])
        prev: tuple[str, int] | None = None
        for e, s in grp.select("event_id", "season").iter_rows():
            name = f"car:{t}:{e}"
            d.param(name, pr.car0 if prev is None else 0.0)
            if prev is not None:
                d.link(name, prev[0], pr.car_reset if s != prev[1] else pr.car_step)
            prev = (name, s)
        assert prev is not None
        team_last[t] = (d.index[prev[0]], str(prev[0].rsplit(":", 1)[-1]))
        team_season[t] = prev[1]
    for drv in df["driver_id"].unique().to_list():
        d.param(f"driver:{drv}", pr.driver)

    n, p = df.height, len(d.names)
    idx = d.index
    ev_ids = df["event_id"].to_list()
    comps = df["compound"].to_list()
    age = df["tyre_age"].to_numpy()
    fuel = df["fuel"].to_numpy()
    traffic = df["traffic"].to_numpy()
    r_i: list[int] = []
    c_i: list[int] = []
    v_i: list[float] = []

    def put(row: int, name: str, value: float) -> None:
        r_i.append(row)
        c_i.append(idx[name])
        v_i.append(value)

    for i, (e, t, drv, c) in enumerate(
        zip(ev_ids, df["team"].to_list(), df["driver_id"].to_list(), comps, strict=True)
    ):
        put(i, f"base:{e}", 1.0)
        put(i, f"car:{t}:{e}", 1.0)
        put(i, f"driver:{drv}", 1.0)
        if c != "MEDIUM":
            put(i, f"compound:{c}", 1.0)
            put(i, f"compound:{c}:{e}", 1.0)
        put(i, f"deg:{c}", age[i])
        put(i, f"deg:{c}:{e}", age[i])
        put(i, f"fuel:{e}", fuel[i])
        put(i, "fuel", fuel[i])
        if traffic[i]:
            put(i, "traffic", 1.0)
    x = sparse.csr_matrix((v_i, (r_i, c_i)), shape=(n, p))
    y = df["y"].to_numpy()
    prec = d.precision()

    # a few hundred parameters: one BLAS thread is fastest, and many concurrent backtests
    # with multi-threaded BLAS oversubscribe the cores (60x slower measured)
    with threadpool_limits(1, user_api="blas"):
        return _irls(x, y, prec, pr, iterations, events, d, team_last, team_season, n, p)


def _irls(
    x: sparse.csr_matrix,
    y: FloatArray,
    prec: FloatArray,
    pr: LapPriors,
    iterations: int,
    events: list[str],
    d: _Design,
    team_last: dict[str, tuple[int, str]],
    team_season: dict[str, int],
    n: int,
    p: int,
) -> LapFit:
    """Student-t IRLS around the Gaussian prior, then the Laplace covariance."""
    w = np.ones(n)
    sigma = 0.01
    beta = np.zeros(p)
    a = prec
    for _ in range(iterations):
        xw = x.T.multiply(w / sigma**2).tocsr()
        a = (xw @ x).toarray() + prec
        beta = np.linalg.solve(a, xw @ y)
        r = y - x @ beta
        sigma = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1e-4)
        w = (pr.nu + 1.0) / (pr.nu + (r / sigma) ** 2)
    cov = np.linalg.inv(a)
    return LapFit(events, d.names, dict(d.index), beta, cov, sigma, team_last, team_season, n)


def window_laps(ctx: EventContext, window: int) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """Laps of the last ``window`` races public at ``ctx.as_of`` (and their event ids)."""
    laps = ctx.known("laps")
    if laps.is_empty():
        return laps, ()
    ev = ctx.tables["events"].select("event_id", "season", "round")
    known = laps.select("event_id").unique().join(ev, on="event_id").sort("season", "round")
    recent = tuple(known.tail(window)["event_id"].to_list())
    return laps.filter(pl.col("event_id").is_in(list(recent))), recent


@dataclass
class LapModelCache:
    """One fit per distinct window of public races, shared by every context that sees it."""

    window: int = 24
    priors: LapPriors = field(default_factory=LapPriors)
    _fits: dict[tuple[int, tuple[str, ...]], LapFit | None] = field(default_factory=dict)
    _table: object | None = None

    def fit_for(self, ctx: EventContext) -> LapFit | None:
        if "laps" not in ctx.tables:
            return None
        table = ctx.tables["laps"]
        if self._table is not table:
            self._fits.clear()
            self._table = table
        laps, recent = window_laps(ctx, self.window)
        if not recent:
            return None
        key = (id(table), recent)
        if key not in self._fits:
            self._fits[key] = fit(laps, ctx.tables, priors=self.priors)
        return self._fits[key]


_DEFAULT_CACHE = LapModelCache()


def car_pace(
    fitted: LapFit, ctx: EventContext, priors: LapPriors | None = None
) -> tuple[FloatArray, FloatArray]:
    """Expected log-lap-time effect (car + driver) of every car in ``ctx`` at its next race,
    and its predictive variance. Unknown teams/drivers get NaN."""
    pr = priors or LapPriors()
    lineage = {
        str(c): str(lin or c)
        for c, lin in ctx.tables["constructor_seasons"]
        .filter(pl.col("season") == ctx.season)
        .select("constructor_id", "lineage_id")
        .iter_rows()
    }
    mean = np.full(ctx.n, np.nan)
    var = np.full(ctx.n, np.nan)
    for k, (drv, con) in enumerate(zip(ctx.drivers, ctx.constructors, strict=True)):
        team = lineage.get(con, con)
        last = fitted.team_last.get(team)
        if last is None:
            continue
        i = last[0]
        step = pr.car_reset if fitted.team_season.get(team) != ctx.season else pr.car_step
        m = float(fitted.mean[i])
        v = float(fitted.cov[i, i]) + step**2
        j = fitted.index.get(f"driver:{drv}")
        if j is not None:
            m += float(fitted.mean[j])
            v += float(fitted.cov[j, j]) + 2.0 * float(fitted.cov[i, j])
        else:
            v += pr.driver**2
        mean[k], var[k] = m, v
    return mean, var


def lap_pace_signal(ctx: EventContext, cache: LapModelCache | None = None) -> FloatArray:
    """Expected pace advantage (% of a lap, centred, higher = faster) from the lap model.
    Cars without lap history get the field mean; zero when no laps are public."""
    fitted = (cache or _DEFAULT_CACHE).fit_for(ctx)
    if fitted is None:
        return np.zeros(ctx.n)
    mean, _ = car_pace(fitted, ctx, (cache or _DEFAULT_CACHE).priors)
    if np.all(np.isnan(mean)):
        return np.zeros(ctx.n)
    fill = float(np.nanmean(mean))
    arr = np.where(np.isnan(mean), fill, mean)
    return -100.0 * (arr - arr.mean())


def pit_loss(laps: pl.DataFrame) -> pl.DataFrame:
    """Per event: the median time lost to a pit stop (in-lap + out-lap minus two of the
    car's median clean laps), in seconds — a simulator input."""
    base = (
        laps.filter(pl.col("clean"))
        .group_by("event_id", "driver_id")
        .agg(pl.col("lap_ms").median().alias("ref_any"))
    )
    pits = laps.filter((pl.col("in_lap") | pl.col("out_lap")) & ~pl.col("neutral")).join(
        base, on=["event_id", "driver_id"]
    )
    per_stop = (
        pits.with_columns(
            pl.when(pl.col("in_lap"))
            .then(pl.col("stint"))
            .otherwise(pl.col("stint") - 1)
            .alias("stop")
        )
        .group_by("event_id", "driver_id", "stop")
        .agg((pl.col("lap_ms") - pl.col("ref_any")).sum().alias("loss_ms"), pl.len().alias("k"))
        .filter(pl.col("k") == 2)
    )
    return per_stop.group_by("event_id").agg(
        (pl.col("loss_ms").median() / 1000.0).alias("pit_loss_s"), pl.len().alias("stops")
    )


def as_of_fit(
    tables: Tables, as_of: datetime, *, window: int = 24, priors: LapPriors | None = None
) -> LapFit | None:
    """Fit the window of races public at ``as_of`` (outside a forecasting context)."""
    from f1p.core.clock import AsOf, known_at

    laps = known_at(tables["laps"], AsOf(as_of))
    ev = tables["events"].select("event_id", "season", "round")
    known = laps.select("event_id").unique().join(ev, on="event_id").sort("season", "round")
    recent = known.tail(window)["event_id"].to_list()
    return fit(laps.filter(pl.col("event_id").is_in(recent)), tables, priors=priors)


def pace_backtest(
    tables: Tables,
    seasons: tuple[int, int],
    *,
    window: int = 24,
    driver_races: int = 5,
    priors: LapPriors | None = None,
) -> pl.DataFrame:
    """Held-out pace: predict every car's clean-lap pace deficit at each race from the races
    before it, and score against the deficit it actually ran (``race_pace``, centred).

    Predictors: the lap model; the exponentially weighted race-pace signal used by
    ``world_v1``; the driver's median deficit over their last ``driver_races`` races.
    One row per (event, predictor) with RMSE (percentage points) and Spearman correlation.
    """
    from f1p.eval.context import build_contexts
    from f1p.eval.metrics import spearman
    from f1p.models.features import race_pace_signal

    cache = LapModelCache(window=window, priors=priors or LapPriors())
    pace = tables["race_pace"]
    ev = tables["events"].select("event_id", "season", "round")
    hist = pace.join(ev, on="event_id")
    rows = []
    for ctx, _, _ in build_contexts(tables, seasons=seasons, checkpoints=("C1",)):
        actual = pace.filter(pl.col("event_id") == ctx.event_id)
        if actual.height < 10:
            continue
        truth = dict(actual.select("driver_id", "pace_deficit_pct").iter_rows())
        keep = [i for i, d in enumerate(ctx.drivers) if d in truth]
        if len(keep) < 10:
            continue
        before = hist.filter(
            (pl.col("season") < ctx.season)
            | ((pl.col("season") == ctx.season) & (pl.col("round") < ctx.round))
        )
        med = {
            d: float(v)
            for d, v in before.sort("season", "round")
            .group_by("driver_id")
            .agg(pl.col("pace_deficit_pct").tail(driver_races).median())
            .iter_rows()
        }
        y = np.array([truth[ctx.drivers[i]] for i in keep])
        y = y - y.mean()
        preds = {
            "lap_model": -lap_pace_signal(ctx, cache),
            "ew_race_pace": -race_pace_signal(ctx),
        }
        dm = np.array([med.get(d, np.nan) for d in ctx.drivers])
        preds["driver_median"] = (
            np.where(np.isnan(dm), np.nanmean(dm), dm)
            if np.isfinite(np.nanmean(dm))
            else np.zeros(ctx.n)
        )
        for name, pred in preds.items():
            p = np.asarray(pred, dtype=np.float64)[keep]
            if not np.any(p):
                continue  # no history for this predictor yet
            p = p - p.mean()
            rows.append(
                {
                    "event_id": ctx.event_id,
                    "season": ctx.season,
                    "predictor": name,
                    "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
                    "spearman": spearman(p, y),
                    "cars": len(keep),
                }
            )
    return pl.DataFrame(rows)
