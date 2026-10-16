"""World model v1 — a Gaussian-filtered Plackett-Luce rating model (plan §7.1, results era).

Each car's log-strength is the sum of a **driver** skill and a **team-lineage** car performance,
both Gaussian beliefs ``N(μ, σ²)`` that:

* drift a little before every race the entity takes part in (form, development);
* drift more at a season boundary, where team strength also mean-reverts — and far more in a
  **regulation-reset** season (the size of that reset effect is *learned*, replacing the
  legacy hand-set 0.35 constant);
* are updated after every race from the classified finishing order by one Newton (Laplace /
  extended-Kalman) step on the exact Plackett-Luce likelihood — the Gaussian-filter view of
  Weng & Lin (2011, JMLR 12) and Glickman's online paired-comparison filters, written in
  natural log-strength units so it is scale-free and well-behaved for any field size.
  Retirements are *censored*: they are left out of the observed order rather than ranked
  last (fixes legacy defect D3).

The filter is causal by construction — the belief before race *r* uses only races before
*r* — and it records every entity's belief history, so the state "as of" any point in time is
recovered exactly without re-running it. Hyper-parameters are fitted by maximising the
**prequential** (one-step-ahead) log-likelihood on seasons strictly before evaluation, inside
bounds that keep every one of them physically meaningful.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Any, ClassVar

import numpy as np
import polars as pl
from numpy.typing import NDArray

from f1p.core.regulations import MAJOR_RESETS, POWER_UNIT_RESETS

FloatArray = NDArray[np.float64]
PROBIT = math.pi / 8.0  # logistic ≈ probit moment-matching constant (MacKay 1992)
VAR_FLOOR = 1e-6
# Every driver enters at the same prior mean and driver beliefs never mean-revert, so a common
# shift of all driver means changes no prediction: the debutant mean is a gauge, fixed here.
# (Team beliefs do revert towards 0, which makes the new-team mean identifiable.)
DRIVER_MU0 = 0.0
ADAPT_CAP = 4.0  # largest variance inflation of the adaptive response in one race
ADAPT_ROUNDS = 8  # R1 acts in the first this many races of a reset season


@dataclass(frozen=True, slots=True)
class RatingParams:
    """Filter hyper-parameters (standard deviations are on the log-strength scale)."""

    beta: float = 0.6  # extra per-race performance noise beyond the Plackett-Luce Gumbel
    beta_quali: float = 0.6  # the same for a qualifying session (one lap: usually less noisy)
    tau_driver: float = 0.03  # driver drift per race
    tau_driver_season: float = 0.25  # extra driver drift per season boundary
    tau_team: float = 0.05  # team drift per race (in-season development)
    tau_team_season: float = 0.5  # team drift per season boundary
    reset_multiplier: float = 1.8  # scales tau_team_season into a regulation-reset season
    team_carryover: float = 0.8  # team mean kept across a season boundary (rest reverts to 0)
    driver_sigma0: float = 0.8  # debutant spread (the debutant mean is the gauge, fixed at 0)
    team_mu0: float = -0.5  # new team-lineage prior mean
    team_sigma0: float = 1.0
    # Regulation resets (docs/plan/RESET_PLAN.md). At their defaults the filter is exactly the
    # one without them; each acts only on the move into, or the races of, a reset season.
    reset_pu_ratio: float = 1.0  # H1: power-unit resets' multiplier / chassis resets'
    reset_carryover_ratio: float = 1.0  # H3: times team_carryover across a reset boundary
    reset_drift: float = 1.0  # H2: per-race team drift times this at the start of a reset season...
    reset_drift_races: float = 3.0  # ... decaying back to 1 over about this many races
    # Adaptive response (docs/plan/CALIBRATION_ADAPTIVE_PLAN.md, ADR): after a race, a team the
    # result surprised (standardised innovation z² > 1) has its variance inflated by
    # min(1 + κ·(z² - 1), ADAPT_CAP). κ = adapt_reset in the first ADAPT_ROUNDS races of a
    # reset season (R1), plus adapt_all in every race (R2). At 0 nothing changes.
    adapt_reset: float = 0.0
    adapt_all: float = 0.0

    # Box constraints for fitting: every value stays physically meaningful.
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "beta": (0.05, 3.0),
        "beta_quali": (0.05, 3.0),
        "tau_driver": (0.001, 0.3),
        "tau_driver_season": (0.01, 1.5),
        "tau_team": (0.001, 0.4),
        "tau_team_season": (0.01, 2.0),
        "reset_multiplier": (1.0, 4.0),
        "team_carryover": (0.0, 1.0),
        "driver_sigma0": (0.05, 3.0),
        "team_mu0": (-3.0, 1.0),
        "team_sigma0": (0.05, 3.0),
        "reset_pu_ratio": (0.5, 4.0),
        "reset_carryover_ratio": (0.0, 1.25),
        "reset_drift": (1.0, 20.0),
        "reset_drift_races": (0.5, 10.0),
        "adapt_reset": (0.0, 5.0),
        "adapt_all": (0.0, 5.0),
    }
    RESET_FIELDS: ClassVar[tuple[str, ...]] = (
        "reset_pu_ratio",
        "reset_carryover_ratio",
        "reset_drift",
        "reset_drift_races",
    )
    ADAPT_FIELDS: ClassVar[tuple[str, ...]] = ("adapt_reset", "adapt_all")

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    def as_vector(self) -> FloatArray:
        """Unconstrained coordinates: each value's logit within its bounds."""
        out = []
        for name in self.names():
            lo, hi = self.BOUNDS[name]
            u = (getattr(self, name) - lo) / (hi - lo)
            u = min(max(u, 1e-6), 1.0 - 1e-6)
            out.append(math.log(u / (1.0 - u)))
        return np.array(out)

    @classmethod
    def from_vector(cls, v: Iterable[float]) -> RatingParams:
        vals = {}
        for name, x in zip(cls.names(), v, strict=True):
            lo, hi = cls.BOUNDS[name]
            vals[name] = lo + (hi - lo) / (1.0 + math.exp(-float(np.clip(x, -40.0, 40.0))))
        return cls(**vals)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RatingParams:
        return cls(**{k: float(v) for k, v in d.items() if k in cls.names()})


# Fitted with ``fit_params(races, train_until=2013, score_from=1980)`` on F1DB 2026-R17
# (prequential log-likelihood 1980-2013: -12451 vs -13219 at the defaults; 2014-2026,
# out of sample: -7795 vs -8005). Only pre-2014 races were used, so evaluating on 2014+
# with these values is honest. The fit says the car dominates: team beliefs move and driver
# skill is near-static.
FITTED_V1 = RatingParams(
    beta=0.8089,
    tau_driver=0.001,
    tau_driver_season=0.01,
    tau_team=0.0955,
    tau_team_season=0.4407,
    reset_multiplier=1.1105,
    team_carryover=0.7913,
    driver_sigma0=0.3432,
    team_mu0=0.5121,
    team_sigma0=1.1338,
)


# --------------------------------------------------------------------------- observations
@dataclass(frozen=True, slots=True)
class RaceObs:
    """One ranked session as the filter sees it: a race, or (``kind="quali"``) a qualifying
    session, which observes the same car+driver strengths with its own noise."""

    event_id: str
    season: int
    round: int
    drivers: tuple[str, ...]  # every starter
    teams: tuple[str, ...]  # team lineage of each starter
    finish_order: NDArray[np.intp]  # indices into drivers of classified finishers, best first
    known_at: datetime | None = None  # when the result became public
    status: tuple[str, ...] = ()  # StatusClass value of each starter
    kind: str = "race"

    @property
    def is_race(self) -> bool:
        return self.kind == "race"


def lineage_map(tables: dict[str, pl.DataFrame]) -> dict[tuple[int, str], str]:
    cs = tables["constructor_seasons"]
    return {
        (int(s), str(c)): str(lin or c)
        for s, c, lin in cs.select("season", "constructor_id", "lineage_id").iter_rows()
    }


def observations(tables: dict[str, pl.DataFrame], *, quali: bool = True) -> list[RaceObs]:
    """Races and (optionally) qualifying sessions, in the order they happened."""
    races = race_observations(tables)
    if not quali:
        return races
    every = (
        races
        + qualifying_observations(tables, "sq")
        + race_observations(tables, session="sprint")
        + qualifying_observations(tables, "quali")
    )
    return sorted(every, key=lambda o: (o.season, o.round, WEEKEND_ORDER[o.kind]))


# Within a weekend: sprint qualifying, sprint, qualifying, race (every non-race session is
# public before the race starts, so their relative order never leaks anything).
WEEKEND_ORDER = {"sq": 0, "sprint": 1, "quali": 2, "race": 3}


def qualifying_observations(
    tables: dict[str, pl.DataFrame], session: str = "quali"
) -> list[RaceObs]:
    """Qualifying orders (classified participants only) as filter observations."""
    lineage = tables["constructor_seasons"].select("season", "constructor_id", "lineage_id")
    events = tables["events"].select("event_id", "season", "round")
    q = (
        tables["qualifying"]
        .filter((pl.col("session") == session) & pl.col("position").is_not_null())
        .join(events, on="event_id")
        .join(lineage, on=["season", "constructor_id"], how="left")
        .with_columns(pl.coalesce("lineage_id", "constructor_id").alias("team"))
        .sort("season", "round", "position")
    )
    out = []
    for (event_id,), g in q.group_by(["event_id"], maintain_order=True):
        g = g.unique("driver_id", keep="first", maintain_order=True)
        if g.height < 2:
            continue
        out.append(
            RaceObs(
                event_id=str(event_id),
                season=int(g["season"][0]),
                round=int(g["round"][0]),
                drivers=tuple(g["driver_id"].to_list()),
                teams=tuple(g["team"].to_list()),
                finish_order=np.arange(g.height, dtype=np.intp),
                known_at=g["observed_at"].max(),  # type: ignore[arg-type]
                kind=session,
            )
        )
    return out


def race_observations(tables: dict[str, pl.DataFrame], *, session: str = "race") -> list[RaceObs]:
    """Chronological race (or sprint) observations: starters, lineage, finish order."""
    lineage = tables["constructor_seasons"].select("season", "constructor_id", "lineage_id")
    r = (
        tables["results"]
        .filter((pl.col("session") == session) & ~pl.col("status").is_in(["dns", "dnq", "dnpq"]))
        .filter(~pl.col("shared_car"))
        .join(lineage, on=["season", "constructor_id"], how="left")
        .with_columns(pl.coalesce("lineage_id", "constructor_id").alias("team"))
        .sort("season", "round", "display_order")
    )
    out = []
    for (event_id,), g in r.group_by(["event_id"], maintain_order=True):
        drivers = tuple(g["driver_id"].to_list())
        if len(set(drivers)) != len(drivers):  # same driver in two cars (pre-1960) — skip
            continue
        fin = g.with_row_index("i").filter(pl.col("classified")).sort("position")
        out.append(
            RaceObs(
                event_id=str(event_id),
                season=int(g["season"][0]),
                round=int(g["round"][0]),
                drivers=drivers,
                teams=tuple(g["team"].to_list()),
                finish_order=fin["i"].to_numpy().astype(np.intp),
                known_at=g["observed_at"].max(),  # type: ignore[arg-type]
                status=tuple(g["status"].to_list()),
                kind=session,
            )
        )
    return out


# --------------------------------------------------------------------------- PL calculus
def pl_grad_hess(eta: FloatArray) -> tuple[FloatArray, FloatArray, float]:
    """Gradient, negative diagonal Hessian and log-likelihood of a full PL order.

    ``eta`` holds log-strengths already arranged best-first. With ``π_ik`` the stage-k choice
    probability of item i, ``g_i = 1 - Σ_{k≤i} π_ik`` and ``h_i = Σ_{k≤i} π_ik (1 - π_ik)``.
    """
    z = eta - eta.max()
    e = np.exp(z)
    suffix = np.cumsum(e[::-1])[::-1]
    a1 = np.cumsum(1.0 / suffix)
    a2 = np.cumsum(1.0 / suffix**2)
    g = 1.0 - e * a1
    h = e * a1 - e**2 * a2
    ll = float(np.sum(z - np.log(suffix)))
    return g, np.maximum(h, 0.0), ll


def predictive_log_strength(mu: FloatArray, var: FloatArray, beta: float) -> FloatArray:
    """Uncertainty-aware log-strengths: centre, then shrink each car by its predictive spread
    against an average rival (probit approximation to integrating the Gaussian belief)."""
    s2 = var + var.mean() + 2.0 * beta**2
    return (mu - mu.mean()) / np.sqrt(1.0 + PROBIT * s2)


# --------------------------------------------------------------------------- filter
@dataclass(slots=True)
class Belief:
    mu: float
    var: float
    season: int


def team_surprise(
    order: NDArray[np.intp],
    teams: Sequence[str],
    drv: Sequence[Belief],
    tms: Sequence[Belief],
    g_eff: FloatArray,
    h_eff: FloatArray,
) -> dict[str, float]:
    """Each team's squared standardised innovation z² from one result, on the prior beliefs.

    Car j's result is the Gaussian pseudo-observation of its log-strength with innovation
    e_j = g_eff_j / h_eff_j and noise 1 / h_eff_j. The team's score G = Σ_j g_eff_j = Σ_j h_j e_j
    over its classified cars has, under the filter's own predictive,
    Var G = Σ h_j + Σ h_j² σ²_driver,j + σ²_team (Σ h_j)²  (the cars share the team belief),
    so z² = G² / Var G has expectation 1 when the filter is right.
    """
    score: dict[str, list[float]] = {}
    for j, i in enumerate(int(k) for k in order):
        s = score.setdefault(teams[i], [0.0, 0.0, 0.0, tms[i].var])
        h = float(h_eff[j])
        s[0] += float(g_eff[j])
        s[1] += h
        s[2] += h * h * drv[i].var
    return {t: g * g / (sh + sd + vt * sh * sh) for t, (g, sh, sd, vt) in score.items() if sh > 0.0}


class BeliefHistory:
    """Post-update belief of one entity after each step it took part in."""

    __slots__ = ("states", "steps")

    def __init__(self) -> None:
        self.steps: list[int] = []
        self.states: list[tuple[float, float, int]] = []

    def at(self, step: int) -> tuple[float, float, int] | None:
        """State after the last update strictly before ``step``."""
        i = bisect_left(self.steps, step) - 1
        return self.states[i] if i >= 0 else None


class RatingFilter:
    def __init__(self, params: RatingParams, *, record: bool = False):
        self.p = params
        self.drivers: dict[str, Belief] = {}
        self.teams: dict[str, Belief] = {}
        self.step = 0
        self.record = record
        self.driver_hist: dict[str, BeliefHistory] = {}
        self.team_hist: dict[str, BeliefHistory] = {}

    # -- time evolution ------------------------------------------------------------------
    def _evolve(
        self,
        mu: float,
        var: float,
        last: int,
        season: int,
        *,
        driver: bool,
        round_: int | None = None,
    ) -> tuple[float, float]:
        """The belief moved from season ``last`` to the race ``round_`` of ``season`` (one
        race's drift, plus the season boundaries crossed)."""
        p = self.p
        if season != last:
            gap = season - last
            if driver:
                var += gap * p.tau_driver_season**2
            else:
                crossed = [s for s in MAJOR_RESETS if last < s <= season]
                mult = 1.0
                if crossed:
                    mult = p.reset_multiplier
                    if any(s in POWER_UNIT_RESETS for s in crossed):
                        mult *= p.reset_pu_ratio  # H1
                tau = p.tau_team_season * mult
                mu *= p.team_carryover**gap * (p.reset_carryover_ratio if crossed else 1.0)  # H3
                var += gap * tau**2
        if driver:
            var += p.tau_driver**2
        else:
            var += (p.tau_team * self._early_reset_factor(season, round_)) ** 2  # H2
        return mu, var

    def _early_reset_factor(self, season: int, round_: int | None) -> float:
        """H2: the in-season team drift is ``reset_drift`` times larger at the first race of a
        reset season and decays back to normal: 1 + (m - 1)·exp(-(r - 1) / L)."""
        p = self.p
        if p.reset_drift == 1.0 or round_ is None or season not in MAJOR_RESETS:
            return 1.0
        return 1.0 + (p.reset_drift - 1.0) * math.exp(-(round_ - 1) / p.reset_drift_races)

    def _adapt_strength(self, race: RaceObs) -> float:
        """ADR's κ for this session: adapt_all in every race, plus adapt_reset in the first
        ADAPT_ROUNDS races of a reset season; 0 for qualifying."""
        if not race.is_race:
            return 0.0
        p = self.p
        early_reset = race.season in MAJOR_RESETS and race.round <= ADAPT_ROUNDS
        return p.adapt_all + (p.adapt_reset if early_reset else 0.0)

    def _prior(self, *, driver: bool) -> tuple[float, float]:
        p = self.p
        return (DRIVER_MU0, p.driver_sigma0**2) if driver else (p.team_mu0, p.team_sigma0**2)

    def _enter(
        self,
        store: dict[str, Belief],
        key: str,
        season: int,
        *,
        driver: bool,
        drift: bool = True,
        round_: int | None = None,
    ) -> Belief:
        b = store.get(key)
        if b is None:
            mu, var = self._prior(driver=driver)
            b = Belief(mu, var, season)
            store[key] = b
            return b
        if drift or season != b.season:
            b.mu, b.var = self._evolve(b.mu, b.var, b.season, season, driver=driver, round_=round_)
            b.season = season
        return b

    def peek(
        self,
        drivers: Sequence[str],
        teams: Sequence[str],
        season: int,
        *,
        before_step: int | None = None,
        round_: int | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Prior (μ, σ²) for each car at ``season`` (race ``round_``) without mutating state.

        With ``before_step`` (requires ``record=True``) the beliefs are those held before race
        number ``before_step`` was assimilated — the filter's state at that moment in time.
        """
        mus, vars_ = [], []
        for d, t in zip(drivers, teams, strict=True):
            md, vd = self._peek_one(d, season, before_step, driver=True, round_=round_)
            mt, vt = self._peek_one(t, season, before_step, driver=False, round_=round_)
            mus.append(md + mt)
            vars_.append(vd + vt)
        return np.array(mus), np.array(vars_)

    def _peek_one(
        self,
        key: str,
        season: int,
        before_step: int | None,
        *,
        driver: bool,
        round_: int | None = None,
    ) -> tuple[float, float]:
        if before_step is None:
            b = (self.drivers if driver else self.teams).get(key)
            state = (b.mu, b.var, b.season) if b is not None else None
        else:
            if not self.record:
                raise ValueError("peek(before_step=...) needs a filter run with record=True")
            h = (self.driver_hist if driver else self.team_hist).get(key)
            state = h.at(before_step) if h is not None else None
        if state is None:
            return self._prior(driver=driver)
        mu, var, last = state
        return self._evolve(mu, var, last, season, driver=driver, round_=round_)

    # -- observation update ----------------------------------------------------------------
    def update(self, race: RaceObs) -> float:
        """Assimilate one session; returns the prequential log-likelihood of its order (races
        only — qualifying sharpens the beliefs but is not what we score)."""
        # Strengths drift once per weekend: qualifying is observed without extra drift and
        # the race that follows it applies the weekend's drift.
        drift = race.is_race
        drv = [
            self._enter(self.drivers, d, race.season, driver=True, drift=drift, round_=race.round)
            for d in race.drivers
        ]
        tms = [
            self._enter(self.teams, t, race.season, driver=False, drift=drift, round_=race.round)
            for t in race.teams
        ]
        order = race.finish_order
        beta = self.p.beta if race.kind in ("race", "sprint") else self.p.beta_quali
        loglik = 0.0
        if order.size >= 2:
            mu = np.array([drv[i].mu + tms[i].mu for i in order])
            var = np.array([drv[i].var + tms[i].var for i in order])
            if race.is_race:
                loglik = _pl_loglik(predictive_log_strength(mu, var, beta))
            g, h, _ = pl_grad_hess(mu)
            # Integrate out the per-session performance noise β (Gaussian pseudo-observation).
            b2 = beta**2
            g_eff = g / (1.0 + b2 * h)
            h_eff = h / (1.0 + b2 * h)
            kappa = self._adapt_strength(race)
            surprise = (
                team_surprise(order, race.teams, drv, tms, g_eff, h_eff) if kappa > 0.0 else {}
            )
            for j, i in enumerate(order):
                for b in (drv[i], tms[i]):  # sequential Kalman updates of the two components
                    v_car = drv[i].var + tms[i].var
                    denom = 1.0 + v_car * h_eff[j]
                    b.mu += b.var * g_eff[j] / denom
                    b.var = max(b.var - b.var**2 * h_eff[j] / denom, VAR_FLOOR)
            for t, z2 in surprise.items():  # ADR: widen the teams the result surprised
                self.teams[t].var *= min(1.0 + kappa * max(0.0, z2 - 1.0), ADAPT_CAP)
        if self.record:
            for name, b in zip(race.drivers, drv, strict=True):
                self._log(self.driver_hist, name, b)
            for name in set(race.teams):
                self._log(self.team_hist, name, self.teams[name])
        self.step += 1
        return loglik

    def _log(self, hist: dict[str, BeliefHistory], key: str, b: Belief) -> None:
        h = hist.get(key)
        if h is None:
            h = hist[key] = BeliefHistory()
        h.steps.append(self.step)
        h.states.append((b.mu, b.var, b.season))


def _pl_loglik(eta_ordered: FloatArray) -> float:
    suffix = np.logaddexp.accumulate(eta_ordered[::-1])[::-1]
    return float(np.sum(eta_ordered - suffix))


def run_filter(
    races: Sequence[RaceObs],
    params: RatingParams,
    *,
    score_from: int | None = None,
    record: bool = False,
) -> tuple[RatingFilter, float]:
    """Run the filter over ``races``; return it and the prequential log-likelihood summed
    over races in seasons ``>= score_from``."""
    f = RatingFilter(params, record=record)
    total = 0.0
    for race in races:
        ll = f.update(race)
        if score_from is None or race.season >= score_from:
            total += ll
    return f, total


def fit_params(
    races: Sequence[RaceObs],
    *,
    train_until: int,
    score_from: int,
    start: RatingParams | None = None,
    max_iter: int = 2000,
) -> tuple[RatingParams, float]:
    """Maximise prequential log-likelihood on seasons [score_from, train_until] (bounded)."""
    from scipy.optimize import minimize

    train = [r for r in races if r.season <= train_until]
    x0 = (start or RatingParams()).as_vector()

    def objective(x: FloatArray) -> float:
        _, ll = run_filter(train, RatingParams.from_vector(x), score_from=score_from)
        return -ll if np.isfinite(ll) else 1e12

    res = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-2, "adaptive": True},
    )
    return RatingParams.from_vector(res.x), float(-res.fun)
