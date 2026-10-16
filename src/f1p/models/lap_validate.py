"""Validation of the Bayesian lap model before any forecasting gate (plan M6 §A.4).

* **Held-out lap score** — the expected log predictive density (elpd) of the next race's
  clean laps under ``lap_v2`` and under the Laplace ``lap_model`` (v1), lap by lap, so the
  two can be compared with a paired test over races. The event's overall level is a nuisance
  shared by every car (and the y-scale is centred on each race's median lap), so both models
  are scored conditionally on it: each predictive is shifted by the median residual of the
  race's laps (per posterior draw for v2).
* **Interval calibration** — coverage of the 50/80/95% predictive intervals of each car's
  centred race pace, against the pace it actually ran (``race_pace``).
* **Simulation-based calibration** (Talts et al. 2018) — datasets drawn from the prior are
  refitted; the rank of each true parameter among its posterior draws must be uniform.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats

from f1p.eval.context import Tables
from f1p.models import lap_model as v1
from f1p.models.lap_data import COMPOUNDS, FUEL_SCALE, WindowData, build_window
from f1p.models.lap_v2 import LapPosterior, SampleConfig, V2Priors, build_model

FloatArray = NDArray[np.float64]
LEVELS = (0.5, 0.8, 0.95)


# -- windows ------------------------------------------------------------------------------------
def window_before(
    tables: Tables, event_id: str, window: int = 24
) -> tuple[pl.DataFrame, list[str]]:
    """Laps of the ``window`` races with laps before ``event_id`` (in calendar order).

    A backtest reads laps as of a checkpoint; here the event's own results are only used as
    the held-out target, never in the fit."""
    ev = tables["events"].select("event_id", "season", "round")
    target = ev.filter(pl.col("event_id") == event_id)
    if target.height == 0:
        raise ValueError(f"unknown event {event_id!r}")
    s, r = int(target["season"][0]), int(target["round"][0])
    have = tables["laps"].select("event_id").unique().join(ev, on="event_id")
    before = have.filter((pl.col("season") < s) | ((pl.col("season") == s) & (pl.col("round") < r)))
    recent = before.sort("season", "round").tail(window)["event_id"].to_list()
    return tables["laps"].filter(pl.col("event_id").is_in(recent)), recent


def holdout_window(tables: Tables, event_id: str, window: int = 24) -> WindowData | None:
    laps, _ = window_before(tables, event_id, window)
    hold = tables["laps"].filter(pl.col("event_id") == event_id)
    return build_window(laps, tables, next_event=event_id, holdout=hold)


# -- held-out lap score -------------------------------------------------------------------------
def _scored_laps(w: WindowData) -> NDArray[np.bool_]:
    """Held-out racing laps (no in-/out-laps: v1 does not model them)."""
    return w.pit[~w.train] == 0


def v2_lap_density(post: LapPosterior) -> FloatArray:
    """Per held-out racing lap: log predictive density under ``lap_v2``, conditional on the
    race's level (each draw's location is shifted by its median residual)."""
    from scipy.special import logsumexp

    w = post.data
    keep = _scored_laps(w)
    y = w.y[~w.train][keep]
    e = w.next_index
    assert e is not None
    mu = post.draws["holdout_mu"][:, keep] - post.draws["base"][:, e][:, None]
    sd = post.draws["holdout_sigma"][:, keep]
    shift = np.median(y[None, :] - mu, axis=1, keepdims=True)
    nu = 2.0 + post.draws["nu_minus_2"][:, None]
    lp = stats.t.logpdf(y[None, :], nu, loc=mu + shift, scale=sd)
    return logsumexp(lp, axis=0) - np.log(lp.shape[0])


def v1_lap_density(w: WindowData, tables: Tables, priors: v1.LapPriors | None = None) -> FloatArray:
    """The same laps under the Laplace lap model fitted on the same window: Student-t(nu) with
    the fitted scale widened by the parameter uncertainty and the priors of the new race's
    own deviations (compound, degradation, fuel)."""
    pr = priors or v1.LapPriors()
    e = w.next_index
    assert e is not None
    train_events = [x for i, x in enumerate(w.events) if i != e]
    laps = tables["laps"].filter(pl.col("event_id").is_in(train_events))
    fitted = v1.fit(laps, tables, priors=pr)
    keep = _scored_laps(w)
    n = int(keep.sum())
    if fitted is None:
        return np.full(n, np.nan)
    hold = np.flatnonzero(~w.train)[keep]
    drivers = [w.drivers[i] for i in w.driver[hold]]
    teams = [w.teams[i] for i in w.team[hold]]
    comp = [COMPOUNDS[int(w.wear_compound[w.wear[i]])] for i in hold]
    age = w.tyre_age[hold]
    fuel_laps = w.fuel[hold] * FUEL_SCALE  # v1's fuel regressor is laps remaining
    gap_close = w.drs[hold]  # v1's traffic term: 1{gap < 1 s}
    mean = np.zeros(n)
    var = np.zeros(n)
    cars = _v1_car(fitted, drivers, teams, w, pr)
    for i in range(n):
        c = comp[i]
        terms = [("fuel", fuel_laps[i]), ("traffic", gap_close[i]), (f"deg:{c}", age[i])]
        if c != "MEDIUM":
            terms.append((f"compound:{c}", 1.0))
        idx = [fitted.index[name] for name, _ in terms if name in fitted.index]
        x = np.array([v for name, v in terms if name in fitted.index])
        mean[i] = cars[0][i] + float(fitted.mean[idx] @ x)
        var[i] = cars[1][i] + float(x @ fitted.cov[np.ix_(idx, idx)] @ x)
        var[i] += (pr.deg_event * age[i]) ** 2 + (pr.fuel_event * fuel_laps[i]) ** 2
        if c != "MEDIUM":
            var[i] += pr.compound_event**2
    y = w.y[hold]
    ok = np.isfinite(mean)
    shift = float(np.median((y - mean)[ok])) if ok.any() else 0.0
    scale = np.sqrt(fitted.sigma**2 + var)
    out = np.full(n, np.nan)
    out[ok] = stats.t.logpdf(y[ok], pr.nu, loc=mean[ok] + shift, scale=scale[ok])
    return out


def _v1_car(
    fitted: v1.LapFit, drivers: list[str], teams: list[str], w: WindowData, pr: v1.LapPriors
) -> tuple[FloatArray, FloatArray]:
    """v1's car + driver mean and variance per lap (as ``lap_model.car_pace`` reads them)."""
    season = int(w.event_season[w.next_index or 0])
    mean = np.full(len(drivers), np.nan)
    var = np.full(len(drivers), np.nan)
    cache: dict[tuple[str, str], tuple[float, float]] = {}
    for k, (d, t) in enumerate(zip(drivers, teams, strict=True)):
        if (d, t) not in cache:
            last = fitted.team_last.get(t)
            if last is None:
                cache[(d, t)] = (np.nan, np.nan)
            else:
                i = last[0]
                step = pr.car_reset if fitted.team_season.get(t) != season else pr.car_step
                m = float(fitted.mean[i])
                v = float(fitted.cov[i, i]) + step**2
                j = fitted.index.get(f"driver:{d}")
                if j is not None:
                    m += float(fitted.mean[j])
                    v += float(fitted.cov[j, j]) + 2.0 * float(fitted.cov[i, j])
                else:
                    v += pr.driver**2
                cache[(d, t)] = (m, v)
        mean[k], var[k] = cache[(d, t)]
    return mean, var


# -- interval calibration ---------------------------------------------------------------------
def realised_pace(tables: Tables, event_id: str) -> dict[str, float]:
    """Each car's centred pace deficit at a race (% of a lap, slower = higher)."""
    pace = tables["race_pace"].filter(pl.col("event_id") == event_id)
    d = dict(pace.select("driver_id", "pace_deficit_pct").iter_rows())
    if not d:
        return {}
    mid = float(np.mean(list(d.values())))
    return {k: float(v) - mid for k, v in d.items()}


def coverage_v2(post: LapPosterior, truth: dict[str, float]) -> dict[float, tuple[int, int]]:
    """Per level: (cars inside the central interval, cars scored)."""
    w = post.data
    drivers = [d for d, _ in w.entries]
    pace = 100.0 * post.draws["pace_next"]  # log effect ≈ % of a lap
    keep = [i for i, d in enumerate(drivers) if d in truth]
    if len(keep) < 5:
        return dict.fromkeys(LEVELS, (0, 0))
    draws = pace[:, keep] - pace[:, keep].mean(axis=1, keepdims=True)
    obs = np.array([truth[drivers[i]] for i in keep])
    obs = obs - obs.mean()
    out = {}
    for lv in LEVELS:
        lo, hi = np.quantile(draws, [(1 - lv) / 2, (1 + lv) / 2], axis=0)
        out[lv] = (int(((obs >= lo) & (obs <= hi)).sum()), len(keep))
    return out


# -- simulation-based calibration ------------------------------------------------------------
SBC_PARAMS = (
    "car_reset",
    "gp_amplitude",
    "car_event_sd",
    "driver0",
    "rookie",
    "fuel",
    "sigma_log",
    "nu_minus_2",
    "wear_event_sd",
    "compound",
    "pace_next",
)


@dataclass
class SBCResult:
    ranks: pl.DataFrame  # param, element, sim, rank, draws
    p_values: dict[str, float]  # chi-square uniformity per parameter (all elements pooled)

    def ok(self, alpha: float = 0.01) -> bool:
        # Bonferroni over the parameters tested
        return all(p >= alpha / max(1, len(self.p_values)) for p in self.p_values.values())


def sbc(
    w: WindowData,
    n_sims: int,
    *,
    cfg: SampleConfig | None = None,
    priors: V2Priors | None = None,
    params: tuple[str, ...] = SBC_PARAMS,
    thin_to: int = 99,
    bins: int = 10,
    seed: int = 0,
) -> SBCResult:
    """Draw ``n_sims`` datasets from the prior on ``w``'s design, refit each (one compiled
    model, data swapped) and rank every true value among ``thin_to`` thinned draws."""
    pm: Any = importlib.import_module("pymc")
    nutpie: Any = importlib.import_module("nutpie")
    cfg = cfg or SampleConfig(draws=500, tune=500, chains=2)
    model = build_model(w, priors)
    with model:
        prior = pm.sample_prior_predictive(draws=n_sims, random_seed=seed)
    compiled = nutpie.compile_pymc_model(model, backend="numba")
    rows = []
    for i in range(n_sims):
        y_sim = np.asarray(prior["prior_predictive"]["obs"])[0, i]
        trace = nutpie.sample(
            compiled.with_data(y=y_sim),
            draws=cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            cores=cfg.cores or cfg.chains,
            seed=seed + 1 + i,
            progress_bar=False,
            save_warmup=False,
        )
        post = trace["posterior"]
        for p in params:
            if p not in post.data_vars:
                continue
            truth = np.asarray(prior["prior"][p])[0, i].ravel()
            arr = np.asarray(post[p])
            flat = arr.reshape(-1, truth.size)
            idx = np.linspace(0, len(flat) - 1, thin_to).round().astype(int)
            sub = flat[idx]
            for j in range(truth.size):
                rows.append(
                    {
                        "param": p,
                        "element": j,
                        "sim": i,
                        "rank": int((sub[:, j] < truth[j]).sum()),
                        "draws": thin_to,
                    }
                )
    ranks = pl.DataFrame(rows)
    p_values = {}
    for (p,), g in ranks.group_by("param"):
        r = g["rank"].to_numpy()
        counts = np.bincount(np.minimum(r * bins // (thin_to + 1), bins - 1), minlength=bins)
        p_values[str(p)] = float(stats.chisquare(counts).pvalue)
    return SBCResult(ranks, p_values)
