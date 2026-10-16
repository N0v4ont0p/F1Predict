"""Calibration and posterior-predictive checks of ``race_sim_v2`` (plan M6 §B.4).

The simulator's sub-models are fitted directly; a few global knobs (:class:`sim_v2.Knobs`)
are not, and are calibrated here against what races look like:

* **summary statistics** per race, computed identically from a simulated race and from the
  lap/result tables: lap-1 |positions gained| and share holding position, overtakes and
  close-following car-laps per racing lap, neutralised periods, stops per finisher and the
  one-stop share, |finish - grid| of finishers, and the finishers' share;
* **simulated method of moments**: the knobs minimise the standardised distance between the
  simulated and observed means of those statistics over a set of races, with common random
  numbers (each race keeps its seed) so the objective is smooth enough for Nelder-Mead;
* **posterior-predictive checks**: per circuit and statistic, the observed race values
  against the simulated distribution (two-sample KS), Benjamini-Hochberg adjusted across
  circuits.

Deviation from the plan, recorded in its build log: the plan names neural posterior
estimation (``sbi``, PyTorch) for this step. The knobs here are five global scalars — the
circuit-specific effects are fitted directly in the sub-models — so a moment calibration
gives the point estimate; NPE (a posterior over the knobs) is M6 follow-up work behind the
same summaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import minimize

from f1p.eval.context import Tables
from f1p.models import sim_fit as sf
from f1p.models import sim_v2 as sv

FloatArray = NDArray[np.float64]

SUMMARIES = (
    "lap1_abs_gain",
    "lap1_hold",
    "overtakes_per_lap",
    "close_per_lap",
    "neutral_periods",
    "stops_mean",
    "one_stop",
    "finish_vs_grid",
    "finished",
)


# -- summaries ----------------------------------------------------------------------------------
def sim_summaries(res: sv.Result, spec: sv.RaceSpec) -> dict[str, FloatArray]:
    """Per simulated race (arrays of length S)."""
    grid = spec.grid
    l1_pos = np.argsort(res.lap1_order, axis=1)
    gain = grid - l1_pos
    racing = max(spec.laps - 1, 1)
    fin = res.finished
    pos = np.argsort(res.orders, axis=1)
    n_fin = np.maximum(fin.sum(axis=1), 1)
    stops = np.where(fin, res.stops, 0)
    return {
        "lap1_abs_gain": np.abs(gain).mean(axis=1),
        "lap1_hold": (gain == 0).mean(axis=1),
        "overtakes_per_lap": res.overtakes / racing,
        "close_per_lap": res.close_pairs / racing,
        "neutral_periods": res.neutral_periods.astype(np.float64),
        "stops_mean": stops.sum(axis=1) / n_fin,
        "one_stop": (fin & (res.stops == 1)).sum(axis=1) / n_fin,
        "finish_vs_grid": np.where(fin, np.abs(pos - grid), 0).sum(axis=1) / n_fin,
        "finished": fin.mean(axis=1),
    }


def observed_summaries(tables: Tables, event_id: str) -> dict[str, float] | None:
    """The same statistics for a real race (None without laps)."""
    laps = tables["laps"].filter(pl.col("event_id") == event_id)
    if laps.is_empty():
        return None
    res = tables["results"].filter(
        (pl.col("event_id") == event_id)
        & (pl.col("session") == "race")
        & ~pl.col("status").is_in(["dns", "dnq", "dnpq"])
        & pl.col("grid").is_not_null()
    )
    if res.height < 10:
        return None
    grid = res["grid"].to_numpy().astype(np.float64)
    grid = np.where(grid <= 0, grid.max() + 1, grid)
    slot = np.argsort(np.argsort(grid, kind="stable"), kind="stable")
    slots = dict(zip(res["driver_id"].to_list(), slot.tolist(), strict=True))
    l1 = laps.filter(pl.col("lap") == 1).select("driver_id", "position").drop_nulls()
    gains = np.array(
        [slots[d] - (p - 1) for d, p in l1.iter_rows() if d in slots], dtype=np.float64
    )
    race_laps = int(laps["lap"].to_numpy().max())
    racing = max(race_laps - 1, 1)
    ov = sf.overtaking_data(tables, laps)
    periods = sf.neutral_periods(laps).height
    stints = laps.group_by("driver_id").agg(pl.col("stint").max(), pl.col("lap").max())
    classified = res.filter(pl.col("classified"))
    fin_ids = set(classified["driver_id"].to_list())
    stops = np.array(
        [s - 1 for d, s, _ in stints.iter_rows() if d in fin_ids and s is not None],
        dtype=np.float64,
    )
    fvg = np.array(
        [
            abs(float(p) - 1 - slots[d])
            for d, p in classified.select("driver_id", "position").iter_rows()
            if d in slots and p is not None
        ]
    )
    return {
        "lap1_abs_gain": float(np.abs(gains).mean()) if gains.size else float("nan"),
        "lap1_hold": float((gains == 0).mean()) if gains.size else float("nan"),
        "overtakes_per_lap": float(ov["passed"].sum()) / racing,
        "close_per_lap": ov.height / racing,
        "neutral_periods": float(periods),
        "stops_mean": float(stops.mean()) if stops.size else float("nan"),
        "one_stop": float((stops == 1).mean()) if stops.size else float("nan"),
        "finish_vs_grid": float(fvg.mean()) if fvg.size else float("nan"),
        "finished": classified.height / res.height,
    }


# -- the calibration set ---------------------------------------------------------------------------
@dataclass
class CalRace:
    event_id: str
    circuit: str
    spec: sv.RaceSpec
    draws: sv.Draws
    models: sv.SubModels
    observed: dict[str, float]
    seed: int


def calibration_races(
    tables: Tables, events: list[str], model: Any, *, sims: int = 500
) -> list[CalRace]:
    """C4 inputs (real grid) for each race of ``events`` with laps, as ``model``
    (:class:`race_sim_v2.RaceSimV2`) would simulate it, and the race's observed summaries."""
    from f1p.eval.context import build_contexts
    from f1p.eval.forecast import stable_seed

    ev = tables["events"].filter(pl.col("event_id").is_in(events))
    season_vals = ev["season"].to_numpy()
    seasons = (int(season_vals.min()), int(season_vals.max()))
    out = []
    for ctx, _, _ in build_contexts(tables, seasons=seasons, checkpoints=("C4",)):
        if ctx.event_id not in events:
            continue
        obs = observed_summaries(tables, ctx.event_id)
        signal = model.prepare(ctx)
        models = model.sub_models(ctx) if signal is not None else None
        if obs is None or signal is None or models is None:
            continue
        seed = stable_seed(ctx.event_id, "calibration")
        spec, draws = model.race(ctx, signal, sims, np.random.default_rng(seed))
        out.append(CalRace(ctx.event_id, ctx.circuit_id, spec, draws, models, obs, seed))
    return out


def simulate_summaries(races: list[CalRace], knobs: sv.Knobs) -> list[dict[str, FloatArray]]:
    """Each race simulated with its own fixed seed (common random numbers across knobs)."""
    return [
        sim_summaries(
            sv.simulate(r.spec, r.draws, r.models, knobs, np.random.default_rng(r.seed)), r.spec
        )
        for r in races
    ]


# -- simulated method of moments -------------------------------------------------------------------
CALIBRATED = ("start_noise", "min_gap", "overtake_shift", "incident_scale")
LOG_SCALE = {"start_noise", "min_gap", "incident_scale"}  # positive knobs: optimise their log


@dataclass
class Calibration:
    knobs: sv.Knobs
    distance: float
    start_distance: float
    observed: dict[str, float]
    simulated: dict[str, float]
    evaluations: int

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "knobs": asdict(self.knobs)}


def _moments(per_race: list[dict[str, FloatArray]]) -> dict[str, float]:
    return {k: float(np.mean([np.nanmean(r[k]) for r in per_race])) for k in SUMMARIES}


def calibrate(
    races: list[CalRace],
    *,
    names: tuple[str, ...] = CALIBRATED,
    start: sv.Knobs | None = None,
    max_evals: int = 80,
) -> Calibration:
    """Knobs minimising Σ ((simulated mean - observed mean) / se)² over the summaries, where
    se is the observed race-to-race sd / √races (the precision the data give each moment)."""
    start = start or sv.Knobs()
    obs = {k: np.array([r.observed[k] for r in races], dtype=np.float64) for k in SUMMARIES}
    obs_mean = {k: float(np.nanmean(v)) for k, v in obs.items()}
    se = {
        k: max(float(np.nanstd(v)) / np.sqrt(max(np.isfinite(v).sum(), 1)), 1e-6)
        for k, v in obs.items()
    }

    def to_knobs(x: FloatArray) -> sv.Knobs:
        vals = {
            n: float(np.exp(v)) if n in LOG_SCALE else float(v)
            for n, v in zip(names, x, strict=True)
        }
        return replace(start, **vals)

    def distance(knobs: sv.Knobs) -> tuple[float, dict[str, float]]:
        sim = _moments(simulate_summaries(races, knobs))
        d = sum(
            ((sim[k] - obs_mean[k]) / se[k]) ** 2 for k in SUMMARIES if np.isfinite(obs_mean[k])
        )
        return float(d), sim

    x0 = np.array(
        [np.log(getattr(start, n)) if n in LOG_SCALE else getattr(start, n) for n in names]
    )
    d0, _ = distance(start)
    res = minimize(
        lambda x: distance(to_knobs(x))[0],
        x0,
        method="Nelder-Mead",
        options={"maxfev": max_evals, "xatol": 1e-2, "fatol": 1e-2},
    )
    best = to_knobs(res.x)
    d, sim = distance(best)
    return Calibration(best, d, d0, obs_mean, sim, int(res.nfev))


# -- posterior-predictive checks -------------------------------------------------------------------
def _bh(p: FloatArray) -> FloatArray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 1.0
    for rank, i in zip(range(m, 0, -1), order[::-1], strict=True):
        run = min(run, p[i] * m / rank)
        adj[i] = run
    return adj


def ppc(races: list[CalRace], knobs: sv.Knobs) -> pl.DataFrame:
    """Per circuit and statistic: the observed race values against the simulated
    distribution (two-sample KS), with Benjamini-Hochberg adjustment across circuits per
    statistic. Plan §B.6 gate 1: every adjusted p > 0.05."""
    sims = simulate_summaries(races, knobs)
    rows = []
    circuits = sorted({r.circuit for r in races})
    for stat in SUMMARIES:
        ps = []
        for c in circuits:
            idx = [i for i, r in enumerate(races) if r.circuit == c]
            observed = np.array([races[i].observed[stat] for i in idx], dtype=np.float64)
            observed = observed[np.isfinite(observed)]
            simulated = np.concatenate([sims[i][stat] for i in idx])
            res: Any = stats.ks_2samp(observed, simulated) if observed.size else None
            p = float(res.pvalue) if res is not None else float("nan")
            ps.append(p)
            rows.append(
                {
                    "statistic": stat,
                    "circuit": c,
                    "races": int(observed.size),
                    "observed": float(observed.mean()) if observed.size else float("nan"),
                    "simulated": float(np.nanmean(simulated)),
                    "p": p,
                }
            )
        adj = _bh(np.nan_to_num(np.array(ps), nan=1.0))
        for j in range(len(circuits)):
            rows[-len(circuits) + j]["p_bh"] = float(adj[j])
    return pl.DataFrame(rows)
