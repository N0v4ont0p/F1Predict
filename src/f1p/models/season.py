"""Season simulator: championship odds from the world model (plan §12.3, Championship Centre).

Every remaining round is forecast at C1 from what is public now. A simulated season then
draws, once, a perturbation of every car's strength from the rating filter's posterior
(form carries from race to race, so seasons are internally consistent, not a string of
independent coin flips), and plays out each remaining race (and sprint) with the same
stage-wise-temperature Plackett-Luce and retirement model used for single races. Points are
scored with the season's real scales, giving title probabilities, final-points fans and
finishing-position distributions for drivers and constructors.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
from numpy.typing import NDArray

from f1p.core import points as pts
from f1p.eval.context import Tables
from f1p.eval.harness import forecast_event
from f1p.models.world_v1 import WorldV1

FloatArray = NDArray[np.float64]
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


@dataclass(frozen=True, slots=True)
class Standing:
    id: str
    current: float
    p_title: float
    expected: float
    quantiles: tuple[float, ...]
    p_position: tuple[float, ...]  # P(finishes the championship 1st, 2nd, …)


@dataclass(frozen=True, slots=True)
class SeasonOutlook:
    season: int
    remaining: tuple[str, ...]
    sprints: int
    n_sims: int
    drivers: tuple[Standing, ...]
    constructors: tuple[Standing, ...]
    team_of: dict[str, str]


def sample_orders_batch(
    log_strength: FloatArray,
    p_finish: FloatArray | None,
    stage_scales: FloatArray | None,
    rng: np.random.Generator,
) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    """Like :func:`f1p.eval.forecast.sample_orders`, but with one log-strength row per draw
    (``log_strength`` is ``(S, n)``) so every simulated season can carry its own form."""
    s, n = log_strength.shape
    finished = (
        np.ones((s, n), dtype=bool) if p_finish is None else rng.random((s, n)) < p_finish[None]
    )
    key = log_strength + rng.gumbel(size=(s, n)) - np.where(finished, 0.0, 1e9)
    if stage_scales is not None and stage_scales.size:
        avail = finished.copy()
        rows = np.arange(s)
        centred = log_strength - log_strength.max(axis=1, keepdims=True)
        for k, scale in enumerate(stage_scales[:n]):
            if not avail.any():
                break
            arrival = rng.standard_exponential((s, n)) * np.exp(-scale * centred)
            arrival[~avail] = np.inf
            pick = arrival.argmin(axis=1)
            ok = avail[rows, pick]
            avail[rows[ok], pick[ok]] = False
            key[rows[ok], pick[ok]] = 1e12 - k
    return np.argsort(-key, axis=1), finished


def _points_by_slot(season: int, session: pts.Session, n: int) -> FloatArray:
    table = np.zeros(n)
    sc = pts.scale(season, session, pts.RaceConditions())
    table[: min(n, len(sc))] = sc[:n]
    return table


def _standings(
    ids: list[str], current: FloatArray, totals: NDArray[np.float64]
) -> tuple[Standing, ...]:
    n_sims, m = totals.shape
    # rank within each simulated season (ties broken at random: countback is unmodelled)
    jitter = np.random.default_rng(0).random(totals.shape) * 1e-6
    order = np.argsort(-(totals + jitter), axis=1)
    rank = np.empty_like(order)
    rank[np.arange(n_sims)[:, None], order] = np.arange(m)[None, :]
    out = []
    for j, ident in enumerate(ids):
        pos = np.bincount(rank[:, j], minlength=m) / n_sims
        out.append(
            Standing(
                id=ident,
                current=float(current[j]),
                p_title=float(pos[0]),
                expected=float(totals[:, j].mean()),
                quantiles=tuple(float(q) for q in np.quantile(totals[:, j], QUANTILES)),
                p_position=tuple(float(p) for p in pos),
            )
        )
    return tuple(sorted(out, key=lambda s: (-s.p_title, -s.expected)))


def simulate_season(
    tables: Tables,
    season: int,
    *,
    model: WorldV1 | None = None,
    n_sims: int = 20_000,
    seed: int = 2026,
    as_of: datetime | None = None,
) -> SeasonOutlook:
    model = model or WorldV1()
    rng = np.random.default_rng(seed)
    events = tables["events"].filter(pl.col("season") == season).sort("round")
    raced = set(
        tables["results"]
        .filter((pl.col("season") == season) & (pl.col("session") == "race"))["event_id"]
        .to_list()
    )
    remaining = [r for r in events.iter_rows(named=True) if r["event_id"] not in raced]

    st = tables["driver_standings"].filter(pl.col("season") == season)
    current_d: dict[str, float] = {}
    if st.height:
        last = st.filter(pl.col("round") == st["round"].max())
        current_d = dict(last.select("driver_id", "points").iter_rows())
    cs = tables["constructor_standings"].filter(pl.col("season") == season)
    current_c: dict[str, float] = {}
    if cs.height:
        last = cs.filter(pl.col("round") == cs["round"].max())
        current_c = dict(last.select("constructor_id", "points").iter_rows())

    forecasts = []
    for ev in remaining:
        fc, ctx = forecast_event(tables, model, ev["event_id"], "C1", as_of=as_of)
        if fc is None or fc.log_strength is None:
            continue
        forecasts.append((ev, fc, ctx))

    drivers = sorted(set(current_d) | {d for _, fc, _ in forecasts for d in fc.drivers})
    index = {d: i for i, d in enumerate(drivers)}
    team_of: dict[str, str] = {}
    for _, fc, ctx in forecasts:
        for d, c in zip(fc.drivers, ctx.constructors, strict=True):
            team_of.setdefault(d, c)
    totals = np.tile(np.array([current_d.get(d, 0.0) for d in drivers]), (n_sims, 1))

    if forecasts:
        # One draw of form per simulated season, from the filter's current uncertainty,
        # expressed on the combiner's scale.
        _, fc0, ctx0 = forecasts[0]
        _, var = model.beliefs(ctx0)
        w_rating = float(fc0.meta.get("w_rating", 1.0))
        sd = {d: w_rating * float(np.sqrt(v)) for d, v in zip(fc0.drivers, var, strict=True)}
        form = {d: rng.normal(0.0, s, n_sims) for d, s in sd.items()}
        for ev, fc, _ in forecasts:
            ls = np.asarray(fc.log_strength)[None, :] + np.column_stack(
                [form.get(d, np.zeros(n_sims)) for d in fc.drivers]
            )
            cols = np.array([index[d] for d in fc.drivers])
            sessions = [pts.Session.RACE] + ([pts.Session.SPRINT] if ev["has_sprint"] else [])
            for session in sessions:
                orders, finished = sample_orders_batch(ls, fc.p_finish, fc.stage_scales, rng)
                table = _points_by_slot(season, session, fc.n)
                # points go to classified cars only, by finishing slot
                slot_points = table[None, :] * np.take_along_axis(finished, orders, axis=1)
                np.add.at(totals, (np.arange(n_sims)[:, None], cols[orders]), slot_points)

    teams = sorted(set(current_c) | set(team_of.values()))
    t_index = {t: i for i, t in enumerate(teams)}
    gained = totals - np.array([current_d.get(d, 0.0) for d in drivers])[None, :]
    team_totals = np.tile(np.array([current_c.get(t, 0.0) for t in teams]), (n_sims, 1))
    for d, j in index.items():
        if d in team_of:
            team_totals[:, t_index[team_of[d]]] += gained[:, j]

    return SeasonOutlook(
        season=season,
        remaining=tuple(ev["event_id"] for ev, _, _ in forecasts),
        sprints=sum(bool(ev["has_sprint"]) for ev, _, _ in forecasts),
        n_sims=n_sims,
        drivers=_standings(drivers, np.array([current_d.get(d, 0.0) for d in drivers]), totals),
        constructors=_standings(
            teams, np.array([current_c.get(t, 0.0) for t in teams]), team_totals
        ),
        team_of=team_of,
    )
