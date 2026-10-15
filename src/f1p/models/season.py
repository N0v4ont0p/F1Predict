"""Season simulator: championship odds from the world model (plan §12.3, Championship Centre).

Every remaining round is forecast at C1 from what is public at the projection time. A
simulated season then draws, once, a perturbation of every car's strength from the rating
filter's posterior (form carries from race to race, so seasons are internally consistent,
not a string of independent coin flips), and plays out each remaining race (and sprint) with
the same stage-wise-temperature Plackett-Luce and retirement model used for single races.
Points are scored with the season's real scales (2014's double-points finale included) and
equal points are split on countback (most wins, then seconds, ...), giving title
probabilities, final-points fans and finishing-position distributions for drivers and
constructors.

A projection can be made **as of any moment**: ``after_round=N`` (or ``as_of=<datetime>``)
cuts every input — results, standings, ratings, race pace — at the time round N's
classification became public, uses the line-up known then, and forecasts every later round
from that information only. That is what the championship history and the title backtest
(:mod:`f1p.models.title_backtest`) are built from.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from f1p.core import points as pts
from f1p.core.clock import AsOf, known_at
from f1p.eval.context import Tables, checkpoint_time, live_context
from f1p.eval.harness import EventForecaster
from f1p.models.world_v1 import WorldV1

FloatArray = NDArray[np.float64]
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
COUNTBACK_PLACES = 5  # wins, seconds, ... used to split equal points
_COUNTBACK_BASE = 64.0  # > any count of one place in a season (two cars x 24+ rounds)


@dataclass(frozen=True, slots=True)
class Standing:
    id: str
    current: float
    p_title: float
    expected: float
    quantiles: tuple[float, ...]
    p_position: tuple[float, ...]  # P(finishes the championship 1st, 2nd, …)
    most_likely: int  # 0-based place in the jointly most likely final order

    @property
    def p_top3(self) -> float:
        return float(sum(self.p_position[:3]))


@dataclass(frozen=True, slots=True)
class SeasonOutlook:
    season: int
    as_of: datetime
    after_round: int  # last round whose result the projection knows (0: pre-season)
    remaining: tuple[str, ...]
    sprints: int
    n_sims: int
    drivers: tuple[Standing, ...]  # in the most likely final order
    constructors: tuple[Standing, ...]
    team_of: dict[str, str]

    def standing(self, kind: str, ident: str) -> Standing | None:
        rows = self.drivers if kind == "driver" else self.constructors
        return next((s for s in rows if s.id == ident), None)


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


def _points_by_slot(
    season: int, session: pts.Session, n: int, multiplier: float = 1.0
) -> FloatArray:
    table = np.zeros(n)
    sc = pts.scale(season, session, pts.RaceConditions(multiplier=multiplier))
    table[: min(n, len(sc))] = sc[:n]
    return table


def countback_key(totals: FloatArray, places: NDArray[np.integer]) -> FloatArray:
    """A single sortable score per simulated season: points first, then countback.

    ``places[..., k]`` counts finishes in place ``k + 1``. Points come in steps of at least
    0.5, so the countback term (< 0.4) never overrides a points difference, and each place
    weighs more than every later place together (counts stay below the base).
    """
    k = places.shape[-1]
    weights = 0.4 / _COUNTBACK_BASE ** np.arange(1, k + 1)
    return totals + places.astype(np.float64) @ weights


def most_likely_order(p_position: FloatArray) -> NDArray[np.intp]:
    """Place of each competitor in the jointly most likely final order.

    Maximises the summed log-probability of one-to-one (competitor, place) pairs — the
    assignment problem — so the table is a single consistent order, unlike sorting by
    expected points (which can rank a long shot above someone more likely to finish ahead).
    """
    cost = -np.log(np.clip(p_position, 1e-12, None))
    rows, cols = linear_sum_assignment(cost)
    place = np.empty(p_position.shape[0], dtype=np.intp)
    place[rows] = cols
    return place


def _standings(
    ids: list[str],
    current: FloatArray,
    totals: FloatArray,
    places: NDArray[np.integer],
    rng: np.random.Generator,
) -> tuple[Standing, ...]:
    n_sims, m = totals.shape
    # points, then countback; anything still level is split at random
    key = countback_key(totals, places) + rng.random(totals.shape) * 1e-12
    order = np.argsort(-key, axis=1)
    rank = np.empty_like(order)
    rank[np.arange(n_sims)[:, None], order] = np.arange(m)[None, :]
    pos = np.stack([np.bincount(rank[:, j], minlength=m) / n_sims for j in range(m)])
    likely = most_likely_order(pos) if m else np.zeros(0, dtype=np.intp)
    out = [
        Standing(
            id=ident,
            current=float(current[j]),
            p_title=float(pos[j, 0]),
            expected=float(totals[:, j].mean()),
            quantiles=tuple(float(q) for q in np.quantile(totals[:, j], QUANTILES)),
            p_position=tuple(float(p) for p in pos[j]),
            most_likely=int(likely[j]),
        )
        for j, ident in enumerate(ids)
    ]
    return tuple(sorted(out, key=lambda s: s.most_likely))


def round_cutoff(tables: Tables, season: int, after_round: int) -> datetime:
    """When round ``after_round``'s race classification became public (round 0: the
    Thursday before the opening round)."""
    events = tables["events"].filter(pl.col("season") == season).sort("round")
    if events.is_empty():
        raise ValueError(f"no events for {season}")
    if after_round <= 0:
        first = events.row(0, named=True)
        return checkpoint_time("C1", first["race_date"], first["race_start_utc"])
    res = tables["results"].filter(
        (pl.col("season") == season)
        & (pl.col("round") == after_round)
        & (pl.col("session") == "race")
    )
    known = res["observed_at"].max() if res.height else None
    if not isinstance(known, datetime):
        raise ValueError(f"{season} round {after_round} has no race result")
    return known


def _place_counts(
    results: pl.DataFrame, key: str, ids: list[str], k: int = COUNTBACK_PLACES
) -> NDArray[np.int16]:
    """Grand-prix finishes in places 1..k per competitor (sprints do not count back)."""
    out = np.zeros((len(ids), k), dtype=np.int16)
    index = {x: i for i, x in enumerate(ids)}
    top = results.filter(
        (pl.col("session") == "race") & pl.col("classified") & (pl.col("position") <= k)
    )
    for ident, position in top.select(key, "position").iter_rows():
        if ident in index and position is not None:
            out[index[ident], int(position) - 1] += 1
    return out


def _latest_points(df: pl.DataFrame, key: str, last_round: int | None) -> dict[str, float]:
    if last_round is None or df.is_empty():
        return {}
    at = df.filter(pl.col("round") == last_round)
    if at.is_empty():
        at = df.filter(pl.col("round") == df["round"].max())
    return {str(a): float(b) for a, b in at.select(key, "points").iter_rows()}


def _lineup(tables: Tables, season: int, last_round: int | None) -> pl.DataFrame:
    """The race line-up public after ``last_round`` (the opening entry list pre-season)."""
    entries = tables["entries"].filter(
        (pl.col("season") == season) & pl.col("round").is_not_null() & ~pl.col("test_driver")
    )
    target = max(last_round or 1, 1)
    at = entries.filter(pl.col("round") == target)
    if at.is_empty():
        at = entries.filter(pl.col("round") <= target)
        at = at.filter(pl.col("round") == at["round"].max()) if at.height else at
    if at.is_empty():
        at = tables["results"].filter(
            (pl.col("season") == season)
            & (pl.col("round") == target)
            & (pl.col("session") == "race")
        )
    return at.select("driver_id", "constructor_id").unique(
        "driver_id", keep="first", maintain_order=True
    )


def simulate_season(
    tables: Tables,
    season: int,
    *,
    model: WorldV1 | None = None,
    n_sims: int = 20_000,
    seed: int = 2026,
    as_of: datetime | None = None,
    after_round: int | None = None,
    uniform: bool = False,
    forecaster: EventForecaster | None = None,
) -> SeasonOutlook:
    """Project the final standings of ``season``.

    ``after_round`` / ``as_of`` make a historical projection (see the module docstring);
    without either, everything public now is used. ``uniform`` replaces the model with
    equally strong, never-retiring cars — the "current points plus a uniform spread" baseline
    of the title backtest. Pass a shared ``forecaster`` to reuse prepared history across
    many projections of the same season.
    """
    model = model or WorldV1()
    rng = np.random.default_rng(seed)
    if after_round is not None:
        as_of = round_cutoff(tables, season, after_round)
    historical = as_of is not None
    cut = AsOf(as_of) if as_of is not None else None

    def public(df: pl.DataFrame) -> pl.DataFrame:
        return known_at(df, cut) if cut is not None else df

    events = tables["events"].filter(pl.col("season") == season).sort("round")
    results = public(tables["results"].filter(pl.col("season") == season))
    raced_rounds = sorted(
        set(results.filter(pl.col("session") == "race")["round"].drop_nulls().to_list())
    )
    last_round = raced_rounds[-1] if raced_rounds else None
    remaining = [r for r in events.iter_rows(named=True) if r["round"] not in set(raced_rounds)]

    current_d = _latest_points(
        public(tables["driver_standings"].filter(pl.col("season") == season)),
        "driver_id",
        last_round,
    )
    current_c: dict[str, float] = {}
    if "constructor_standings" in tables:
        current_c = _latest_points(
            public(tables["constructor_standings"].filter(pl.col("season") == season)),
            "constructor_id",
            last_round,
        )

    lineup = _lineup(tables, season, last_round) if historical else None
    fcr = forecaster or EventForecaster(tables, model, "C1")
    forecasts = []
    for ev in remaining:
        if uniform:
            ctx = live_context(tables, ev["event_id"], "C1", as_of, lineup=lineup)
            forecasts.append((ev, None, ctx))
            continue
        fc, ctx = fcr.forecast(ev["event_id"], as_of=as_of, lineup=lineup)
        if fc is None or fc.log_strength is None:
            continue
        forecasts.append((ev, fc, ctx))

    drivers = sorted(set(current_d) | {d for _, _, ctx in forecasts for d in ctx.drivers})
    index = {d: i for i, d in enumerate(drivers)}
    team_of: dict[str, str] = {}
    for _, _, ctx in forecasts:
        for d, c in zip(ctx.drivers, ctx.constructors, strict=True):
            team_of.setdefault(d, c)
    for d, c in (
        results.filter(pl.col("session") == "race")
        .sort("round", descending=True)
        .select("driver_id", "constructor_id")
        .iter_rows()
    ):
        team_of.setdefault(d, c)  # drivers who have left keep their last team
    base_d = np.array([current_d.get(d, 0.0) for d in drivers])
    totals = np.tile(base_d, (n_sims, 1))
    places = np.tile(_place_counts(results, "driver_id", drivers), (n_sims, 1, 1))
    sims = np.arange(n_sims)[:, None]

    form: dict[str, FloatArray] = {}
    if forecasts and not uniform:
        # One draw of form per simulated season, from the filter's uncertainty at the
        # projection time, expressed on the combiner's scale.
        _, fc0, ctx0 = forecasts[0]
        assert fc0 is not None
        _, var = model.beliefs(ctx0)
        w_rating = float(fc0.meta.get("w_rating", 1.0))
        form = {
            d: rng.normal(0.0, w_rating * float(np.sqrt(v)), n_sims)
            for d, v in zip(ctx0.drivers, var, strict=True)
        }
    for ev, fc, ctx in forecasts:
        n = ctx.n
        if fc is None:
            ls = np.zeros((n_sims, n))
            p_finish, stages = None, None
        else:
            ls = np.asarray(fc.log_strength)[None, :] + np.column_stack(
                [form.get(d, np.zeros(n_sims)) for d in ctx.drivers]
            )
            p_finish, stages = fc.p_finish, fc.stage_scales
        cols = np.array([index[d] for d in ctx.drivers])
        multiplier = pts.EVENT_MULTIPLIERS.get((season, ev["circuit_id"]), 1.0)
        sessions = [pts.Session.RACE] + ([pts.Session.SPRINT] if ev["has_sprint"] else [])
        for session in sessions:
            orders, finished = sample_orders_batch(ls, p_finish, stages, rng)
            table = _points_by_slot(
                season, session, n, multiplier if session is pts.Session.RACE else 1.0
            )
            # points go to classified cars only, by finishing slot
            done = np.take_along_axis(finished, orders, axis=1)
            np.add.at(totals, (sims, cols[orders]), table[None, :] * done)
            if session is pts.Session.RACE:
                for k in range(min(COUNTBACK_PLACES, n)):
                    np.add.at(places[:, :, k], (sims[:, 0], cols[orders[:, k]]), done[:, k])

    teams = sorted(set(current_c) | set(team_of.values()))
    t_index = {t: i for i, t in enumerate(teams)}
    base_c = np.array([current_c.get(t, 0.0) for t in teams])
    team_totals = np.tile(base_c, (n_sims, 1))
    team_places = np.tile(_place_counts(results, "constructor_id", teams), (n_sims, 1, 1))
    gained = totals - base_d[None, :]
    base_places = _place_counts(results, "driver_id", drivers)
    gained_places = places - base_places[None]
    for d, j in index.items():
        if d in team_of:
            team_totals[:, t_index[team_of[d]]] += gained[:, j]
            team_places[:, t_index[team_of[d]]] += gained_places[:, j]

    return SeasonOutlook(
        season=season,
        as_of=as_of or datetime.now(UTC),
        after_round=last_round or 0,
        remaining=tuple(ev["event_id"] for ev, _, _ in forecasts),
        sprints=sum(bool(ev["has_sprint"]) for ev, _, _ in forecasts),
        n_sims=n_sims,
        drivers=_standings(drivers, base_d, totals, places, rng),
        constructors=_standings(teams, base_c, team_totals, team_places, rng),
        team_of=team_of,
    )
