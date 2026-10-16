"""Races as sequences (plan M6 §D.2): the data of the ``lapseq`` sequence model.

One :class:`RaceSeq` is a race as live timing shows it — per lap and car: lap time, gap to
the leader, running position, tyres, stops, neutralisation, still running — plus the grid,
a pre-race pace covariate and the classification. Real races come from the ``laps`` table
(:func:`real_race`); simulated ones from ``race_sim_v2``'s trace (:func:`simulated_races`),
in the same form, so a model can be pretrained on simulations and fine-tuned on real laps.

:func:`tokens` turns a race (or its first ``k`` laps) into the model's input: token
``t = 0`` is the grid, token ``t >= 1`` the state at the end of lap ``t + 1`` — live timing
has no lap-1 times, so lap 1 is folded into the lap-2 token. Every input is
scale-free (lap times relative to the field's lap, gaps on a log scale) so real and
simulated races, and circuits of any length, share one representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from f1p.models.lap_data import COMPOUNDS

FloatArray = NDArray[np.float32]
C_IDX = {c: i for i, c in enumerate(COMPOUNDS)}
UNKNOWN = C_IDX["UNKNOWN"]

FEATURES = (
    "position",  # /20
    "gap",  # log1p(s behind the leader) / log1p(120)
    "gap_ahead",  # s to the car ahead, capped at 5, /5
    "residual",  # lap time / the field's lap - 1, x10 (clipped)
    "field_slow",  # the field's lap / its usual lap - 1, x5 (neutralisations)
    "tyre_age",  # /40
    *(f"tyre_{c.lower()}" for c in COMPOUNDS),  # one-hot (UNKNOWN: no compound data)
    "stops",  # /3
    "pit",  # stopped on this lap
    "out_lap",  # stopped on the previous lap
    "neutral",  # this lap under SC / VSC
    "progress",  # lap / race laps
    "laps_left",  # /70
    "grid",  # /20
    "pace",  # pre-race pace deficit, % of a lap, /5
    "pace_missing",
    "is_grid",  # the pre-race token
)
F = len(FEATURES)
_FI = {f: i for i, f in enumerate(FEATURES)}


@dataclass
class RaceSeq:
    """One race, laps ``1..L`` on axis 0 (lap ``l`` at index ``l - 1``), cars on axis 1."""

    lap_s: FloatArray  # (L, n) NaN: no time (lap 1, retired, timing gap)
    gap_s: FloatArray  # (L, n) behind the leader
    position: NDArray[np.int16]  # (L, n) 0: not running / unknown
    compound: NDArray[np.int8]  # (L, n) COMPOUNDS index
    tyre_age: FloatArray  # (L, n)
    pit: NDArray[np.bool_]  # (L, n) stopped on this lap
    alive: NDArray[np.bool_]  # (L, n) running at the end of the lap
    neutral: NDArray[np.bool_]  # (L,)
    grid: NDArray[np.int16]  # (n,) starting slot, 0 = pole
    pace: FloatArray  # (n,) pre-race pace deficit, % (NaN: unknown)
    finish: NDArray[np.int16]  # (n,) classified position (0 = winner), -1 not classified
    race_laps: int
    drivers: tuple[str, ...] = ()
    event_id: str = ""

    @property
    def n(self) -> int:
        return int(self.lap_s.shape[1])

    @property
    def laps(self) -> int:
        return int(self.lap_s.shape[0])


@dataclass
class Tokens:
    """A race's model input and targets: ``T`` tokens, ``n`` cars."""

    x: FloatArray  # (T, n, F)
    running: NDArray[np.bool_]  # (T, n)
    next_res: FloatArray  # (T, n) next lap's residual (NaN: not a clean lap)
    next_pit: FloatArray  # (T, n) 1/0, NaN: no next lap
    next_out: FloatArray  # (T, n) retires before the end of the next lap; NaN: n/a
    next_neutral: FloatArray  # (T,) a neutralisation starts next lap; NaN: n/a
    finish: NDArray[np.int16]  # (n,)


# -- features ----------------------------------------------------------------------------------
def _field_lap(seq: RaceSeq) -> FloatArray:
    """The field's lap per lap: the median time of the cars running on track (no stop on
    this lap or the previous one); NaN where there is none."""
    out_lap = np.zeros_like(seq.pit)
    out_lap[1:] = seq.pit[:-1]
    ok = seq.alive & ~seq.pit & ~out_lap & np.isfinite(seq.lap_s)
    vals = np.where(ok, seq.lap_s, np.nan)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmedian(vals, axis=1).astype(np.float32)


def tokens(seq: RaceSeq, upto: int | None = None) -> Tokens:
    """The model input for laps ``1..upto`` (default: the whole race) and its targets.

    Token 0 is the grid; token ``t >= 1`` is the end of lap ``t + 1``. Only what was known
    at each token goes into its features (the usual-lap reference is a running median);
    targets look one lap ahead, and the classification is the race's."""
    laps = seq.laps if upto is None else min(upto, seq.laps)
    n = seq.n
    x = np.zeros((laps, n, F), dtype=np.float32)
    running = np.zeros((laps, n), dtype=bool)

    field = _field_lap(seq)
    pit = seq.pit
    out_lap = np.zeros_like(pit)
    out_lap[1:] = pit[:-1]
    stops = np.cumsum(pit, axis=0)
    grid = seq.grid.astype(np.float32)
    pace = seq.pace.astype(np.float32)

    # the static channels, every token
    x[:, :, _FI["grid"]] = grid[None, :] / 20.0
    has_pace = np.isfinite(pace)
    x[:, :, _FI["pace"]] = np.where(has_pace, np.clip(pace, 0.0, 5.0) / 5.0, 0.0)[None, :]
    x[:, :, _FI["pace_missing"]] = (~has_pace).astype(np.float32)[None, :]

    # token 0: the grid
    x[0, :, _FI["position"]] = (grid + 1.0) / 20.0
    x[0, :, _FI["is_grid"]] = 1.0
    x[0, :, _FI["laps_left"]] = seq.race_laps / 70.0
    x[0, :, _FI["tyre_" + COMPOUNDS[UNKNOWN].lower()]] = 1.0
    running[0] = True

    usual: list[float] = []
    for t in range(1, laps):
        i = t  # lap t + 1 lives at index t
        alive = seq.alive[i]
        running[t] = alive
        if np.isfinite(field[i]):
            usual.append(float(field[i]))
        base = float(np.median(usual)) if usual else np.nan
        lap = seq.lap_s[i]
        with np.errstate(invalid="ignore", divide="ignore"):
            res = np.clip(lap / field[i] - 1.0, -0.05, 0.3) * 10.0
            slow = np.clip(field[i] / base - 1.0, -0.05, 1.0) * 5.0
        x[t, :, _FI["position"]] = seq.position[i] / 20.0
        gap = np.where(np.isfinite(seq.gap_s[i]), np.maximum(seq.gap_s[i], 0.0), 0.0)
        x[t, :, _FI["gap"]] = np.clip(np.log1p(gap) / np.log1p(120.0), 0.0, 1.5)
        x[t, :, _FI["gap_ahead"]] = _gap_ahead(seq.position[i], gap, alive)
        x[t, :, _FI["residual"]] = np.where(np.isfinite(res), res, 0.0)
        x[t, :, _FI["field_slow"]] = slow if np.isfinite(slow) else 0.0
        age = seq.tyre_age[i]
        x[t, :, _FI["tyre_age"]] = np.where(np.isfinite(age), np.clip(age, 0, 80) / 40.0, 0.0)
        comp = seq.compound[i].astype(np.int64)
        x[t, np.arange(n), _FI["tyre_medium"] + comp] = 1.0
        x[t, :, _FI["stops"]] = np.minimum(stops[i], 6) / 3.0
        x[t, :, _FI["pit"]] = pit[i]
        x[t, :, _FI["out_lap"]] = out_lap[i]
        x[t, :, _FI["neutral"]] = float(seq.neutral[i])
        x[t, :, _FI["progress"]] = (i + 1) / seq.race_laps
        x[t, :, _FI["laps_left"]] = (seq.race_laps - i - 1) / 70.0
    x[~running] = 0.0  # nothing about a car that is not running

    # targets: the lap after each token (lap t + 2, index t + 1)
    next_res = np.full((laps, n), np.nan, dtype=np.float32)
    next_pit = np.full((laps, n), np.nan, dtype=np.float32)
    next_out = np.full((laps, n), np.nan, dtype=np.float32)
    next_neutral = np.full(laps, np.nan, dtype=np.float32)
    for t in range(laps):
        j = t + 1
        if j >= seq.laps:
            break
        live = running[t]
        next_pit[t] = np.where(live, pit[j], np.nan)
        next_out[t] = np.where(live, ~seq.alive[j], np.nan)
        clean = live & seq.alive[j] & ~pit[j] & ~out_lap[j] & ~seq.neutral[j]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.clip(seq.lap_s[j] / field[j] - 1.0, -0.05, 0.3) * 10.0
        next_res[t] = np.where(clean & np.isfinite(r), r, np.nan)
        prev_neutral = bool(seq.neutral[j - 1]) if j - 1 >= 1 else False
        if not prev_neutral:
            next_neutral[t] = float(seq.neutral[j])
    return Tokens(x, running, next_res, next_pit, next_out, next_neutral, seq.finish)


def _gap_ahead(position: NDArray[Any], gap: NDArray[Any], alive: NDArray[np.bool_]) -> Any:
    out = np.zeros(position.shape, dtype=np.float32)
    idx = np.flatnonzero(alive & (position > 0))
    if idx.size < 2:
        return out
    order = idx[np.argsort(position[idx], kind="stable")]
    ahead = np.diff(gap[order], prepend=gap[order[0]])
    out[order] = np.clip(ahead, 0.0, 5.0) / 5.0
    return out


# -- real races ---------------------------------------------------------------------------------
def real_race(
    laps: pl.DataFrame,
    results: pl.DataFrame,
    drivers: tuple[str, ...],
    race_laps: int,
    *,
    pace: dict[str, float] | None = None,
    upto: int | None = None,
    event_id: str = "",
    grid: NDArray[Any] | None = None,
    outcome: bool = True,
) -> RaceSeq:
    """A race from its ``laps`` rows (one event) and race ``results`` rows, for the cars
    ``drivers`` (the field, in any order). ``upto``: only laps ``1..upto`` (in the race).

    ``grid`` (ranks aligned to ``drivers``, 1 = pole) overrides the results' grid. Without
    ``outcome`` the classification is not read at all (a forecast's input): a car runs while
    it has laps completed, and nobody is classified."""
    last = race_laps if upto is None else min(upto, race_laps)
    n = len(drivers)
    col = {d: j for j, d in enumerate(drivers)}
    lap_s = np.full((last, n), np.nan, dtype=np.float32)
    gap_ahead = np.full((last, n), np.nan, dtype=np.float32)
    position = np.zeros((last, n), dtype=np.int16)
    compound = np.full((last, n), UNKNOWN, dtype=np.int8)
    age = np.full((last, n), np.nan, dtype=np.float32)
    pit = np.zeros((last, n), dtype=bool)
    seen = np.zeros((last, n), dtype=bool)
    neutral_votes = np.zeros((last, 2), dtype=np.int64)
    rows = laps.filter(pl.col("lap") <= last).select(
        "driver_id",
        "lap",
        "lap_ms",
        "position",
        "gap_ahead_s",
        "compound",
        "tyre_age",
        "in_lap",
        "neutral",
    )
    for d, lap, ms, pos, ga, comp, ta, inl, neu in rows.iter_rows():
        j = col.get(d)
        if j is None or lap < 1:
            continue
        i = lap - 1
        seen[i, j] = True
        lap_s[i, j] = ms / 1000.0 if ms is not None else np.nan
        position[i, j] = pos or 0
        gap_ahead[i, j] = ga if ga is not None else np.nan
        compound[i, j] = C_IDX.get(comp, UNKNOWN) if comp else UNKNOWN
        age[i, j] = ta if ta is not None else np.nan
        pit[i, j] = bool(inl)
        neutral_votes[i, int(bool(neu))] += 1
    lap_s[0] = np.nan  # the start lap is not comparable (and live timing has none)

    # running: from the classification's laps completed (a classified car — lapped or not
    # — runs to the flag); timing records otherwise
    res = results.select("driver_id", "laps", "classified")
    done = {d: (int(lp or 0), bool(c) and outcome) for d, lp, c in res.iter_rows()}
    alive = np.zeros((last, n), dtype=bool)
    for d, j in col.items():
        if d in done:
            completed, classified = done[d]
            upto_lap = race_laps if classified else completed
        else:
            upto_lap = int(np.flatnonzero(seen[:, j]).max() + 1) if seen[:, j].any() else 0
        alive[: min(upto_lap, last), j] = True

    # carry the last record forward over timing gaps (positions, tyres)
    for i in range(1, last):
        gap_rows = alive[i] & ~seen[i]
        position[i] = np.where(gap_rows, position[i - 1], position[i])
        compound[i] = np.where(gap_rows, compound[i - 1], compound[i])
        age[i] = np.where(gap_rows, age[i - 1] + 1.0, age[i])
    position = np.where(alive, position, 0).astype(np.int16)
    gap_s = _gap_matrix(lap_s, position, gap_ahead, alive)

    grid_slot = np.full(n, n, dtype=np.float64)
    finish = np.full(n, -1, dtype=np.int16)
    for d, g, classified, pos in results.select(
        "driver_id", "grid", "classified", "position"
    ).iter_rows():
        j = col.get(d)
        if j is None:
            continue
        if g is not None and g > 0:
            grid_slot[j] = g - 1
        if outcome and classified and pos is not None:
            finish[j] = pos
    if grid is not None:
        grid_slot = np.asarray(grid, dtype=np.float64)
    # grid slots as ranks (pit-lane starts behind); finishing positions from 0
    grid = np.argsort(np.argsort(grid_slot, kind="stable"), kind="stable").astype(np.int16)
    ranked = np.flatnonzero(finish >= 0)
    order = ranked[np.argsort(finish[ranked], kind="stable")]
    finish[:] = -1
    finish[order] = np.arange(order.size, dtype=np.int16)
    pace_arr = np.array([(pace or {}).get(d, np.nan) for d in drivers], dtype=np.float32)
    neutral = neutral_votes[:, 1] > neutral_votes[:, 0]
    return RaceSeq(
        lap_s,
        gap_s,
        position,
        compound,
        age,
        pit,
        alive,
        neutral,
        grid,
        pace_arr,
        finish,
        race_laps,
        tuple(drivers),
        event_id,
    )


def _gap_matrix(
    lap_s: FloatArray, position: NDArray[np.int16], gap_ahead: FloatArray, alive: Any
) -> FloatArray:
    """Seconds behind the leader per lap: the chain of intervals in running order; where an
    interval is missing, the summed lap times (from lap 2) to the leader, never less than
    the car ahead's gap."""
    laps, n = lap_s.shape
    total = np.nancumsum(np.nan_to_num(lap_s, nan=0.0), axis=0)
    out = np.full((laps, n), np.nan, dtype=np.float32)
    for i in range(laps):
        idx = np.flatnonzero(alive[i] & (position[i] > 0))
        if idx.size == 0:
            continue
        order = idx[np.argsort(position[i, idx], kind="stable")]
        acc, lead = 0.0, float(total[i, order[0]])
        out[i, order[0]] = 0.0
        for j in order[1:]:
            ga = float(gap_ahead[i, j])
            if np.isfinite(ga):
                acc += max(ga, 0.0)
            else:
                acc = max(acc, float(total[i, j]) - lead)
            out[i, j] = acc
    return out


def recent_pace(
    race_pace: pl.DataFrame,
    events: pl.DataFrame,
    drivers: tuple[str, ...],
    season: int,
    round_: int,
    *,
    last: int = 3,
) -> dict[str, float]:
    """Each driver's mean clean-lap pace deficit (% of a lap) over their ``last`` races
    before ``(season, round)`` (``race_pace`` must already be cut at the checkpoint)."""
    ev = events.select("event_id", "season", "round")
    rp = (
        race_pace.join(ev, on="event_id")
        .filter(
            (
                (pl.col("season") < season)
                | ((pl.col("season") == season) & (pl.col("round") < round_))
            )
            & pl.col("driver_id").is_in(list(drivers))
            & pl.col("pace_deficit_pct").is_not_null()
        )
        .sort("season", "round")
    )
    if rp.is_empty():
        return {}
    agg = rp.group_by("driver_id").agg(pl.col("pace_deficit_pct").tail(last).mean())
    return {str(d): float(v) for d, v in agg.iter_rows() if v is not None}


# -- simulated races -----------------------------------------------------------------------------
def simulated_races(
    trace: Any,
    orders: NDArray[np.int64],
    finished: NDArray[np.bool_],
    grid: NDArray[np.int64],
    pace: NDArray[np.float64],
    race_laps: int,
    rng: np.random.Generator,
    *,
    pace_noise: float = 0.3,
) -> list[RaceSeq]:
    """``race_sim_v2``'s traced races as :class:`RaceSeq` (one per simulated race). The pace
    covariate is each car's drawn pace as a deficit to the fastest, % of a lap, plus noise
    the size of a real estimate's error."""
    s, _, n = trace.lap_s.shape
    out = []
    for i in range(s):
        deficit = 100.0 * (pace[i] - pace[i].min()) + pace_noise * rng.standard_normal(n)
        finish = np.full(n, -1, dtype=np.int16)
        k = int(finished[i].sum())
        finish[orders[i, :k]] = np.arange(k, dtype=np.int16)
        out.append(
            RaceSeq(
                trace.lap_s[i].astype(np.float32),
                trace.gap_s[i].astype(np.float32),
                trace.position[i].astype(np.int16),
                trace.compound[i].astype(np.int8),
                trace.tyre_age[i].astype(np.float32),
                trace.pit[i].copy(),
                trace.alive[i].copy(),
                trace.neutral[i].copy(),
                np.argsort(np.argsort(grid[i], kind="stable"), kind="stable").astype(np.int16),
                np.maximum(deficit, 0.0).astype(np.float32),
                finish,
                race_laps,
            )
        )
    return out
