"""``race_sim_v2``'s race (plan M6 §B.2): the fitted sub-models of :mod:`f1p.models.sim_fit`
and the Bayesian lap model's posterior, run lap by lap over many simulated races at once.

Per simulated race, one posterior draw (each car's pace, the compound wear curves with a new
wear rate for this race, compound offsets, fuel, pit loss, lap noise and the traffic terms).
Per lap:

1. **lap 1** — the grid becomes the lap-1 order through the start model (a Plackett-Luce
   ranking of start utilities); starting compounds from the compound model;
2. **incidents** — the fitted hazard (higher on lap 1, after restarts); an incident retires
   0-3 cars, chosen by running position, and neutralises the race (always when nobody
   retires, else with the fitted probability) for an empirical number of laps;
   **mechanical** retirements come from each car's reliability;
3. **strategy** — each car pits with the fitted per-lap policy (tyre age, race left, stops
   made, neutralised lap, compound); a dry race forces a stop on the last lap but one; the
   next compound comes from the compound model;
4. **lap times** — posterior pace + wear(compound, age) + compound offset + fuel + noise;
   a car within 3 s of the car ahead loses the traffic term (dirty air, DRS) the lap model
   measured; behind a safety car everyone runs the neutralised pace;
5. **overtaking** — in running order, a car within 1 s at the line attempts to pass with the
   fitted probability (circuit, typical pace advantage, gap, tyre-age and softness
   advantages); one that is not past is held behind;
6. **restarts** pack the field behind the leader.

This module is the NumPy oracle. The knobs calibrated by simulation-based inference (plan
§B.4) are in :class:`Knobs`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from f1p.models.lap_data import COMPOUNDS, GAP_HINGES, ispline_basis
from f1p.models.sim_fit import (
    DRY,
    SOFTNESS,
    CompoundModel,
    IncidentModel,
    LogitFit,
    StartModel,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
C_IDX = {c: i for i, c in enumerate(COMPOUNDS)}
DRY_IDX = np.array([C_IDX[c] for c in DRY], dtype=np.int64)
SOFT_OF = np.array([SOFTNESS.get(c, np.nan) for c in COMPOUNDS])
RETIRED_KEY = 1e5
MAX_AGE = 80
# Traffic when no lap_v2 posterior is at hand: the posterior means of the first real lap_v2
# fit (24 races to 2024-R02), per GAP_HINGES (0.5/1/2/3 s) and for DRS range, as fractions of
# a lap. Following closely costs time (dirty air), DRS range gives a little back.
TRAFFIC_GAP = (-0.0010, 0.0061, 0.0004, 0.0034)
TRAFFIC_DRS = -0.0012


@dataclass(frozen=True, slots=True)
class Knobs:
    """Global settings the data cannot fit directly (calibrated by SBI, plan §B.4)."""

    pace_scale: float = 1.0  # multiplies the posterior pace gaps
    overtake_shift: float = 0.0  # added to every overtaking logit
    incident_scale: float = 1.0  # multiplies the incident odds
    start_gap_s: float = 0.45  # s per position at the end of lap 1
    # scale of the start model's Gumbel noise: the Plackett-Luce fit alone shuffles lap 1 too
    # much (mean |positions gained| 2.2 vs 1.48 observed; 0.6 gives 1.41)
    start_noise: float = 0.6
    # s: a car held behind ends the lap this far back (observed close-following gaps have a
    # median of ~0.86 s; only 4.7% of racing laps are within 0.5 s of the car ahead)
    min_gap: float = 0.7
    follow_window: float = 1.0  # s at the line: closer than this, a pass may be attempted
    sc_slow: float = 1.4  # neutralised lap time / base lap
    sc_gap: float = 0.8  # s between cars at a restart
    sc_pit_factor: float = 0.55  # share of the pit loss paid under neutralisation


@dataclass
class SubModels:
    overtaking: LogitFit
    strategy: LogitFit
    start: StartModel
    incidents: IncidentModel
    compounds: CompoundModel


@dataclass
class Draws:
    """Per simulated race: parameters drawn from the lap model's posterior."""

    pace: FloatArray  # (S, n) fraction of a lap (lower = faster), centred per race
    wear_w: FloatArray  # (S, C, K) I-spline weights per compound (COMPOUNDS order)
    level: FloatArray  # (S, C) compound offsets vs MEDIUM
    fuel: FloatArray  # (S,) effect of 50 laps of fuel
    noise_sd: FloatArray  # (S,) lap noise, fraction of a lap
    pit_loss_s: FloatArray  # (S,) seconds lost to a stop
    gap_coef: FloatArray  # (S, len(GAP_HINGES)) traffic hinge effects
    drs: FloatArray  # (S,) the DRS-range term


def draws_from_posterior(
    post: dict[str, FloatArray],
    order: list[int],
    s: int,
    base_lap_s: float,
    rng: np.random.Generator,
) -> Draws:
    """``s`` races' parameters from ``lap_v2`` draws (as :func:`lap_v2.load_draws` returns
    them). ``order`` maps the race's cars to the posterior's entries."""
    d = rng.integers(0, len(post["pace_next"]), s)
    pace = post["pace_next"][d][:, order]
    pace = pace - pace.mean(axis=1, keepdims=True)
    rate = post["wear_event_sd"][d][:, None, None] * rng.standard_normal((s, len(COMPOUNDS), 1))
    wear_w = np.exp(post["wear_log"][d] + rate)  # this race's wear rate per compound
    level = np.zeros((s, len(COMPOUNDS)))
    level[:, 1:] = post["compound"][d]
    pit = post["pit"][d] + post["pit_event_sd"][d][:, None] * rng.standard_normal((s, 2))
    pit_loss = (np.expm1(pit[:, 0]) + np.expm1(pit[:, 1])) * base_lap_s
    return Draws(
        pace,
        wear_w,
        level,
        post["fuel"][d],
        np.exp(post["sigma_log"][d]),
        pit_loss,
        post["gap"][d],
        post["drs"][d],
    )


def draws_from_laplace(
    pace_mean: FloatArray,
    pace_sd: FloatArray,
    s: int,
    rng: np.random.Generator,
    *,
    deg: dict[str, float],
    offset: dict[str, float],
    fuel: float,
    pit_loss_s: float,
    noise_sd: float = 0.004,
    gap_coef: tuple[float, ...] = TRAFFIC_GAP,
    drs: float = TRAFFIC_DRS,
) -> Draws:
    """Fallback without a ``lap_v2`` posterior: the Laplace lap model's pace (mean, sd),
    linear degradation per compound (an I-spline fit of the line) and typical traffic
    effects (:data:`TRAFFIC_GAP`, :data:`TRAFFIC_DRS`)."""
    n = len(pace_mean)
    pace = pace_mean[None, :] + rng.standard_normal((s, n)) * pace_sd[None, :]
    pace = pace - pace.mean(axis=1, keepdims=True)
    ages = np.arange(0.0, 41.0)
    basis = ispline_basis(ages)
    k = basis.shape[1]
    wear_w = np.zeros((s, len(COMPOUNDS), k))
    for c, i in C_IDX.items():
        slope = deg.get(c, deg.get("MEDIUM", 0.0008))
        w, *_ = np.linalg.lstsq(basis, slope * ages, rcond=None)
        wear_w[:, i, :] = np.clip(w, 0.0, None)[None, :]
    level = np.zeros((s, len(COMPOUNDS)))
    for c, i in C_IDX.items():
        level[:, i] = offset.get(c, 0.0)
    return Draws(
        pace,
        wear_w,
        level,
        np.full(s, fuel),
        np.full(s, noise_sd),
        np.full(s, pit_loss_s),
        np.tile(np.asarray(gap_coef, dtype=np.float64), (s, 1)),
        np.full(s, drs),
    )


@dataclass
class RaceSpec:
    circuit: str
    laps: int
    base_lap_s: float
    teams: list[str]  # start-model team labels (lineage:season), one per car
    p_finish: FloatArray  # (n,) reliability
    grid: IntArray  # (S, n) starting slots, 0 = pole
    dry: bool = True


@dataclass
class Result:
    orders: IntArray  # (S, n) finishing order
    finished: NDArray[np.bool_]  # (S, n)
    stops: IntArray  # (S, n) stops made
    neutral_periods: IntArray  # (S,)
    overtakes: IntArray  # (S,) passes on track (excluding lap 1 and pit cycles)
    lap1_order: IntArray  # (S, n)
    close_pairs: IntArray  # (S,) green-flag car-laps started within 1 s of the car ahead
    extra: dict[str, FloatArray] = field(default_factory=dict)


def _logit(x: FloatArray) -> FloatArray:
    """The logistic function (numerically stable for any x)."""
    return expit(x)


@dataclass
class _State:
    """The race state carried from lap to lap (one row per simulated race)."""

    cum: FloatArray  # (S, n) time behind the leader
    comp: IntArray  # (S, n) compound (COMPOUNDS index)
    age: FloatArray  # (S, n) laps on the current tyres
    stops: IntArray  # (S, n)
    alive: NDArray[np.bool_]  # (S, n)
    done: IntArray  # (S, n) laps completed
    neutral_left: IntArray  # (S,) neutralised laps still to run
    restart: NDArray[np.bool_]  # (S,) this lap follows a neutralisation
    periods: IntArray  # (S,) neutralisations so far
    passes: IntArray  # (S,) overtakes so far
    close: IntArray  # (S,) laps a car started within the follow window of the car ahead


@dataclass
class _Race:
    """Everything constant during one batch of races."""

    spec: RaceSpec
    draws: Draws
    models: SubModels
    knobs: Knobs
    s: int
    n: int
    pace: FloatArray
    age_basis: FloatArray
    b_ov: dict[str, float]
    b_st: dict[str, float]
    b_in: dict[str, float]
    circ_inc: float
    circ_str: float
    circ_ov: float
    mech: FloatArray  # (n,) per-lap mechanical retirement hazard


def _start(r: _Race, rng: np.random.Generator) -> tuple[_State, IntArray]:
    """Lap 1: the start model turns the grid into the lap-1 order; starting compounds."""
    sm, spec, s, n = r.models.start, r.spec, r.s, r.n
    z = (sm.run.get(spec.circuit, sm.run_mean) - sm.run_mean) / max(sm.run_sd, 1e-9)
    slope = sm.w_slot + sm.w_run * z + sm.circuit.get(spec.circuit, 0.0)
    team = np.array([sm.team.get(t, 0.0) for t in spec.teams])
    util = slope * spec.grid.astype(np.float64) + sm.w_clean * (spec.grid % 2 == 0)
    util = util + team[None, :] + r.knobs.start_noise * rng.gumbel(size=(s, n))
    lap1_order = np.argsort(-util, axis=1)
    lap1_rank = np.argsort(lap1_order, axis=1)
    draw = rng.choice(len(DRY), size=(s, n), p=r.models.compounds.start)
    comp = DRY_IDX[draw] if spec.dry else np.full((s, n), C_IDX["INTERMEDIATE"], dtype=np.int64)
    state = _State(
        cum=r.knobs.start_gap_s * lap1_rank.astype(np.float64),
        comp=comp.astype(np.int64),
        age=np.ones((s, n)),  # one lap done on the starting set
        stops=np.zeros((s, n), dtype=np.int64),
        alive=np.ones((s, n), dtype=bool),
        done=np.ones((s, n), dtype=np.int64),
        neutral_left=np.zeros(s, dtype=np.int64),
        restart=np.zeros(s, dtype=bool),
        periods=np.zeros(s, dtype=np.int64),
        passes=np.zeros(s, dtype=np.int64),
        close=np.zeros(s, dtype=np.int64),
    )
    return state, lap1_order.astype(np.int64)


def _incidents(
    st: _State, r: _Race, lap: int, rng: np.random.Generator
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """This lap's incidents: (cars retiring at the end of the lap, neutralised this lap)."""
    inc, s, n, laps = r.models.incidents, r.s, r.n, r.spec.laps
    logit_inc = (
        inc.hazard.coef[0]
        + r.circ_inc
        + r.b_in["lap1"] * (lap == 1)
        + r.b_in["restart"] * st.restart
        + r.b_in["progress"] * (lap / laps)
        + np.log(r.knobs.incident_scale)
    )
    incident = (rng.random(s) < _logit(logit_inc)) & (st.neutral_left == 0)
    k = np.where(incident, rng.choice(len(inc.dnf_counts), size=s, p=inc.dnf_counts), 0)
    goes_neutral = incident & ((k == 0) | (rng.random(s) < inc.p_neutral_given_dnf))
    if lap >= laps - 1:
        goes_neutral[:] = False  # no new neutralisation on the last laps
    length = inc.lengths[rng.integers(0, len(inc.lengths), s)]
    if lap == 1:
        # a neutralisation drawn on lap 1 runs from lap 2
        st.neutral_left = np.where(goes_neutral, np.minimum(length, laps - 1), st.neutral_left)
    else:
        st.neutral_left = np.where(
            goes_neutral, np.minimum(length, laps - lap + 1), st.neutral_left
        )
    st.periods = st.periods + goes_neutral
    # victims: k cars by running position (weights by decile), Gumbel top-k
    running = np.argsort(np.argsort(np.where(st.alive, st.cum, np.inf), axis=1), axis=1)
    decile = np.clip((running * 10) // max(n, 1), 0, 9)
    noise = rng.gumbel(size=(s, n))
    score = np.where(st.alive, np.log(inc.position_weights[decile]) + noise, -np.inf)
    pick_rank = np.argsort(np.argsort(-score, axis=1), axis=1)
    crashed = st.alive & (pick_rank < k[:, None])
    broke = st.alive & (rng.random((s, n)) < r.mech[None, :])
    neutral = (st.neutral_left > 0) & (lap > 1)
    return crashed | broke, neutral


def _strategy(
    st: _State, r: _Race, lap: int, neutral: NDArray[np.bool_], rng: np.random.Generator
) -> NDArray[np.bool_]:
    """Who pits this lap (and on which compound they rejoin)."""
    s, n, laps, b = r.s, r.n, r.spec.laps, r.b_st
    left = (laps - lap) / laps
    a20 = st.age / 20.0
    logit_pit = (
        r.models.strategy.coef[0]
        + r.circ_str
        + b["age"] * a20
        + b["age2"] * a20**2
        + b["left"] * left
        + b["stops"] * np.minimum(st.stops, 4)
        + b["neutral"] * neutral[:, None]
        + b["soft"] * (st.comp == C_IDX["SOFT"])
        + b["hard"] * (st.comp == C_IDX["HARD"])
        + b["wet"] * np.isin(st.comp, [C_IDX["INTERMEDIATE"], C_IDX["WET"]])
    )
    pit = st.alive & (rng.random((s, n)) < _logit(logit_pit)) & (lap < laps)
    if r.spec.dry:
        pit |= st.alive & (st.stops == 0) & (lap == laps - 1)  # two dry compounds required
        stint = np.minimum(st.stops + 2, 4) - 2  # the stint the stop starts: 2, 3, 4+
        bucket = r.models.compounds.left_bucket(np.full((s, n), left))
        probs = r.models.compounds.next[stint, bucket]  # (S, n, 3)
        u = rng.random((s, n, 1))
        new_comp = DRY_IDX[np.minimum((u > np.cumsum(probs, axis=2)).sum(axis=2), 2)]
        st.comp = np.where(pit, new_comp, st.comp)
    st.stops = st.stops + pit
    return pit


def _lap_times(
    st: _State,
    r: _Race,
    lap: int,
    pit: NDArray[np.bool_],
    neutral: NDArray[np.bool_],
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    """(lap time, its expected part without noise) per car, in seconds."""
    s, n, laps, base, d = r.s, r.n, r.spec.laps, r.spec.base_lap_s, r.draws
    a_idx = np.minimum(st.age, MAX_AGE).astype(np.int64)
    w_car = d.wear_w[np.arange(s)[:, None], st.comp]  # (S, n, K)
    wear = np.einsum("snk,snk->sn", w_car, r.age_basis[a_idx])
    level = np.take_along_axis(d.level, st.comp, axis=1)
    expected = r.pace + wear + level
    lap_time = base * (
        1.0
        + expected
        + d.fuel[:, None] * (laps - lap) / 50.0
        + d.noise_sd[:, None] * rng.standard_normal((s, n))
    )
    loss = np.where(neutral, r.knobs.sc_pit_factor, 1.0)[:, None] * d.pit_loss_s[:, None]
    lap_time = np.where(neutral[:, None], base * r.knobs.sc_slow, lap_time) + pit * loss
    st.age = np.where(pit, 0.0, st.age + 1.0)
    return lap_time, base * expected


def _overtaking(
    st: _State,
    r: _Race,
    lap_time: FloatArray,
    expected_s: FloatArray,
    pit: NDArray[np.bool_],
    neutral: NDArray[np.bool_],
    rng: np.random.Generator,
) -> None:
    """Running order, traffic, and one overtaking attempt per car on the car ahead."""
    s, n, base, k, d = r.s, r.n, r.spec.base_lap_s, r.knobs, r.draws
    road = np.argsort(np.where(st.alive, st.cum, np.inf), axis=1)
    cum_r = np.take_along_axis(st.cum, road, axis=1)
    with np.errstate(invalid="ignore"):  # retired cars (inf - inf): masked below
        gap_before = np.diff(cum_r, axis=1, prepend=-np.inf)  # to the car ahead at the line
    hinges = np.asarray(GAP_HINGES)[None, None, :]
    hinge = np.clip(hinges - gap_before[..., None], 0.0, None) / hinges
    traffic = (hinge * d.gap_coef[:, None, :]).sum(axis=2) + d.drs[:, None] * (gap_before < 1.0)
    traffic = np.where(np.isfinite(traffic), traffic, 0.0)  # retired cars: no traffic
    lt_r = np.take_along_axis(lap_time, road, axis=1)
    lt_r = lt_r + np.where(neutral[:, None], 0.0, base * traffic)
    exp_r = np.take_along_axis(expected_s, road, axis=1)
    age_r = np.take_along_axis(st.age, road, axis=1)
    soft_r = np.nan_to_num(SOFT_OF[np.take_along_axis(st.comp, road, axis=1)])
    pit_r = np.take_along_axis(pit, road, axis=1)
    new_r = cum_r + lt_r
    b, ov = r.b_ov, r.models.overtaking
    cols: list[FloatArray] = [new_r[:, 0]]
    ref_new, ref_exp, ref_age = new_r[:, 0], exp_r[:, 0], age_r[:, 0]
    ref_soft, ref_pit = soft_r[:, 0], pit_r[:, 0]
    passes = np.zeros(s, dtype=np.int64)
    n_close = np.zeros(s, dtype=np.int64)
    for p in range(1, n):
        mine = new_r[:, p]
        live = np.isfinite(mine) & np.isfinite(ref_new)
        # an attempt: both cars racing on track (passing a car in the pit lane is free, and
        # the data's close pairs exclude pit laps the same way)
        close = live & (gap_before[:, p] < k.follow_window) & ~neutral & ~pit_r[:, p] & ~ref_pit
        logit_pass = (
            ov.coef[0]
            + r.circ_ov
            + b["pace_adv_s"] * (ref_exp - exp_r[:, p])
            + b["gap_s"] * gap_before[:, p]
            + b["age_adv"] * (ref_age - age_r[:, p])
            + b["soft_adv"] * (soft_r[:, p] - ref_soft)
        )
        passed = close & (rng.random(s) < _logit(logit_pass))
        passes = passes + passed
        n_close = n_close + close
        # past: ahead of the car it attacked; not past: held behind it when both are on
        # track or both in the pit lane (they queue); a car in the pit lane holds up nobody
        # on track, and a car on track is not held by one in the pit lane
        mine = np.where(passed, np.minimum(mine, ref_new - 0.1), mine)
        same_lane = pit_r[:, p] == ref_pit
        held = live & ~passed & (mine < ref_new + k.min_gap) & same_lane
        mine = np.where(held, ref_new + k.min_gap, mine)
        cols.append(mine)
        # the next car races the car now directly ahead of it: the attacked car if this one
        # got past (it is behind the attacker), else this one
        ref_new = np.where(passed, ref_new, mine)
        ref_exp = np.where(passed, ref_exp, exp_r[:, p])
        ref_age = np.where(passed, ref_age, age_r[:, p])
        ref_soft = np.where(passed, ref_soft, soft_r[:, p])
        ref_pit = np.where(passed, ref_pit, pit_r[:, p])
    st.cum = np.take_along_axis(np.stack(cols, axis=1), np.argsort(road, axis=1), axis=1)
    st.passes = st.passes + passes
    st.close = st.close + n_close


def _end_of_lap(
    st: _State, r: _Race, lap: int, out: NDArray[np.bool_], neutral: NDArray[np.bool_]
) -> None:
    """Restarts pack the field; retirements leave; times re-centred on the leader."""
    st.neutral_left = np.maximum(st.neutral_left - neutral, 0)
    st.restart = neutral & (st.neutral_left == 0)
    rank = np.argsort(np.argsort(st.cum, axis=1), axis=1)
    lead = np.min(np.where(st.alive, st.cum, np.inf), axis=1, keepdims=True)
    packed = lead + r.knobs.sc_gap * rank
    st.cum = np.where(st.restart[:, None] & st.alive, packed, st.cum)
    st.alive = st.alive & ~out
    st.cum = np.where(out, np.inf, st.cum)
    st.done = np.where(st.alive, lap, st.done)
    lead = np.min(np.where(st.alive, st.cum, np.inf), axis=1, keepdims=True)
    st.cum = np.where(st.alive, st.cum - np.where(np.isfinite(lead), lead, 0.0), st.cum)


def simulate(
    spec: RaceSpec,
    draws: Draws,
    models: SubModels,
    knobs: Knobs,
    rng: np.random.Generator,
) -> Result:
    s, n = spec.grid.shape
    laps = spec.laps
    inc = models.incidents
    st_fit, ov_fit = models.strategy, models.overtaking
    r = _Race(
        spec=spec,
        draws=draws,
        models=models,
        knobs=knobs,
        s=s,
        n=n,
        pace=knobs.pace_scale * draws.pace,
        age_basis=ispline_basis(np.arange(MAX_AGE + 1, dtype=np.float64)),
        b_ov=dict(zip(ov_fit.names, ov_fit.coef[1:].tolist(), strict=True)),
        b_st=dict(zip(st_fit.names, st_fit.coef[1:].tolist(), strict=True)),
        b_in=dict(zip(inc.hazard.names, inc.hazard.coef[1:].tolist(), strict=True)),
        circ_inc=inc.hazard.offset(spec.circuit),
        circ_str=st_fit.offset(spec.circuit),
        circ_ov=ov_fit.offset(spec.circuit) + knobs.overtake_shift,
        mech=1.0 - (1.0 - (1.0 - spec.p_finish) * inc.mechanical_share) ** (1.0 / laps),
    )
    st, lap1_order = _start(r, rng)
    for lap in range(1, laps + 1):
        out, neutral = _incidents(st, r, lap, rng)
        if lap == 1:
            # the start lap: the order is set; retirements leave at the end of it
            st.alive = st.alive & ~out
            st.cum = np.where(out, np.inf, st.cum)
            continue
        pit = _strategy(st, r, lap, neutral, rng)
        lap_time, expected_s = _lap_times(st, r, lap, pit, neutral, rng)
        _overtaking(st, r, lap_time, expected_s, pit, neutral, rng)
        _end_of_lap(st, r, lap, out, neutral)
    key = np.where(st.alive, st.cum, RETIRED_KEY - st.done + np.arange(n)[None, :] / 128.0)
    orders = np.argsort(key, axis=1, kind="stable")
    return Result(orders, st.alive, st.stops, st.periods, st.passes, lap1_order, st.close)
