"""Race simulator core for ``race_sim_v2`` (plan M6 §B.3): v1's race, vectorised over
simulations, with one implementation for NumPy and for MLX (Metal on Apple silicon).

The race is the one :func:`f1p.models.race_sim.simulate` runs — the same start, stop plan,
safety-car hazard and pit pull-forward, lap times, front-to-back overtaking resolution,
restart packing and retirements — rewritten so that

* every per-lap operation is a whole-array operation over (simulations x cars);
* overtaking is resolved in *road order*: the cars are gathered into running order once per
  lap, resolved front to back column by column (each car gets one attempt on the car ahead,
  as in v1), and scattered back; there is no per-position scatter;
* the per-lap random numbers are inputs, so the NumPy and MLX twins can be run on the same
  draws (and must then give the same finishing orders);
* times are kept relative to the leader (re-centred every lap): every decision depends only
  on time differences, so this changes nothing but keeps float32 exact to ~10 µs.

Design note (deviation from the plan's odd-even transposition network): repeated adjacent
compare-and-swap passes would give a car several overtaking attempts per lap, changing the
race v1 is calibrated on, and would not reduce the sequential depth (n - 1 rounds either
way). The speed comes instead from vectorising over simulations and, on MLX, compiling the
whole lap step (the column loop is unrolled into one graph).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from f1p.models.race_sim import CircuitProfile, SimParams, _stop_plan

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
NO_STOP = 10_000  # padding for stop slots that are never used
MAX_STOPS = 3
RETIRED_KEY = 1e5  # retirements sort behind every classified car (exact in float32)


@dataclass
class RaceInputs:
    """Everything a batch of simulated races needs before lap 1 (host arrays)."""

    pace: FloatArray  # (S, n) fraction of a lap, centred and scaled (lower = faster)
    start: FloatArray  # (S, n) seconds behind the leader on the grid, with start chaos
    # (S, n, MAX_STOPS + 1) planned stop laps, ascending, padded with NO_STOP; the last column
    # is always NO_STOP, so a car that has made all its stops points at "never"
    stops: NDArray[np.int64]
    retire_lap: NDArray[np.int64]  # (S, n) lap of retirement (laps + 1: finishes)

    @property
    def shape(self) -> tuple[int, int]:
        s, n = self.pace.shape
        return s, n


def stop_slots(plan: NDArray[np.bool_]) -> NDArray[np.int64]:
    """v1's (S, n, laps + 1) stop plan as (S, n, MAX_STOPS + 1) stop laps, ascending, with a
    final NO_STOP column."""
    s, n, _ = plan.shape
    out = np.full((s, n, MAX_STOPS + 1), NO_STOP, dtype=np.int64)
    lap = np.arange(plan.shape[2])
    ranked = np.where(plan, lap[None, None, :], NO_STOP)
    ranked.sort(axis=2)
    k = min(MAX_STOPS, plan.shape[2])
    out[:, :, :k] = ranked[:, :, :k]
    return out


def draw_inputs(
    pace: FloatArray,
    grid: IntArray,
    p_finish: FloatArray,
    profile: CircuitProfile,
    params: SimParams,
    rng: np.random.Generator,
) -> RaceInputs:
    """The pre-race draws, distributed exactly as in v1's ``simulate``."""
    s, n = pace.shape
    laps = profile.laps
    centred = params.pace_scale * (pace - pace.mean(axis=1, keepdims=True))
    start = 0.2 * grid.astype(np.float64) + rng.normal(0.0, params.start_sd, (s, n))
    plan = _stop_plan(profile, s, n, rng)
    retire = np.where(
        rng.random((s, n)) < p_finish[None, :], laps + 1, rng.integers(1, laps + 1, (s, n))
    )
    return RaceInputs(centred, start, stop_slots(plan), retire.astype(np.int64))


@dataclass
class LapRandoms:
    sc: Any  # (S,) uniforms: does a safety-car period start?
    noise: Any  # (S, n) standard normals: lap-time noise
    passing: Any  # (S, n - 1) uniforms: overtaking attempts, by road position


def numpy_randoms(rng: np.random.Generator, s: int, n: int) -> Callable[[int], LapRandoms]:
    def draw(_: int) -> LapRandoms:
        return LapRandoms(rng.random(s), rng.normal(0.0, 1.0, (s, n)), rng.random((s, n - 1)))

    return draw


def mlx_randoms(seed: int, s: int, n: int) -> Callable[[int], LapRandoms]:
    """Per-lap random numbers drawn on the device from counter-based keys: lap ``k``'s draws
    depend only on (seed, k), so a run is reproducible and nothing crosses to the host."""
    import mlx.core as mx

    def draw(lap: int) -> LapRandoms:
        lap_key = mx.random.key((seed * 1_000_003 + lap) % (2**32))
        k_sc, k_noise, k_pass = mx.random.split(lap_key, 3)
        return LapRandoms(
            mx.random.uniform(shape=(s,), key=k_sc),
            mx.random.normal(shape=(s, n), key=k_noise),
            mx.random.uniform(shape=(s, n - 1), key=k_pass),
        )

    return draw


def simulate(
    pace: FloatArray,
    grid: IntArray,
    p_finish: FloatArray,
    profile: CircuitProfile,
    params: SimParams,
    seed: int,
    *,
    backend: str = "numpy",
) -> SimResult:
    """Drop-in for v1's ``simulate`` (plus safety-car counts): pre-race draws on the host,
    per-lap draws from the back end's own generator."""
    rng = np.random.default_rng(seed)
    inputs = draw_inputs(pace, grid, p_finish, profile, params, rng)
    s, n = pace.shape
    randoms = numpy_randoms(rng, s, n) if backend == "numpy" else mlx_randoms(seed, s, n)
    return run(inputs, profile, params, randoms, backend=backend)


# -- the array back ends --------------------------------------------------------------------------
class _NumpyOps:
    float_t = np.float64
    inf = np.inf

    @staticmethod
    def asarray(x: Any, dtype: Any = None) -> Any:
        return np.asarray(x, dtype=dtype)

    where = staticmethod(np.where)
    argsort = staticmethod(lambda x: np.argsort(x, axis=1, kind="stable"))
    take = staticmethod(np.take_along_axis)
    isfinite = staticmethod(np.isfinite)
    maximum = staticmethod(np.maximum)
    stack = staticmethod(lambda xs: np.stack(xs, axis=1))

    @staticmethod
    def sigmoid(x: Any) -> Any:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def row_min(x: Any) -> Any:
        return np.min(x, axis=1, keepdims=True)

    @staticmethod
    def any_last(x: Any) -> Any:
        return np.any(x, axis=-1)

    @staticmethod
    def to_numpy(x: Any) -> np.ndarray:
        return np.asarray(x)


class _MlxOps:
    def __init__(self) -> None:
        import mlx.core as mx

        self.mx = mx
        self.float_t = mx.float32
        self.inf = float("inf")

    def asarray(self, x: Any, dtype: Any = None) -> Any:
        if isinstance(x, self.mx.array):  # already on the device: never round-trip via NumPy
            return x.astype(dtype) if dtype is not None else x
        arr = np.asarray(x)
        if dtype is None:
            dtype = self.float_t if arr.dtype.kind == "f" else None
        if arr.dtype == np.int64:
            arr = arr.astype(np.int32)
        out = self.mx.array(arr)
        return out.astype(dtype) if dtype is not None else out

    def where(self, c: Any, a: Any, b: Any) -> Any:
        return self.mx.where(c, a, b)

    def argsort(self, x: Any) -> Any:
        return self.mx.argsort(x, axis=1)

    def take(self, x: Any, idx: Any, axis: int) -> Any:
        return self.mx.take_along_axis(x, idx, axis=axis)

    def isfinite(self, x: Any) -> Any:
        return self.mx.isfinite(x)

    def maximum(self, a: Any, b: Any) -> Any:
        return self.mx.maximum(a, b)

    def stack(self, xs: list[Any]) -> Any:
        return self.mx.stack(xs, axis=1)

    def sigmoid(self, x: Any) -> Any:
        return self.mx.sigmoid(x)

    def row_min(self, x: Any) -> Any:
        return self.mx.min(x, axis=1, keepdims=True)

    def any_last(self, x: Any) -> Any:
        return self.mx.any(x, axis=-1)

    def to_numpy(self, x: Any) -> np.ndarray:
        self.mx.eval(x)
        return np.array(x)


# -- one lap -----------------------------------------------------------------------------------
def _lap(
    ops: Any,
    state: dict[str, Any],
    rand: LapRandoms,
    lap: int,
    c: dict[str, Any],
) -> dict[str, Any]:
    """Advance every simulated race by one lap (the race of v1's ``simulate`` loop body)."""
    cum, age, alive = state["cum"], state["age"], state["alive"]
    sc_left, ptr, done, sc_count = state["sc_left"], state["ptr"], state["done"], state["sc_count"]
    stops = c["stops"]
    laps = c["laps"]
    n = c["n"]
    base = c["base"]

    # safety car: a new period may start (not on lap 1, not on the last laps)
    start = (sc_left == 0) & (rand.sc < c["sc_hazard"]) & (lap > 1) & (lap < laps - 2)
    sc_left = ops.where(start, c["sc_laps"], sc_left)
    sc_count = sc_count + start
    under = sc_left > 0

    # the next planned stop of each car; under a new safety car, a stop due within the
    # window is taken now instead (v1's pull-forward)
    next_stop = ops.take(stops, ptr[:, :, None], axis=2)[:, :, 0]
    due = (next_stop == lap) | (start[:, None] & (next_stop <= lap + c["sc_window"]))
    pit = due & alive
    ptr = ptr + pit

    lap_time = base * (
        1.0 + c["pace"] + c["deg"] * age + c["fuel"] * (laps - lap) + c["lap_noise"] * rand.noise
    )
    lap_time = ops.where(under[:, None], base * c["sc_slow"], lap_time)
    loss = ops.where(under[:, None], c["sc_pit_factor"], 1.0) * c["pit_loss"]
    lap_time = lap_time + pit * loss
    age = ops.where(pit, 0.0, age + 1.0)

    # track position: running order, then each car against the effective car ahead
    road = ops.argsort(ops.where(alive, cum, ops.inf))
    new = ops.where(alive, cum + lap_time, ops.inf)
    new_r = ops.take(new, road, axis=1)
    lt_r = ops.take(lap_time, road, axis=1)
    cols = [new_r[:, 0]]
    prev_new, prev_lt = new_r[:, 0], lt_r[:, 0]
    for p in range(1, n):
        mine, lt = new_r[:, p], lt_r[:, p]
        close = ops.isfinite(mine) & ops.isfinite(prev_new) & (mine < prev_new + c["min_gap"])
        p_pass = ops.sigmoid(c["pass_a"] + c["pass_b"] * (prev_lt - lt))
        passed = close & ~under & (rand.passing[:, p - 1] < p_pass)
        held = close & ~passed
        mine = ops.where(held, prev_new + c["min_gap"], mine)
        cols.append(mine)
        better = passed & (mine < prev_new)
        prev_new = ops.where(better, prev_new, mine)
        prev_lt = ops.where(better, prev_lt, lt)
    cum = ops.take(ops.stack(cols), ops.argsort(road), axis=1)

    # the safety car ends: the field closes up behind the leader
    sc_left = ops.maximum(sc_left - 1, 0)
    restart = under & (sc_left == 0)
    rank = ops.argsort(ops.argsort(cum))
    lead = ops.row_min(cum)
    packed = lead + c["sc_gap"] * rank
    cum = ops.where(restart[:, None] & alive, packed, cum)

    # retirements
    out = alive & (c["retire_lap"] == lap)
    alive = alive & ~out
    cum = ops.where(out, ops.inf, cum)
    done = ops.where(alive, lap, done)

    # relative to the leader (decisions depend only on differences)
    lead = ops.row_min(ops.where(alive, cum, ops.inf))
    cum = ops.where(alive, cum - ops.where(ops.isfinite(lead), lead, 0.0), cum)
    return {
        "cum": cum,
        "age": age,
        "alive": alive,
        "sc_left": sc_left,
        "ptr": ops.where(ptr > MAX_STOPS, MAX_STOPS, ptr),  # the NO_STOP column
        "done": done,
        "sc_count": sc_count,
    }


def _constants(
    ops: Any, inputs: RaceInputs, profile: CircuitProfile, params: SimParams
) -> dict[str, Any]:
    laps = profile.laps
    _, n = inputs.shape
    return {
        "n": n,
        "laps": laps,
        "base": profile.base_lap_s,
        "pace": ops.asarray(inputs.pace),
        "deg": profile.deg_per_lap,
        "fuel": profile.fuel_per_lap,
        "lap_noise": params.lap_noise,
        "sc_hazard": min(profile.sc_per_race / max(laps - 3, 1), 0.5),
        "sc_laps": params.sc_laps,
        "sc_slow": params.sc_slow,
        "sc_window": params.sc_pit_window,
        "sc_pit_factor": params.sc_pit_factor,
        "sc_gap": params.sc_gap,
        "pit_loss": profile.pit_loss_s,
        "min_gap": params.min_gap,
        "pass_a": params.pass_a,
        "pass_b": params.pass_b,
        "stops": ops.asarray(inputs.stops),
        "retire_lap": ops.asarray(inputs.retire_lap),
    }


@dataclass
class SimResult:
    orders: IntArray  # (S, n) finishing order (car indices)
    finished: NDArray[np.bool_]  # (S, n)
    sc_count: NDArray[np.int64]  # (S,) safety-car periods


def run(
    inputs: RaceInputs,
    profile: CircuitProfile,
    params: SimParams,
    randoms: Callable[[int], LapRandoms],
    *,
    backend: str = "numpy",
) -> SimResult:
    """Run the races of ``inputs`` on ``backend`` (``numpy`` or ``mlx``), drawing each lap's
    random numbers from ``randoms(lap)``."""
    ops: Any = _NumpyOps() if backend == "numpy" else _MlxOps()
    s, n = inputs.shape
    c = _constants(ops, inputs, profile, params)
    zeros_i = np.zeros((s, n), dtype=np.int64)
    state: dict[str, Any] = {
        "cum": ops.asarray(inputs.start),
        "age": ops.asarray(np.zeros((s, n))),
        "alive": ops.asarray(np.ones((s, n), dtype=bool)),
        "sc_left": ops.asarray(np.zeros(s, dtype=np.int64)),
        "ptr": ops.asarray(zeros_i),
        "done": ops.asarray(zeros_i),
        "sc_count": ops.asarray(np.zeros(s, dtype=np.int64)),
    }
    if backend == "mlx":
        mx = ops.mx

        def body(st: dict[str, Any], sc: Any, noise: Any, passing: Any, lap: Any) -> dict[str, Any]:
            return _lap(ops, st, LapRandoms(sc, noise, passing), lap, c)

        compiled = mx.compile(body)
        for lap in range(1, profile.laps + 1):
            r = randoms(lap)
            state = compiled(
                state,
                ops.asarray(r.sc),
                ops.asarray(r.noise),
                ops.asarray(r.passing),
                mx.array(lap, dtype=mx.int32),
            )
            mx.eval(state)
    else:
        for lap in range(1, profile.laps + 1):
            state = _lap(ops, state, randoms(lap), lap, c)
    alive = ops.to_numpy(state["alive"]).astype(bool)
    cum = ops.to_numpy(state["cum"]).astype(np.float64)
    done = ops.to_numpy(state["done"]).astype(np.int64)
    idx = np.arange(n)[None, :] / 128.0
    key = np.where(alive, cum, RETIRED_KEY - done + idx)
    orders = np.argsort(key, axis=1, kind="stable")
    return SimResult(orders, alive, ops.to_numpy(state["sc_count"]).astype(np.int64))
