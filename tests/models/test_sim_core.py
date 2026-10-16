from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import stats

from f1p.models import sim_core as sc
from f1p.models.race_sim import CircuitProfile, SimParams, Trace, simulate

PROFILE = CircuitProfile(
    laps=57, base_lap_s=92.0, pit_loss_s=22.0, sc_per_race=1.2, stops=(0.05, 0.6, 0.3, 0.05)
)
PARAMS = SimParams()
N = 20


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS p-value."""
    res: Any = stats.ks_2samp(a, b)
    return float(res.pvalue)


def _field(s: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pace_mean = np.linspace(-0.008, 0.012, N)
    pace = pace_mean[None, :] + rng.normal(0.0, 0.002, (s, N))
    grid = np.argsort(np.argsort(pace + rng.normal(0, 0.002, (s, N)), axis=1), axis=1)
    p_finish = np.linspace(0.95, 0.85, N)
    return pace, grid, p_finish


def test_stop_slots() -> None:
    plan = np.zeros((1, 2, 10), dtype=bool)
    plan[0, 0, [3, 7]] = True
    slots = sc.stop_slots(plan)
    assert slots.shape == (1, 2, sc.MAX_STOPS + 1)
    assert slots[0, 0].tolist() == [3, 7, sc.NO_STOP, sc.NO_STOP]
    assert np.all(slots[0, 1] == sc.NO_STOP)
    assert np.all(slots[..., -1] == sc.NO_STOP)  # "never" after the last stop


def test_numpy_race_is_sane() -> None:
    s = 3000
    pace, grid, pf = _field(s, 0)
    rng = np.random.default_rng(1)
    inputs = sc.draw_inputs(pace, grid, pf, PROFILE, PARAMS, rng)
    res = sc.run(inputs, PROFILE, PARAMS, sc.numpy_randoms(rng, s, N))
    assert res.orders.shape == (s, N)
    assert np.all(np.sort(res.orders, axis=1) == np.arange(N))  # each order a permutation
    # finishers come before every retirement, retirements by laps completed
    pos = np.argsort(res.orders, axis=1)
    fin = res.finished
    assert np.all(np.where(fin, pos, -1).max(axis=1) < np.where(~fin, pos, N).min(axis=1))
    # the fastest cars win most
    wins = np.bincount(res.orders[:, 0], minlength=N)
    assert wins[:3].sum() > wins[-3:].sum() * 5
    assert 0.5 < res.sc_count.mean() < 2.0  # the circuit's safety-car rate


@pytest.fixture(scope="module")
def shared() -> tuple[sc.RaceInputs, list[sc.LapRandoms]]:
    s = 2000
    pace, grid, pf = _field(s, 2)
    rng = np.random.default_rng(3)
    inputs = sc.draw_inputs(pace, grid, pf, PROFILE, PARAMS, rng)
    draw = sc.numpy_randoms(rng, s, N)
    return inputs, [draw(lap) for lap in range(1, PROFILE.laps + 1)]


def test_numpy_and_mlx_run_the_same_race(
    shared: tuple[sc.RaceInputs, list[sc.LapRandoms]],
) -> None:
    pytest.importorskip("mlx.core")
    inputs, draws = shared
    a = sc.run(inputs, PROFILE, PARAMS, lambda lap: draws[lap - 1], backend="numpy")
    b = sc.run(inputs, PROFILE, PARAMS, lambda lap: draws[lap - 1], backend="mlx")
    np.testing.assert_array_equal(a.sc_count, b.sc_count)
    np.testing.assert_array_equal(a.finished, b.finished)
    same = np.all(a.orders == b.orders, axis=1).mean()
    # float32 vs float64 can flip a decision that lands exactly on a threshold, rarely
    assert same >= 0.99


def test_core_matches_v1_in_distribution() -> None:
    """The rewrite runs v1's race: same per-car finishing-position distributions, finishers
    and safety-car counts (two independent samples)."""
    s = 20_000
    pace, grid, pf = _field(s, 4)
    rng1 = np.random.default_rng(5)
    trace = Trace(s)
    o1, f1 = simulate(pace, grid, pf, PROFILE, PARAMS, rng1, trace)
    sc1 = np.stack(trace.sc, axis=1)
    starts1 = (sc1[:, 1:] & ~sc1[:, :-1]).sum(axis=1) + sc1[:, 0]
    rng2 = np.random.default_rng(6)
    inputs = sc.draw_inputs(pace, grid, pf, PROFILE, PARAMS, rng2)
    res = sc.run(inputs, PROFILE, PARAMS, sc.numpy_randoms(rng2, s, N))
    pos1 = np.argsort(o1, axis=1)
    pos2 = np.argsort(res.orders, axis=1)
    p_values = [_ks(pos1[:, k], pos2[:, k]) for k in range(N)]
    assert min(p_values) > 0.01 / N  # Bonferroni over the cars
    se = np.sqrt(pos1.var(axis=0) / s + pos2.var(axis=0) / s)
    assert np.all(np.abs(pos1.mean(axis=0) - pos2.mean(axis=0)) < 4 * se)
    assert abs(f1.mean() - res.finished.mean()) < 0.01
    # v1's trace counts a period that starts on lap 1 too; the core's never does (neither
    # does v1: lap > 1 is required), so the counts are directly comparable
    assert _ks(starts1, res.sc_count) > 0.01


def test_mlx_device_randoms_are_reproducible_and_distributed_alike() -> None:
    pytest.importorskip("mlx.core")
    s = 20_000
    pace, grid, pf = _field(s, 7)
    a = sc.simulate(pace, grid, pf, PROFILE, PARAMS, seed=11, backend="mlx")
    b = sc.simulate(pace, grid, pf, PROFILE, PARAMS, seed=11, backend="mlx")
    np.testing.assert_array_equal(a.orders, b.orders)  # same seed, same races
    c = sc.simulate(pace, grid, pf, PROFILE, PARAMS, seed=12, backend="numpy")
    pa, pc = np.argsort(a.orders, axis=1), np.argsort(c.orders, axis=1)
    p_values = [_ks(pa[:, k], pc[:, k]) for k in range(N)]
    assert min(p_values) > 0.01 / N
    assert _ks(a.sc_count, c.sc_count) > 0.01
