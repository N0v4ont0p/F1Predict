from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from f1p.eval import harness
from f1p.eval.baselines import Uniform
from f1p.eval.forecast import pl_sample_orders
from f1p.lake.curate_f1db import Curated
from f1p.models.race_sim import (
    CircuitProfile,
    RaceSim,
    SimParams,
    circuit_profile,
    implied_log_strength,
    simulate,
)
from synthetic import synthetic_laps

PROFILE = CircuitProfile(
    laps=50, base_lap_s=90.0, pit_loss_s=22.0, sc_per_race=0.5, stops=(0.0, 1.0, 0.0, 0.0)
)


def _run(pace: np.ndarray, params: SimParams, s: int = 3000, seed: int = 0, grid=None):  # type: ignore[no-untyped-def]
    n = pace.size
    rng = np.random.default_rng(seed)
    g = np.broadcast_to(np.arange(n) if grid is None else grid, (s, n))
    return simulate(np.tile(pace, (s, 1)), g, np.ones(n), PROFILE, params, rng)


def test_faster_cars_win_more_and_orders_are_valid() -> None:
    pace = np.linspace(0.0, 0.02, 10)[::-1]  # pole-sitter is the slowest
    orders, finished = _run(pace, SimParams())
    assert finished.all()
    assert np.all(np.sort(orders, axis=1) == np.arange(10))
    wins = np.bincount(orders[:, 0], minlength=10)
    assert wins[9] > wins[0]  # the fastest car, from the back, still wins more often


def test_track_position_matters_when_passing_is_hard() -> None:
    pace = np.zeros(10)
    pace[5] = -0.003  # a quicker car starting sixth
    easy, _ = _run(pace, SimParams(pass_a=1.5))
    hard, _ = _run(pace, SimParams(pass_a=-4.0))
    p_easy = (easy[:, 0] == 5).mean()
    p_hard = (hard[:, 0] == 5).mean()
    assert p_easy > p_hard
    # with equal cars, pole converts more often when passing is hard
    eq = np.zeros(10)
    assert (_run(eq, SimParams(pass_a=-4.0))[0][:, 0] == 0).mean() > (
        _run(eq, SimParams(pass_a=1.5))[0][:, 0] == 0
    ).mean()


def test_retirements_follow_p_finish() -> None:
    n, s = 10, 4000
    rng = np.random.default_rng(1)
    p_finish = np.linspace(0.6, 0.95, n)
    orders, finished = simulate(
        np.zeros((s, n)), np.broadcast_to(np.arange(n), (s, n)), p_finish, PROFILE, SimParams(), rng
    )
    np.testing.assert_allclose(finished.mean(axis=0), p_finish, atol=0.03)
    # retirements are classified behind every finisher
    pos_of = np.argsort(orders, axis=1)
    last_finisher = np.where(finished, pos_of, -1).max(axis=1)
    first_dnf = np.where(~finished, pos_of, n).min(axis=1)
    assert np.all(last_finisher < first_dnf)


def test_implied_strengths_recover_plackett_luce() -> None:
    rng = np.random.default_rng(3)
    true = np.array([1.5, 1.0, 0.5, 0.0, -0.5, -1.0])
    orders = pl_sample_orders(true, 6000, rng)
    est = implied_log_strength(orders, 6)
    np.testing.assert_allclose(est - est.mean(), true - true.mean(), atol=0.08)


def test_circuit_profile_from_laps(curated: Curated) -> None:
    laps = synthetic_laps(curated)
    events = curated.tables["events"]
    circuit = events.filter(pl.col("event_id") == "2026-R05")["circuit_id"][0]
    prof = circuit_profile(laps, events, circuit)
    assert prof.laps == 50
    assert 80 < prof.base_lap_s < 110
    assert sum(prof.stops) == pytest.approx(1.0)
    assert prof.stops[1] > 0.5  # every synthetic car stops once
    assert 12.0 <= prof.pit_loss_s <= 40.0


@pytest.fixture(scope="module")
def result(curated: Curated) -> harness.EvalResult:
    tables = {**curated.tables, "laps": synthetic_laps(curated)}
    sim = RaceSim(n_sims=1500, calibration_sims=150, calibration_events=6)
    return harness.run(
        tables,
        [sim, Uniform()],
        seasons=(2026, 2026),
        history_from=2026,
        keep_forecasts=True,
    )


def test_simulator_in_the_harness(result: harness.EvalResult) -> None:
    pe = result.per_event.filter(pl.col("model") == "race_sim_v1")
    assert set(pe["checkpoint"].unique().to_list()) == {"C1", "C4"}
    assert pe.height >= 2 * 14  # every race once laps are public (not the opener)
    fc = next(v for (m, cp, _), v in result.forecasts.items() if m == "race_sim_v1" and cp == "C4")
    assert fc.p_win.sum() == pytest.approx(1.0)
    assert fc.log_strength is not None
    assert fc.p_finish is not None
    for cp in ("C1", "C4"):
        cmp = result.compare("race_sim_v1", "uniform", cp, "winner_logloss")
        assert cmp.mean_diff < 0, (cp, cmp)
