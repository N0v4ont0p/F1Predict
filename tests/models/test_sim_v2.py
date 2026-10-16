from __future__ import annotations

import numpy as np
import pytest

from f1p.models import sim_fit as sf
from f1p.models import sim_v2 as sv
from f1p.models.lap_data import COMPOUNDS, GAP_HINGES

N, S, LAPS = 20, 400, 40


def _models(*, pass_shift: float = 0.0, incident: float = -3.4) -> sv.SubModels:
    """Sub-models with coefficients close to the fitted ones (no lake in tests)."""
    overtaking = sf.LogitFit(
        sf.OVERTAKE_FEATURES, np.array([0.8 + pass_shift, 1.2, -5.6, 0.08, 0.04])
    )
    strategy = sf.LogitFit(
        sf.STRATEGY_FEATURES, np.array([-6.8, 3.9, -1.0, 2.0, -0.2, 2.3, 0.3, -0.6, 0.0, -0.3])
    )
    start = sf.StartModel(-0.6, 0.3, 0.2, {}, {}, 400.0, 200.0, {}, {}, 0.03)
    hazard = sf.LogitFit(sf.INCIDENT_FEATURES, np.array([incident, 2.2, -0.25, -0.3]))
    incidents = sf.IncidentModel(
        hazard,
        np.array([0.4, 0.5, 0.08, 0.02]),
        0.5,
        np.array([2, 3, 4, 5], dtype=np.int64),
        np.ones(10),
        0.5,
    )
    compounds = sf.CompoundModel(np.array([0.2, 0.65, 0.15]), np.full((3, 3, 3), 1.0 / 3.0))
    return sv.SubModels(overtaking, strategy, start, incidents, compounds)


def _race(p_finish: float = 0.92, seed: int = 0) -> tuple[sv.RaceSpec, sv.Draws]:
    rng = np.random.default_rng(seed)
    pace = np.linspace(-0.006, 0.010, N)
    grid = np.tile(np.arange(N), (S, 1))
    draws = sv.draws_from_laplace(
        pace,
        np.full(N, 0.002),
        S,
        rng,
        deg={"SOFT": 0.0012, "MEDIUM": 0.0008, "HARD": 0.0005},
        offset={"SOFT": -0.004, "HARD": 0.004},
        fuel=0.03,
        pit_loss_s=22.0,
    )
    spec = sv.RaceSpec("somewhere", LAPS, 90.0, ["x"] * N, np.full(N, p_finish), grid)
    return spec, draws


def test_race_invariants() -> None:
    spec, draws = _race()
    res = sv.simulate(spec, draws, _models(), sv.Knobs(), np.random.default_rng(1))
    assert np.all(np.sort(res.orders, axis=1) == np.arange(N))
    pos = np.argsort(res.orders, axis=1)
    fin = res.finished
    assert np.all(np.where(fin, pos, -1).max(axis=1) < np.where(~fin, pos, N).min(axis=1))
    assert np.all(res.stops[fin] >= 1)  # a dry race needs two compounds
    assert res.stops.max() <= 6
    assert 10 < fin.sum(axis=1).mean() < N
    wins = np.bincount(res.orders[:, 0], minlength=N)
    assert wins[:3].sum() > 5 * wins[-3:].sum()  # the fast cars win
    assert res.overtakes.mean() > 0
    assert res.close_pairs.mean() >= res.overtakes.mean()


def test_without_passing_or_incidents_the_lap1_order_holds() -> None:
    spec, draws = _race(p_finish=1.0)
    models = _models(pass_shift=-50.0, incident=-50.0)
    # nobody pits by choice: everyone takes the forced stop on the same lap
    models.strategy.coef[0] = -50.0
    res = sv.simulate(spec, draws, models, sv.Knobs(), np.random.default_rng(2))
    assert res.finished.all()
    assert res.overtakes.sum() == 0
    assert res.neutral_periods.sum() == 0
    np.testing.assert_array_equal(res.orders, res.lap1_order)


def test_start_without_noise_keeps_the_grid() -> None:
    spec, draws = _race()
    res = sv.simulate(spec, draws, _models(), sv.Knobs(start_noise=1e-9), np.random.default_rng(3))
    np.testing.assert_array_equal(res.lap1_order, spec.grid)
    noisy = sv.simulate(spec, draws, _models(), sv.Knobs(start_noise=1.0), np.random.default_rng(3))
    assert np.abs(np.argsort(noisy.lap1_order, axis=1) - spec.grid).mean() > 1.0


def test_incident_rate_scales_neutralisations() -> None:
    spec, draws = _race()
    calm = sv.simulate(
        spec, draws, _models(), sv.Knobs(incident_scale=0.2), np.random.default_rng(4)
    )
    busy = sv.simulate(
        spec, draws, _models(), sv.Knobs(incident_scale=5.0), np.random.default_rng(4)
    )
    assert busy.neutral_periods.mean() > 2 * calm.neutral_periods.mean()
    assert busy.finished.sum() < calm.finished.sum()


def test_draws_from_posterior() -> None:
    rng = np.random.default_rng(5)
    d, k = 50, 5
    post = {
        "pace_next": rng.normal(0, 0.005, (d, 3)),
        "wear_log": rng.normal(-5.5, 0.3, (d, len(COMPOUNDS), k)),
        "wear_event_sd": np.full(d, 0.3),
        "compound": rng.normal(0, 0.01, (d, len(COMPOUNDS) - 1)),
        "pit": np.tile([0.05, 0.2], (d, 1)),
        "pit_event_sd": np.full(d, 0.02),
        "fuel": np.full(d, 0.03),
        "sigma_log": np.full(d, -5.5),
        "gap": np.zeros((d, len(GAP_HINGES))),
        "drs": np.zeros(d),
    }
    draws = sv.draws_from_posterior(post, [2, 0, 1], 100, 90.0, rng)
    assert draws.pace.shape == (100, 3)
    assert np.allclose(draws.pace.mean(axis=1), 0.0)
    assert draws.wear_w.shape == (100, len(COMPOUNDS), k)
    assert np.all(draws.level[:, 0] == 0.0)  # MEDIUM is the reference
    assert draws.pit_loss_s.mean() == pytest.approx((np.expm1(0.05) + np.expm1(0.2)) * 90, rel=0.05)


def test_trace_follows_the_race() -> None:
    spec, draws = _race()
    plain = sv.simulate(spec, draws, _models(), sv.Knobs(), np.random.default_rng(5))
    res = sv.simulate(spec, draws, _models(), sv.Knobs(), np.random.default_rng(5), trace=True)
    # tracing draws nothing: the same races
    np.testing.assert_array_equal(plain.orders, res.orders)
    tr = res.trace
    assert tr is not None
    assert tr.lap_s.shape == (S, LAPS, N)
    assert np.isnan(tr.lap_s[:, 0]).all()  # no lap-1 time
    alive = tr.alive
    # alive only ever turns off; the last lap's survivors are the finishers
    assert not (alive[:, 1:] & ~alive[:, :-1]).any()
    np.testing.assert_array_equal(alive[:, -1], res.finished)
    # positions are 1..(cars running) each lap, the leader 0 s ahead of everyone
    running = alive.sum(axis=2)
    pos_sorted = np.sort(np.where(alive, tr.position, 99), axis=2)
    expect = np.arange(1, N + 1)[None, None, :]
    assert np.all((pos_sorted == expect) | (expect > running[..., None]))
    lead = tr.position == 1
    np.testing.assert_allclose(tr.gap_s[lead], 0.0, atol=1e-6)
    assert np.all(tr.gap_s[alive] >= -1e-6)
    # the final lap's running order is the classification
    last = np.where(alive[:, -1], tr.position[:, -1], 99)
    order = np.argsort(last, axis=1, kind="stable")
    k = res.finished.sum(axis=1)
    for i in range(0, S, 37):
        np.testing.assert_array_equal(order[i, : k[i]], res.orders[i, : k[i]])
    # stops add up; a lap time is the change in time behind the leader plus his lap
    np.testing.assert_array_equal(tr.pit.sum(axis=1), res.stops)
    green = alive[:, 1:] & alive[:, :-1] & ~tr.neutral[:, 1:, None] & ~tr.pit[:, 1:]
    laps_ok = tr.lap_s[:, 1:][green]
    assert np.all((laps_ok > 80.0) & (laps_ok < 110.0))
    assert np.all(tr.lap_s[:, 1:][tr.pit[:, 1:] & alive[:, 1:]] > 100.0)  # the stop costs
    # tyres: age resets on the stop lap
    assert np.all(tr.tyre_age[:, 1:][tr.pit[:, 1:]] == 0.0)
