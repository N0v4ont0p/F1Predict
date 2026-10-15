from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from f1p.lake.curate_f1db import Curated
from f1p.models.ratings import (
    DRIVER_MU0,
    RaceObs,
    RatingFilter,
    RatingParams,
    pl_grad_hess,
    predictive_log_strength,
    race_observations,
    run_filter,
)


def _race(
    event_id: str, season: int, drivers: list[str], teams: list[str], finish: list[int]
) -> RaceObs:
    return RaceObs(
        event_id, season, 1, tuple(drivers), tuple(teams), np.array(finish, dtype=np.intp)
    )


def _pl_ll(eta: np.ndarray) -> float:
    suffix = np.logaddexp.accumulate(eta[::-1])[::-1]
    return float(np.sum(eta - suffix))


@given(st.lists(st.floats(-3, 3), min_size=2, max_size=12))
def test_grad_hess_match_finite_differences(values: list[float]) -> None:
    eta = np.array(values)
    g, h, ll = pl_grad_hess(eta)
    assert ll == pytest.approx(_pl_ll(eta), abs=1e-9)
    eps = 1e-5
    for i in range(eta.size):
        e = np.zeros_like(eta)
        e[i] = eps
        num_g = (_pl_ll(eta + e) - _pl_ll(eta - e)) / (2 * eps)
        num_h = -(_pl_ll(eta + e) - 2 * _pl_ll(eta) + _pl_ll(eta - e)) / eps**2
        assert g[i] == pytest.approx(num_g, abs=1e-5)
        assert h[i] == pytest.approx(num_h, abs=1e-3)
    assert np.all(h >= 0)


@given(st.lists(st.floats(-5, 5), min_size=11, max_size=11))
def test_param_vector_round_trip_stays_in_bounds(x: list[float]) -> None:
    p = RatingParams.from_vector(x)
    for name, (lo, hi) in RatingParams.BOUNDS.items():
        assert lo <= getattr(p, name) <= hi
    again = RatingParams.from_vector(p.as_vector())
    for name in RatingParams.names():
        assert getattr(again, name) == pytest.approx(getattr(p, name), rel=1e-4, abs=1e-6)
    assert RatingParams.from_dict(p.to_dict()) == p


def test_update_rewards_winner_and_penalises_last() -> None:
    f = RatingFilter(RatingParams())
    race = _race("x", 2020, ["a", "b", "c", "d"], ["A", "B", "C", "D"], [0, 1, 2])
    f.update(race)
    mu = {d: f.drivers[d].mu for d in "abcd"}
    assert mu["a"] > mu["b"] > mu["c"]
    assert mu["a"] > DRIVER_MU0 > mu["c"]
    # d retired: censored, so no information about its pace
    assert mu["d"] == pytest.approx(DRIVER_MU0)
    for d in "abc":
        assert f.drivers[d].var < RatingParams().driver_sigma0 ** 2 + RatingParams().tau_driver ** 2
    assert f.drivers["d"].var == pytest.approx(RatingParams().driver_sigma0 ** 2)


def test_repeated_wins_converge_and_shrink_uncertainty() -> None:
    f = RatingFilter(RatingParams())
    for _ in range(30):
        f.update(_race("x", 2020, ["a", "b"], ["A", "B"], [0, 1]))
    assert f.drivers["a"].mu + f.teams["A"].mu > f.drivers["b"].mu + f.teams["B"].mu + 2.0
    p = RatingParams()
    prior_total = p.driver_sigma0**2 + p.team_sigma0**2
    assert f.drivers["a"].var + f.teams["A"].var < 0.6 * prior_total


def test_reset_season_inflates_team_uncertainty_more() -> None:
    p = RatingParams(reset_multiplier=3.0)
    f = RatingFilter(p)
    f.update(_race("x", 2012, ["a", "b"], ["A", "B"], [0, 1]))
    _, v_normal = f.peek(["a"], ["A"], 2013)
    _, v_reset = f.peek(["a"], ["A"], 2014)  # 2014 is a major reset
    assert v_reset[0] - v_normal[0] > (p.tau_team_season**2) * 2


def test_recorded_history_reproduces_state_at_every_step(curated: Curated) -> None:
    races = race_observations(curated.tables)
    params = RatingParams()
    full, _ = run_filter(races, params, record=True)
    for k in (0, 1, 5, len(races) // 2, len(races) - 1, len(races)):
        partial, _ = run_filter(races[:k], params)
        target = races[min(k, len(races) - 1)]
        a = full.peek(target.drivers, target.teams, target.season, before_step=k)
        b = partial.peek(target.drivers, target.teams, target.season)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])


def test_peek_before_step_requires_record() -> None:
    with pytest.raises(ValueError, match="record=True"):
        RatingFilter(RatingParams()).peek(["a"], ["A"], 2020, before_step=0)


def test_race_observations_from_curated(curated: Curated) -> None:
    races = race_observations(curated.tables)
    assert races
    keys = [(r.season, r.round) for r in races]
    assert keys == sorted(keys)
    for r in races:
        assert len(r.drivers) == len(r.teams)
        assert r.finish_order.size <= len(r.drivers)
        assert r.known_at is not None
    r21 = next(r for r in races if r.event_id == "2021-R22")
    assert r21.drivers[r21.finish_order[0]] == "max_verstappen"


def test_predictive_log_strength_is_shift_invariant_and_shrinks() -> None:
    mu = np.array([1.0, 0.0, -1.0])
    var = np.array([0.1, 0.1, 0.1])
    a = predictive_log_strength(mu, var, 0.5)
    b = predictive_log_strength(mu + 7.0, var, 0.5)
    np.testing.assert_allclose(a, b)
    assert abs(a[0] - a[2]) < 2.0
    wide = predictive_log_strength(mu, np.array([3.0, 0.1, 0.1]), 0.5)
    assert wide[0] < a[0]  # an uncertain favourite is pulled toward the field


def test_filter_beats_uniform_prequentially(curated: Curated) -> None:
    races = race_observations(curated.tables)
    _, ll = run_filter(races, RatingParams(), score_from=2026)
    uniform = 0.0
    for r in races:
        if r.season >= 2026:
            uniform += _pl_ll(np.zeros(r.finish_order.size))
    assert ll > uniform


def test_driver_prior_mean_is_a_gauge(curated: Curated, monkeypatch: pytest.MonkeyPatch) -> None:
    """Shifting every driver's prior mean changes no prediction (why it is not fitted)."""
    import f1p.models.ratings as ratings

    races = race_observations(curated.tables)
    _, base = run_filter(races, RatingParams())
    monkeypatch.setattr(ratings, "DRIVER_MU0", 2.5)
    _, shifted = run_filter(races, RatingParams())
    assert shifted == pytest.approx(base, rel=1e-9)


def test_weekend_observations_in_session_order(curated: Curated) -> None:
    from f1p.models.ratings import observations

    obs = observations(curated.tables)
    weekend = [o.kind for o in obs if o.event_id == "2026-R17"]
    assert weekend == ["sq", "sprint", "quali", "race"]
    for o in obs:
        if o.event_id == "2026-R17" and o.kind != "race":
            race = next(r for r in obs if r.event_id == o.event_id and r.is_race)
            assert o.known_at is not None
            assert race.known_at is not None
            assert o.known_at <= race.known_at
    plain = race_observations(curated.tables)
    assert [o.event_id for o in observations(curated.tables, quali=False)] == [
        o.event_id for o in plain
    ]


def test_qualifying_sharpens_but_is_not_scored(curated: Curated) -> None:
    from f1p.models.ratings import observations

    params = RatingParams()
    _, races_only = run_filter(race_observations(curated.tables), params, score_from=2026)
    both, with_quali = run_filter(observations(curated.tables), params, score_from=2026)
    assert with_quali != races_only  # sharper beliefs change the race predictions
    plain, _ = run_filter(race_observations(curated.tables), params)
    d = "max_verstappen"
    assert both.drivers[d].var < plain.drivers[d].var
