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


@given(
    st.lists(
        st.floats(-5, 5), min_size=len(RatingParams.names()), max_size=len(RatingParams.names())
    )
)
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


# -- regulation resets (docs/plan/RESET_PLAN.md) -------------------------------------------------
class _OldFilter(RatingFilter):
    """The filter's evolution as it was before the reset parameters (the reference)."""

    def _evolve(self, mu, var, last, season, *, driver, round_=None):  # type: ignore[no-untyped-def,override]
        from f1p.core.regulations import MAJOR_RESETS

        p = self.p
        if season != last:
            gap = season - last
            if driver:
                var += gap * p.tau_driver_season**2
            else:
                reset = any(last < s <= season for s in MAJOR_RESETS)
                tau = p.tau_team_season * (p.reset_multiplier if reset else 1.0)
                mu *= p.team_carryover**gap
                var += gap * tau**2
        var += (p.tau_driver if driver else p.tau_team) ** 2
        return mu, var


def _seasons(seasons: range, seed: int = 0) -> list[RaceObs]:
    """Six teams, two drivers each, eight races a season; the order follows team strength."""
    rng = np.random.default_rng(seed)
    out = []
    for s in seasons:
        strength = rng.normal(0, 1, 6)
        for r in range(1, 9):
            eta = np.repeat(strength, 2) + rng.gumbel(size=12)
            out.append(
                RaceObs(
                    f"{s}-R{r:02d}",
                    s,
                    r,
                    tuple(f"d{i}" for i in range(12)),
                    tuple(f"t{i // 2}" for i in range(12)),
                    np.argsort(-eta).astype(np.intp),
                )
            )
    return out


def test_reset_parameters_at_their_defaults_change_nothing() -> None:
    """At the defaults the filter is bit for bit the one without the reset parameters."""
    from f1p.models.ratings import FITTED_V1

    races = _seasons(range(2008, 2024))  # crosses 2009, 2014, 2017 and 2022
    new, old = RatingFilter(FITTED_V1), _OldFilter(FITTED_V1)
    a = np.array([new.update(r) for r in races])
    b = np.array([old.update(r) for r in races])
    assert np.array_equal(a, b)
    for t in ("t0", "t3"):
        assert new.teams[t].mu == old.teams[t].mu
        assert new.teams[t].var == old.teams[t].var


def test_reset_parameters_act_only_on_reset_seasons() -> None:
    """Before the first reset, any reset parameters give the same filter."""
    races = _seasons(range(2010, 2014))  # no reset in 2010-2013
    p = RatingParams(
        reset_pu_ratio=3.0, reset_carryover_ratio=0.2, reset_drift=8.0, reset_drift_races=2.0
    )
    a = np.array([RatingFilter(p).update(r) for r in races])
    b = np.array([RatingFilter(RatingParams()).update(r) for r in races])
    assert np.array_equal(a, b)


def test_power_unit_resets_widen_more() -> None:
    """H1: across a power-unit reset (2014) the team variance grows by the extra ratio; across
    a chassis reset (2017) it does not."""
    p = RatingParams(reset_pu_ratio=2.0)
    f = RatingFilter(p)
    for season in (2013, 2016):
        f.update(_race(f"{season}-R01", season, ["a", "b"], ["A", "B"], [0, 1]))
    base = RatingParams()
    g = RatingFilter(base)
    for season in (2013, 2016):
        g.update(_race(f"{season}-R01", season, ["a", "b"], ["A", "B"], [0, 1]))
    # the same evolution formula with the multiplier doubled for 2014, unchanged for 2017
    for season, ratio in ((2014, 2.0), (2017, 1.0)):
        last = season - 1
        b = f.teams["A"] if season == 2017 else None
        mu0, var0 = 0.3, 0.2
        m1, v1 = f._evolve(mu0, var0, last, season, driver=False)
        m0, _ = g._evolve(mu0, var0, last, season, driver=False)
        assert m1 == m0  # the mean is not touched by H1
        tau = base.tau_team_season * base.reset_multiplier
        expected = var0 + (ratio * tau) ** 2 + base.tau_team**2
        assert v1 == pytest.approx(expected)
        assert b is None or b.var > 0


def test_reset_carryover_scales_the_mean_at_resets_only() -> None:
    """H3: across a reset boundary the team mean keeps carryover x ratio; elsewhere carryover."""
    p = RatingParams(reset_carryover_ratio=0.5)
    f = RatingFilter(p)
    m_reset, _ = f._evolve(1.0, 0.1, 2021, 2022, driver=False)
    m_plain, _ = f._evolve(1.0, 0.1, 2020, 2021, driver=False)
    assert m_reset == pytest.approx(p.team_carryover * 0.5)
    assert m_plain == pytest.approx(p.team_carryover)
    # drivers carry no team mean
    md, _ = f._evolve(1.0, 0.1, 2021, 2022, driver=True)
    assert md == 1.0


def test_early_reset_drift_decays() -> None:
    """H2: the per-race team drift is reset_drift times larger at round 1 of a reset season,
    decays with the round, and is unchanged in other seasons and for drivers."""
    p = RatingParams(reset_drift=5.0, reset_drift_races=2.0)
    f = RatingFilter(p)

    def drift(season: int, rnd: int, *, driver: bool = False) -> float:
        _, v = f._evolve(0.0, 0.0, season, season, driver=driver, round_=rnd)
        return float(np.sqrt(v))

    assert drift(2022, 1) == pytest.approx(5.0 * p.tau_team)
    assert drift(2022, 3) == pytest.approx(p.tau_team * (1 + 4.0 * np.exp(-1.0)))
    assert drift(2022, 1) > drift(2022, 2) > drift(2022, 5) > drift(2022, 15) > p.tau_team
    assert drift(2023, 1) == pytest.approx(p.tau_team)
    assert drift(2022, 1, driver=True) == pytest.approx(p.tau_driver)
    # no round given (old callers): no early drift
    _, v = f._evolve(0.0, 0.0, 2022, 2022, driver=False)
    assert np.sqrt(v) == pytest.approx(p.tau_team)


def test_peek_matches_the_update_with_the_round() -> None:
    """The forecast's prior for race r (peek with round_) is each entity's belief moved to race
    r — H2 included. (``update`` enters a team once per car, so a two-car team drifts twice
    per race there; that pre-dates the reset parameters and the fitted values absorb it. The
    comparison here is per entity, entered once.)"""
    p = RatingParams(reset_drift=6.0, reset_drift_races=1.5, reset_carryover_ratio=0.6)
    races = _seasons(range(2020, 2023))
    f = RatingFilter(p, record=True)
    for k, race in enumerate(races[:40]):
        mu, var = f.peek(race.drivers, race.teams, race.season, before_step=k, round_=race.round)
        g = RatingFilter(p)
        for r in races[:k]:
            g.update(r)
        drv = {
            d: g._enter(g.drivers, d, race.season, driver=True, round_=race.round)
            for d in race.drivers
        }
        tms = {
            t: g._enter(g.teams, t, race.season, driver=False, round_=race.round)
            for t in dict.fromkeys(race.teams)
        }
        want_mu = [drv[d].mu + tms[t].mu for d, t in zip(race.drivers, race.teams, strict=True)]
        want_var = [drv[d].var + tms[t].var for d, t in zip(race.drivers, race.teams, strict=True)]
        np.testing.assert_allclose(mu, want_mu, rtol=1e-12)
        np.testing.assert_allclose(var, want_var, rtol=1e-12)
        f.update(race)


# -- adaptive response (docs/plan/CALIBRATION_ADAPTIVE_PLAN.md, ADR) ---------------------------
def test_adaptive_strength_is_gated() -> None:
    """R1 acts in the first ADAPT_ROUNDS races of a reset season only; R2 in every race;
    qualifying never."""
    from f1p.models.ratings import ADAPT_ROUNDS

    f = RatingFilter(RatingParams(adapt_reset=2.0, adapt_all=0.5))

    def obs(season: int, rnd: int, kind: str = "race") -> RaceObs:
        return RaceObs("x", season, rnd, ("a",), ("A",), np.array([0], dtype=np.intp), kind=kind)

    assert f._adapt_strength(obs(2022, 1)) == 2.5
    assert f._adapt_strength(obs(2022, ADAPT_ROUNDS)) == 2.5
    assert f._adapt_strength(obs(2022, ADAPT_ROUNDS + 1)) == 0.5
    assert f._adapt_strength(obs(2023, 1)) == 0.5
    assert f._adapt_strength(obs(2022, 1, "quali")) == 0.0


def test_adaptive_response_off_changes_nothing() -> None:
    """κ = 0 (the default) is the filter without the response; R1 does nothing before the
    first reset season."""
    races = _seasons(range(2010, 2014))

    def run(p: RatingParams) -> np.ndarray:
        f = RatingFilter(p)
        return np.array([f.update(r) for r in races])

    a = run(RatingParams())
    assert np.array_equal(a, run(RatingParams(adapt_reset=3.0)))
    assert not np.array_equal(a, run(RatingParams(adapt_all=1.0)))


def test_team_surprise_has_unit_expectation_under_the_model() -> None:
    """Sampling innovations from the filter's own predictive gives E[z²] = 1."""
    from f1p.models.ratings import Belief, team_surprise

    rng = np.random.default_rng(3)
    h = rng.uniform(0.1, 0.6, 3)
    v_d = rng.uniform(0.05, 0.5, 3)
    v_t = 0.4
    cov = np.diag(v_d + 1.0 / h) + v_t
    drv = [Belief(0.0, float(v), 2020) for v in v_d]
    tms = [Belief(0.0, v_t, 2020)] * 3
    order = np.arange(3, dtype=np.intp)
    e = rng.multivariate_normal(np.zeros(3), cov, size=40_000)
    z2 = [team_surprise(order, ["T"] * 3, drv, tms, h * ei, h)["T"] for ei in e]
    assert np.mean(z2) == pytest.approx(1.0, abs=0.03)


def test_adaptive_response_inflates_surprised_teams_with_a_cap() -> None:
    """After one race, each team's variance is the plain update's times
    min(1 + κ·max(0, z² - 1), ADAPT_CAP); the means are untouched."""
    import copy

    from f1p.models.ratings import ADAPT_CAP, team_surprise

    races = _seasons(range(2019, 2021))
    before = RatingFilter(RatingParams())
    for r in races[:-1]:
        before.update(r)
    race = races[-1]
    for kappa in (0.7, 50.0):
        plain = copy.deepcopy(before)
        adapt = copy.deepcopy(before)
        adapt.p = RatingParams(adapt_all=kappa)
        # the surprise on the prior beliefs, computed as the update does
        probe = copy.deepcopy(before)
        drv = [probe._enter(probe.drivers, d, race.season, driver=True) for d in race.drivers]
        tms = [probe._enter(probe.teams, t, race.season, driver=False) for t in race.teams]
        mu = np.array([drv[i].mu + tms[i].mu for i in race.finish_order])
        g, hh, _ = pl_grad_hess(mu)
        b2 = probe.p.beta**2
        z2 = team_surprise(
            race.finish_order, race.teams, drv, tms, g / (1 + b2 * hh), hh / (1 + b2 * hh)
        )
        plain.update(race)
        adapt.update(race)
        factors = []
        for t, z in z2.items():
            want = min(1.0 + kappa * max(0.0, z - 1.0), ADAPT_CAP)
            factors.append(want)
            assert adapt.teams[t].var == pytest.approx(plain.teams[t].var * want, rel=1e-12)
            assert adapt.teams[t].mu == plain.teams[t].mu
        assert any(f > 1.0 for f in factors)
        if kappa == 50.0:
            assert any(f == ADAPT_CAP for f in factors)
