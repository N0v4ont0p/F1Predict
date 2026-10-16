from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from f1p.lake.curate_f1db import Curated
from f1p.models import lap_data as ld
from synthetic import synthetic_laps


def test_ispline_is_monotone_from_zero() -> None:
    age = np.linspace(0, 80, 161)
    b = ld.ispline_basis(age)
    assert np.all(b[0] == 0)  # every curve starts at zero tyre age
    assert np.all(b[age <= 1.0] < 0.1)  # flat start: no early column aliasing an offset
    assert np.all(np.diff(b, axis=0) >= -1e-12)  # each column non-decreasing
    assert np.allclose(b[age >= ld.AGE_KNOTS[-1]], 1.0)  # flat (saturated) beyond the last knot
    w = np.abs(np.random.default_rng(0).normal(size=b.shape[1]))
    assert np.all(np.diff(b @ w) >= -1e-12)  # positive weights → monotone wear


def test_evolution_basis_is_pure_curvature() -> None:
    u = np.linspace(0, 1, 201)
    e = ld.evolution_basis(u)
    assert e.shape == (201, ld.EVO_BASIS - 2)
    lin = np.column_stack([np.ones_like(u), u])
    coef, *_ = np.linalg.lstsq(lin, e, rcond=None)
    assert np.abs(coef).max() < 1e-8  # no constant or slope: cannot trade off with fuel
    assert np.allclose(np.sqrt((e**2).mean(axis=0)), 1.0, atol=1e-6)


def test_gap_basis() -> None:
    hinges, drs = ld.gap_basis(np.array([0.2, 0.9, 1.5, 2.9, 5.0, np.nan]))
    assert drs.tolist() == [1, 1, 0, 0, 0, 0]
    assert np.all(hinges[4:] == 0)  # far behind or nobody ahead: no traffic
    assert np.all(np.diff(hinges.sum(axis=1)[:5]) < 0)  # closer is more traffic


def test_hsgp_matches_exact_matern() -> None:
    x = np.arange(25.0)  # a 24-race window plus the forecast race, at the edge
    basis = ld.hsgp_basis(x)
    amp = 0.004
    # long lengthscales lose the near-constant component, which the team-season level carries
    for ls, tol in ((3.5, 0.015), (6.0, 0.005), (12.0, 0.04)):
        s = ld.matern32_sqrt_psd(basis.sqrt_eig, amp, ls)
        approx = (basis.phi * s**2) @ basis.phi.T
        r = np.abs(x[:, None] - x[None, :]) * np.sqrt(3.0) / ls
        exact = amp**2 * (1 + r) * np.exp(-r)
        assert np.abs(approx - exact).max() < tol * amp**2, ls


def test_walk_matrix() -> None:
    c = ld.walk_matrix(np.array([0, 0, 0, 1, 1]), np.array([2023, 2024, 2025, 2024, 2025]))
    assert c.tolist() == [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1],
    ]


@pytest.fixture(scope="module")
def tables(curated: Curated) -> dict[str, pl.DataFrame]:
    return {**curated.tables, "laps": synthetic_laps(curated)}


def test_build_window(tables: dict[str, pl.DataFrame]) -> None:
    laps = tables["laps"]
    w = ld.build_window(laps, tables)
    assert w is not None
    s = w.summary()
    assert s["events"] == len(set(laps["event_id"].to_list()))
    assert s["holdout_laps"] == 0
    assert w.next_index is None
    assert w.entries == []
    # centred: every event's median racing lap is 0
    for e in range(s["events"]):
        assert np.median(w.y[(w.ev == e) & (w.pit == 0)]) == pytest.approx(0.0, abs=1e-12)
    assert set(np.unique(w.pit).tolist()) == {0, 1, 2}  # in- and out-laps are modelled
    assert w.age.shape == (w.n, len(ld.AGE_KNOTS))
    assert w.tyre_age.min() >= 0
    assert w.hsgp.phi.shape[0] == s["events"]
    assert w.circuit_x.shape == (s["events"], len(ld.CIRCUIT_FEATURES))
    assert w.ts_walk.shape == (s["team_seasons"],) * 2
    # offsets are contrasts against each race's most-used compound, which has none
    lap_comp = w.wear_compound[w.wear]
    for e in range(s["events"]):
        racing = (w.ev == e) & (w.pit == 0)
        ref = np.bincount(lap_comp[racing], minlength=len(ld.COMPOUNDS)).argmax()
        assert np.all(w.off[(w.ev == e) & (lap_comp == ref)] == -1)
        assert np.all(w.off[(w.ev == e) & (lap_comp != ref)] >= 0)
    assert np.all(w.off_compound != w.off_ref)
    assert np.all(w.off_compound[w.off[w.off >= 0]] == lap_comp[w.off >= 0])


def test_forecast_race_and_holdout(tables: dict[str, pl.DataFrame]) -> None:
    events = (
        tables["events"]
        .filter(pl.col("event_id").is_in(tables["laps"]["event_id"].unique().to_list()))
        .sort("season", "round")["event_id"]
        .to_list()
    )
    window, target = events[:4], events[4]
    laps = tables["laps"].filter(pl.col("event_id").is_in(window))
    hold = tables["laps"].filter(pl.col("event_id") == target)
    w = ld.build_window(laps, tables, next_event=target, holdout=hold)
    assert w is not None
    assert w.events[-1] == target
    assert w.next_index == len(w.events) - 1
    assert not w.train[w.ev == w.next_index].any()  # the target race is never fitted
    assert w.train[w.ev != w.next_index].all()
    assert w.summary()["holdout_laps"] > 0
    # every held-out car is an entry, with a driver-season row at the target's season
    assert {w.drivers[d] for d in w.driver[~w.train]} <= {d for d, _ in w.entries}
    assert np.all(w.ds_present[w.entry_ds, w.next_index] == 1)
    assert np.all(w.team_present[w.entry_team, w.next_index] == 1)
    # the race is centred on its own median (scored on relative times)
    racing = (~w.train) & (w.pit == 0)
    assert np.median(w.y[racing]) == pytest.approx(0.0, abs=1e-12)
    # forecasting (no laps yet): entries alone define the target column
    entries = [(d, "mclaren") for d in ("norris", "piastri")]
    f = ld.build_window(laps, tables, next_event=target, entries=entries)
    assert f is not None
    assert f.summary()["holdout_laps"] == 0
    assert [d for d, _ in f.entries] == ["norris", "piastri"]
    assert f.team_present[:, f.next_index].sum() == 1


def test_circuit_features_are_standardised(tables: dict[str, pl.DataFrame]) -> None:
    f = ld.circuit_features(tables)
    assert f.columns == ["circuit_id", *ld.CIRCUIT_FEATURES]
    assert f.null_count().sum_horizontal().item() == 0
