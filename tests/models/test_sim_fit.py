from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from f1p.models import sim_fit as sf


def test_fit_logit_recovers_slopes_and_pooled_offsets() -> None:
    rng = np.random.default_rng(0)
    n = 40_000
    x = rng.normal(size=(n, 2))
    groups = rng.choice(["a", "b", "c", "d"], size=n).tolist()
    true_u = {"a": -1.0, "b": 0.0, "c": 0.5, "d": 0.5}
    eta = -1.0 + x @ np.array([1.5, -0.7]) + np.array([true_u[g] for g in groups])
    y = rng.random(n) < 1.0 / (1.0 + np.exp(-eta))
    fit = sf.fit_logit(x, y, ["x1", "x2"], groups=groups, l2=1e-3, group_sd=2.0)
    assert fit.coef[1:] == pytest.approx([1.5, -0.7], abs=0.05)
    # offsets are identified up to the intercept: compare differences
    assert fit.offset("a") - fit.offset("b") == pytest.approx(-1.0, abs=0.1)
    assert fit.offset("c") - fit.offset("d") == pytest.approx(0.0, abs=0.1)
    assert fit.offset("unseen") == 0.0
    assert fit.prob(x[:5], groups[:5]).shape == (5,)


def test_fit_logit_refuses_non_finite_inputs() -> None:
    x = np.array([[0.0], [np.nan]])
    with pytest.raises(ValueError, match="non-finite"):
        sf.fit_logit(x, np.array([0, 1]), ["x"])


def _laps(rows: list[dict[str, object]]) -> pl.DataFrame:
    base = {
        "event_id": "E",
        "car_number": "0",
        "lap_ms": 90_000,
        "stint": 1,
        "compound": "MEDIUM",
        "tyre_age": 5,
        "in_lap": False,
        "out_lap": False,
        "neutral": False,
        "clean": True,
        "source": "test",
        "observed_at": datetime(2024, 1, 1, tzinfo=UTC),
    }
    return pl.DataFrame([{**base, **r} for r in rows])


TABLES = {"events": pl.DataFrame({"event_id": ["E"], "circuit_id": ["c1"], "season": [2024]})}


def test_overtaking_rows_and_outcome() -> None:
    rows = []
    # lap 2: A leads, B 0.5 s behind; lap 3: B ahead; C far behind throughout
    for lap, (pa, pb) in {2: (1, 2), 3: (2, 1), 4: (2, 1)}.items():
        rows += [
            {"driver_id": "A", "lap": lap, "position": pa, "gap_ahead_s": None if pa == 1 else 0.6},
            {"driver_id": "B", "lap": lap, "position": pb, "gap_ahead_s": None if pb == 1 else 0.5},
            {"driver_id": "C", "lap": lap, "position": 3, "gap_ahead_s": 5.0},
        ]
    d = sf.overtaking_data(TABLES, _laps(rows))
    first = d.filter(pl.col("lap") == 2)
    assert first.height == 1  # only the close pair (C is 5 s behind)
    assert first["passed"].item() is True
    assert first["gap_s"].item() == pytest.approx(0.5)
    later = d.filter(pl.col("lap") == 3)
    assert later["passed"].item() is False  # A, now behind, does not re-pass
    # a pit lap or a neutralised lap removes the pair
    pit = [{**r, "in_lap": True} if r["lap"] == 3 and r["driver_id"] == "A" else r for r in rows]
    assert sf.overtaking_data(TABLES, _laps(pit)).filter(pl.col("lap") == 2).height == 0
    sc = [{**r, "neutral": True} if r["lap"] == 3 else r for r in rows]
    assert sf.overtaking_data(TABLES, _laps(sc)).filter(pl.col("lap") == 2).height == 0


def test_strategy_rows() -> None:
    rows = []
    for lap in range(1, 11):
        stint = 1 if lap <= 5 else 2
        rows.append(
            {
                "driver_id": "A",
                "lap": lap,
                "position": 1,
                "gap_ahead_s": None,
                "stint": stint,
                "tyre_age": lap if stint == 1 else lap - 5,
                "in_lap": lap == 5,
                "out_lap": lap == 6,
                "compound": None if stint == 1 else "HARD",
            }
        )
    d = sf.strategy_data(TABLES, _laps(rows))
    assert d.height == 9  # every lap but the out-lap
    assert d["pit"].sum() == 1
    assert d.filter(pl.col("pit"))["age"].item() == pytest.approx(5 / 20)
    assert d.filter(pl.col("unknown") == 1).height == 5  # stint 1 (laps 1-5): no compound
    assert d.filter(pl.col("hard") == 1).height == 4  # laps 7-10 (6 is the out-lap)
    assert d.select(sf.STRATEGY_FEATURES).null_count().sum_horizontal().item() == 0


def _race(
    event: str, neutral_laps: set[int], crash: dict[str, int]
) -> tuple[pl.DataFrame, list[dict[str, object]]]:
    """Three cars, 10 laps; ``crash``: driver → laps completed before an incident DNF."""
    rows, results = [], []
    for d, start in (("A", 1), ("B", 2), ("C", 3)):
        last = crash.get(d, 10)
        for lap in range(1, last + 1):
            rows.append(
                {
                    "event_id": event,
                    "driver_id": d,
                    "lap": lap,
                    "position": start,
                    "gap_ahead_s": None if start == 1 else 2.0,
                    "neutral": lap in neutral_laps,
                }
            )
        results.append(
            {
                "event_id": event,
                "session": "race",
                "driver_id": d,
                "status": "dnf_incident" if d in crash else "finished",
                "laps": last,
            }
        )
    return _laps(rows), results


def test_neutral_periods() -> None:
    laps, _ = _race("E", {5, 6, 9}, {})
    periods = sf.neutral_periods(laps)
    assert periods.select("start", "length").rows() == [(5, 2), (9, 1)]


def test_incidents_link_crashes_to_the_neutralisation_they_cause() -> None:
    # E1: C crashes during lap 4 (3 laps done); the field is neutralised from lap 5;
    # a second, unexplained neutralisation on lap 8.  E2: a quiet race.
    l1, r1 = _race("E1", {5, 6, 8}, {"C": 3})
    l2, r2 = _race("E2", set(), {})
    tables = {
        "events": pl.DataFrame({"event_id": ["E1", "E2"], "circuit_id": ["c", "c"]}),
        "results": pl.DataFrame(r1 + r2),
    }
    inc = sf.fit_incidents(tables, pl.concat([l1, l2]))
    assert inc.hazard.events == 2  # the crash (with its neutralisation) and the lap-8 one
    assert inc.dnf_counts.tolist() == pytest.approx([0.5, 0.5, 0.0, 0.0])
    assert inc.p_neutral_given_dnf == pytest.approx(1.0)
    assert sorted(inc.lengths.tolist()) == [1, 2]
    assert inc.mechanical_share == pytest.approx(0.0)
    assert inc.position_weights.shape == (10,)


def test_start_model_utilities() -> None:
    m = sf.StartModel(
        w_slot=-0.6,
        w_clean=0.3,
        w_run=0.2,
        circuit={"monaco": -1.0},
        team={"ferrari:2024": 0.5},
        run_mean=400.0,
        run_sd=200.0,
        run={"monaco": 100.0},
        lap1_dnf={"monaco": 0.01},
        lap1_dnf_global=0.03,
    )
    slot = np.arange(4.0)
    u = m.utilities(slot, "monza", ["x", "x", "x", "x"])
    assert np.all(np.diff(u[::2]) < 0)  # further back on the same side: lower utility
    assert u[0] - u[1] > -m.w_slot  # P1 (clean side) ahead of P2 by more than the slot gap
    tight = m.utilities(slot, "monaco", ["x"] * 4)
    assert np.all(np.diff(tight[::2]) < np.diff(u[::2]))  # Monaco keeps grid order more
    assert m.utilities(slot, "monza", ["ferrari:2024", "x", "x", "x"])[0] == pytest.approx(
        u[0] + 0.5
    )
    assert m.p_lap1_dnf("monaco") == 0.01
    assert m.p_lap1_dnf("nowhere") == 0.03
