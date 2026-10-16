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
