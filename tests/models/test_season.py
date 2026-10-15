from __future__ import annotations

import numpy as np
import pytest

from f1p.eval.forecast import position_matrix, sample_orders
from f1p.lake.curate_f1db import Curated
from f1p.models.season import sample_orders_batch, simulate_season


def test_batch_sampler_matches_single_sampler() -> None:
    rng = np.random.default_rng(0)
    ls = np.array([1.5, 1.0, 0.3, -0.4, -1.0])
    pf = np.full(5, 0.9)
    sc = np.exp(-0.2 * np.arange(5))
    batch, fin_b = sample_orders_batch(np.tile(ls, (40_000, 1)), pf, sc, rng)
    single, fin_s = sample_orders(ls, 40_000, rng, p_finish=pf, stage_scales=sc)
    np.testing.assert_allclose(position_matrix(batch, 5), position_matrix(single, 5), atol=0.012)
    assert fin_b.mean() == pytest.approx(fin_s.mean(), abs=0.01)


@pytest.fixture(scope="module")
def outlook(curated: Curated):  # type: ignore[no-untyped-def]
    return simulate_season(curated.tables, 2026, n_sims=4000)


def test_season_outlook_is_coherent(outlook, curated: Curated) -> None:  # type: ignore[no-untyped-def]
    assert outlook.remaining[0] == "2026-R18"
    assert len(outlook.remaining) == 6
    assert sum(s.p_title for s in outlook.drivers) == pytest.approx(1.0, abs=1e-6)
    assert sum(s.p_title for s in outlook.constructors) == pytest.approx(1.0, abs=1e-6)
    for s in outlook.drivers:
        assert s.expected >= s.current - 1e-9  # points never go down
        assert list(s.quantiles) == sorted(s.quantiles)
        assert sum(s.p_position) == pytest.approx(1.0, abs=1e-6)
    # at most 26 points per remaining race (25 + nothing else in 2026)
    leader = max(outlook.drivers, key=lambda s: s.current)
    assert max(s.quantiles[-1] for s in outlook.drivers) <= leader.current + 25 * 6 + 1e-6


def test_points_distributed_per_race(outlook) -> None:  # type: ignore[no-untyped-def]
    gained = sum(s.expected - s.current for s in outlook.drivers)
    # every remaining race hands out at most 101 points (25+18+…+1) - less if cars retire
    assert 0.8 * 101 * 6 < gained <= 101 * 6 + 1e-6


def test_countback_splits_equal_points() -> None:
    from f1p.models.season import countback_key

    totals = np.array([[100.0, 100.0, 100.0, 99.5]])
    places = np.array([[[2, 0, 0, 0, 0], [2, 1, 0, 0, 0], [1, 9, 9, 9, 9], [30, 0, 0, 0, 0]]])
    key = countback_key(totals, places)[0]
    # more seconds beats fewer at equal wins; one more win beats any number of seconds;
    # half a point beats any countback
    assert list(np.argsort(-key)) == [1, 0, 2, 3]


def test_most_likely_order_is_a_consistent_permutation() -> None:
    from f1p.models.season import most_likely_order

    # Sorted by expected place (A 0.45, B 1.0, C 1.45) the order is A, B, C, but B is never
    # third, so the jointly most likely order puts B first: P = .45 x .45 x .55 beats
    # .55 x .10 x .55.
    p = np.array(
        [
            [0.55, 0.45, 0.00],
            [0.45, 0.10, 0.45],
            [0.00, 0.45, 0.55],
        ]
    )
    place = most_likely_order(p)
    assert place.tolist() == [1, 0, 2]


def _truncate(tables: dict, season: int, last_round: int) -> dict:  # type: ignore[type-arg]
    import polars as pl

    out = dict(tables)
    for name in ("results", "driver_standings", "constructor_standings", "grid", "qualifying"):
        if name in out and "round" in out[name].columns:
            df = out[name]
            out[name] = df.filter(~((pl.col("season") == season) & (pl.col("round") > last_round)))
    return out


def test_projection_after_a_round_sees_nothing_later(curated: Curated) -> None:
    import polars as pl

    a = simulate_season(curated.tables, 2026, after_round=10, n_sims=2000, seed=7)
    cut = _truncate(curated.tables, 2026, 10)
    b = simulate_season(cut, 2026, after_round=10, n_sims=2000, seed=7)
    assert a.after_round == 10
    assert a.remaining[0] == "2026-R11"
    assert len(a.remaining) == 13
    assert [s.id for s in a.drivers] == [s.id for s in b.drivers]
    for x, y in zip(a.drivers, b.drivers, strict=True):
        assert x.p_title == pytest.approx(y.p_title)
        assert x.expected == pytest.approx(y.expected)
    st = curated.tables["driver_standings"].filter(
        (pl.col("season") == 2026) & (pl.col("round") == 10)
    )
    for d, pts_ in st.select("driver_id", "points").iter_rows():
        s = a.standing("driver", d)
        assert s is not None
        assert s.current == pytest.approx(pts_)
    # the most likely order is a permutation, and the table is sorted by it
    assert [s.most_likely for s in a.drivers] == list(range(len(a.drivers)))
    assert [s.most_likely for s in a.constructors] == list(range(len(a.constructors)))


def test_preseason_and_uniform_baseline(curated: Curated) -> None:
    pre = simulate_season(curated.tables, 2026, after_round=0, n_sims=1000)
    assert pre.after_round == 0
    assert len(pre.remaining) == 23
    assert all(s.current == 0 for s in pre.drivers)
    uni = simulate_season(curated.tables, 2026, after_round=15, n_sims=2000, uniform=True)
    assert len(uni.remaining) == 8
    assert sum(s.p_title for s in uni.drivers) == pytest.approx(1.0, abs=1e-6)
    leader = max(uni.drivers, key=lambda s: s.current)
    assert leader.p_title == max(s.p_title for s in uni.drivers)
