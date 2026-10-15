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
