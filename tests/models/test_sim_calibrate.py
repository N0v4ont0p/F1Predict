from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from scipy import stats
from test_sim_v2 import N, _models, _race

from f1p.models import sim_calibrate as sc
from f1p.models import sim_v2 as sv


def test_benjamini_hochberg_matches_scipy() -> None:
    p = np.array([0.001, 0.04, 0.03, 0.2, 0.5, 0.012])
    np.testing.assert_allclose(sc._bh(p), stats.false_discovery_control(p))


def test_sim_summaries() -> None:
    spec, draws = _race()
    res = sv.simulate(spec, draws, _models(), sv.Knobs(), np.random.default_rng(0))
    out = sc.sim_summaries(res, spec)
    assert set(out) == set(sc.SUMMARIES)
    for v in out.values():
        assert v.shape == (spec.grid.shape[0],)
        assert np.isfinite(v).all()
    assert np.all((out["lap1_hold"] >= 0) & (out["lap1_hold"] <= 1))
    assert np.all(out["stops_mean"] >= 1)  # dry race: every finisher stops
    assert np.all(out["close_per_lap"] >= out["overtakes_per_lap"])


def _cal_races(knobs: sv.Knobs, k: int = 6) -> list[sc.CalRace]:
    """Races whose "observed" summaries come from the simulator itself at ``knobs``."""
    races = []
    for i in range(k):
        spec, draws = _race(seed=10 + i)
        models = _models()
        truth = sc.sim_summaries(
            sv.simulate(spec, draws, models, knobs, np.random.default_rng(100 + i)), spec
        )
        observed = {key: float(v.mean()) for key, v in truth.items()}
        races.append(sc.CalRace(f"E{i}", "c", spec, draws, models, observed, seed=200 + i))
    return races


def test_calibration_recovers_known_knobs() -> None:
    truth = sv.Knobs(start_noise=0.4)
    races = _cal_races(truth)
    start = sv.Knobs(start_noise=1.0)
    cal = sc.calibrate(races, names=("start_noise",), start=start, max_evals=30)
    assert cal.distance < cal.start_distance
    assert cal.knobs.start_noise == pytest.approx(0.4, rel=0.15)
    assert cal.simulated["lap1_hold"] == pytest.approx(cal.observed["lap1_hold"], abs=0.02)


def test_ppc_flags_a_misfit() -> None:
    truth = sv.Knobs()
    races = _cal_races(truth)
    good = sc.ppc(races, truth)
    assert good.height == len(sc.SUMMARIES)  # one circuit
    assert good.filter(pl.col("statistic") == "lap1_abs_gain")["p_bh"].item() > 0.05
    bad = sc.ppc(races, replace(truth, start_noise=3.0))
    assert bad.filter(pl.col("statistic") == "lap1_abs_gain")["p_bh"].item() < 0.05
    assert N == 20
