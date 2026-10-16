from __future__ import annotations

import numpy as np
import pytest

from f1p.eval import harness
from f1p.eval.baselines import RankPL, grid_ranks, standings_ranks
from f1p.lake.curate_f1db import Curated
from f1p.models.stacking import fit_calibrator, stack_result
from f1p.models.world_v1 import WorldV1


@pytest.fixture(scope="module")
def result(curated: Curated) -> harness.EvalResult:
    return harness.run(
        curated.tables,
        [
            WorldV1(),
            RankPL("standings", standings_ranks, frozenset({"C1", "C4"})),
            RankPL("grid", grid_ranks, frozenset({"C4"})),  # a member of C4 only
        ],
        seasons=(2021, 2026),
        history_from=2013,
        keep_forecasts=True,
    )


def test_members_per_checkpoint_and_recalibration(
    curated: Curated, result: harness.EvalResult
) -> None:
    members = ["world_v1", "standings", "grid"]
    stacked, fits = stack_result(
        curated.tables, result, members, seasons=(2026, 2026), training_seasons=6
    )
    # C1 pools the members that forecast C1; C4 all three
    assert fits[(2026, "C1")].members == ("world_v1", "standings")
    assert fits[(2026, "C4")].members == ("world_v1", "standings", "grid")
    pe = stacked.per_event
    c1 = set(pe.filter(pe["checkpoint"] == "C1")["model"].unique().to_list())
    assert c1 == {"world_v1", "standings", "stack", "stack_iso"}
    assert pe.filter(pe["model"] == "stack_iso").height == pe.filter(pe["model"] == "stack").height
    # the recalibrated P(win) is a distribution, keeps the favourites' order, and only P(win)
    # changes (the full-order scores are the pool's)
    for cp in ("C1", "C4"):
        fit = fits[(2026, cp)]
        assert fit.calibrator is not None
        a = pe.filter((pe["model"] == "stack") & (pe["checkpoint"] == cp)).sort("event_id")
        b = pe.filter((pe["model"] == "stack_iso") & (pe["checkpoint"] == cp)).sort("event_id")
        np.testing.assert_allclose(a["finisher_pl_loglik"], b["finisher_pl_loglik"])
        np.testing.assert_allclose(a["podium_brier"], b["podium_brier"])
    summary = stacked.summary()
    assert "ece_top1" in summary.columns
    assert summary["ece_top1"].drop_nulls().len() == summary.height


def test_without_recalibration(curated: Curated, result: harness.EvalResult) -> None:
    stacked, fits = stack_result(
        curated.tables,
        result,
        ["world_v1", "standings"],
        seasons=(2026, 2026),
        training_seasons=6,
        recalibrate=False,
    )
    assert "stack_iso" not in stacked.per_event["model"].unique().to_list()
    assert all(f.calibrator is None for f in fits.values())


def test_calibrator_is_monotone_and_needs_both_outcomes() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.6, 4000)
    won = (rng.uniform(size=p.size) < np.sqrt(p)).astype(float)  # an under-confident forecast
    iso = fit_calibrator(p, won)
    q = iso.predict(np.sort(p))
    assert np.all(np.diff(q) >= 0)
    assert iso.predict(np.array([0.25]))[0] == pytest.approx(0.5, abs=0.06)
    assert fit_calibrator(p, np.zeros_like(p)) is None
