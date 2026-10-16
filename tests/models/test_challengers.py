from __future__ import annotations

import numpy as np
import pytest

from f1p.eval import harness
from f1p.eval.baselines import RankPL, Uniform, standings_ranks
from f1p.lake.curate_f1db import Curated
from f1p.models.challengers import RankerModel, feature_names, relevance
from f1p.models.stacking import stack_result
from f1p.models.world_v1 import WorldV1

try:  # the optional `ml` extra; LightGBM also needs OpenMP at import time
    import catboost  # noqa: F401
    import lightgbm  # noqa: F401
except Exception as exc:
    pytest.skip(f"gradient-boosting libraries unavailable: {exc}", allow_module_level=True)


@pytest.fixture(scope="module")
def result(curated: Curated) -> harness.EvalResult:
    return harness.run(
        curated.tables,
        [RankerModel("lgbm"), RankerModel("catboost"), Uniform()],
        seasons=(2026, 2026),
        history_from=2021,
        keep_forecasts=True,
    )


def test_feature_table_is_causal_and_complete(curated: Curated) -> None:
    from f1p.eval.context import build_contexts

    model = RankerModel("lgbm")
    contexts = build_contexts(curated.tables, seasons=(2026, 2026))
    by = {(c.event_id, c.checkpoint): (c, o) for c, o, _ in contexts}
    for cp in ("C1", "C4"):
        ctx, outcome = by[("2026-R10", cp)]
        sig = model.prepare(ctx)
        assert sig.x.shape == (ctx.n, len(feature_names(cp)))
        assert np.all(np.isfinite(sig.x))
        rel = relevance(ctx, outcome)
        assert rel.max() == len(outcome.classified)
        assert (rel == 0).sum() == ctx.n - len(outcome.classified)


def test_rankers_forecast_and_beat_uniform(result: harness.EvalResult) -> None:
    for name in ("lgbm_rank", "catboost_rank"):
        pe = result.per_event.filter(result.per_event["model"] == name)
        assert set(pe["checkpoint"].unique().to_list()) == {"C1", "C4"}
        assert pe.height == 2 * 17
        for cp in ("C1", "C4"):
            cmp = result.compare(name, "uniform", cp, "finisher_pl_loglik")
            assert cmp.mean_diff > 0, (name, cp, cmp)
    fc = result.forecasts[("lgbm_rank", "C4", "2026-R17")]
    assert fc.p_win.sum() == pytest.approx(1.0)
    assert fc.p_finish is not None
    assert fc.stage_scales is not None


def test_stack_fits_on_earlier_seasons(curated: Curated) -> None:
    members = ["world_v1", "standings"]
    standings = RankPL("standings", standings_ranks, frozenset({"C1", "C4"}))
    res = harness.run(
        curated.tables,
        [WorldV1(), standings],
        seasons=(2021, 2026),
        history_from=2013,
        keep_forecasts=True,
    )
    stacked, fits = stack_result(
        curated.tables, res, members, seasons=(2026, 2026), training_seasons=6
    )
    assert set(fits) == {(2026, "C1"), (2026, "C4")}
    pe = stacked.per_event
    assert set(pe["model"].unique().to_list()) == {"world_v1", "standings", "stack", "stack_iso"}
    assert pe.filter(pe["model"] == "stack").height == 2 * 17
    for fit in fits.values():
        # weights are on each member's own strength scale: only their signs are comparable
        assert all(np.isfinite(w) and w >= 0.0 for w in fit.weights.values())
        assert fit.weights["world_v1"] > 0.0
    # the pool is never significantly worse than its best member
    for cp in ("C1", "C4"):
        cmp = stacked.compare("stack", "world_v1", cp, "finisher_pl_loglik")
        assert cmp.hi > 0, (cp, cmp)
    # weights come from earlier seasons only: scrambling this season's forecasts of a member
    # changes the stack's forecasts but not its weights
    scrambled = dict(res.forecasts)
    for (m, cp, eid), fc in res.forecasts.items():
        if m == "standings" and eid.startswith("2026"):
            assert fc.log_strength is not None
            scrambled[(m, cp, eid)] = type(fc).from_log_strength(
                eid, cp, m, fc.drivers, fc.log_strength[::-1].copy()
            )
    res2 = harness.EvalResult(res.per_event, res.calibration, scrambled)
    _, fits2 = stack_result(curated.tables, res2, members, seasons=(2026, 2026), training_seasons=6)
    for key, fit in fits.items():
        np.testing.assert_allclose(fit.fit.w, fits2[key].fit.w)
