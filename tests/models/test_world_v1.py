from __future__ import annotations

import numpy as np
import pytest

from f1p.eval import harness
from f1p.eval.baselines import Uniform
from f1p.eval.context import build_contexts
from f1p.lake.curate_f1db import Curated
from f1p.models.ratings import race_observations, run_filter
from f1p.models.registry import registered_models
from f1p.models.world_v1 import WorldV1, ratings_table


@pytest.fixture(scope="module")
def result(curated: Curated) -> harness.EvalResult:
    return harness.run(
        curated.tables, [WorldV1(), Uniform()], seasons=(2026, 2026), history_from=2021
    )


def test_registered() -> None:
    assert "world_v1" in registered_models()


def test_scored_on_every_event_at_both_checkpoints(result: harness.EvalResult) -> None:
    pe = result.per_event
    for cp in ("C1", "C4"):
        n = pe.filter((pe["model"] == "world_v1") & (pe["checkpoint"] == cp)).height
        assert n == 17


def test_beats_uniform(result: harness.EvalResult) -> None:
    for cp in ("C1", "C4"):
        for metric in ("winner_logloss", "finisher_pl_loglik"):
            cmp = result.compare("world_v1", "uniform", cp, metric)
            assert harness.better(metric, cmp.mean_diff), (cp, metric, cmp)


def test_beliefs_use_only_public_results(curated: Curated) -> None:
    """At C1 of 2026-R05 the model knows exactly the races before it — no more, no less."""
    model = WorldV1()
    contexts = build_contexts(curated.tables, seasons=(2026, 2026), checkpoints=("C1", "C4"))
    races = race_observations(curated.tables)
    for ctx, _, _ in contexts:
        if ctx.event_id not in ("2026-R01", "2026-R05", "2026-R17"):
            continue
        before = [r for r in races if (r.season, r.round) < (ctx.season, ctx.round)]
        filt, _ = run_filter(before, model.params)
        st = model._state(ctx.tables)
        teams = [st.lineage.get((ctx.season, c), c) for c in ctx.constructors]
        expect = filt.peek(ctx.drivers, teams, ctx.season)
        got = model.beliefs(ctx)
        np.testing.assert_allclose(got[0], expect[0])
        np.testing.assert_allclose(got[1], expect[1])


def test_c4_uses_qualifying_signals(result: harness.EvalResult, curated: Curated) -> None:
    model = WorldV1()
    contexts = build_contexts(curated.tables, seasons=(2026, 2026), checkpoints=("C4",))
    ctx = contexts[-1][0]
    sig = model.prepare(ctx)
    x = sig.x
    assert x.shape == (ctx.n, 3)
    assert np.all(np.isfinite(x))
    assert x[:, 1].std() > 0  # grid signal present
    assert sig.finish_logit.shape == (ctx.n,)


def test_forecasts_retirements(result: harness.EvalResult, curated: Curated) -> None:
    model = WorldV1()
    r = harness.run(
        curated.tables, [model], seasons=(2026, 2026), history_from=2021, keep_forecasts=True
    )
    fc = next(iter(r.forecasts.values()))
    assert fc.p_finish is not None
    assert np.all((fc.p_finish > 0.3) & (fc.p_finish < 1.0))
    assert fc.p_win.sum() == pytest.approx(1.0)
    # a car that may retire cannot be more likely to win than if it always finished
    from f1p.eval.forecast import pl_win_probabilities

    assert fc.log_strength is not None
    top = int(np.argmax(fc.log_strength))
    assert fc.p_win[top] <= pl_win_probabilities(fc.log_strength)[top] + 1e-9
    assert "finish_a" in fc.meta


def test_without_retirements_matches_plain_pl(curated: Curated) -> None:
    r = harness.run(
        curated.tables,
        [WorldV1(retirements=False)],
        seasons=(2026, 2026),
        history_from=2021,
        keep_forecasts=True,
    )
    fc = next(iter(r.forecasts.values()))
    assert fc.p_finish is None
    assert fc.win_probs is None


def test_ratings_table(curated: Curated) -> None:
    df = ratings_table(WorldV1(), curated.tables, top=5)
    assert set(df["kind"].to_list()) == {"driver", "team"}
    assert df.height <= 10
