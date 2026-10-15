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
    assert x.shape == (ctx.n, 4)  # rating, grid, quali, lap pace
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
    # nobody wins without finishing
    assert np.all(fc.p_win <= fc.p_finish + 0.01)
    assert "finish_a" in fc.meta


def test_without_retirements_matches_plain_pl(curated: Curated) -> None:
    r = harness.run(
        curated.tables,
        [WorldV1(retirements=False, smooth_stages=False)],
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


def test_qualifying_observations_respect_checkpoints(curated: Curated) -> None:
    """With qualifying assimilated, C4 beliefs include this weekend's sessions; C1 never do."""
    model = WorldV1(quali_obs=True)
    contexts = build_contexts(curated.tables, seasons=(2026, 2026), checkpoints=("C1", "C4"))
    by = {(c.event_id, c.checkpoint): c for c, _, _ in contexts}
    st = model._state(curated.tables)
    c1, c4 = by[("2026-R17", "C1")], by[("2026-R17", "C4")]
    step1 = st.step_at(c1.as_of.timestamp())
    step4 = st.step_at(c4.as_of.timestamp())
    seen1 = {(o.event_id, o.kind) for o in st.races[:step1]}
    seen4 = {(o.event_id, o.kind) for o in st.races[:step4]}
    assert ("2026-R16", "race") in seen1
    assert not any(e == "2026-R17" for e, _ in seen1)
    assert {("2026-R17", k) for k in ("sq", "sprint", "quali")} <= seen4
    assert ("2026-R17", "race") not in seen4
    mu1, _ = model.beliefs(c1)
    mu4, _ = model.beliefs(c4)
    assert not np.allclose(mu1, mu4[: mu1.size]) or c1.drivers != c4.drivers


def test_stage_temperatures_fall_down_the_order(curated: Curated) -> None:
    model = WorldV1()
    r = harness.run(
        curated.tables, [model], seasons=(2026, 2026), history_from=2021, keep_forecasts=True
    )
    fc = next(iter(r.forecasts.values()))
    assert fc.stage_scales is not None
    assert fc.stage_scales[0] == 1.0  # the combiner's own scale is the winner's
    assert np.all(np.diff(fc.stage_scales[:15]) <= 1e-12)
    assert fc.p_win.sum() == pytest.approx(1.0)


def test_rain_signal_respects_knowability(curated: Curated) -> None:
    from datetime import UTC, datetime

    import polars as pl

    from f1p.models.world_v1 import rain_signal

    contexts = build_contexts(curated.tables, seasons=(2026, 2026), checkpoints=("C1", "C4"))
    by = {(c.event_id, c.checkpoint): c for c, _, _ in contexts}
    c1, c4 = by[("2026-R17", "C1")], by[("2026-R17", "C4")]
    weather = pl.DataFrame(
        {
            "event_id": ["2026-R17", "2026-R17"],
            "kind": ["climatology", "forecast"],
            "wet_prob": [0.3, 0.9],
            "precip_mm": [1.0, 4.0],
            "temp_c": [30.0, 28.0],
            "observed_at": [datetime(2026, 1, 1, tzinfo=UTC), c4.as_of],
        }
    )
    tables = {**curated.tables, "weather_forecast": weather}
    c1 = type(c1)(**{**c1.__dict__, "tables": tables, "_cache": {}})
    c4 = type(c4)(**{**c4.__dict__, "tables": tables, "_cache": {}})
    assert rain_signal(c1) == pytest.approx(0.3)  # only climatology is public on Thursday
    assert rain_signal(c4) == pytest.approx(0.9)  # race-morning forecast
    assert rain_signal(by[("2026-R16", "C1")]) == 0.0  # no weather table at all


def test_race_pace_signal_uses_only_public_races(curated: Curated) -> None:
    from datetime import datetime

    import polars as pl

    from f1p.models.features import race_pace_signal

    res = curated.tables["results"].filter(
        (pl.col("season") == 2026) & (pl.col("session") == "race") & pl.col("classified")
    )
    # synthetic pace: in R16 Verstappen is fastest; in R17 (not yet public at R17 C1) Norris is
    rows = []
    for r in res.iter_rows(named=True):
        if r["round"] not in (16, 17):
            continue
        fast = "max_verstappen" if r["round"] == 16 else "lando_norris"
        rows.append(
            {
                "event_id": r["event_id"],
                "driver_id": r["driver_id"],
                "car_number": r["car_number"],
                "clean_laps": 40,
                "median_lap_ms": 95000,
                "pace_deficit_pct": 0.0 if r["driver_id"] == fast else 1.5,
                "observed_at": r["observed_at"],
            }
        )
    pace = pl.DataFrame(rows).with_columns(pl.col("clean_laps").cast(pl.Int32))
    tables = {**curated.tables, "race_pace": pace}
    contexts = build_contexts(tables, seasons=(2026, 2026), checkpoints=("C1",))
    ctx = next(c for c, _, _ in contexts if c.event_id == "2026-R17")
    sig = race_pace_signal(ctx)
    best = ctx.drivers[int(np.argmax(sig))]
    assert best == "max_verstappen"  # R17's own pace is not public before its weekend
    assert abs(float(sig.mean())) < 1e-9
    published = pace.filter(pl.col("event_id") == "2026-R17")["observed_at"].min()
    assert isinstance(published, datetime)
    assert ctx.as_of < published
