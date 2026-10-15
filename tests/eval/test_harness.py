from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from f1p.eval import harness
from f1p.eval.baselines import RankPL, Uniform, default_baselines, golden_section_max, quali_ranks
from f1p.eval.context import build_contexts, checkpoint_time
from f1p.lake.curate_f1db import Curated


@pytest.fixture(scope="module")
def result(curated: Curated) -> harness.EvalResult:
    return harness.run(curated.tables, default_baselines(), seasons=(2026, 2026), history_from=2021)


def test_contexts_respect_knowability(curated: Curated) -> None:
    contexts = build_contexts(curated.tables, seasons=(2026, 2026))
    assert contexts
    for ctx, outcome, outcome_known in contexts:
        assert outcome_known is not None
        assert outcome_known > ctx.as_of
        for name in ("results", "qualifying", "grid", "driver_standings"):
            known = ctx.known(name)
            latest = known["observed_at"].max()
            if latest is not None:
                assert isinstance(latest, datetime)
                assert latest <= ctx.as_of
        # this event's own race result is never visible
        own = ctx.known("results").filter(
            (pl.col("event_id") == ctx.event_id) & (pl.col("session") == "race")
        )
        assert own.is_empty()
        if ctx.checkpoint == "C1":
            assert ctx.known("qualifying").filter(pl.col("event_id") == ctx.event_id).is_empty()
        assert set(outcome.classified) <= set(ctx.drivers)


def test_c4_sees_qualifying(curated: Curated) -> None:
    contexts = build_contexts(curated.tables, seasons=(2026, 2026), checkpoints=("C4",))
    ctx = next(c for c, _, _ in contexts if c.event_id == "2026-R17")
    ranks = quali_ranks(ctx)
    assert ranks is not None
    assert ranks.min() == 1.0


def test_checkpoint_order() -> None:
    from datetime import date

    d = date(2026, 10, 11)
    assert checkpoint_time("C1", d, None) < checkpoint_time("C4", d, None)
    with pytest.raises(ValueError, match="unknown checkpoint"):
        checkpoint_time("C9", d, None)


def test_every_model_scored_on_every_event(result: harness.EvalResult) -> None:
    pe = result.per_event
    n_events = pe.filter(pl.col("model") == "uniform")["event_id"].n_unique()
    assert n_events == 17
    counts = pe.group_by("model", "checkpoint").len()
    assert set(counts["len"].to_list()) == {17}


def test_informative_baselines_beat_uniform(result: harness.EvalResult) -> None:
    for model, cp in (("quali", "C4"), ("grid", "C4"), ("standings", "C1")):
        cmp = result.compare(model, "uniform", cp, "winner_logloss")
        assert cmp.mean_diff < 0, (model, cp, cmp)


def test_history_is_strictly_earlier(curated: Curated) -> None:
    seen = []

    class Spy(Uniform):
        name = "spy"

        def forecast(self, ctx, signal, history):  # type: ignore[no-untyped-def]
            for c, _, _ in history:
                seen.append((ctx.as_of, c.as_of, c.race_date, ctx.race_date))
            return super().forecast(ctx, signal, history)

    harness.run(curated.tables, [Spy()], seasons=(2026, 2026), history_from=2021)
    assert seen
    assert all(hist_date < date for _, _, hist_date, date in seen)
    assert all(h_asof < asof + timedelta(seconds=1) for asof, h_asof, _, _ in seen)


def test_beta_fit_is_causal_and_cached(curated: Curated) -> None:
    model = RankPL("quali", quali_ranks, frozenset({"C4"}))
    harness.run(
        curated.tables, [model], seasons=(2026, 2026), history_from=2021, checkpoints=("C4",)
    )
    assert set(model._beta) == {(2026, "C4")}
    assert 0.0 < model._beta[(2026, "C4")] < 3.0


def test_summary_and_calibration(result: harness.EvalResult) -> None:
    s = result.summary()
    assert {"ece_top3", "ece_top10", "winner_logloss"} <= set(s.columns)
    assert s.filter(pl.col("model") == "quali")["checkpoint"].to_list() == ["C4"]


def test_golden_section_finds_maximum() -> None:
    assert golden_section_max(lambda x: -((x - 1.3) ** 2), 0.0, 3.0) == pytest.approx(1.3, abs=1e-3)
