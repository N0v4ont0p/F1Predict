from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from f1p.eval import harness
from f1p.eval.baselines import default_baselines
from f1p.eval.context import build_contexts, outcome_for
from f1p.lake.curate_f1db import Curated
from f1p.models.features import practice_pace, sprint_quali_ranks
from f1p.models.quali_v1 import QualiV1, knockout_probs
from f1p.models.registry import registered_models


@pytest.fixture(scope="module")
def result(curated: Curated) -> harness.EvalResult:
    return harness.run(
        curated.tables,
        [QualiV1(), *default_baselines("quali")],
        seasons=(2026, 2026),
        history_from=2021,
        target="quali",
        keep_forecasts=True,
    )


def test_registered_for_the_quali_target_only() -> None:
    models = registered_models()
    assert models["quali_v1"].targets == frozenset({"quali"})
    assert harness.model_targets(models["world_v1"]) == frozenset({"race"})


def test_quali_outcome_is_the_qualifying_order(curated: Curated) -> None:
    outcome, known = outcome_for(curated.tables, "2026-R17", "quali")
    q = curated.tables["qualifying"].filter(
        (pl.col("event_id") == "2026-R17")
        & (pl.col("session") == "quali")
        & pl.col("position").is_not_null()  # cars without a qualifying place are not ranked
    )
    assert outcome.winner == q.sort("position")["driver_id"][0]
    assert len(outcome.classified) == q.height
    assert known == q["observed_at"].max()


def test_checkpoints_cut_the_weekend_correctly(curated: Curated) -> None:
    contexts = build_contexts(curated.tables, seasons=(2026, 2026), target="quali")
    by = {(c.event_id, c.checkpoint): c for c, _, _ in contexts}
    c1, c2 = by[("2026-R17", "C1")], by[("2026-R17", "C2")]
    assert c1.target == c2.target == "quali"
    assert c1.as_of < c2.as_of
    # C2 is just before qualifying: practice and sprint qualifying are public, qualifying not
    seen = c2.known("qualifying").filter(pl.col("event_id") == "2026-R17")
    assert set(seen["session"].to_list()) == {"sq"}
    assert c2.known("practice").filter(pl.col("event_id") == "2026-R17").height > 0
    assert c1.known("practice").filter(pl.col("event_id") == "2026-R17").is_empty()
    assert np.all(practice_pace(c1) == 0)
    assert practice_pace(c2).std() > 0
    assert sprint_quali_ranks(c1) is None
    assert sprint_quali_ranks(c2) is not None


def test_beats_baselines(result: harness.EvalResult) -> None:
    for cp, ref in (("C1", "last_quali"), ("C2", "last_quali"), ("C2", "uniform")):
        for metric in ("winner_logloss", "finisher_pl_loglik"):
            cmp = result.compare("quali_v1", ref, cp, metric)
            assert harness.better(metric, cmp.mean_diff), (cp, ref, metric, cmp)
    assert set(result.per_event["target"].unique().to_list()) == {"quali"}


def test_knockout_probabilities(result: harness.EvalResult) -> None:
    fc = result.forecasts[("quali_v1", "C2", "2026-R17")]
    ko = knockout_probs(fc, 2026)
    assert ko is not None
    np.testing.assert_allclose(ko.q3 + ko.out_q2 + ko.out_q1, 1.0, atol=1e-9)
    assert ko.q3.sum() == pytest.approx(10.0, abs=1e-6)
    assert ko.out_q1.sum() == pytest.approx(6.0, abs=1e-6)  # 22 cars: 6 out in Q1
    assert np.all(ko.out_q2 >= -1e-9)
