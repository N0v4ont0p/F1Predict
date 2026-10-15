from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from f1p.lake.curate_f1db import Curated
from f1p.models import title_backtest as tb
from f1p.models.season import Standing


def _standing(ident: str, p: list[float]) -> Standing:
    return Standing(ident, 0.0, p[0], 0.0, (0.0,) * 5, tuple(p), 0)


def test_score_snapshot() -> None:
    rows = (_standing("a", [0.7, 0.3]), _standing("b", [0.3, 0.7]))
    s = tb.score_snapshot(rows, {"a": 0, "b": 1}, n_sims=1000)
    assert s["title_logloss"] == pytest.approx(-np.log(0.7))
    assert s["position_brier"] == pytest.approx(2 * 0.3**2)
    assert s["champion_favourite"] == 1.0
    # an impossible champion is clipped at 1 / n_sims, not infinite
    s = tb.score_snapshot(
        (_standing("a", [1.0, 0.0]), _standing("b", [0.0, 1.0])), {"b": 0, "a": 1}, n_sims=1000
    )
    assert s["title_logloss"] == pytest.approx(np.log(1000))


def test_called_from() -> None:
    df = pl.DataFrame(
        {
            "season": [2020] * 4,
            "after_round": [0, 1, 2, 3],
            "rounds": [4] * 4,
            "kind": ["driver"] * 4,
            "model": ["world_v1"] * 4,
            "p_champion": [0.6, 0.4, 0.55, 0.9],
        }
    )
    c = tb.called_from(df).row(0, named=True)
    assert c["called_after"] == 2
    assert c["p_preseason"] == pytest.approx(0.6)


@pytest.fixture(scope="module")
def snapshots(curated: Curated) -> pl.DataFrame:
    return tb.run(curated.tables, (2021, 2021), n_sims=300)


def test_backtest_scores_completed_seasons_only(snapshots: pl.DataFrame, curated: Curated) -> None:
    assert tb.run(curated.tables, (2026, 2026)).is_empty()  # 2026 is still running
    assert set(snapshots["season"].to_list()) == {2021}
    rounds = sorted(set(snapshots["after_round"].to_list()))
    assert rounds == list(range(22))  # pre-season .. after the penultimate round
    assert snapshots.height == 22 * 2 * 2
    assert snapshots["title_logloss"].is_finite().all()


def test_model_calls_2021_better_than_a_uniform_spread(snapshots: pl.DataFrame) -> None:
    summary = {(s.kind, s.metric): s for s in tb.summarise(snapshots)}
    brier = summary[("driver", "position_brier")]
    assert brier.model < brier.baseline
