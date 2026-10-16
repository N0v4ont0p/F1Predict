from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from f1p.lake.curate_f1db import Curated

optuna = pytest.importorskip("optuna")

from f1p.tuning import search  # noqa: E402
from f1p.tuning.spaces import COMBINER, SPACES  # noqa: E402

SEASON = (2021, 2021)  # the fixture's tune season


def test_score_config_reports_each_season(curated: Curated) -> None:
    seen: list[tuple[int, float]] = []
    s = search.score_config(
        curated.tables,
        COMBINER.build(COMBINER.baseline),
        SEASON,
        report=lambda step, value: seen.append((step, value)),
    )
    assert s.events
    assert len(s.events) == len(s.losses)
    assert [step for step, _ in seen] == [0]
    assert seen[0][1] == pytest.approx(s.mean)
    # finisher log-likelihood is maximised: the loss is its negative
    assert search.loss_sign("finisher_pl_loglik") == -1.0
    assert search.loss_sign("winner_logloss") == 1.0


def test_optimize_and_select(curated: Curated, tmp_path: Path) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    path = search.study_path(tmp_path, "t")
    study = search.optimize(
        path, "t", "combiner", trials=4, seasons=SEASON, tables=curated.tables, cache_dir=tmp_path
    )
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(done) >= 2
    assert study.trials[0].user_attrs.get("baseline") is True  # the hand-set config is trial 0
    assert study.trials[0].params == COMBINER.baseline
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE  # never pruned
    # resumable: reopening the journal sees the same trials
    again = search.open_study(path, "t", COMBINER)
    assert len(again.trials) == len(study.trials)
    sel = search.select(study, "combiner", curated.tables, top=3, seasons=SEASON, null_flips=200)
    best = min(c.select for c in sel.candidates)
    assert sel.chosen.select <= best + sel.chosen.se + 1e-12  # within one SE of the best
    within = [c for c in sel.candidates if c.select <= best + c.se + 1e-12]
    assert sel.chosen.complexity == min(c.complexity for c in within)  # and the simplest
    assert any(c.baseline for c in sel.candidates)
    out = search.save_selection(sel, tmp_path / "sel.json")
    assert out.exists()


def test_search_never_sees_the_holdout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="holdout"):
        search.optimize(tmp_path / "x.journal", "x", "combiner", trials=1, seasons=(2023, 2024))


def _study(effect: float, trials: int = 20, events: int = 200, seed: int = 0) -> object:
    """A study whose trials' per-event losses are the baseline's plus noise, and ``effect``
    better for the last trial."""
    rng = np.random.default_rng(seed)
    study = optuna.create_study(direction="minimize")
    names = [f"E{i}" for i in range(events)]
    base = rng.normal(1.0, 0.3, events)
    dist = {"l2": optuna.distributions.FloatDistribution(0.05, 20.0, log=True)}
    for k in range(trials):
        losses = base if k == 0 else base + rng.normal(0.0, 0.05, events)
        if k == trials - 1:
            losses = losses - effect
        study.add_trial(
            optuna.trial.create_trial(
                params={"l2": 1.0},
                distributions=dist,
                value=float(np.mean(losses)),
                user_attrs={
                    "events": names,
                    "losses": losses.tolist(),
                    "baseline": k == 0,
                },
            )
        )
    return study


def test_null_inflation() -> None:
    flat = search.null_inflation(_study(0.0), flips=1000)
    assert flat["p"] > 0.05  # the best of 19 noise trials is what chance produces
    assert flat["best_improvement"] > 0  # ... and still looks like an improvement
    real = search.null_inflation(_study(0.05), flips=1000)
    assert real["p"] < 0.01
    assert real["best_improvement"] > real["null_q95"]


def test_spaces_are_well_formed() -> None:
    for space in SPACES.values():
        assert space.sampler in ("tpe", "cmaes", "gp")
        assert set(space.baseline) == {p.name for p in space.params}
        for p in space.params:
            assert p.low < p.high
            assert 0.0 <= p.unit(space.baseline[p.name]) <= 1.0, (space.name, p.name)
        assert space.complexity(space.baseline) == 0.0
