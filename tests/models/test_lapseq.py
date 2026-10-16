from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

mx = pytest.importorskip("mlx.core")

from scipy import stats  # noqa: E402
from test_sim_v2 import _models, _race  # noqa: E402

from f1p.eval import harness  # noqa: E402
from f1p.eval.context import build_contexts  # noqa: E402
from f1p.eval.forecast import pl_log_likelihood  # noqa: E402
from f1p.lake.curate_f1db import Curated  # noqa: E402
from f1p.models import lapseq as lq  # noqa: E402
from f1p.models import lapseq_data as ld  # noqa: E402
from f1p.models import lapseq_model as lm  # noqa: E402
from f1p.models import sim_v2 as sv  # noqa: E402
from synthetic import synthetic_laps, with_positions  # noqa: E402

TINY = lq.LapSeqConfig(
    d=16, heads=2, blocks=1, mlp=2, dropout=0.0, batch=8, pretrain_epochs=1, finetune_epochs=3
)


@pytest.fixture(scope="module")
def sims() -> list[ld.RaceSeq]:
    spec, draws = _race()
    res = sv.simulate(spec, draws, _models(), sv.Knobs(), np.random.default_rng(3), trace=True)
    assert res.trace is not None
    return ld.simulated_races(
        res.trace,
        res.orders,
        res.finished,
        spec.grid,
        draws.pace,
        spec.laps,
        np.random.default_rng(0),
    )


@pytest.fixture(scope="module")
def tables(curated: Curated) -> dict[str, pl.DataFrame]:
    laps = pl.concat([synthetic_laps(curated, season=y, seed=y) for y in (2021, 2026)])
    return {**curated.tables, "laps": with_positions(laps)}


# -- data -----------------------------------------------------------------------------------------
def test_tokens_only_look_back(sims: list[ld.RaceSeq]) -> None:
    """A token's features depend only on laps up to it: the first k tokens of a whole race
    are the tokens of its first k laps."""
    race = sims[0]
    whole = ld.tokens(race)
    assert whole.x.shape == (race.laps, race.n, ld.F)
    for k in (1, 2, 7, race.laps - 1):
        part = ld.tokens(race, upto=k)
        np.testing.assert_array_equal(part.x, whole.x[:k])
        np.testing.assert_array_equal(part.running, whole.running[:k])
    assert np.all(whole.x[0, :, ld.FEATURES.index("is_grid")] == 1.0)
    assert np.all(whole.x[1:, :, ld.FEATURES.index("is_grid")] == 0.0)
    assert np.isfinite(whole.x).all()
    # one compound per running car-token
    tyres = [i for i, f in enumerate(ld.FEATURES) if f.startswith("tyre_") and f != "tyre_age"]
    onehot = whole.x[1:][:, :, tyres].sum(axis=2)
    np.testing.assert_array_equal(onehot[whole.running[1:]], 1.0)


def test_targets_are_the_next_lap(sims: list[ld.RaceSeq]) -> None:
    race = sims[1]
    tok = ld.tokens(race)
    t = 5
    live = tok.running[t]
    np.testing.assert_array_equal(tok.next_pit[t][live], race.pit[t + 1][live])
    np.testing.assert_array_equal(tok.next_out[t][live], ~race.alive[t + 1][live])
    assert np.isnan(tok.next_pit[t][~live]).all()
    assert np.isnan(tok.next_pit[-1]).all()  # no lap after the last
    # a pit or neutralised lap is not a clean lap
    j = t + 1
    dirty = race.pit[j] | race.pit[j - 1] | race.neutral[j]
    assert np.isnan(tok.next_res[t][dirty]).all()


def test_real_race(tables: dict[str, pl.DataFrame]) -> None:
    ctx, outcome, _ = next(
        c for c in build_contexts(tables, seasons=(2026, 2026), checkpoints=("C4",))
    )
    seq = lm.race_seq(ctx, outcome=True)
    assert seq is not None
    idx = ctx.index()
    # the classification, from 0
    for pos, d in enumerate(outcome.classified):
        assert seq.finish[idx[d]] == pos
    assert (seq.finish >= 0).sum() == len(outcome.classified)
    # classified cars run to the end; the grid is a permutation
    assert seq.alive[-1][seq.finish >= 0].all()
    assert sorted(seq.grid.tolist()) == list(range(ctx.n))
    # positions are 1..(running) on a timed lap, gaps grow down the order
    lap = 10
    live = seq.alive[lap] & (seq.position[lap] > 0)
    order = np.argsort(seq.position[lap][live])
    assert np.all(np.diff(seq.gap_s[lap][live][order]) >= 0)
    # a forecast's input has no result in it
    blind = lm.race_seq(ctx, outcome=False)
    assert blind is not None
    assert (blind.finish == -1).all()
    assert blind.laps == 1


# -- the network and its losses ------------------------------------------------------------------
def test_lgamma_matches_math() -> None:
    xs = np.array([1.0, 1.5, 2.0, 3.7, 10.0, 51.0])
    got = np.array(lq.lgamma(mx.array(xs)))
    np.testing.assert_allclose(got, [math.lgamma(x) for x in xs], rtol=1e-5, atol=1e-6)


def test_student_t_nll_matches_scipy() -> None:
    y = np.array([0.1, -0.3, 2.0])
    mu = np.array([0.0, 0.1, 0.5])
    log_s = np.array([-1.0, -0.5, 0.2])
    nu = 4.3
    got = np.array(lq.student_t_nll(mx.array(y), mx.array(mu), mx.array(log_s), mx.array(nu)))
    want = -stats.t.logpdf(y, df=nu, loc=mu, scale=np.exp(log_s))
    np.testing.assert_allclose(got, want, rtol=1e-4)


def test_pl_loss_is_the_plackett_luce_likelihood() -> None:
    rng = np.random.default_rng(0)
    b, t, n = 2, 3, 6
    s = rng.normal(size=(b, t, n)).astype(np.float32)
    order = np.array([[3, 1, 0, 5, 0, 0], [2, 4, 0, 1, 3, 5]], dtype=np.int32)
    ok = np.array([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=bool)
    running = np.ones((b, t, n), dtype=bool)
    running[1, 2, 4] = False  # a finisher missing at a token: dropped from that ranking
    got = float(lq.pl_loss(mx.array(s), mx.array(order), mx.array(ok), mx.array(running)).item())
    want = []
    for i in range(b):
        for j in range(t):
            o = [k for k in order[i][ok[i]] if running[i, j, k]]
            want.append(pl_log_likelihood(s[i, j].astype(np.float64), np.array(o)) / len(o))
    assert got == pytest.approx(-np.mean(want), rel=1e-5)


def _batch(races: list[ld.RaceSeq]) -> dict:  # type: ignore[type-arg]
    return lq.collate([ld.tokens(r) for r in races])


def test_network_is_causal_and_ignores_padding(sims: list[ld.RaceSeq]) -> None:
    net = lq.build_network(TINY)
    mx.eval(net.parameters())
    net.eval()
    b = _batch(sims[:2])
    out = net(b["x"], b["running"])
    # change everything after token 10: outputs up to it do not move
    x2 = np.array(b["x"])
    x2[:, 11:] = np.random.default_rng(1).normal(size=x2[:, 11:].shape)
    out2 = net(mx.array(x2), b["running"])
    for k in ("strength", "mu", "pit", "classified", "neutral"):
        np.testing.assert_allclose(np.array(out[k])[:, :11], np.array(out2[k])[:, :11], atol=1e-5)
    # a padded (absent) car changes nothing for the others
    x3 = np.concatenate([np.array(b["x"]), np.ones((2, x2.shape[1], 1, ld.F), np.float32)], 2)
    r3 = np.concatenate([np.array(b["running"]), np.zeros((2, x2.shape[1], 1), bool)], 2)
    out3 = net(mx.array(x3), mx.array(r3))
    np.testing.assert_allclose(
        np.array(out["strength"]), np.array(out3["strength"])[:, :, :-1], atol=1e-5
    )


def test_training_learns_and_round_trips(sims: list[ld.RaceSeq], tmp_path: Path) -> None:
    real, synthetic = sims[:60], sims[60:200]
    cfg = lq.LapSeqConfig(**{**TINY.__dict__, "finetune_epochs": 6})
    untrained = lq.build_network(cfg)
    mx.eval(untrained.parameters())
    before = lq.evaluate(untrained, [ld.tokens(r) for r in real[-9:]], cfg)
    trained = lq.train(real, synthetic, cfg)
    assert [x.phase for x in trained.log][:1] == ["pretrain"]
    assert trained.val_terms["total"] < before["total"]
    assert trained.val_terms["order"] < before["order"]
    # the strength head orders the field: correlated with the finish at the last token
    tok = ld.tokens(real[0])
    p = lq.predict(trained.net, tok)
    fin = real[0].finish >= 0
    rho = stats.spearmanr(-p.strength[fin], real[0].finish[fin]).statistic
    assert rho > 0.3
    # saved and loaded: the same predictions
    lq.save(trained, tmp_path / "m", {"season": 0})
    net, cfg2, meta = lq.load(tmp_path / "m")
    assert cfg2 == cfg
    assert meta["season"] == 0
    np.testing.assert_allclose(lq.predict(net, tok).strength, p.strength, atol=1e-6)


def test_config_parameter_count() -> None:
    from mlx.utils import tree_flatten

    for cfg in (TINY, lq.LapSeqConfig(d=32, heads=4, blocks=2)):
        net = lq.build_network(cfg)
        sizes = [getattr(v, "size", 0) for _, v in tree_flatten(net.parameters())]
        assert sum(sizes) == cfg.params()
    assert 5e6 < lq.FULL.params() < 30e6  # in the plan's 5-30M range


# -- the forecaster --------------------------------------------------------------------------------
def test_lapseq_through_the_harness(tables: dict[str, pl.DataFrame], tmp_path: Path) -> None:
    model = lm.LapSeq(
        config=lq.LapSeqConfig(**{**TINY.__dict__, "finetune_epochs": 2}),
        training_seasons=6,
        pretrain=False,
        min_races=10,
        root=str(tmp_path / "lapseq"),
    )
    res = harness.run(
        tables,
        [model],
        seasons=(2026, 2026),
        history_from=2026,
        checkpoints=("C4", "C5"),
        keep_forecasts=True,
    )
    df = res.per_event
    assert set(df["checkpoint"].unique().to_list()) == {"C4", "C5"}
    assert df["winner_logloss"].is_finite().all()
    # one network for the season, saved once and reused
    saved = list((tmp_path / "lapseq").glob("*/2026/weights.safetensors"))
    assert len(saved) == 1
    for (_, cp, _), fc in res.forecasts.items():
        assert fc.p_finish is not None
        assert np.isclose(fc.p_win.sum(), 1.0)
        if cp == "C5":
            assert fc.meta["lap"] > 0
    # no training races (no laps before the season): no forecast, no error
    early = harness.run(
        tables, [model], seasons=(2021, 2021), history_from=2021, checkpoints=("C4",)
    )
    assert early.per_event.is_empty()
