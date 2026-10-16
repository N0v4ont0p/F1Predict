from __future__ import annotations

import itertools
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
from f1p.eval.metrics import spearman  # noqa: E402
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
    return lq.MLXBackend.arrays(lq.collate([ld.tokens(r) for r in races]))


MLX_TINY = lq.LapSeqConfig(**{**TINY.__dict__, "backend": "mlx"})


@pytest.mark.parametrize("temporal", ["attention", "ssm"])
def test_network_is_causal_and_ignores_padding(sims: list[ld.RaceSeq], temporal: str) -> None:
    net = lq.build_network(lq.LapSeqConfig(**{**TINY.__dict__, "temporal": temporal}))
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


@pytest.mark.parametrize(
    ("backend", "temporal"), [("mlx", "attention"), ("torch", "attention"), ("torch", "ssm")]
)
def test_training_learns_and_round_trips(
    sims: list[ld.RaceSeq], tmp_path: Path, backend: str, temporal: str
) -> None:
    if backend == "torch":
        pytest.importorskip("torch")
    real, synthetic = sims[:60], sims[60:200]
    cfg = lq.LapSeqConfig(
        **{**TINY.__dict__, "finetune_epochs": 6, "backend": backend, "temporal": temporal}
    )
    before = lq.evaluate(lq.make_backend(cfg), [ld.tokens(r) for r in real[-9:]])
    trained = lq.train(real, synthetic, cfg)
    assert trained.backend.name == backend
    assert [x.phase for x in trained.log][:1] == ["pretrain"]
    assert trained.val_terms["total"] < before["total"]
    assert trained.val_terms["order"] < before["order"]
    # the strength head orders the field: correlated with the finish at the last token
    tok = ld.tokens(real[0])
    p = lq.predict(trained.backend, tok)
    fin = real[0].finish >= 0
    rho = spearman(-p.strength[fin], real[0].finish[fin].astype(np.float64))
    assert rho > 0.3
    # saved and loaded: the same predictions
    lq.save(trained, tmp_path / "m", {"season": 0})
    be, cfg2, meta = lq.load(tmp_path / "m")
    assert cfg2 == cfg
    assert meta["season"] == 0
    assert meta["trained_on"] == backend
    np.testing.assert_allclose(lq.predict(be, tok).strength, p.strength, atol=1e-5)


def _compare(a: lq.Backend, b: lq.Backend, batch: dict, what: str) -> None:  # type: ignore[type-arg]
    a.set_training(False)
    b.set_training(False)
    ha, hb = a.heads(batch), b.heads(batch)
    for k in ha:
        np.testing.assert_allclose(ha[k], hb[k], atol=2e-4, rtol=1e-4, err_msg=f"{what}: {k}")
    la, lb = a.losses(batch), b.losses(batch)
    for k in la:
        assert la[k] == pytest.approx(lb[k], rel=1e-4, abs=1e-5), (what, k)


def _cfg(backend: str, temporal: str = "attention") -> lq.LapSeqConfig:
    return lq.LapSeqConfig(
        **{**TINY.__dict__, "blocks": 2, "backend": backend, "temporal": temporal}
    )


@pytest.mark.parametrize("temporal", ["attention", "ssm"])
def test_mlx_and_torch_are_twins(sims: list[ld.RaceSeq], temporal: str) -> None:
    """The same weights give the same heads and the same losses in both back ends (the
    arithmetic, on the CPU)."""
    pytest.importorskip("torch")
    m = lq.make_backend(_cfg("mlx", temporal), device="cpu")
    t = lq.make_backend(_cfg("torch", temporal), device="cpu")
    t.load(m.export())
    assert set(t.export()) == set(m.export())
    batch = lq.collate([ld.tokens(r) for r in sims[:3]])
    _compare(m, t, batch, "mlx vs torch")
    # and a network trained in PyTorch runs in MLX
    m.load(t.export())
    _compare(m, t, batch, "torch weights in mlx")


@pytest.mark.parametrize("temporal", ["attention", "ssm"])
def test_gpu_matches_cpu(sims: list[ld.RaceSeq], temporal: str) -> None:
    """On Apple silicon: MLX on the GPU (Metal) computes what it does on the CPU."""
    if not mx.metal.is_available():
        pytest.skip("no Metal device")
    cpu = lq.make_backend(_cfg("mlx", temporal), device="cpu")
    gpu = lq.make_backend(_cfg("mlx", temporal), device="gpu")
    gpu.load(cpu.export())
    batch = lq.collate([ld.tokens(r) for r in sims[:3]])
    _compare(cpu, gpu, batch, f"mlx cpu vs gpu ({temporal})")


def test_torch_runs_on_the_cpu() -> None:
    """PyTorch's MPS results disagreed with its CPU results on the macOS runner (by up to
    1.0 on the heads, while MLX on Metal agreed): the PyTorch back end refuses the GPU."""
    pytest.importorskip("torch")
    be = lq.make_backend(_cfg("torch"))
    assert str(be.device) == "cpu"  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="CPU"):
        lq.make_backend(_cfg("torch"), device="gpu")


def test_torch_losses_match_the_references() -> None:
    torch = pytest.importorskip("torch")
    from f1p.models import lapseq_torch as lt

    y, mu, log_s, nu = (
        np.array([0.1, -0.3, 2.0]),
        np.array([0.0, 0.1, 0.5]),
        np.array([-1.0, -0.5, 0.2]),
        4.3,
    )
    got = lt.student_t_nll(torch.tensor(y), torch.tensor(mu), torch.tensor(log_s), torch.tensor(nu))
    want = -stats.t.logpdf(y, df=nu, loc=mu, scale=np.exp(log_s))
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-5)


def test_learning_rate_schedule() -> None:
    cfg = lq.LapSeqConfig(lr=1e-3, warmup=10)
    total = 200
    lrs = [lq.lr_at(cfg, s, total) for s in range(total + 1)]
    assert lrs[0] == pytest.approx(1e-3 / 50)
    assert max(lrs) == pytest.approx(1e-3)
    assert lrs[-1] == pytest.approx(1e-3 / 20)
    assert all(a <= b for a, b in itertools.pairwise(lrs[:11]))  # warm-up rises
    assert all(a >= b for a, b in itertools.pairwise(lrs[10:]))  # then decays


def test_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        lq.resolve_backend(lq.LapSeqConfig(backend="jax"))


def test_config_parameter_count() -> None:
    from mlx.utils import tree_flatten

    cfgs = [TINY, lq.LapSeqConfig(d=32, heads=4, blocks=2)]
    cfgs += [lq.LapSeqConfig(**{**c.__dict__, "temporal": "ssm"}) for c in cfgs]
    cfgs.append(lq.LapSeqConfig(d=32, blocks=1, temporal="ssm", ssm_state=8, ssm_expand=2))
    for cfg in cfgs:
        net = lq.build_network(cfg)
        sizes = [getattr(v, "size", 0) for _, v in tree_flatten(net.parameters())]
        assert sum(sizes) == cfg.params(), cfg
    assert 5e6 < lq.FULL.params() < 30e6  # in the plan's 5-30M range
    # the state-space variant of the small network is within 3% of its size
    ssm = lq.LapSeqConfig(temporal="ssm")
    assert abs(ssm.params() / lq.SMALL.params() - 1.0) < 0.03


def test_ssm_settings_and_identity() -> None:
    with pytest.raises(ValueError, match="temporal"):
        lq.LapSeqConfig(temporal="rnn")
    # an attention network's identity is the settings it was first trained with (no
    # state-space fields), so its cached networks stay valid; an SSM network's has them
    att = lq.LapSeqConfig().identity()
    assert "temporal" not in att
    assert att.startswith("LapSeqConfig(d=128, heads=4")
    ssm = lq.LapSeqConfig(temporal="ssm").identity()
    assert "temporal='ssm'" in ssm
    assert "ssm_state=16" in ssm
    assert lq.digest(lq.LapSeqConfig(), 8) != lq.digest(lq.LapSeqConfig(temporal="ssm"), 8)
    assert lq.digest(lq.LapSeqConfig(), 8) == lq.digest(lq.LapSeqConfig(), 8)


def test_ssm_step_is_the_full_pass(sims: list[ld.RaceSeq]) -> None:
    """Lap-by-lap updates of the state-space layer equal its pass over the whole race —
    the live form: one update per lap, no re-reading."""
    torch = pytest.importorskip("torch")
    from f1p.models import lapseq_torch as lt

    torch.manual_seed(0)
    layer = lt.SSM(lq.LapSeqConfig(d=16, temporal="ssm", ssm_state=4, ssm_conv=3))
    x = torch.randn(5, 30, 16)
    with torch.no_grad():
        full = layer(x)
        state = None
        steps = []
        for t in range(x.shape[1]):
            y, state = layer.step(x[:, t], state)
            steps.append(y)
    np.testing.assert_allclose(torch.stack(steps, 1).numpy(), full.numpy(), atol=1e-5)
    # and causal: changing lap 20 leaves laps 0..19 alone
    x2 = x.clone()
    x2[:, 20:] += 3.0
    with torch.no_grad():
        np.testing.assert_allclose(layer(x2)[:, :20].numpy(), full[:, :20].numpy(), atol=1e-6)


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
    saved = list((tmp_path / "lapseq").glob("*/2026/weights.npz"))
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
