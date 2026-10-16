"""``lapseq`` — a sequence model over laps (plan M6 §D.2), in MLX or PyTorch.

**Architecture.** A factorised spatio-temporal transformer: each block attends over the laps
of each car (causally: a token sees its own car's earlier laps) and then across the cars of
the same lap, then a feed-forward layer (pre-norm, residual). The tokens are
:func:`f1p.models.lapseq_data.tokens`: the grid, then the race lap by lap. Nothing at a
token depends on a later lap, so the model reads a race in progress exactly as it reads a
finished one.

**Heads**, per car and token (each about the next lap, except the last two):

* next-lap time: a Student-t on the lap's residual to the field (location, scale, and a
  learned degrees of freedom);
* P(pit next lap); P(retire before the end of the next lap);
* P(a neutralisation starts next lap) — per race, from the cars' mean state;
* P(classified | running) and a **Plackett-Luce strength** for the final classification.
  The finishing order is scored at every token, so the strength head is trained to
  forecast the result from any point of the race: the grid token gives a pre-race (C4)
  forecast, the lap-``k`` token the in-race (C5) one.

**Training** (:func:`train`): pretraining on races simulated by ``race_sim_v2`` across its
posterior (the simulator as the prior, a prior-fitted network), then fine-tuning on real
races, with early stopping on the most recent real races held out. One model per forecast
season, on the seasons before it only.

**Back ends.** The same network in MLX (the GPU on Apple silicon) and PyTorch
(:mod:`f1p.models.lapseq_torch`: MPS, or fast CPU elsewhere — MLX's Linux CPU build has no
optimised matrix kernels). ``backend="auto"`` takes MLX on a Metal GPU, else PyTorch if
installed. Weights are stored back-end neutral (NumPy arrays under MLX's parameter names),
so a network trained on one runs on the other; a parity test holds them to the same
outputs.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.special import expit

from f1p.core.logging import get_logger
from f1p.models.lapseq_data import F, RaceSeq, Tokens, tokens

log = get_logger("models.lapseq")

HEADS = ("mu", "log_sigma", "pit", "retire", "classified", "strength")  # per car, in order


@dataclass(frozen=True)
class LapSeqConfig:
    d: int = 128  # model width
    heads: int = 4
    blocks: int = 4
    mlp: int = 4  # feed-forward width, x d
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 200  # steps
    batch: int = 8  # races per step
    pretrain_epochs: int = 3
    finetune_epochs: int = 40
    patience: int = 5  # epochs without a better validation loss
    val_share: float = 0.15  # the most recent real races, for early stopping
    sims_per_race: int = 32  # pretraining: simulated races per real training race
    pl_weight: float = 1.0
    seed: int = 0
    backend: str = "auto"  # mlx, torch or auto

    def params(self) -> int:
        """Parameter count (embedding, blocks, heads)."""
        d, m = self.d, self.mlp
        attn = 2 * (4 * d * d + 4 * d)
        ff = 2 * m * d * d + m * d + d
        norms = 3 * 2 * d
        heads = (d * len(HEADS) + len(HEADS)) + (d + 1) + 1
        return (F * d + d) + self.blocks * (attn + ff + norms) + 2 * d + heads


SMALL = LapSeqConfig()  # 1.1M parameters
FULL = LapSeqConfig(d=256, heads=8, blocks=6, sims_per_race=256)  # 6.3M (plan §D.2)


def resolve_backend(cfg: LapSeqConfig) -> str:
    import importlib.util

    if cfg.backend in ("mlx", "torch"):
        return cfg.backend
    if cfg.backend != "auto":
        raise ValueError(f"unknown backend {cfg.backend!r}; choose mlx, torch or auto")
    if importlib.util.find_spec("mlx") is not None:
        import mlx.core as mx

        if mx.metal.is_available():
            return "mlx"
    if importlib.util.find_spec("torch") is not None:
        return "torch"
    return "mlx"


# -- batches ------------------------------------------------------------------------------------
def collate(batch: Sequence[Tokens]) -> dict[str, np.ndarray]:
    """Pad races to one array per input/target (padded cars and laps are not running); each
    target comes with its 0/1 mask."""
    b = len(batch)
    t = max(k.x.shape[0] for k in batch)
    n = max(k.x.shape[1] for k in batch)
    x = np.zeros((b, t, n, F), dtype=np.float32)
    running = np.zeros((b, t, n), dtype=bool)
    res = np.full((b, t, n), np.nan, dtype=np.float32)
    pit = np.full((b, t, n), np.nan, dtype=np.float32)
    out = np.full((b, t, n), np.nan, dtype=np.float32)
    neu = np.full((b, t), np.nan, dtype=np.float32)
    order = np.zeros((b, n), dtype=np.int32)
    order_ok = np.zeros((b, n), dtype=bool)
    cls = np.zeros((b, n), dtype=np.float32)
    for i, k in enumerate(batch):
        ti, ni = k.running.shape
        x[i, :ti, :ni] = k.x
        running[i, :ti, :ni] = k.running
        res[i, :ti, :ni] = k.next_res
        pit[i, :ti, :ni] = k.next_pit
        out[i, :ti, :ni] = k.next_out
        neu[i, :ti] = k.next_neutral
        fin = np.flatnonzero(k.finish >= 0)
        o = fin[np.argsort(k.finish[fin], kind="stable")]
        order[i, : o.size] = o
        order_ok[i, : o.size] = True
        cls[i, :ni] = k.finish >= 0
    arrays: dict[str, np.ndarray] = {
        "x": x,
        "running": running,
        "order": order,
        "order_ok": order_ok,
        "cls": cls,
    }
    for name, a in (("res", res), ("pit", pit), ("out", out), ("neu", neu)):
        ok = np.isfinite(a)
        arrays[name] = np.where(ok, a, 0.0).astype(np.float32)
        arrays[f"{name}_m"] = ok.astype(np.float32)
    return arrays


# -- the back-end interface ---------------------------------------------------------------------
class Backend(Protocol):
    """One network and its optimiser, in one array library."""

    name: str
    net: Any

    def step(self, batch: dict[str, np.ndarray]) -> float: ...

    def losses(self, batch: dict[str, np.ndarray]) -> dict[str, float]: ...

    def heads(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...

    def snapshot(self) -> Any: ...

    def restore(self, snap: Any) -> None: ...

    def set_training(self, on: bool) -> None: ...

    def export(self) -> dict[str, np.ndarray]: ...

    def load(self, weights: dict[str, np.ndarray]) -> None: ...


def make_backend(cfg: LapSeqConfig, total_steps: int = 1, device: str | None = None) -> Backend:
    """The configuration's back end; ``device`` ("cpu", or "gpu" for Metal/MPS) overrides the
    default (the GPU when there is one)."""
    if resolve_backend(cfg) == "torch":
        from f1p.models.lapseq_torch import TorchBackend

        return TorchBackend(cfg, total_steps, device)
    return MLXBackend(cfg, total_steps, device)


def lr_at(cfg: LapSeqConfig, step: int, total: int) -> float:
    """Linear warm-up, then cosine decay to 1/20 of the peak (both back ends)."""
    warm = min(cfg.warmup, max(total // 10, 1))
    if step < warm:
        return cfg.lr / 50.0 + (cfg.lr - cfg.lr / 50.0) * step / warm
    frac = min((step - warm) / max(total - warm, 1), 1.0)
    low = cfg.lr / 20.0
    return low + 0.5 * (cfg.lr - low) * (1.0 + math.cos(math.pi * frac))


# -- MLX ----------------------------------------------------------------------------------------
_LANCZOS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)


def lgamma(x: Any) -> Any:
    """log Γ(x) for x > 0.5 (Lanczos, g = 7; MLX has no lgamma)."""
    import mlx.core as mx

    x = x - 1.0
    a = mx.full(x.shape, _LANCZOS[0])
    for i, c in enumerate(_LANCZOS[1:], start=1):
        a = a + c / (x + i)
    t = x + 7.5
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * mx.log(t) - t + mx.log(a)


def build_network(cfg: LapSeqConfig) -> Any:
    """The MLX network."""
    import mlx.core as mx
    import mlx.nn as nn

    class Attention(nn.Module):
        def __init__(self, d: int, heads: int) -> None:
            super().__init__()
            self.heads = heads
            self.qkv = nn.Linear(d, 3 * d)
            self.out = nn.Linear(d, d)

        def __call__(self, x: Any, mask: Any) -> Any:
            b, length, d = x.shape
            h = self.heads
            q, k, v = mx.split(self.qkv(x), 3, axis=-1)
            q, k, v = (z.reshape(b, length, h, d // h).transpose(0, 2, 1, 3) for z in (q, k, v))
            # explicit masked attention, not the fused kernel: the masks broadcast over
            # singleton axes, and the GPU kernel's broadcasting is not the CPU's (the PyTorch
            # twin computes exactly this)
            scores = (q @ k.transpose(0, 1, 3, 2)) * (d // h) ** -0.5 + mask
            o = mx.softmax(scores, axis=-1, precise=True) @ v
            return self.out(o.transpose(0, 2, 1, 3).reshape(b, length, d))

    class Block(nn.Module):
        def __init__(self, c: LapSeqConfig) -> None:
            super().__init__()
            self.ln_t = nn.LayerNorm(c.d)
            self.time = Attention(c.d, c.heads)
            self.ln_c = nn.LayerNorm(c.d)
            self.cars = Attention(c.d, c.heads)
            self.ln_f = nn.LayerNorm(c.d)
            self.ff = nn.Sequential(
                nn.Linear(c.d, c.mlp * c.d),
                nn.GELU(),
                nn.Dropout(c.dropout),
                nn.Linear(c.mlp * c.d, c.d),
            )
            self.drop = nn.Dropout(c.dropout)

        def __call__(self, h: Any, time_mask: Any, car_mask: Any) -> Any:
            b, t, n, d = h.shape
            x = self.ln_t(h).transpose(0, 2, 1, 3).reshape(b * n, t, d)
            x = self.time(x, time_mask).reshape(b, n, t, d).transpose(0, 2, 1, 3)
            h = h + self.drop(x)
            y = self.cars(self.ln_c(h).reshape(b * t, n, d), car_mask).reshape(b, t, n, d)
            h = h + self.drop(y)
            return h + self.drop(self.ff(self.ln_f(h)))

    class LapSeqNet(nn.Module):
        def __init__(self, c: LapSeqConfig) -> None:
            super().__init__()
            self.embed = nn.Linear(F, c.d)
            self.blocks = [Block(c) for _ in range(c.blocks)]
            self.ln = nn.LayerNorm(c.d)
            self.car_head = nn.Linear(c.d, len(HEADS))
            self.race_head = nn.Linear(c.d, 1)  # a neutralisation starts
            self.nu_raw = mx.array([1.0])

        def __call__(self, x: Any, running: Any) -> dict[str, Any]:
            b, t, n, _ = x.shape
            neg = mx.array(-1e9, dtype=x.dtype)
            live = running.astype(x.dtype)
            # over laps: causal, and only tokens of the car while it runs
            causal = mx.triu(mx.full((t, t), -1e9, dtype=x.dtype), k=1)
            keys_t = mx.where(running.transpose(0, 2, 1), 0.0, neg).reshape(b * n, 1, 1, t)
            time_mask = causal[None, None] + keys_t
            # across cars: the cars running on that lap
            car_mask = mx.where(running, 0.0, neg).reshape(b * t, 1, 1, n)
            h = self.embed(x)
            for blk in self.blocks:
                h = blk(h, time_mask, car_mask)
            h = self.ln(h)
            out = self.car_head(h)
            pooled = (h * live[..., None]).sum(axis=2) / mx.maximum(live.sum(axis=2), 1.0)[
                ..., None
            ]
            heads = {k: out[..., i] for i, k in enumerate(HEADS)}
            heads["log_sigma"] = mx.clip(heads["log_sigma"], -8.0, 3.0)
            heads["neutral"] = self.race_head(pooled)[..., 0]
            heads["nu"] = 2.0 + nn.softplus(self.nu_raw)[0]
            return heads

    return LapSeqNet(cfg)


def _bce(logit: Any, y: Any) -> Any:
    import mlx.core as mx

    return mx.logaddexp(mx.zeros_like(logit), logit) - y * logit


def _masked_mean(v: Any, m: Any) -> Any:
    import mlx.core as mx

    return (v * m).sum() / mx.maximum(m.sum(), 1.0)


def student_t_nll(y: Any, mu: Any, log_sigma: Any, nu: Any) -> Any:
    import mlx.core as mx

    z = (y - mu) / mx.exp(log_sigma)
    return -(
        lgamma((nu + 1.0) / 2.0)
        - lgamma(nu / 2.0)
        - 0.5 * mx.log(nu * math.pi)
        - log_sigma
        - (nu + 1.0) / 2.0 * mx.log1p(z * z / nu)
    )


def pl_loss(strength: Any, order: Any, order_ok: Any, running: Any) -> Any:
    """Mean over (race, token) of the negative Plackett-Luce log-likelihood per finisher of
    the classification, among the classified cars running at the token."""
    import mlx.core as mx

    b, t, n = strength.shape
    idx = mx.broadcast_to(order[:, None, :], (b, t, n))
    s = mx.take_along_axis(strength, idx, axis=2)
    ok = order_ok[:, None, :] & mx.take_along_axis(running, idx, axis=2)
    okf = ok.astype(s.dtype)
    suffix = mx.logcumsumexp(mx.where(ok, s, -1e9), axis=2, reverse=True)
    terms = (s - suffix) * okf
    count = okf.sum(axis=2)
    per = terms.sum(axis=2) / mx.maximum(count, 1.0)
    use = (count >= 2).astype(s.dtype)
    return -(per * use).sum() / mx.maximum(use.sum(), 1.0)


def loss_terms(net: Any, batch: dict[str, Any], cfg: LapSeqConfig) -> dict[str, Any]:
    """The loss terms (MLX arrays) of one batch of MLX arrays."""
    out = net(batch["x"], batch["running"])
    run = batch["running"].astype(batch["x"].dtype)
    terms = {
        "time": _masked_mean(
            student_t_nll(batch["res"], out["mu"], out["log_sigma"], out["nu"]), batch["res_m"]
        ),
        "pit": _masked_mean(_bce(out["pit"], batch["pit"]), batch["pit_m"]),
        "retire": _masked_mean(_bce(out["retire"], batch["out"]), batch["out_m"]),
        "neutral": _masked_mean(_bce(out["neutral"], batch["neu"]), batch["neu_m"]),
        "classified": _masked_mean(_bce(out["classified"], batch["cls"][:, None, :]), run),
        "order": pl_loss(out["strength"], batch["order"], batch["order_ok"], batch["running"]),
    }
    terms["total"] = (
        terms["time"]
        + terms["pit"]
        + terms["retire"]
        + terms["neutral"]
        + terms["classified"]
        + cfg.pl_weight * terms["order"]
    )
    return terms


class MLXBackend:
    name = "mlx"

    def __init__(self, cfg: LapSeqConfig, total_steps: int, device: str | None = None) -> None:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim

        mx.random.seed(cfg.seed)
        self.cfg = cfg
        self.device = None if device is None else (mx.cpu if device == "cpu" else mx.gpu)
        self.net = build_network(cfg)
        mx.eval(self.net.parameters())
        self.opt = optim.AdamW(
            learning_rate=lambda step: mx.array(lr_at(cfg, int(np.asarray(step)), total_steps)),
            weight_decay=cfg.weight_decay,
        )

        def objective(model: Any, batch: dict[str, Any]) -> Any:
            return loss_terms(model, batch, cfg)["total"]

        self._grad = nn.value_and_grad(self.net, objective)

    @staticmethod
    def arrays(batch: dict[str, np.ndarray]) -> dict[str, Any]:
        import mlx.core as mx

        return {k: mx.array(v) for k, v in batch.items()}

    def _on(self) -> Any:
        import contextlib

        import mlx.core as mx

        return contextlib.nullcontext() if self.device is None else mx.stream(self.device)

    def step(self, batch: dict[str, np.ndarray]) -> float:
        import mlx.core as mx
        import mlx.optimizers as optim

        with self._on():
            loss, grads = self._grad(self.net, self.arrays(batch))
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            self.opt.update(self.net, grads)
            mx.eval(self.net.parameters(), self.opt.state, loss)
        return float(loss.item())

    def losses(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        import mlx.core as mx

        with self._on():
            terms = loss_terms(self.net, self.arrays(batch), self.cfg)
            mx.eval(terms)
        return {k: float(v.item()) for k, v in terms.items()}

    def heads(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        import mlx.core as mx

        with self._on():
            a = self.arrays(batch)
            out = self.net(a["x"], a["running"])
            mx.eval(out)
        return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}

    def snapshot(self) -> Any:
        return self.net.parameters()  # MLX updates are functional: the old arrays stay valid

    def restore(self, snap: Any) -> None:
        self.net.update(snap)

    def set_training(self, on: bool) -> None:
        self.net.train(on)

    def export(self) -> dict[str, np.ndarray]:
        from mlx.utils import tree_flatten

        return {k: np.array(v) for k, v in tree_flatten(self.net.parameters())}

    def load(self, weights: dict[str, np.ndarray]) -> None:
        import mlx.core as mx

        self.net.load_weights([(k, mx.array(v)) for k, v in weights.items()])


# -- training (back-end neutral) -------------------------------------------------------------------
@dataclass
class TrainLog:
    phase: str
    epoch: int
    train: float
    val: float | None
    seconds: float


@dataclass
class Trained:
    backend: Backend
    config: LapSeqConfig
    log: list[TrainLog] = field(default_factory=list)
    val_terms: dict[str, float] = field(default_factory=dict)

    @property
    def net(self) -> Any:
        return self.backend.net


def _batches(items: Sequence[Tokens], size: int, rng: np.random.Generator) -> list[list[Tokens]]:
    idx = rng.permutation(len(items))
    return [[items[i] for i in idx[j : j + size]] for j in range(0, len(items), size)]


def evaluate(backend: Backend, items: Sequence[Tokens], batch: int = 8) -> dict[str, float]:
    """Mean loss terms over ``items`` (weighted by race; dropout off)."""
    backend.set_training(False)
    sums: dict[str, float] = {}
    for j in range(0, len(items), batch):
        chunk = items[j : j + batch]
        for k, v in backend.losses(collate(chunk)).items():
            sums[k] = sums.get(k, 0.0) + v * len(chunk)
    return {k: v / max(len(items), 1) for k, v in sums.items()}


def train(
    real: Sequence[RaceSeq],
    simulated: Sequence[RaceSeq] = (),
    cfg: LapSeqConfig = SMALL,
    *,
    progress: Callable[[TrainLog], None] | None = None,
) -> Trained:
    """Pretrain on ``simulated`` races (if any), then fine-tune on ``real`` ones (given in
    time order; the last ``val_share`` are held out for early stopping, and the network
    keeps the weights of its best validation epoch)."""
    rng = np.random.default_rng(cfg.seed)
    real_tok = [tokens(r) for r in real]
    n_val = max(1, round(cfg.val_share * len(real_tok))) if len(real_tok) > 1 else 0
    fit, val = real_tok[: len(real_tok) - n_val], real_tok[len(real_tok) - n_val :]
    sim_tok = [tokens(r) for r in simulated]
    steps = cfg.pretrain_epochs * math.ceil(len(sim_tok) / cfg.batch) + cfg.finetune_epochs * (
        math.ceil(len(fit) / cfg.batch)
    )
    be = make_backend(cfg, steps)
    out = Trained(be, cfg)

    def run_epoch(phase: str, e: int, items: Sequence[Tokens]) -> float:
        t0 = time.perf_counter()
        be.set_training(True)
        total = 0.0
        for chunk in _batches(items, cfg.batch, rng):
            total += be.step(collate(chunk)) * len(chunk)
        tr = total / max(len(items), 1)
        v = evaluate(be, val, cfg.batch)["total"] if val else None
        entry = TrainLog(phase, e, tr, v, time.perf_counter() - t0)
        out.log.append(entry)
        log.info("lapseq %s epoch %d: train %.4f val %s", phase, e, tr, v)
        if progress is not None:
            progress(entry)
        return tr if v is None else v

    for e in range(cfg.pretrain_epochs if sim_tok else 0):
        run_epoch("pretrain", e, sim_tok)
    best, snap, stale = math.inf, None, 0
    for e in range(cfg.finetune_epochs if fit else 0):
        v = run_epoch("finetune", e, fit)
        if v < best - 1e-4:
            best, stale, snap = v, 0, be.snapshot()
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if snap is not None:
        be.restore(snap)
    be.set_training(False)
    if val:
        out.val_terms = evaluate(be, val, cfg.batch)
    return out


# -- prediction ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Prediction:
    strength: np.ndarray  # (n,) Plackett-Luce log-strength at the last token
    p_classified: np.ndarray  # (n,) given running
    running: np.ndarray  # (n,)
    mu: np.ndarray  # (n,) next-lap residual location (x10 scale)
    sigma: np.ndarray
    nu: float
    p_pit: np.ndarray
    p_retire: np.ndarray
    p_neutral: float


def predict(backend: Backend, tok: Tokens) -> Prediction:
    """The heads at the last token of ``tok`` (one race)."""
    backend.set_training(False)
    out = backend.heads(collate([tok]))
    last = tok.x.shape[0] - 1

    def at(k: str) -> np.ndarray:
        return out[k][0, last]

    return Prediction(
        at("strength"),
        expit(at("classified")),
        tok.running[last].copy(),
        at("mu"),
        np.exp(at("log_sigma")),
        float(np.asarray(out["nu"]).reshape(-1)[0]),
        expit(at("pit")),
        expit(at("retire")),
        float(expit(out["neutral"][0, last])),
    )


# -- persistence --------------------------------------------------------------------------------
def save(trained: Trained, root: Path, meta: dict[str, Any]) -> Path:
    """Weights (back-end neutral ``.npz``) and the configuration, log and metadata."""
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / "weights.tmp.npz"
    np.savez(tmp, allow_pickle=False, **trained.backend.export())
    tmp.replace(root / "weights.npz")
    (root / "meta.json").write_text(
        json.dumps(
            {
                "config": asdict(trained.config),
                "trained_on": trained.backend.name,
                "log": [asdict(x) for x in trained.log],
                "val_terms": trained.val_terms,
                **meta,
            },
            indent=1,
        )
    )
    return root


def load(root: Path, backend: str | None = None) -> tuple[Backend, LapSeqConfig, dict[str, Any]]:
    """A saved network on ``backend`` (default: the configuration's, resolved here)."""
    meta = json.loads((root / "meta.json").read_text())
    cfg = LapSeqConfig(**meta["config"])
    if backend is not None:
        cfg = LapSeqConfig(**{**asdict(cfg), "backend": backend})
    be = make_backend(cfg)
    with np.load(root / "weights.npz") as w:
        be.load({k: w[k] for k in w.files})
    be.set_training(False)
    return be, cfg, meta


class FileLock:
    """An exclusive lock on ``path`` (worker processes training the same season wait for the
    first, then load its model)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w")
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc: object) -> None:
        fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._fh.close()


def digest(*parts: Any) -> str:
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]
