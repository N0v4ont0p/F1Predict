"""``lapseq`` — a sequence model over laps (plan M6 §D.2), in MLX.

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
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from f1p.core.logging import get_logger
from f1p.models.lapseq_data import F, RaceSeq, Tokens, tokens

log = get_logger("models.lapseq")


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

    def params(self) -> int:
        """Parameter count (embedding, blocks, heads)."""
        d, m = self.d, self.mlp
        attn = 2 * (4 * d * d + 4 * d)
        ff = 2 * m * d * d + m * d + d
        norms = 3 * 2 * d
        return (F * d + d) + self.blocks * (attn + ff + norms) + 2 * d + (d * 6 + 6) + d + 1 + 1


SMALL = LapSeqConfig()
FULL = LapSeqConfig(d=256, heads=8, blocks=6, sims_per_race=256)  # 6.3M parameters (plan §D.2)


# -- the network (MLX) --------------------------------------------------------------------------
def _mlx() -> Any:
    import mlx.core as mx

    return mx


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
    mx = _mlx()
    x = x - 1.0
    a = mx.full(x.shape, _LANCZOS[0])
    for i, c in enumerate(_LANCZOS[1:], start=1):
        a = a + c / (x + i)
    t = x + 7.5
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * mx.log(t) - t + mx.log(a)


def build_network(cfg: LapSeqConfig) -> Any:
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
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=(d // h) ** -0.5, mask=mask)
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
            self.car_head = nn.Linear(c.d, 6)  # mu, log sigma, pit, retire, classified, strength
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
            return {
                "mu": out[..., 0],
                "log_sigma": mx.clip(out[..., 1], -8.0, 3.0),
                "pit": out[..., 2],
                "retire": out[..., 3],
                "classified": out[..., 4],
                "strength": out[..., 5],
                "neutral": self.race_head(pooled)[..., 0],
                "nu": 2.0 + nn.softplus(self.nu_raw)[0],
            }

    return LapSeqNet(cfg)


# -- batches and the loss -----------------------------------------------------------------------
def collate(batch: Sequence[Tokens]) -> dict[str, Any]:
    """Pad races to one array per input/target (padded cars and laps are not running)."""
    mx = _mlx()
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

    def pair(a: np.ndarray) -> tuple[Any, Any]:
        ok = np.isfinite(a)
        return mx.array(np.where(ok, a, 0.0)), mx.array(ok.astype(np.float32))

    res_v, res_m = pair(res)
    pit_v, pit_m = pair(pit)
    out_v, out_m = pair(out)
    neu_v, neu_m = pair(neu)
    return {
        "x": mx.array(x),
        "running": mx.array(running),
        "res": res_v,
        "res_m": res_m,
        "pit": pit_v,
        "pit_m": pit_m,
        "out": out_v,
        "out_m": out_m,
        "neu": neu_v,
        "neu_m": neu_m,
        "order": mx.array(order),
        "order_ok": mx.array(order_ok),
        "cls": mx.array(cls),
    }


def _bce(logit: Any, y: Any) -> Any:
    mx = _mlx()
    return mx.logaddexp(mx.zeros_like(logit), logit) - y * logit


def _masked_mean(v: Any, m: Any) -> Any:
    mx = _mlx()
    return (v * m).sum() / mx.maximum(m.sum(), 1.0)


def student_t_nll(y: Any, mu: Any, log_sigma: Any, nu: Any) -> Any:
    mx = _mlx()
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
    mx = _mlx()
    b, t, n = strength.shape
    idx = mx.broadcast_to(order[:, None, :], (b, t, n))
    s = mx.take_along_axis(strength, idx, axis=2)
    ok = order_ok[:, None, :] & mx.take_along_axis(running, idx, axis=2)
    okf = ok.astype(s.dtype)
    s_masked = mx.where(ok, s, -1e9)
    suffix = mx.logcumsumexp(s_masked, axis=2, reverse=True)
    terms = (s - suffix) * okf
    count = okf.sum(axis=2)
    per = terms.sum(axis=2) / mx.maximum(count, 1.0)
    use = (count >= 2).astype(s.dtype)
    return -(per * use).sum() / mx.maximum(use.sum(), 1.0)


def loss_terms(net: Any, batch: dict[str, Any], cfg: LapSeqConfig) -> dict[str, Any]:
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


# -- training -----------------------------------------------------------------------------------
@dataclass
class TrainLog:
    phase: str
    epoch: int
    train: float
    val: float | None
    seconds: float


@dataclass
class Trained:
    net: Any
    config: LapSeqConfig
    log: list[TrainLog] = field(default_factory=list)
    val_terms: dict[str, float] = field(default_factory=dict)


def _batches(items: Sequence[Tokens], size: int, rng: np.random.Generator) -> list[list[Tokens]]:
    idx = rng.permutation(len(items))
    return [[items[i] for i in idx[j : j + size]] for j in range(0, len(items), size)]


def evaluate(net: Any, items: Sequence[Tokens], cfg: LapSeqConfig) -> dict[str, float]:
    """Mean loss terms over ``items`` (weighted by race; dropout off)."""
    mx = _mlx()
    net.eval()
    sums: dict[str, float] = {}
    for j in range(0, len(items), cfg.batch):
        chunk = items[j : j + cfg.batch]
        terms = loss_terms(net, collate(chunk), cfg)
        mx.eval(terms)
        for k, v in terms.items():
            sums[k] = sums.get(k, 0.0) + float(v.item()) * len(chunk)
    net.train()
    return {k: v / max(len(items), 1) for k, v in sums.items()}


def train(
    real: Sequence[RaceSeq],
    simulated: Sequence[RaceSeq] = (),
    cfg: LapSeqConfig = SMALL,
) -> Trained:
    """Pretrain on ``simulated`` races (if any), then fine-tune on ``real`` ones (given in
    time order; the last ``val_share`` are held out for early stopping)."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    net = build_network(cfg)
    mx.eval(net.parameters())
    real_tok = [tokens(r) for r in real]
    n_val = max(1, round(cfg.val_share * len(real_tok))) if len(real_tok) > 1 else 0
    fit, val = real_tok[: len(real_tok) - n_val], real_tok[len(real_tok) - n_val :]
    sim_tok = [tokens(r) for r in simulated]
    steps = cfg.pretrain_epochs * math.ceil(len(sim_tok) / cfg.batch) + cfg.finetune_epochs * (
        math.ceil(len(fit) / cfg.batch)
    )
    warm = min(cfg.warmup, max(steps // 10, 1))
    schedule = optim.join_schedules(
        [
            optim.linear_schedule(cfg.lr / 50.0, cfg.lr, warm),
            optim.cosine_decay(cfg.lr, max(steps - warm, 1), cfg.lr / 20.0),
        ],
        [warm],
    )
    opt = optim.AdamW(learning_rate=schedule, weight_decay=cfg.weight_decay)
    net.train()

    def objective(model: Any, batch: dict[str, Any]) -> Any:
        return loss_terms(model, batch, cfg)["total"]

    step_fn = nn.value_and_grad(net, objective)

    def epoch(items: Sequence[Tokens]) -> float:
        total, count = 0.0, 0
        for chunk in _batches(items, cfg.batch, rng):
            loss, grads = step_fn(net, collate(chunk))
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            opt.update(net, grads)
            mx.eval(net.parameters(), opt.state, loss)
            total += float(loss.item()) * len(chunk)
            count += len(chunk)
        return total / max(count, 1)

    out = Trained(net, cfg)
    for e in range(cfg.pretrain_epochs if sim_tok else 0):
        t0 = time.perf_counter()
        tr = epoch(sim_tok)
        v = evaluate(net, val, cfg)["total"] if val else None
        out.log.append(TrainLog("pretrain", e, tr, v, time.perf_counter() - t0))
        log.info("lapseq pretrain epoch %d: train %.4f val %s", e, tr, v)
    best, best_params, stale = math.inf, None, 0
    for e in range(cfg.finetune_epochs if fit else 0):
        t0 = time.perf_counter()
        tr = epoch(fit)
        v = evaluate(net, val, cfg)["total"] if val else tr
        out.log.append(TrainLog("finetune", e, tr, v, time.perf_counter() - t0))
        log.info("lapseq finetune epoch %d: train %.4f val %.4f", e, tr, v)
        if v < best - 1e-4:
            best, stale = v, 0
            best_params = _copy(net.parameters())
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_params is not None:
        net.update(best_params)
    net.eval()
    if val:
        out.val_terms = evaluate(net, val, cfg)
        net.eval()
    return out


def _copy(params: Any) -> Any:
    from mlx.utils import tree_map

    mx = _mlx()
    return tree_map(lambda a: mx.array(a) if hasattr(a, "shape") else a, params)


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


def predict(net: Any, tok: Tokens) -> Prediction:
    """The heads at the last token of ``tok`` (one race)."""
    mx = _mlx()
    from scipy.special import expit

    net.eval()
    b = collate([tok])
    out = net(b["x"], b["running"])
    mx.eval(out)
    last = tok.x.shape[0] - 1

    def at(k: str) -> np.ndarray:
        return np.asarray(out[k][0, last], dtype=np.float64)

    return Prediction(
        at("strength"),
        expit(at("classified")),
        tok.running[last].copy(),
        at("mu"),
        np.exp(at("log_sigma")),
        float(out["nu"].item()),
        expit(at("pit")),
        expit(at("retire")),
        float(expit(np.asarray(out["neutral"][0, last]))),
    )


# -- persistence --------------------------------------------------------------------------------
def save(trained: Trained, root: Path, meta: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    trained.net.save_weights(str(root / "weights.safetensors"))
    (root / "meta.json").write_text(
        json.dumps(
            {
                "config": asdict(trained.config),
                "log": [asdict(x) for x in trained.log],
                "val_terms": trained.val_terms,
                **meta,
            },
            indent=1,
        )
    )
    return root


def load(root: Path) -> tuple[Any, LapSeqConfig, dict[str, Any]]:
    meta = json.loads((root / "meta.json").read_text())
    cfg = LapSeqConfig(**meta["config"])
    net = build_network(cfg)
    net.load_weights(str(root / "weights.safetensors"))
    net.eval()
    return net, cfg, meta


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
