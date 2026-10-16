"""The ``lapseq`` network in PyTorch (plan §D.2's fallback) — the twin of the MLX network in
:mod:`f1p.models.lapseq`, layer for layer, under the same parameter names (MLX's
``ff.layers.i`` is PyTorch's ``ff.i``), so weights move between the two.

It runs on the **CPU**. On the macOS CI runner, PyTorch's MPS device gave heads up to 1.0
away from PyTorch's own CPU on the same weights, while MLX on Metal matched its CPU to 2e-4
(``test_gpu_matches_cpu``). The Apple GPU path is therefore MLX (the ``auto`` choice there);
MPS is refused until it computes what the CPU does."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import torch
import torch.nn.functional as tf
from torch import nn

from f1p.models.lapseq import HEADS, LapSeqConfig, lr_at
from f1p.models.lapseq_data import F


def device(dev: str | None = None) -> torch.device:
    if dev in (None, "cpu"):
        return torch.device("cpu")
    raise ValueError(
        f"the PyTorch back end runs on the CPU (asked for {dev!r}): its MPS results disagree "
        "with its CPU results; use backend='mlx' for the Apple GPU"
    )


class Attention(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, length, d = x.shape
        h = self.heads
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (z.reshape(b, length, h, d // h).transpose(1, 2) for z in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) * (d // h) ** -0.5 + mask  # as the MLX twin
        o = scores.softmax(-1) @ v
        return self.out(o.transpose(1, 2).contiguous().reshape(b, length, d))


class SSM(nn.Module):
    """The selective state-space layer over laps (the MLX twin's, parameter for parameter):
    input-dependent Δ, B, C; diagonal state; causal depthwise convolution; SiLU gating."""

    def __init__(self, c: LapSeqConfig) -> None:
        super().__init__()
        e, n, r, k = c.ssm_inner, c.ssm_state, c.dt_rank, c.ssm_conv
        self.k, self.e, self.n, self.r = k, e, n, r
        self.in_proj = nn.Linear(c.d, 2 * e)
        self.conv_w = nn.Parameter(torch.empty(e, k).uniform_(-(k**-0.5), k**-0.5))
        self.conv_b = nn.Parameter(torch.zeros(e))
        self.x_proj = nn.Linear(e, r + 2 * n, bias=False)
        self.dt_proj = nn.Linear(r, e)
        with torch.no_grad():
            dt = torch.exp(torch.empty(e).uniform_(math.log(1e-3), math.log(1e-1)))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        a = torch.arange(1, n + 1, dtype=torch.float32).expand(e, n)
        self.A_log = nn.Parameter(torch.log(a).clone())
        self.D = nn.Parameter(torch.ones(e))
        self.out_proj = nn.Linear(e, c.d)

    def _mix(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(Δ, B, C) from the convolved input."""
        proj = self.x_proj(u)
        dt = tf.softplus(self.dt_proj(proj[..., : self.r]))
        return dt, proj[..., self.r : self.r + self.n], proj[..., self.r + self.n :]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        del mask  # causal by construction
        b, t, _ = x.shape
        u, z = self.in_proj(x).split(self.e, dim=-1)
        pad = torch.cat([u.new_zeros(b, self.k - 1, self.e), u], dim=1)
        conv = self.conv_b + sum(pad[:, j : j + t, :] * self.conv_w[:, j] for j in range(self.k))
        u = tf.silu(conv)
        dt, bm, cm = self._mix(u)
        a = -torch.exp(self.A_log)
        h = u.new_zeros(b, self.e, self.n)
        ys = []
        for i in range(t):
            da = torch.exp(dt[:, i, :, None] * a[None])
            h = da * h + dt[:, i, :, None] * bm[:, i, None, :] * u[:, i, :, None]
            ys.append((h * cm[:, i, None, :]).sum(-1))
        y = torch.stack(ys, dim=1) + u * self.D
        return self.out_proj(y * tf.silu(z))

    def step(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """One lap: ``x`` (b, d) and the state (the last ``k-1`` inputs, the hidden state)
        → (output (b, d), new state). Iterating it equals :meth:`forward` over the laps —
        the live-forecasting form: one update per lap, no re-reading of the race."""
        b = x.shape[0]
        if state is None:
            state = (x.new_zeros(b, self.k - 1, self.e), x.new_zeros(b, self.e, self.n))
        window, h = state
        u, z = self.in_proj(x).split(self.e, dim=-1)
        window = torch.cat([window, u[:, None, :]], dim=1)  # (b, k, e)
        conv = self.conv_b + (window * self.conv_w.T[None]).sum(1)
        u = tf.silu(conv)
        dt, bm, cm = self._mix(u)
        a = -torch.exp(self.A_log)
        h = torch.exp(dt[:, :, None] * a[None]) * h + dt[:, :, None] * bm[:, None, :] * u[..., None]
        y = (h * cm[:, None, :]).sum(-1) + u * self.D
        return self.out_proj(y * tf.silu(z)), (window[:, 1:], h)


class Block(nn.Module):
    def __init__(self, c: LapSeqConfig) -> None:
        super().__init__()
        self.ln_t = nn.LayerNorm(c.d)
        self.time = SSM(c) if c.temporal == "ssm" else Attention(c.d, c.heads)
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

    def forward(
        self, h: torch.Tensor, time_mask: torch.Tensor, car_mask: torch.Tensor
    ) -> torch.Tensor:
        b, t, n, d = h.shape
        # contiguous before every reshape of a transposed view (MPS has had view bugs)
        x = self.ln_t(h).transpose(1, 2).contiguous().reshape(b * n, t, d)
        x = self.time(x, time_mask).reshape(b, n, t, d).transpose(1, 2).contiguous()
        h = h + self.drop(x)
        y = self.cars(self.ln_c(h).reshape(b * t, n, d), car_mask).reshape(b, t, n, d)
        h = h + self.drop(y)
        return h + self.drop(self.ff(self.ln_f(h)))


class TorchNet(nn.Module):
    def __init__(self, c: LapSeqConfig) -> None:
        super().__init__()
        self.embed = nn.Linear(F, c.d)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.blocks)])
        self.ln = nn.LayerNorm(c.d)
        self.car_head = nn.Linear(c.d, len(HEADS))
        self.race_head = nn.Linear(c.d, 1)
        self.nu_raw = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x: torch.Tensor, running: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t, n, _ = x.shape
        neg = torch.tensor(-1e9, dtype=x.dtype, device=x.device)
        zero = torch.zeros((), dtype=x.dtype, device=x.device)
        live = running.to(x.dtype)
        causal = torch.triu(torch.full((t, t), -1e9, dtype=x.dtype, device=x.device), 1)
        keys_t = torch.where(running.transpose(1, 2).contiguous(), zero, neg).reshape(
            b * n, 1, 1, t
        )
        time_mask = causal[None, None] + keys_t
        car_mask = torch.where(running, zero, neg).reshape(b * t, 1, 1, n)
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, time_mask, car_mask)
        h = self.ln(h)
        out = self.car_head(h)
        pooled = (h * live[..., None]).sum(2) / live.sum(2).clamp_min(1.0)[..., None]
        heads = {k: out[..., i] for i, k in enumerate(HEADS)}
        heads["log_sigma"] = heads["log_sigma"].clamp(-8.0, 3.0)
        heads["neutral"] = self.race_head(pooled)[..., 0]
        heads["nu"] = 2.0 + tf.softplus(self.nu_raw)[0]
        return heads


# -- losses (as in the MLX twin) ------------------------------------------------------------------
def _bce(logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return tf.softplus(logit) - y * logit


def _masked_mean(v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (v * m).sum() / m.sum().clamp_min(1.0)


def student_t_nll(
    y: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor, nu: torch.Tensor
) -> torch.Tensor:
    z = (y - mu) / log_sigma.exp()
    return -(
        torch.lgamma((nu + 1.0) / 2.0)
        - torch.lgamma(nu / 2.0)
        - 0.5 * torch.log(nu * math.pi)
        - log_sigma
        - (nu + 1.0) / 2.0 * torch.log1p(z * z / nu)
    )


def pl_loss(
    strength: torch.Tensor, order: torch.Tensor, order_ok: torch.Tensor, running: torch.Tensor
) -> torch.Tensor:
    b, t, n = strength.shape
    idx = order[:, None, :].expand(b, t, n).long()
    s = strength.gather(2, idx)
    ok = order_ok[:, None, :] & running.gather(2, idx)
    okf = ok.to(s.dtype)
    masked = torch.where(ok, s, torch.full_like(s, -1e9))
    suffix = masked.flip(2).logcumsumexp(2).flip(2)
    terms = (s - suffix) * okf
    count = okf.sum(2)
    per = terms.sum(2) / count.clamp_min(1.0)
    use = (count >= 2).to(s.dtype)
    return -(per * use).sum() / use.sum().clamp_min(1.0)


def loss_terms(
    net: TorchNet, batch: dict[str, torch.Tensor], cfg: LapSeqConfig
) -> dict[str, torch.Tensor]:
    out = net(batch["x"], batch["running"])
    run = batch["running"].to(batch["x"].dtype)
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


# -- parameter names shared with MLX ---------------------------------------------------------------
def to_mlx_name(name: str) -> str:
    return re.sub(r"\.ff\.(\d+)\.", r".ff.layers.\1.", name)


def from_mlx_name(name: str) -> str:
    return re.sub(r"\.ff\.layers\.(\d+)\.", r".ff.\1.", name)


class TorchBackend:
    name = "torch"

    def __init__(self, cfg: LapSeqConfig, total_steps: int, dev: str | None = None) -> None:
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.device = device(dev)
        self.net = TorchNet(cfg).to(self.device)
        self.opt = torch.optim.AdamW(
            self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.total = total_steps
        self.steps = 0

    def tensors(self, batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {k: torch.from_numpy(v).to(self.device) for k, v in batch.items()}

    def step(self, batch: dict[str, np.ndarray]) -> float:
        for group in self.opt.param_groups:
            group["lr"] = lr_at(self.cfg, self.steps, self.total)
        self.opt.zero_grad(set_to_none=True)
        loss = loss_terms(self.net, self.tensors(batch), self.cfg)["total"]
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        self.steps += 1
        return float(loss.detach().cpu())

    def losses(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        with torch.no_grad():
            terms = loss_terms(self.net, self.tensors(batch), self.cfg)
        return {k: float(v.cpu()) for k, v in terms.items()}

    def heads(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        t = self.tensors(batch)
        with torch.no_grad():
            out = self.net(t["x"], t["running"])
        return {k: v.cpu().double().numpy() for k, v in out.items()}

    def snapshot(self) -> Any:
        return {k: v.detach().clone() for k, v in self.net.state_dict().items()}

    def restore(self, snap: Any) -> None:
        self.net.load_state_dict(snap)

    def set_training(self, on: bool) -> None:
        self.net.train(on)

    def export(self) -> dict[str, np.ndarray]:
        return {to_mlx_name(k): v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}

    def load(self, weights: dict[str, np.ndarray]) -> None:
        state = {from_mlx_name(k): torch.from_numpy(np.asarray(v)) for k, v in weights.items()}
        self.net.load_state_dict(state)
        self.net.to(self.device)
