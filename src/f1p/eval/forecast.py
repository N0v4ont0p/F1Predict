"""The common output of every model: a probability distribution over finishing orders.

A :class:`Forecast` stores, for each driver in the field, the probability of finishing in each
position (``position_probs[i, k]`` = P(driver i finishes k+1-th)). Plackett-Luce models also
keep their log-strengths so the exact likelihood of an observed order can be scored.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

DEFAULT_SAMPLES = 20_000

FloatArray = NDArray[np.float64]


def stable_seed(*parts: str) -> int:
    """A reproducible seed per forecast, so Monte Carlo noise is not shared across events."""
    return zlib.crc32("|".join(parts).encode())


def pl_sample_orders(
    log_strength: FloatArray, n_samples: int, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Sample finishing orders from a Plackett-Luce model (Gumbel-max trick).

    Returns an ``(n_samples, n)`` array; row ``s`` lists driver indices from P1 to last.
    """
    gumbel = rng.gumbel(size=(n_samples, log_strength.size))
    return np.argsort(-(log_strength[None, :] + gumbel), axis=1)


def pl_sample_orders_with_retirements(
    log_strength: FloatArray, p_finish: FloatArray, n_samples: int, rng: np.random.Generator
) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    """Sample (orders, finished): each car is classified with probability ``p_finish``; the
    classified cars are ordered by Plackett-Luce and the retirements fill the slots behind."""
    finished = rng.random((n_samples, log_strength.size)) < p_finish[None, :]
    gumbel = rng.gumbel(size=(n_samples, log_strength.size))
    key = log_strength[None, :] + gumbel - np.where(finished, 0.0, 1e9)
    return np.argsort(-key, axis=1), finished


def rao_blackwell_win(log_strength: FloatArray, finished: NDArray[np.bool_]) -> FloatArray:
    """P(win) averaged exactly over the Plackett-Luce race given each sampled finisher set —
    far less Monte Carlo noise than counting sampled winners."""
    e = np.exp(log_strength - log_strength.max())
    num = finished * e[None, :]
    tot = num.sum(axis=1, keepdims=True)
    ok = tot[:, 0] > 0
    if not ok.any():
        return np.full(log_strength.size, 1.0 / log_strength.size)
    p = (num[ok] / tot[ok]).mean(axis=0)
    return p / p.sum()


def position_matrix(orders: NDArray[np.intp], n: int) -> FloatArray:
    """Empirical P(driver i finishes position k) from sampled orders."""
    counts = np.zeros((n, n), dtype=np.float64)
    positions = np.broadcast_to(np.arange(n), orders.shape)
    np.add.at(counts, (orders, positions), 1.0)
    return counts / orders.shape[0]


def pl_win_probabilities(log_strength: FloatArray) -> FloatArray:
    """Exact P(win) under Plackett-Luce: softmax of the log-strengths."""
    z = log_strength - log_strength.max()
    e = np.exp(z)
    return e / e.sum()


def pl_log_likelihood(log_strength: FloatArray, order: NDArray[np.intp]) -> float:
    """Exact log-probability of a (partial) order under Plackett-Luce.

    ``order`` lists the indices of the drivers whose relative order was observed (e.g. the
    classified finishers), best first. Plackett-Luce is closed under marginalisation, so the
    probability of a sub-ranking only involves the drivers in it — retirements are censored.
    """
    s = log_strength[order]
    # log-sum-exp of each suffix s[i:], computed stably from the back.
    suffix = np.logaddexp.accumulate(s[::-1])[::-1]
    return float(np.sum(s - suffix))


@dataclass(frozen=True, slots=True)
class Forecast:
    event_id: str
    checkpoint: str
    model: str
    drivers: tuple[str, ...]
    position_probs: FloatArray
    log_strength: FloatArray | None = None
    meta: dict[str, float] = field(default_factory=dict)
    win_probs: FloatArray | None = None  # exact/Rao-Blackwellised P(win) when available
    p_finish: FloatArray | None = None  # P(classified) when the model forecasts retirements

    def __post_init__(self) -> None:
        n = len(self.drivers)
        if self.position_probs.shape != (n, n):
            raise ValueError(f"position_probs must be {(n, n)}, got {self.position_probs.shape}")
        if not np.allclose(self.position_probs.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("each driver's position probabilities must sum to 1")
        if not np.allclose(self.position_probs.sum(axis=0), 1.0, atol=1e-6):
            raise ValueError("each position must be filled with probability 1")

    @property
    def n(self) -> int:
        return len(self.drivers)

    @property
    def p_win(self) -> FloatArray:
        if self.win_probs is not None:
            return self.win_probs
        if self.log_strength is not None:
            return pl_win_probabilities(self.log_strength)
        return self.position_probs[:, 0]

    def p_top(self, k: int) -> FloatArray:
        return self.position_probs[:, : min(k, self.n)].sum(axis=1)

    @property
    def expected_position(self) -> FloatArray:
        return self.position_probs @ np.arange(1, self.n + 1, dtype=np.float64)

    @classmethod
    def from_log_strength(
        cls,
        event_id: str,
        checkpoint: str,
        model: str,
        drivers: tuple[str, ...],
        log_strength: FloatArray,
        *,
        n_samples: int = DEFAULT_SAMPLES,
        seed: int | None = None,
        meta: dict[str, float] | None = None,
        p_finish: FloatArray | None = None,
    ) -> Forecast:
        """Plackett-Luce forecast; with ``p_finish`` each car may also retire (classified
        cars are ordered by Plackett-Luce, retirements fill the slots behind them)."""
        if seed is None:
            seed = stable_seed(event_id, checkpoint, model)
        rng = np.random.default_rng(seed)
        ls = np.asarray(log_strength, dtype=np.float64)
        win = None
        pf = None
        if p_finish is None:
            orders = pl_sample_orders(ls, n_samples, rng)
        else:
            pf = np.asarray(p_finish, dtype=np.float64)
            orders, finished = pl_sample_orders_with_retirements(ls, pf, n_samples, rng)
            win = rao_blackwell_win(ls, finished)
        return cls(
            event_id,
            checkpoint,
            model,
            drivers,
            position_matrix(orders, len(drivers)),
            ls,
            meta or {},
            win,
            pf,
        )
