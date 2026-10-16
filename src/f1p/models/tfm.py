"""A tabular foundation model as a stack member (plan M6 §D.1): TabICL, in-context learning
on the shared feature table of the challengers.

Licence check (plan §D.1, done first): **TabICL** — code and weights — is BSD-3-Clause
(Inria Soda team), so it is the member here. TabPFN-2.5 and later weights are licensed
non-commercially (no production use; gated), so they are not used by default.

TFMs predict per row; a race is an order, so two framings, both evaluated:

* **pointwise** — regression on each car's normalised finishing score (winner 1, last
  classified ~1/m, retirements 0);
* **pairwise** — classification of "car A beats car B" from the feature difference, then a
  Bradley-Terry fit per race turns the pair probabilities into strengths (suits a
  classification-native TFM).

Everything else is the challengers' pipeline (:class:`challengers.RankerModel`): the causal
feature table, a "fit" per season on seasons S-8..S-1 (for a TFM: the context), and the
out-of-sample score → Plackett-Luce strength mapping fitted on the latest season it did not
see. The pairwise context keeps the most recent races (pairs grow as n²).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from f1p.eval.context import EventContext
from f1p.eval.metrics import Outcome
from f1p.models.challengers import RankerModel, RankSignal, relevance

FloatArray = NDArray[np.float64]
Rows = list[tuple[EventContext, RankSignal, Outcome]]


def bradley_terry(p: FloatArray, iters: int = 200) -> FloatArray:
    """Log-strengths ``s`` (centred) maximising sum_{i!=j} p_ij log sigmoid(s_i - s_j): the
    Bradley-Terry fit to a matrix of win probabilities (Newton steps on a concave
    objective; ``p[i, j]`` = P(i beats j), with ``p[j, i] = 1 - p[i, j]``)."""
    n = p.shape[0]
    s = np.zeros(n)
    off = ~np.eye(n, dtype=bool)
    for _ in range(iters):
        q = expit(s[:, None] - s[None, :])
        # gradient: dL/ds_k = 2 sum_j (p_kj - q_kj); Hessian: off-diagonal +2 q(1-q),
        # diagonal minus the row sums (negative semi-definite)
        g = 2.0 * np.where(off, p - q, 0.0).sum(axis=1)
        w = np.where(off, q * (1.0 - q), 0.0)
        h = w + w.T
        np.fill_diagonal(h, -(w + w.T).sum(axis=1))
        h -= 1e-6 * np.eye(n)  # the common level is unidentified: a tiny ridge
        step = np.linalg.solve(h, -g)
        s = s + step
        s -= s.mean()
        if np.max(np.abs(step)) < 1e-8:
            break
    return s


@dataclass
class _Pointwise:
    est: Any

    def predict(self, x: FloatArray) -> FloatArray:
        return np.asarray(self.est.predict(x), dtype=np.float64)


@dataclass
class _Pairwise:
    est: Any

    def predict(self, x: FloatArray) -> FloatArray:
        n = x.shape[0]
        i, j = np.where(~np.eye(n, dtype=bool))
        proba = np.asarray(self.est.predict_proba(x[i] - x[j]), dtype=np.float64)
        classes = list(getattr(self.est, "classes_", [0, 1]))
        win = proba[:, classes.index(1)] if 1 in classes else proba[:, -1]
        p = np.full((n, n), 0.5)
        p[i, j] = win
        p = 0.5 * (p + (1.0 - p.T))  # A beats B and B loses to A: one probability
        np.fill_diagonal(p, 0.0)
        return bradley_terry(np.clip(p, 1e-4, 1 - 1e-4))


@dataclass
class TFMRanker(RankerModel):
    """TabICL on the challengers' feature table. ``framing`` is ``pointwise`` or
    ``pairwise``."""

    kind: str = "tabicl"
    framing: str = "pointwise"
    max_context_races: int = 40  # pairwise: the most recent races (~231 pairs each)
    n_estimators: int = 4  # TabICL's internal ensemble (feature/class shuffles)

    def __post_init__(self) -> None:
        if self.framing not in ("pointwise", "pairwise"):
            raise ValueError(f"unknown framing {self.framing!r}")
        if not self.name:
            self.name = f"tabicl_{self.framing}"

    def _estimator(self, task: str) -> Any:
        """TabICL (on Apple silicon: the MPS back end); a regressor or a classifier."""
        import torch
        from tabicl import TabICLClassifier, TabICLRegressor

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        cls = TabICLRegressor if task == "regression" else TabICLClassifier
        return cls(n_estimators=self.n_estimators, device=device, random_state=self.seed)

    def _train(self, rows: Rows) -> Any:
        rows = sorted(rows, key=lambda r: (r[0].season, r[0].round))
        if self.framing == "pointwise":
            x = np.concatenate([sig.x for _, sig, _ in rows])
            y = np.concatenate([_normalised(c, o) for c, _, o in rows])
            est = self._estimator("regression")
            est.fit(x, y)
            return _Pointwise(est)
        xs, ys = [], []
        for c, sig, o in rows[-self.max_context_races :]:
            rel = relevance(c, o)
            i, j = np.where(rel[:, None] != rel[None, :])  # ordered pairs with an outcome
            xs.append(sig.x[i] - sig.x[j])
            ys.append((rel[i] > rel[j]).astype(np.int64))
        est = self._estimator("classification")
        est.fit(np.concatenate(xs), np.concatenate(ys))
        return _Pairwise(est)


def _normalised(ctx: EventContext, outcome: Outcome) -> FloatArray:
    rel = relevance(ctx, outcome).astype(np.float64)
    top = rel.max()
    return rel / top if top > 0 else rel
