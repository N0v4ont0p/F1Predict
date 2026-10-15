"""Paired bootstrap over events (plan §10.4): is model A really better than model B?"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Comparison:
    mean_diff: float
    lo: float
    hi: float
    n: int

    @property
    def significant(self) -> bool:
        """The 95% interval excludes zero."""
        return self.lo > 0 or self.hi < 0


def paired_bootstrap(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
    clusters: NDArray[np.int64] | None = None,
) -> Comparison:
    """Mean of ``a - b`` over events with a percentile bootstrap interval.

    Events are resampled with replacement, keeping each event's pair together, so the
    interval reflects event-to-event variation — the honest unit of evidence in F1. With
    ``clusters`` (e.g. the season of each title snapshot) whole clusters are resampled
    instead, for samples that are correlated within a cluster.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same shape")
    keep = ~(np.isnan(a) | np.isnan(b))
    d = a[keep] - b[keep]
    if d.size == 0:
        return Comparison(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, d.size, size=(n_resamples, d.size))
        means = d[idx].mean(axis=1)
    else:
        _, label = np.unique(np.asarray(clusters)[keep], return_inverse=True)
        k = int(label.max()) + 1
        sums = np.bincount(label, weights=d, minlength=k)
        sizes = np.bincount(label, minlength=k).astype(np.float64)
        pick = rng.integers(0, k, size=(n_resamples, k))
        means = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return Comparison(float(d.mean()), float(lo), float(hi), int(d.size))
