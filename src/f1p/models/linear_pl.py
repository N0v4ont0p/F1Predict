"""Linear Plackett-Luce: ``log s_i = x_i · w`` with ``w`` fitted by penalised maximum likelihood.

The combiner every world-model forecaster uses to turn per-car signals (rating, grid,
qualifying, …) into one calibrated ranking distribution. Only classified finishers enter the
likelihood (retirements are censored), and the gradient is exact, so L-BFGS converges in a few
dozen evaluations even over hundreds of races.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Sample = tuple[FloatArray, NDArray[np.intp]]  # (features n x p, finisher indices best-first)


def _pl_grad(eta: FloatArray) -> tuple[FloatArray, float]:
    z = eta - eta.max()
    e = np.exp(z)
    suffix = np.cumsum(e[::-1])[::-1]
    g = 1.0 - e * np.cumsum(1.0 / suffix)
    return g, float(np.sum(z - np.log(suffix)))


def loglik_and_grad(w: FloatArray, samples: Sequence[Sample]) -> tuple[float, FloatArray]:
    total, grad = 0.0, np.zeros_like(w)
    for x, order in samples:
        if order.size < 2:
            continue
        xo = x[order]
        g, ll = _pl_grad(xo @ w)
        total += ll
        grad += xo.T @ g
    return total, grad


def fit(
    samples: Sequence[Sample],
    *,
    prior: FloatArray,
    l2: float = 1.0,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
) -> FloatArray:
    """MAP weights under a Gaussian prior ``N(prior, 1/l2)`` (prior returned if no data)."""
    usable = [(x, o) for x, o in samples if o.size >= 2]
    if not usable:
        return prior.copy()
    from scipy.optimize import minimize

    def objective(w: FloatArray) -> tuple[float, FloatArray]:
        ll, g = loglik_and_grad(w, usable)
        d = w - prior
        return -ll + 0.5 * l2 * float(d @ d), -g + l2 * d

    res = minimize(objective, prior.copy(), jac=True, method="L-BFGS-B", bounds=bounds)
    return np.asarray(res.x, dtype=np.float64)
